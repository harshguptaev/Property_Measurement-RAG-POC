#!/usr/bin/env python3

"""
Bedrock Claude Sonnet 3.5 helper for extracting house width/height from an
orthographic top image using the prompt in `prompts/getWidthHeightOfHouse.py`.

Usage from other modules:

    from llm import get_required_dimensions_from_llm
    dims = get_required_dimensions_from_llm(lat_str, lon_str)
    # dims -> {"required_width": int, "required_height": int, "image_path": str}

Environment:
- AWS credentials are read from environment (already exported by user)
- Region is us-east-1
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import boto3

from prompts.getWidthHeightOfHouse import PROMPT as WIDTH_HEIGHT_PROMPT
from PIL import Image


# MODEL_ID = "anthropic.claude-3-7-sonnet-20250219-v1:0"
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
REGION = "us-east-1"


def _resolve_image_path(lat: str, lon: str, base_dir: str = "final_data") -> str:
    """
    Resolve the path to the top-down image inside final_data/<lat>_<lon>/.
    Looks for case variants: top.jpg, Top.jpg, top.png.
    """
    folder = Path(base_dir) / f"{lat}_{lon}"
    # Try common names
    candidates = [
        folder / "top.jpg",
        folder / "Top.jpg",
        folder / "top.png",
        folder / "Top.png",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No top image found in {folder}")


def _build_prompt_for_image(image_path: str) -> str:
    """Fill the width/height placeholders in the prompt based on the image."""
    with Image.open(image_path) as img:
        width, height = img.size
        print("Image dimensions: ", width, height)
    prompt_text = WIDTH_HEIGHT_PROMPT.replace("{{image_width}}", str(width)).replace("{{image_height}}", str(height))
    return prompt_text


def _image_format_from_path(image_path: str) -> str:
    ext = Path(image_path).suffix.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "jpeg"
    if ext in ("png",):
        return "png"
    # Fallback
    return "jpeg"


def _call_bedrock_vision(prompt_text: str, image_path: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """
    Call Bedrock Converse API for Claude 3.5 Sonnet with one image + text.
    Returns the assistant text output.
    """
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_format = _image_format_from_path(image_path)

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
                    {"text": prompt_text},
                ],
            }
        ],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": 0.9},
    )

    # The text is typically in response['output']['message']['content'][0]['text']
    try:
        parts = response["output"]["message"]["content"]
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
        return "\n".join([t for t in texts if t])
    except Exception as e:
        raise RuntimeError(f"Unexpected Bedrock response structure: {e}")


def _parse_dimensions_from_text(text: str) -> Dict[str, int]:
    """
    Parse the assistant's response which should be a JSON object with
    width_px and height_px. Returns required_width/required_height ints.
    """
    # Try direct JSON load first
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find first JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end + 1])
        else:
            raise

    width = int(data.get("width_px") or data.get("width") or data.get("required_width") or 0)
    height = int(data.get("height_px") or data.get("height") or data.get("required_height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions in model output: {data}")
    return {"required_width": width, "required_height": height}


def get_required_dimensions_from_llm(lat: str, lon: str, base_dir: str = "final_data") -> Dict[str, int]:
    """
    Find final_data/<lat>_<lon>/top.jpg, build prompt with image dimensions, call
    Claude Sonnet 3.5 (Bedrock) with image + prompt, and return required dims.

    Returns: {"required_width": int, "required_height": int, "image_path": str}
    """
    image_path = _resolve_image_path(lat, lon, base_dir)
    prompt_text = _build_prompt_for_image(image_path)
    llm_text = _call_bedrock_vision(prompt_text, image_path)
    dims = _parse_dimensions_from_text(llm_text)
    dims["image_path"] = image_path
    return dims

