import os
import json
import base64
from pathlib import Path
from typing import List

import boto3
from pdf2image import convert_from_path
import io

# Configuration
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
REGION_NAME = "us-east-1"

def invoke_bedrock_llm(prompt: str, image_bytes_list: List[bytes] = None, max_tokens: int = 8000) -> str:
    """
    Invoke AWS Bedrock LLM with the given prompt and optional image content

    Args:
        prompt: The prompt to send to the LLM
        image_bytes_list: Optional list of image bytes
        max_tokens: Maximum tokens in response

    Returns:
        LLM response as string
    """
    bedrock_client = boto3.client('bedrock-runtime', region_name=REGION_NAME)

    try:
        # Prepare message content
        message_content = [{"type": "text", "text": prompt}]

        # Add images if provided
        if image_bytes_list:
            for i, image_bytes in enumerate(image_bytes_list):
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_base64
                    }
                })

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        }

        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body)
        )

        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']

    except Exception as e:
        print(f"Error invoking Bedrock LLM: {e}")
        raise

def convert_pdf_to_images(pdf_path: str) -> List[bytes]:
    """
    Convert PDF pages to images

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of image bytes for each page
    """
    print(f"📄 Converting PDF to images: {pdf_path}")

    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=200)

    image_bytes_list = []
    for i, image in enumerate(images):
        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_bytes = img_buffer.getvalue()
        image_bytes_list.append(image_bytes)
        print(f"  Page {i+1}: {len(image_bytes)} bytes")

    return image_bytes_list

def process_pdf(pdf_path: str) -> str:
    """
    Process a PDF file and get LLM response

    Args:
        pdf_path: Path to the PDF file

    Returns:
        LLM response as string
    """
    print(f"🚀 Processing PDF: {pdf_path}")

    # Convert PDF to images
    image_bytes_list = convert_pdf_to_images(pdf_path)

    # Create the detailed prompt for specific sections
    prompt = f"""You are a document-to-JSON transformation agent specializing in roofing reports.
Extract roofing report data and return as JSON array with this exact format:

First object should be property information:
{{
  "property_id": "PROP_[REPORT_ID]",
  "address": "[FULL_ADDRESS]",
  "latitude": [LATITUDE],
  "longitude": [LONGITUDE],
  "owner": "[OWNER_NAME_OR_EMPTY]"
}}

Then create chunks for these specific sections ONLY if the data is present:

1. "House Measurements" chunk:
{{
  "chunk_id": "C001",
  "property_id": "PROP_[REPORT_ID]",
  "section": "House Measurements",
  "type": "text",
  "data": {{
    "number_of_stories": "[ACTUAL_VALUE_OR_LEAVE_EMPTY]",
    "total_roof_facets": [ACTUAL_NUMBER_OR_LEAVE_EMPTY],
    "structure_complexity": "[Simple/Normal/Complex_OR_LEAVE_EMPTY]",
    "estimated_attic": "[AREA_VALUE_OR_LEAVE_EMPTY]",
    "total_roof_obstructions": [ACTUAL_NUMBER_OR_LEAVE_EMPTY],
    "number_of_structures" : 1
  }}
}}

2. "Roof Measurements - All Structures" chunk:
{{
  "chunk_id": "C002",
  "property_id": "PROP_[REPORT_ID]",
  "section": "Roof Measurements - All Structures",
  "type": "text",
  "data": {{
    "structure": "All Structures",
    "total_area": "[AREA_VALUE_OR_LEAVE_EMPTY]",
    "total_roof_facets": [ACTUAL_NUMBER_OR_LEAVE_EMPTY],
    "predominant_pitch": "[PITCH_VALUE_OR_LEAVE_EMPTY]",
    "ridges": "[LENGTH_VALUE_OR_LEAVE_EMPTY]",
    "hips": "[LENGTH_VALUE_OR_LEAVE_EMPTY]",
    "valleys": "[LENGTH_VALUE_OR_LEAVE_EMPTY]",
    "rakes": "[LENGTH_VALUE_OR_LEAVE_EMPTY]",
    "eaves_starter": "[LENGTH_VALUE_OR_LEAVE_EMPTY]",
    "drip_edge": "[LENGTH_VALUE_OR_LEAVE_EMPTY]",
    "flashing": "[LENGTH_VALUE_OR_LEAVE_EMPTY]",
    "step_flashing": "[LENGTH_VALUE_OR_LEAVE_EMPTY]",
    "parapet_walls": "[LENGTH_VALUE_OR_LEAVE_EMPTY]",
    "roof_obstructions_perimeter": "[LENGTH_VALUE_OR_LEAVE_EMPTY]",
    "roof_obstructions_area": "[AREA_VALUE_OR_LEAVE_EMPTY]",
    "net_roof_area": "[AREA_VALUE_OR_LEAVE_EMPTY]"
  }}
}}

CRITICAL REQUIREMENTS:
- Extract property information first (address, coordinates, owner)
- ONLY create the chunks shown above if that specific data is present in the PDF
- For any field where data is not available, leave it empty but keep the field
- Use the exact chunk_ids "C001" and "C002" as shown
- Return ONLY a valid JSON array with the objects shown above
- Do not add any extra chunks or sections beyond these two
- If data for a section is completely missing, do not include that chunk at all

Please analyze the attached PDF document pages and extract only the property information and these two specific measurement sections."""

    # Get LLM response with image content
    print("🤖 Sending PDF pages as images to LLM for processing...")
    response = invoke_bedrock_llm(prompt, image_bytes_list)

    print(f"✅ LLM processing complete. Response length: {len(response)} characters")
    return response

def save_response(response: str, report_id: str):
    """
    Save LLM response to file

    Args:
        response: LLM response text
        report_id: Report ID
    """
    # Save to input_data/<reportid>/<reportid>.json
    output_path = Path("input_data") / report_id / f"{report_id}.json"

    # Try to parse as JSON first to validate
    try:
        parsed_response = json.loads(response)
        # If successful, save as formatted JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_response, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved formatted JSON response: {output_path}")
    except json.JSONDecodeError:
        # If not valid JSON, save as text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response)
        print(f"💾 Saved text response: {output_path}")

def export_chunk_data_for_report(report_id: str):
    """Main function to export chunk data for a report"""
    print(f"Exporting chunk data for report {report_id}")

    # Load the pdf file from input_data/Report-<report_id>/<report_id>.pdf
    pdf_path = os.path.join("input_data", f"{report_id}", f"{report_id}.pdf")
    if not os.path.exists(pdf_path):
        print(f"PDF file not found for report {report_id}: {pdf_path}")
        return

    print(f"Found PDF file: {pdf_path}")

    # Process the PDF with LLM
    try:
        response = process_pdf(pdf_path)

        # Save the response
        save_response(response, report_id)

        print(f"✅ Successfully processed report {report_id}")

    except Exception as e:
        print(f"❌ Error processing report {report_id}: {e}")