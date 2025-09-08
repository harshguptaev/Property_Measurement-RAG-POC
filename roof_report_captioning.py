#!/usr/bin/env python3
"""Utility script to batch analyze roof report images using the comprehensive Gemini prompt.

Usage:
  python roof_report_captioning.py --dir extracted_images/report_44995431 \
      --pattern "*.png" --output analysis_report_44995431.json

Requires GEMINI_API_KEY in environment.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List

from src.gemini_client import GeminiVisionClient, is_gemini_available


def collect_images(directory: Path, pattern: str) -> List[Path]:
    return sorted(directory.rglob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Analyze roof report images with Gemini")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--pattern", type=str, default="*.png", help="Glob pattern (default: *.png)")
    parser.add_argument("--output", type=str, default="roof_image_analysis.json", help="Output JSON file")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit of images to process")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    img_dir = Path(args.dir)
    if not img_dir.exists():
        raise SystemExit(f"Directory not found: {img_dir}")

    images = collect_images(img_dir, args.pattern)
    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        raise SystemExit("No images found with given pattern.")

    if not is_gemini_available():
        raise SystemExit("Gemini API not available. Ensure google-generativeai is installed and GEMINI_API_KEY is set.")

    client = GeminiVisionClient()
    results = client.batch_analyze_roof_report(images)

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Wrote analysis for {len(results)} images to {out_path}")


if __name__ == "__main__":
    main()
