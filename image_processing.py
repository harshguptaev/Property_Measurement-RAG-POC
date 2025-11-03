#!/usr/bin/env python3
"""
Image Preprocessor Module
Handles image preprocessing: grayscale conversion and size normalization
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple
import boto3
from botocore.exceptions import ClientError

# Configuration
TARGET_SIZE = (512, 512)
PROCESSED_DIR = "input_data"  # Using input_data directory like other files

# S3 Configuration
S3_BUCKET = "evtech-us-east-2-pg-test-sunsitecomplete"
S3_REGION = "us-east-2"
S3_BASE_PATH = "property-data/source_data"

def create_processed_dir():
    """Create processed directory if it doesn't exist"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_image(report_id: str) -> Optional[str]:
    """
    Preprocess a roof image: convert to grayscale and normalize size

    Args:
        input_path: Path to the input image file
        report_id: Report ID to organize files

    Returns:
        Path to the processed image file if successful, None otherwise
    """
    input_path = os.path.join("input_data", report_id, "DDD.png")
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Input file does not exist: {input_path}")
            return None

        print(f"Preprocessing image: {input_path}")

        # Load image
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not load image: {input_path}")
            return None

        # Apply preprocessing
        processed_img = preprocess_roof_image(img)

        # Create report-specific directory
        report_dir = os.path.join(PROCESSED_DIR, report_id)
        os.makedirs(report_dir, exist_ok=True)

        # Save with specific name "processed_ddd"
        output_filename = "processed_DDD.png"
        output_path = os.path.join(report_dir, output_filename)

        # Save processed image
        success = cv2.imwrite(output_path, processed_img)
        if success:
            print(f"Successfully processed: {output_path}")

            # Upload to S3
            s3_key = f"{S3_BASE_PATH}/processed/{report_id}/{output_filename}"
            if upload_to_s3(output_path, s3_key):
                print(f"Successfully uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
            else:
                print(f"Failed to upload to S3: {s3_key}")

            return output_path
        else:
            print(f"Failed to save processed image: {output_path}")
            return None

    except Exception as e:
        print(f"Error preprocessing {input_path}: {str(e)}")
        return None

def preprocess_roof_image(img: np.ndarray) -> np.ndarray:
    """
    Core preprocessing function: convert to grayscale and normalize size

    Args:
        img: Input image as numpy array

    Returns:
        Processed image as numpy array
    """
    # 1️⃣ Convert to grayscale
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()

    # 2️⃣ Normalize size to target dimensions
    resized_img = cv2.resize(gray_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # 3️⃣ Normalize pixel values to 0-1 range (optional but good for ML)
    normalized_img = resized_img.astype(np.float32) / 255.0

    # Convert back to 0-255 range for saving
    final_img = (normalized_img * 255).astype(np.uint8)

    return final_img

def upload_to_s3(local_file_path: str, s3_key: str) -> bool:
    """
    Upload a file to S3

    Args:
        local_file_path: Path to local file
        s3_key: S3 key for the file

    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client = boto3.client('s3', region_name=S3_REGION)
        print(f"Uploading {local_file_path} to s3://{S3_BUCKET}/{s3_key}")
        s3_client.upload_file(local_file_path, S3_BUCKET, s3_key)
        print(f"Successfully uploaded to S3: {s3_key}")
        return True
    except ClientError as e:
        print(f"S3 upload failed for {s3_key}: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error uploading {s3_key}: {str(e)}")
        return False
