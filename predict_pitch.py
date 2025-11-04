#!/usr/bin/env python3

import json
import os
import uuid
from typing import Dict

import boto3


def _pitch_endpoint_client(region: str = "us-east-2"):
    # Use S3-specific AWS credentials for SageMaker runtime
    s3_session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_S3"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_S3"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN_S3"),
        region_name=os.getenv("AWS_REGION_S3", region),
    )
    return s3_session.client("sagemaker-runtime")


def save_pitch_data(latitude: float, longitude: float) -> Dict[str, dict]:
    print("Saving pitch data for latitude: ", latitude, " and longitude: ", longitude)
    predominant_pitch = 0.0
    # Output directory
    lat_lon_folder = f"{latitude}_{longitude}"
    base_dir = os.path.join(os.getcwd(), "final_data", lat_lon_folder)
    os.makedirs(base_dir, exist_ok=True)

    # Bucket and S3 prefix
    bucket = "evtech-us-east-2-pg-test-sunsitecomplete"
    prefix = f"property-data/LatLongData/{lat_lon_folder}"

    # Files to process
    files = [
        "East_cropped.png",
        "West_cropped.png",
        "North_cropped.png",
        "South_cropped.png",
    ]

    # SageMaker endpoint configuration
    endpoint_name = "app-test-2x0-ep-pitch"
    client = _pitch_endpoint_client(region="us-east-2")

    aggregated: Dict[str, dict] = {}

    for filename in files:
        s3_uri = f"s3://{bucket}/{prefix}/{filename}"
        payload = {
            "image_uri": s3_uri,
            "jobId": str(uuid.uuid4()),
        }
        try:
            response = client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=json.dumps(payload),
                ContentType="application/json",
            )
            result = json.loads(response["Body"].read().decode("utf-8"))
            pitch = result["predictions"][0]["value"]
            predominant_pitch += pitch
            print(result)
        except Exception as exc:  # keep going for other images
            result = {"error": str(exc), "image_uri": s3_uri}
            print(result)

        key = filename.split("_")[0].lower()  # east/west/north/south
        aggregated[key] = result

    # Save one combined JSON
    out_path = os.path.join(base_dir, "pitch_response.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print("Pitch response saved to:", out_path)

    return predominant_pitch/4