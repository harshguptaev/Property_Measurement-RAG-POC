#!/usr/bin/env python3

import json
import os
import uuid
from typing import Dict, Tuple

import boto3


def _letr_endpoint_client(region: str = "us-east-2"):
    return boto3.client("sagemaker-runtime", region_name=region)


def save_letr_data(latitude: float, longitude: float) -> Tuple[str, str]:
    print("Saving LETR data for latitude: ", latitude, " and longitude: ", longitude)
    # Output directory
    lat_lon_folder = f"{latitude}_{longitude}"
    base_dir = os.path.join(os.getcwd(), "final_data", lat_lon_folder)
    os.makedirs(base_dir, exist_ok=True)

    # Bucket and S3 prefix
    bucket = "evtech-us-east-2-pg-test-sunsitecomplete"
    prefix = f"property-data/LatLongData/{lat_lon_folder}"

    # Files to process
    files = [
        "Top_cropped.jpg"
    ]

    # SageMaker endpoint configuration
    endpoint_name = "app-test-2x0-ep-letr-inference-container"
    client = _letr_endpoint_client(region="us-east-2")

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
            outline_result = json.loads(response["Body"].read().decode("utf-8"))
            # save the outline_result to the base_dir
            with open(os.path.join(base_dir, "top_outline.json"), "w") as f:
                json.dump(outline_result, f, indent=2)
        except Exception as exc:  # keep going for other images
            result = {"error": str(exc), "image_uri": s3_uri}
            print(result)


    return bucket, prefix