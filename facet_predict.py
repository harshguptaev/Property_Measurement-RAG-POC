#!/usr/bin/env python3

import os
import json
import boto3

from s3_utils import S3Client
from facets_visualizer import (
    draw_facets_canvas,
    overlay_facets_on_image,
    overlay_facets_with_lengths_on_image,
)
from area_calculator import save_area_json


def run_facet_predict(lat: float, lon: float, s3_url: str, region: str = "us-east-2") -> None:
    """
    Run the facet SageMaker inference pipeline and generate artifacts under
    final_data/<lat>_<lon>/.
    """
    # Use S3-specific AWS credentials for both S3 and SageMaker
    s3_session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_S3"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_S3"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN_S3"),
        region_name=os.getenv("AWS_REGION_S3", region),
    )
    s3_client = S3Client(region=region, session=s3_session)
    num_facets = 0
    # Create output directory structure
    output_dir = "final_data"
    folder_name = f"{lat}_{lon}"
    target_dir = os.path.join(output_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    print(f"Created/using output directory: {target_dir}")

    # SageMaker configuration with S3 session
    endpoint_name = "app-test-2x0-ep-facet"
    sagemaker_runtime = s3_session.client("sagemaker-runtime")

    # Prepare the payload
    payload = {
        "image_uri": s3_url,
        "jobId": "cedf7e94-e64f-432a-bc0c-994b42e895aa",
        "simplify_threshold": 0.005,
    }

    print("Calling SageMaker endpoint...")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=json.dumps(payload),
            ContentType="application/json",
        )

        # Parse the response
        result = json.loads(response["Body"].read().decode("utf-8"))
        score_threshold = 0.70
        for prediction in result["predictions"]:
            if prediction["score"] >= score_threshold:
                num_facets += 1
        # Save response to JSON file
        response_json_path = os.path.join(target_dir, "response.json")
        with open(response_json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Response saved to: {response_json_path}")

        # Compute facet-wise and total areas -> area.json
        try:
            area_json_path = os.path.join(target_dir, "area.json")
            gserve_json_path = os.path.join(target_dir, "gserve_response.json")
            save_area_json(
                response_json_path,
                area_json_path,
                gserve_json_path=gserve_json_path,
                score_threshold=0.70,
            )
            print(f"Area results saved to: {area_json_path}")
        except Exception as area_err:
            print(f"Warning: Failed to compute areas: {area_err}")

        # Download the image from S3 to the same directory (optional if already local)
        image_filename = s3_client.get_image_filename(s3_url)
        image_path = os.path.join(target_dir, image_filename)
        s3_client.download_image(s3_url, image_path)

        # Try to download sibling pdf.pdf (if present)
        try:
            s3_client.download_sibling_if_exists(s3_url, "pdf.pdf", target_dir)
        except Exception as pdf_err:
            print(f"Warning: Failed to download PDF: {pdf_err}")

        # Visualize facets
        try:
            # 1) Draw facets on blank canvas using JSON image size -> response.png
            response_png = os.path.join(target_dir, "response.png")
            draw_facets_canvas(response_json_path, response_png)

            # 2) Overlay facets on top of the downloaded image -> response-top.png
            response_top_png = os.path.join(target_dir, "response-top.png")
            overlay_facets_on_image(image_path, response_json_path, response_top_png)
            print(f"Visualization saved: {response_png} and {response_top_png}")

            # 3) Overlay facets with length labels using gserve_response.json -> response-top-lengths.png
            gserve_json_path = os.path.join(target_dir, "gserve_response.json")
            response_top_lengths_png = os.path.join(target_dir, "response-top-lengths.png")
            overlay_facets_with_lengths_on_image(
                image_path=image_path,
                response_json_path=response_json_path,
                gserve_json_path=gserve_json_path,
                output_path=response_top_lengths_png,
                outline_width=2,
            )
            print(f"Length-labeled overlay saved: {response_top_lengths_png}")
        except Exception as viz_err:
            print(f"Warning: Failed to generate visualizations: {viz_err}")

    except Exception as e:
        print(f"Error calling SageMaker: {str(e)}")
        # Save error to file as well
        error_data = {"error": str(e)}
        response_json_path = os.path.join(target_dir, "response.json")
        with open(response_json_path, "w") as f:
            json.dump(error_data, f, indent=2)
        print(f"Error saved to: {response_json_path}")

    return num_facets


