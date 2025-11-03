#!/usr/bin/env python3

import boto3
import os
from urllib.parse import urlparse
from botocore.exceptions import ClientError

class S3Client:
    def __init__(self, region="us-east-2"):
        """
        Initialize S3 client with AWS credentials from environment variables
        """
        self.client = boto3.client('s3', region_name=region)
        self.region = region

    def parse_s3_url(self, s3_url):
        """
        Parse S3 URL to extract bucket and key
        Returns: (bucket_name, key)
        """
        parsed = urlparse(s3_url)
        if not parsed.scheme == 's3':
            raise ValueError(f"Invalid S3 URL format: {s3_url}")

        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key

    def extract_folder_name(self, s3_url):
        """
        Extract the folder name from S3 URL.
        For s3://bucket/path/to/folder/file.png, returns 'folder'
        """
        parsed = urlparse(s3_url)
        path_parts = parsed.path.strip('/').split('/')
        # Return the second-to-last part (index -2)
        if len(path_parts) >= 2:
            return path_parts[-2]
        else:
            return 'default'

    def download_image(self, s3_url, local_path):
        """
        Download image from S3 URL to local path
        """
        try:
            bucket, key = self.parse_s3_url(s3_url)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            print(f"Downloading image from {s3_url} to {local_path}")
            self.client.download_file(bucket, key, local_path)
            print(f"Image downloaded successfully to {local_path}")

            return True

        except Exception as e:
            print(f"Error downloading image from S3: {str(e)}")
            return False

    def get_image_filename(self, s3_url):
        """
        Extract filename from S3 URL
        """
        parsed = urlparse(s3_url)
        path_parts = parsed.path.strip('/').split('/')
        return path_parts[-1] if path_parts else 'image.png'

    def get_parent_prefix(self, s3_url):
        """
        Return the parent key prefix (folder) for the given S3 URL's key.
        Example: s3://bucket/a/b/file.png -> 'a/b'
        """
        _, key = self.parse_s3_url(s3_url)
        parts = key.split('/')
        if len(parts) <= 1:
            return ''
        return '/'.join(parts[:-1])

    def object_exists(self, bucket: str, key: str) -> bool:
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            # 404 Not Found
            if e.response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 404 or e.response.get('Error', {}).get('Code') in {"404", "NoSuchKey"}:
                return False
            # Other errors propagate
            return False

    def download_sibling_if_exists(self, s3_url: str, sibling_filename: str, local_dir: str) -> bool:
        """
        If an object named `sibling_filename` exists in the same S3 'folder' as `s3_url`,
        download it to `local_dir` and return True. Otherwise return False.
        """
        bucket, _key = self.parse_s3_url(s3_url)
        parent = self.get_parent_prefix(s3_url)
        sibling_key = f"{parent}/{sibling_filename}" if parent else sibling_filename
        if self.object_exists(bucket, sibling_key):
            local_path = os.path.join(local_dir, sibling_filename)
            os.makedirs(local_dir, exist_ok=True)
            print(f"Downloading sibling file s3://{bucket}/{sibling_key} -> {local_path}")
            self.client.download_file(bucket, sibling_key, local_path)
            return True
        else:
            print(f"Sibling file not found in S3: s3://{bucket}/{sibling_key}")
            return False

    def upload_file(self, local_path: str, s3_url: str) -> None:
        """
        Upload a single local file to an S3 URL (s3://bucket/key).
        """
        bucket, key = self.parse_s3_url(s3_url)
        self.client.upload_file(local_path, bucket, key)

    def upload_cropped_images_to_s3(self, latitude: float, longitude: float):
        """
        Upload all files from local final_data/<lat>_<lon>/ to
        s3://evtech-us-east-2-pg-test-sunsitecomplete/property-data/LatLongData/<lat>_<lon>/
        except 'pictometry_response.json'.
        """
        lat_lon_folder = f"{latitude}_{longitude}"
        local_dir = os.path.join(os.getcwd(), "final_data", lat_lon_folder)
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Local folder not found: {local_dir}")

        bucket = "evtech-us-east-2-pg-test-sunsitecomplete"
        base_prefix = f"property-data/LatLongData/{lat_lon_folder}/"

        # Create a folder marker (optional in S3, but harmless)
        try:
            self.client.put_object(Bucket=bucket, Key=base_prefix)
        except Exception:
            pass

        excluded = {"pictometry_response.json"}

        for filename in os.listdir(local_dir):
            local_path = os.path.join(local_dir, filename)
            if not os.path.isfile(local_path):
                continue
            if filename in excluded or filename.startswith('.'):
                continue

            key = base_prefix + filename
            s3_url = f"s3://{bucket}/{key}"
            print(f"Uploading {local_path} -> {s3_url}")
            self.upload_file(local_path, s3_url)



def upload_ddd_to_s3(report_id: str):
    """
    Upload DDD to s3 bucket
    """
    local_path = os.path.join(os.getcwd(), "input_data", report_id, "DDD.png")
    processed_path = os.path.join(os.getcwd(), "input_data", report_id, "processed_DDD.png")
    if not os.path.exists(processed_path):
        print(f"Processed DDD file not found for report {report_id}")
        return
    if not os.path.exists(local_path):
        print(f"DDD file not found for report {report_id}")
        return
    s3_url = f"s3://evtech-us-east-2-pg-test-sunsitecomplete/property-data/Level3Data/{report_id}/DDD.png"
    s3_processed_url = f"s3://evtech-us-east-2-pg-test-sunsitecomplete/property-data/Level3Data/{report_id}/processed_DDD.png"
    s3_client = S3Client()
    s3_client.upload_file(local_path, s3_url)
    s3_client.upload_file(processed_path, s3_processed_url)
    return s3_processed_url