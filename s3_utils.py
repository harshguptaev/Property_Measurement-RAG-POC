#!/usr/bin/env python3

import boto3
import os
from urllib.parse import urlparse
from botocore.exceptions import ClientError

class S3Client:
    def __init__(self, region="us-east-2", session=None):
        """
        Initialize S3 client with AWS credentials.
        If session is provided, use it; otherwise use default boto3 session.
        """
        if session:
            self.client = session.client('s3', region_name=region)
        else:
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

    def upload_all_files_to_s3(self, local_dir: str, bucket: str = None, s3_prefix: str = "", excluded_files: set = None, recursive: bool = True):
        """
        Upload all files from a local directory to S3, optionally recursively.

        Args:
            local_dir: Local directory path to upload files from
            bucket: S3 bucket name (defaults to evtech-us-east-2-pg-test-sunsitecomplete)
            s3_prefix: S3 prefix/key path (e.g., 'property-data/LatLongData/folder/')
            excluded_files: Set of filenames to exclude from upload
            recursive: Whether to upload files from subdirectories recursively
        """
        if bucket is None:
            bucket = "evtech-us-east-2-pg-test-sunsitecomplete"

        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Local folder not found: {local_dir}")

        if excluded_files is None:
            excluded_files = set()

        # Ensure s3_prefix ends with '/'
        if s3_prefix and not s3_prefix.endswith('/'):
            s3_prefix += '/'

        # Create a folder marker (optional in S3, but harmless)
        try:
            self.client.put_object(Bucket=bucket, Key=s3_prefix)
        except Exception:
            pass

        uploaded_count = 0

        if recursive:
            # Walk through all files recursively
            for root_dir, dirs, files in os.walk(local_dir):
                # Calculate relative path from local_dir
                rel_path = os.path.relpath(root_dir, local_dir)
                if rel_path == '.':
                    current_prefix = s3_prefix
                else:
                    current_prefix = s3_prefix + rel_path.replace(os.sep, '/') + '/'

                # Create folder marker for subdirectories
                if rel_path != '.':
                    try:
                        self.client.put_object(Bucket=bucket, Key=current_prefix)
                    except Exception:
                        pass

                for filename in files:
                    if filename in excluded_files or filename.startswith('.'):
                        continue

                    local_path = os.path.join(root_dir, filename)
                    key = current_prefix + filename
                    s3_url = f"s3://{bucket}/{key}"
                    print(f"Uploading {local_path} -> {s3_url}")
                    self.upload_file(local_path, s3_url)
                    uploaded_count += 1
        else:
            # Original non-recursive behavior
            for filename in os.listdir(local_dir):
                local_path = os.path.join(local_dir, filename)
                if not os.path.isfile(local_path):
                    continue
                if filename in excluded_files or filename.startswith('.'):
                    continue

                key = s3_prefix + filename
                s3_url = f"s3://{bucket}/{key}"
                print(f"Uploading {local_path} -> {s3_url}")
                self.upload_file(local_path, s3_url)
                uploaded_count += 1

        print(f"✅ Uploaded {uploaded_count} files from {local_dir} to s3://{bucket}/{s3_prefix}")
        return uploaded_count

    def upload_cropped_images_to_s3(self, latitude: float, longitude: float):
        """
        Upload all files from local final_data/<lat>_<lon>/ to
        s3://evtech-us-east-2-pg-test-sunsitecomplete/property-data/LatLongData/<lat>_<lon>/
        except 'pictometry_response.json'.
        """
        lat_lon_folder = f"{latitude}_{longitude}"
        local_dir = os.path.join(os.getcwd(), "final_data", lat_lon_folder)
        bucket = "evtech-us-east-2-pg-test-sunsitecomplete"
        base_prefix = f"property-data/LatLongData/{lat_lon_folder}/"

        excluded_files = {"pictometry_response.json"}
        return self.upload_all_files_to_s3(local_dir, bucket, base_prefix, excluded_files)



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