import boto3
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class S3UploadClient:
    def __init__(self, bucket_name: str = "evtech-us-east-2-pg-test-sunsitecomplete", region: str = "us-east-2",
                 aws_access_key_id: str = None, aws_secret_access_key: str = None, aws_session_token: str = None):
        self.bucket_name = bucket_name
        self.region = region

        # Create boto3 client with provided credentials
        client_kwargs = {'region_name': region}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs['aws_access_key_id'] = aws_access_key_id
            client_kwargs['aws_secret_access_key'] = aws_secret_access_key
            if aws_session_token:
                client_kwargs['aws_session_token'] = aws_session_token

        self.s3_client = boto3.client('s3', **client_kwargs)

    def upload_file(self, local_path: str, s3_key: str) -> str:
        """Upload a single file to S3 and return its URL"""
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"Uploaded {local_path} to {s3_key}")
            return url
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return ""

    def upload_directory(self, local_dir: str, s3_prefix: str) -> List[str]:
        """Upload all files from a local directory to S3 with the given prefix"""
        uploaded_urls = []
        local_path = Path(local_dir)

        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                # Calculate relative path from local_dir
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}"

                url = self.upload_file(str(file_path), s3_key)
                if url:
                    uploaded_urls.append(url)

        return uploaded_urls

    def upload_extracted_images(self) -> Dict[str, List[str]]:
        """Upload all extracted images and return URLs organized by report"""
        project_root = Path(__file__).parent.parent.parent
        extracted_images_dir = project_root / "extracted_images"
        s3_prefix = "property-data/extracted_images"

        uploaded_urls = self.upload_directory(str(extracted_images_dir), s3_prefix)

        # Organize URLs by report
        report_urls = {}
        for url in uploaded_urls:
            # Extract report ID from URL
            parts = url.split('/')
            if len(parts) >= 2:
                report_id = parts[-2]  # Second to last part should be report folder
                if report_id.startswith('report_'):
                    if report_id not in report_urls:
                        report_urls[report_id] = []
                    report_urls[report_id].append(url)

        return report_urls

    def upload_final_chunks(self) -> Dict[str, str]:
        """Upload all final chunk JSON files and return URLs"""
        project_root = Path(__file__).parent.parent.parent
        final_chunks_dir = project_root / "Final_Chunks"
        s3_prefix = "property-data/final_chunks"

        uploaded_urls = self.upload_directory(str(final_chunks_dir), s3_prefix)

        # Organize URLs by report
        chunk_urls = {}
        for url in uploaded_urls:
            # Extract report ID from URL
            filename = url.split('/')[-1]
            if filename.endswith('.json'):
                report_id = filename.replace('.json', '').replace('report_', 'report_')
                chunk_urls[report_id] = url

        return chunk_urls

    def _get_image_mappings(self) -> Dict[str, Dict[str, str]]:
        """Get mapping of image filenames to chunk metadata"""
        return {
            "Lengths_Diagram.png": {
                "section": "Lengths Diagram",
                "description": "Diagram showing roof lengths and measurements"
            },
            "Pitch_Diagram.png": {
                "section": "Pitch Diagram",
                "description": "Diagram showing roof pitch measurements"
            },
            "Area_Diagram.png": {
                "section": "Area Diagram",
                "description": "Diagram showing roof area calculations"
            },
            "Top_View.png": {
                "section": "Top View",
                "description": "Top view diagram of the roof structure"
            },
            "North_Side.png": {
                "section": "North Side View",
                "description": "North side view of the property"
            },
            "South_Side.png": {
                "section": "South Side View",
                "description": "South side view of the property"
            },
            "East_Side.png": {
                "section": "East Side View",
                "description": "East side view of the property"
            },
            "West_Side.png": {
                "section": "West Side View",
                "description": "West side view of the property"
            }
        }

    def _get_next_chunk_number(self, chunks: List[Dict[str, Any]]) -> int:
        """Find the next available chunk number"""
        existing_chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        max_chunk_num = 0
        for chunk_id in existing_chunk_ids:
            try:
                num = int(chunk_id.split('_')[-1])
                max_chunk_num = max(max_chunk_num, num)
            except (ValueError, IndexError):
                continue
        return max_chunk_num

    def _extract_report_number(self, report_id: str) -> str:
        """Extract numeric part from report_id (e.g., 'report_32249000' -> '32249000')"""
        return report_id.split('_')[-1] if '_' in report_id else report_id

    def _update_single_report_chunks(self, report_id: str, urls: List[str]) -> None:
        """Update chunks for a single report"""
        project_root = Path(__file__).parent.parent.parent
        final_chunks_dir = project_root / "Final_Chunks"
        chunk_file = final_chunks_dir / f"{report_id}.json"

        if not chunk_file.exists():
            logger.warning(f"Chunk file not found: {chunk_file}")
            return

        # Load existing chunks
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        image_mappings = self._get_image_mappings()
        max_chunk_num = self._get_next_chunk_number(chunks)
        report_number = self._extract_report_number(report_id)

        # Add image chunks
        for url in urls:
            filename = url.split('/')[-1]
            if filename in image_mappings:
                mapping = image_mappings[filename]
                max_chunk_num += 1

                image_chunk = {
                    "chunk_id": f"{report_number}_chunk_{max_chunk_num}",
                    "section": mapping["section"],
                    "type": "image",
                    "data": {
                        "description": mapping["description"],
                        "image_file": url
                    }
                }
                chunks.append(image_chunk)
                logger.info(f"Added image chunk for {filename} to {report_id}")

        # Save updated chunks
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    def update_chunks_with_image_urls(self, image_urls: Dict[str, List[str]]) -> None:
        """Update Final_Chunks JSON files with S3 image URLs"""
        for report_id, urls in image_urls.items():
            self._update_single_report_chunks(report_id, urls)

    def upload_all(self) -> Dict[str, Dict]:
        """Upload both extracted images and final chunks, return all URLs"""
        logger.info("Starting upload of extracted images...")
        image_urls = self.upload_extracted_images()

        logger.info("Starting upload of final chunks...")
        chunk_urls = self.upload_final_chunks()

        logger.info("Updating Final_Chunks with S3 image URLs...")
        self.update_chunks_with_image_urls(image_urls)

        return {
            "extracted_images": image_urls,
            "final_chunks": chunk_urls
        }

def main():
    """Example usage"""
    # Get AWS credentials from environment variables
    aws_access_key_id = os.getenv('S3_AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('S3_AWS_SECRET_ACCESS_KEY')
    aws_session_token = os.getenv('S3_AWS_SESSION_TOKEN')

    client = S3UploadClient(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )

    print("Uploading all data to S3...")
    result = client.upload_all()

    print("\nExtracted Images URLs:")
    for report, urls in result["extracted_images"].items():
        print(f"{report}: {len(urls)} images")
        for url in urls[:2]:  # Show first 2 URLs per report
            print(f"  {url}")
        if len(urls) > 2:
            print(f"  ... and {len(urls) - 2} more")

    print("\nFinal Chunks URLs:")
    for report, url in result["final_chunks"].items():
        print(f"{report}: {url}")

    print("\nFinal_Chunks JSON files have been updated with S3 image URLs.")

if __name__ == "__main__":
    main()
