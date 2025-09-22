import logging
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Iterable

import boto3
from botocore.exceptions import ClientError
from PIL import Image

from .poc_image_similarity import _embed_image_with_bedrock, image_file_to_base64


logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Resolve repository root from this file location."""
    # .../Property_Measurement-RAG-POC/src/image_similarity_poc/s3_client.py -> root is parents[2]
    return Path(__file__).resolve().parents[2]


def get_s3_client(region_name: str = "us-east-2"):
    """Create and return a boto3 S3 client for the given region."""
    # Relies on AWS creds from environment/instance profile
    return boto3.client("s3", region_name=region_name)


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Split an S3 URI into (bucket, key). Raises ValueError on invalid input."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI (must start with s3://): {s3_uri}")
    without_scheme = s3_uri[5:]
    parts = without_scheme.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid S3 URI (missing bucket or key): {s3_uri}")
    bucket, key = parts[0], parts[1]
    return bucket, key


def download_s3_to_path(s3_uri: str, destination_path: Path, *, region_name: str = "us-east-2") -> bool:
    """
    Download a single S3 object (by s3_uri) to a specific local file path.
    Returns True on success, False otherwise.
    """
    bucket, key = parse_s3_uri(s3_uri)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = get_s3_client(region_name=region_name)
    try:
        logger.info("Downloading %s to %s", s3_uri, str(destination_path))
        s3.download_file(bucket, key, str(destination_path))
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "404":
            logger.error("Object not found: %s", s3_uri)
        else:
            logger.error("Failed to download %s: %s", s3_uri, e)
        return False
    except Exception as e:
        logger.error("Unexpected error downloading %s: %s", s3_uri, e)
        return False


def download_report_backgroundImage(
    report_id: str,
    *,
    bucket: str = "evtech-us-east-2-pg-prod-sunsitecomplete",
    region_name: str = "us-east-2",
    root_output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Download the backgroundImage.jpg for a given report ID and save to
    test_data/{report_id}/backgroundImage.jpg under repo root (or root_output_dir if provided).
    Returns the saved file path on success, otherwise None.
    """
    if root_output_dir is None:
        root_output_dir = _project_root() / "test_data"

    key = f"{report_id}/deliverable/backgroundImage.jpg"
    s3_uri = f"s3://{bucket}/{key}"

    dest_path = root_output_dir / report_id / "backgroundImage.jpg"
    if dest_path.exists():
        return dest_path
    ok = download_s3_to_path(s3_uri, dest_path, region_name=region_name)
    return dest_path if ok else None


def _save_webp_and_base64(
    jpg_path: Path,
    *,
    webp_name: str = "backgroundImage.webp",
    b64_name: str = "backgroundImage_base64",
) -> Tuple[Optional[Path], Optional[Path]]:
    if not jpg_path.exists():
        return None, None
    try:
        img = Image.open(str(jpg_path)).convert("RGB")
        webp_path = jpg_path.parent / webp_name
        img.save(str(webp_path), format="WEBP", quality=80, method=6)
    except Exception as e:
        logger.error("Failed to create WEBP for %s: %s", str(jpg_path), e)
        return None, None

    try:
        b64 = image_file_to_base64(str(webp_path))
        b64_path = jpg_path.parent / b64_name
        with open(b64_path, "w", encoding="utf-8") as f:
            f.write(b64)
    except Exception as e:
        logger.error("Failed to write base64 for %s: %s", str(webp_path), e)
        b64_path = None

    return webp_path, b64_path


def embed_backgroundImage(
    report_folder: Path,
    *,
    webp_name: str = "backgroundImage.webp",
    embeddings_name: str = "backgroundImage_embedings",
) -> Optional[Path]:
    """
    Generate Titan image embedding for extended ortho WEBP and save JSON array
    to {report_folder}/{embeddings_name}. Returns saved path or None on failure.
    """
    webp_path = report_folder / webp_name
    if not webp_path.exists():
        return None
    try:
        emb = _embed_image_with_bedrock(str(webp_path))
        if not emb:
            return None
        out_path = report_folder / embeddings_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(emb, f)
        return out_path
    except Exception as e:
        logger.error("Embedding failed for %s: %s", str(webp_path), e)
        return None


def _iter_report_ids_from_file(reports_txt_path: Path) -> Iterable[str]:
    with reports_txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            report_id = line.strip()
            if not report_id:
                continue
            if report_id.startswith("#"):
                continue
            yield report_id


def batch_download_from_reports(
    *,
    reports_txt_path: Path = _project_root() / "src" / "image_similarity_poc" / "Reports.txt",
    bucket: str = "evtech-us-east-2-pg-prod-sunsitecomplete",
    region_name: str = "us-east-2",
    root_output_dir: Optional[Path] = None,
) -> None:
    """
    Read report IDs from Reports.txt and download each backgroundImage.jpg
    into test_data/{report_id}/backgroundImage.jpg.
    """
    if root_output_dir is None:
        root_output_dir = _project_root() / "test_data"

    if not reports_txt_path.exists():
        raise FileNotFoundError(f"Reports file not found: {reports_txt_path}")

    success_count = 0
    total = 0
    for report_id in _iter_report_ids_from_file(reports_txt_path):
        total += 1
        saved = download_report_backgroundImage(
            report_id,
            bucket=bucket,
            region_name=region_name,
            root_output_dir=root_output_dir,
        )
        if saved is not None:
            success_count += 1
            logger.info("Saved: %s", str(saved))
            # After saving JPG, produce WEBP + base64 + embeddings
            webp_path, _ = _save_webp_and_base64(saved)
            if webp_path is not None:
                embed_path = embed_backgroundImage(saved.parent)
                if embed_path is not None:
                    logger.info("Embeddings saved: %s", str(embed_path))
                else:
                    logger.warning("Embedding failed for report_id=%s", report_id)
            else:
                logger.warning("WEBP/base64 generation failed for report_id=%s", report_id)
        else:
            logger.warning("Failed: report_id=%s", report_id)

    logger.info("Download complete. %d/%d succeeded.", success_count, total)