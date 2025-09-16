import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Iterable

import boto3
from botocore.exceptions import ClientError


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


def download_report_extended_ortho(
    report_id: str,
    *,
    bucket: str = "evtech-us-east-2-pg-prod-sunsitecomplete",
    region_name: str = "us-east-2",
    root_output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Download the extendedOrthoImage.jpg for a given report ID and save to
    test_data/{report_id}/extendedOrthoImage.jpg under repo root (or root_output_dir if provided).
    Returns the saved file path on success, otherwise None.
    """
    if root_output_dir is None:
        root_output_dir = _project_root() / "test_data"

    key = f"{report_id}/deliverable/extendedOrthoImage.jpg"
    s3_uri = f"s3://{bucket}/{key}"

    dest_path = root_output_dir / report_id / "extendedOrthoImage.jpg"
    ok = download_s3_to_path(s3_uri, dest_path, region_name=region_name)
    return dest_path if ok else None


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
    Read report IDs from Reports.txt and download each extendedOrthoImage.jpg
    into test_data/{report_id}/extendedOrthoImage.jpg.
    """
    if root_output_dir is None:
        root_output_dir = _project_root() / "test_data"

    if not reports_txt_path.exists():
        raise FileNotFoundError(f"Reports file not found: {reports_txt_path}")

    success_count = 0
    total = 0
    for report_id in _iter_report_ids_from_file(reports_txt_path):
        total += 1
        saved = download_report_extended_ortho(
            report_id,
            bucket=bucket,
            region_name=region_name,
            root_output_dir=root_output_dir,
        )
        if saved is not None:
            success_count += 1
            logger.info("Saved: %s", str(saved))
        else:
            logger.warning("Failed: report_id=%s", report_id)

    logger.info("Download complete. %d/%d succeeded.", success_count, total)