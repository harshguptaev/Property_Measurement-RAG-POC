import os
import requests
from typing import Optional, Dict, Any


# API credentials from Postman collection
REPORTS_USERNAME = "C8E9F401-7D41-44A3-9E39-B971DE7E3593"
REPORTS_PASSWORD = "A288CEFD0DB1E7D74395C9D966F3CF06AB8D0F9E55326A7D95E7CC2102E40D15"

# Base URLs
REPORT_INFO_BASE_URL = "https://operationsextranet.cmh.reportsprod.evinternal.net/ReportInfo"
GET_REPORT_FILE_BASE_URL = "https://intranetrest.cmh.reportsprod.evinternal.net/GetReportFile"


def get_report_info(report_id: str, timeout_seconds: int = 30) -> Dict[str, Any]:
    """
    Get report information from the ReportInfo API.

    Args:
        report_id: The report ID to fetch information for
        timeout_seconds: Request timeout in seconds

    Returns:
        Parsed JSON response as a dictionary
    """
    url = f"{REPORT_INFO_BASE_URL}/{report_id}"

    response = requests.get(
        url,
        auth=(REPORTS_USERNAME, REPORTS_PASSWORD),
        timeout=timeout_seconds
    )
    response.raise_for_status()
    return response.json()


def download_report_file(
    report_id: str,
    file_type_id: str,
    filename: str,
    timeout_seconds: int = 30
) -> str:
    """
    Download a report file from the GetReportFile API and save it to ReportsData/{report_id}/ directory.

    Args:
        report_id: The report ID
        file_type_id: The file type ID
        filename: The filename to save the file as (without extension)
        timeout_seconds: Request timeout in seconds

    Returns:
        The full path where the file was saved
    """
    # Create the directory path
    directory_path = os.path.join("input_data", report_id)
    os.makedirs(directory_path, exist_ok=True)

    # Build the URL with query parameters
    url = GET_REPORT_FILE_BASE_URL
    params = {
        "reportId": report_id,
        "fileTypeId": file_type_id
    }

    # Make the request
    response = requests.get(
        url,
        params=params,
        auth=(REPORTS_USERNAME, REPORTS_PASSWORD),
        timeout=timeout_seconds
    )
    response.raise_for_status()

    # Read first 512 bytes to detect file type based on content
    content_sample = response.content[:512]

    # Detect file type based on magic bytes/content
    if content_sample.startswith(b"%PDF"):
        extension = ".pdf"
    elif content_sample.startswith(b"\xff\xd8\xff"):  # JPEG magic bytes
        extension = ".jpg"
    elif content_sample.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG magic bytes
        extension = ".png"
    elif content_sample.startswith(b"RIFF") and b"WEBP" in content_sample[:12]:  # WebP
        extension = ".webp"
    elif content_sample.startswith(b"{") or content_sample.startswith(b"["):  # JSON
        extension = ".json"
    else:
        extension = ""  # no extension for unknown types

    # Create the full file path
    if extension == "":
        return None
    full_filename = f"{filename}"
    file_path = os.path.join(directory_path, full_filename)

    # Save the file
    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_path


__all__ = [
    "get_report_info",
    "download_report_file",
]
