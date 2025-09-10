import os
from typing import Dict, Optional, Tuple, Union

import requests


PICTOMETRY_BASE_URL = (
    "https://api.cmh.platform-test.evinternal.net/solar/"
    "sunpath-pictometry-service/v1/pictometry/latestimageinfo"
)


def getpictometryresponse(
    latitude: float,
    longitude: float,
    timeout_seconds: int = 30,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Call the Pictometry latestimageinfo endpoint with given latitude and longitude.

    Args:
        latitude: Latitude value.
        longitude: Longitude value.
        timeout_seconds: Request timeout in seconds.
        extra_headers: Optional headers to include in the request.

    Returns:
        Parsed JSON response as a dictionary.
    """
    params = {"latLng": f"{latitude},{longitude}"}
    headers = {"Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    response = requests.get(PICTOMETRY_BASE_URL, params=params, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def parseresponse(api_response: Dict, image_size: Tuple[int, int] = (1024, 1024)) -> Dict[str, str]:
    """
    Parse the API response and construct downloadable image URLs for each orientation.

    The API returns orientations like "East", "West", "North", "South", "Top" with
    an "imageResource" URL. The actual image can be fetched by appending
    "/width:{w};height:{h}" to that URL.

    Args:
        api_response: Response dictionary from getpictometryresponse.
        image_size: Desired (width, height) tuple to append to the URL.

    Returns:
        A mapping of orientation (lowercase) to downloadable image URL.
    """
    width, height = image_size
    orientations = ["East", "West", "North", "South", "Top"]
    result: Dict[str, str] = {}

    for orientation in orientations:
        data = api_response.get(orientation)
        if not data:
            continue
        image_resource = data.get("imageResource")
        if not image_resource:
            continue
        # Ensure we only append size segment once
        size_segment = f"/width:{width};height:{height}"
        url = image_resource if image_resource.endswith(size_segment) else image_resource + size_segment
        result[orientation.lower()] = url

    return result


def saveimagesfrompictometry(
    latitude_or_latlng: Union[float, Tuple[float, float]],
    longitude: Optional[float] = None,
    output_root: Optional[str] = None,
    image_size: Tuple[int, int] = (1024, 1024),
    timeout_seconds: int = 60,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Fetch pictometry images for the given lat/lon and save them to disk.

    The images are saved in a directory named with the combination of latitude and
    longitude, e.g., "38.0_-41.0", containing files named
    "east.jpg", "west.jpg", "north.jpg", "south.jpg", and "top.jpg" if available.

    Args:
        latitude_or_latlng: Either latitude as float, or a (lat, lon) tuple.
        longitude: Longitude value when latitude is provided as a float.
        output_root: Optional root directory to place the lat/lon folder. If not provided,
            attempts to use an existing "extracted_images" directory in the project; if not
            present, creates a local "pictometry_images" directory.
        image_size: Desired (width, height) for fetched images.
        timeout_seconds: Per-image download timeout in seconds.
        extra_headers: Optional headers to include when downloading images.

    Returns:
        Mapping of orientation (lowercase) to the saved file path.
    """
    # Normalize inputs
    if isinstance(latitude_or_latlng, (tuple, list)) and longitude is None:
        latitude, longitude_val = float(latitude_or_latlng[0]), float(latitude_or_latlng[1])
    else:
        if longitude is None:
            raise ValueError("longitude must be provided when latitude is a float")
        latitude, longitude_val = float(latitude_or_latlng), float(longitude)

    # Step 1: Get API response
    api_response = getpictometryresponse(latitude, longitude_val, extra_headers=extra_headers)
    # Step 2: Parse response for image URLs
    image_urls = parseresponse(api_response, image_size=image_size)
    # Determine output root
    if output_root is None:
        default_root = os.path.join(os.getcwd(), "pictometry_images")
        output_root = default_root if os.path.isdir(default_root) else os.path.join(os.getcwd(), "pictometry_images")

    # Create directory named as combination of lat and lon
    folder_name = f"{latitude}_{longitude_val}"
    target_dir = os.path.join(output_root, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    saved_paths: Dict[str, str] = {}
    session = requests.Session()
    headers = {"Accept": "image/*"}
    if extra_headers:
        headers.update(extra_headers)

    for orientation, url in image_urls.items():
        try:
            resp = session.get(url, headers=headers, timeout=timeout_seconds, stream=True)
            resp.raise_for_status()

            # Determine file extension from Content-Type if present
            content_type = resp.headers.get("Content-Type", "").lower()
            ext = ".jpg"
            if "png" in content_type:
                ext = ".png"
            elif "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            elif "webp" in content_type:
                ext = ".webp"

            file_path = os.path.join(target_dir, f"{orientation}{ext}")

            with open(file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            saved_paths[orientation] = file_path
        except Exception as exc:  # noqa: BLE001 - broad to continue others
            # Skip failed orientation but continue others
            continue

    return saved_paths


__all__ = [
    "getpictometryresponse",
    "parseresponse",
    "saveimagesfrompictometry",
]


