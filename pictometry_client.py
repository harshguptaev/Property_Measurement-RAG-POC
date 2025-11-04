import os
from typing import Dict, Optional, Tuple, Union
from gserve_client import get_image_metadata
import requests
import json
from src.utils.gserveUtil import get_relevant_metadata_from_gserve
from PIL import Image

PICTOMETRY_BASE_URL = (
    "https://api.cmh.platform-prod.evinternal.net/solar/"
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

    # create final_data directory if not exists
    os.makedirs(os.path.join(os.getcwd(), "final_data"), exist_ok=True)
    # create final_data/{latitude}_{longitude} directory if not exists
    os.makedirs(os.path.join(os.getcwd(), "final_data", f"{latitude}_{longitude}"), exist_ok=True)
    # save the response to final_data/{latitude}_{longitude}/pictometry_response.json
    with open(os.path.join(os.getcwd(), "final_data", f"{latitude}_{longitude}", "pictometry_response.json"), "w") as f:
        json.dump(response.json(), f, indent=2)
    return response.json()


def parseresponse(api_response: Dict, latitude: float, longitude: float, image_size: Tuple[int, int] = (1000, 1000)) -> Dict[str, str]:
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
    orientations = ["Top"]
    result: Dict[str, str] = {}
    entire_metadata = {}

    # Create directory first
    os.makedirs(os.path.join(os.getcwd(), "final_data", f"{latitude}_{longitude}"), exist_ok=True)

    for orientation in orientations:
        # print(orientation)
        data = api_response.get(orientation)
        if not data:
            continue
        image_resource = data.get("imageResource")
        image_urn = data.get("imageId")

        # Get full metadata for this image
        full_metadata = get_image_metadata(image_urn)
        relevant_metadata = get_relevant_metadata_from_gserve(full_metadata)

        # Map orientation names to lowercase keys
        orientation_key = orientation.lower()

        # Store in entire_metadata dict
        entire_metadata[orientation_key] = relevant_metadata

        # Save individual MetadataJSON at Pictometry_Data/{latitude}_{longitude}/{orientation}_gserve_metadata.json
        # with open(os.path.join(os.getcwd(), "final_data", f"{latitude}_{longitude}", f"{orientation}_gserve_metadata.json"), "w") as f:
        #     json.dump(relevant_metadata, f)

        if not image_resource:
            continue
        # Ensure we only append size segment once
        size_segment = f"/width:{width};height:{height}"
        url = image_resource + size_segment
        result[orientation.lower()] = url

    # Save the complete metadata structure
    with open(os.path.join(os.getcwd(), "final_data", f"{latitude}_{longitude}", "gserve_response.json"), "w") as f:
        json.dump(entire_metadata["top"], f, indent=2)


    gserve_orientation = ["Top", "North", "South", "East", "West"]
    entire_gserve_metadata = {}
    for orientation in gserve_orientation:
        data = api_response.get(orientation)
        if not data:
            continue
        image_resource = data.get("imageResource")
        image_urn = data.get("imageId")

        # Get full metadata for this image
        full_metadata = get_image_metadata(image_urn)
        relevant_metadata = get_relevant_metadata_from_gserve(full_metadata)

        # Map orientation names to lowercase keys
        orientation_key = orientation.lower()

        # Store in entire_metadata dict
        entire_gserve_metadata[orientation_key] = relevant_metadata

    with open(os.path.join(os.getcwd(), "final_data", f"{latitude}_{longitude}", "gserve_entire_response.json"), "w") as f:
        json.dump(entire_gserve_metadata, f, indent=2)

    return result


def saveimagesfrompictometry(
    latitude_or_latlng: Union[float, Tuple[float, float]],
    longitude: Optional[float] = None,
    output_root: Optional[str] = "final_data/",
    image_size: Tuple[int, int] = (1000, 1000),
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
    image_urls = parseresponse(api_response, latitude, longitude_val, image_size=image_size)
    # Determine output root
    if output_root is None:
        default_root = os.path.join(os.getcwd(), "final_data")
        output_root = default_root if os.path.isdir(default_root) else os.path.join(os.getcwd(), "final_data")

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
            # print(content_type)
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



def save_cropped_images(dimensions: Dict[str, int], latitude: float, longitude: float):

    top_width = dimensions['required_width'] + 50
    top_height = dimensions['required_height'] + 50
    print("Top width: ", top_width, "Top height: ", top_height)
    others_width = top_width - 100
    others_height = top_height - 100
    print("Others width: ", others_width, "Others height: ", others_height)
    # read the pictometry_response.json file
    with open(os.path.join(os.getcwd(), "final_data", f"{latitude}_{longitude}", "pictometry_response.json"), "r") as f:
        pictometry_response = json.load(f)

    # crop the images to the dimensions
    for orientation, url in pictometry_response.items():
        image_url = url['imageResource']
        metadata = url['metadataResource']
        if orientation == "Top":
            crop_image(image_url, top_width, top_height, latitude, longitude, orientation, metadata)
        else:
            crop_image(image_url, others_width, others_height, latitude, longitude, orientation, metadata)


def crop_image(image_url: str, width: int, height: int, latitude: float, longitude: float, orientation: str, metadata: str):
    print("Cropping image to dimensions: ", width, height)

    session = requests.Session()
    headers = {"Accept": "image/*"}

    try:
        resp = session.get(image_url + f"/width:{width};height:{height}", headers=headers, timeout=60, stream=True)
        resp.raise_for_status()

        # Always save as PNG regardless of response content type
        file_path = os.path.join(os.getcwd(), "final_data", f"{latitude}_{longitude}", f"{orientation}_cropped.png")

        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Fetch Pictometry metadata and update geometry in gserve_entire_response.json
        try:
            # Append same dimensions to metadata URL as image URL
            metadata_url = metadata + f"/width:{width};height:{height}"
            metadata_resp = session.get(metadata_url, headers={"Accept": "application/json"}, timeout=60)
            metadata_resp.raise_for_status()
            pictometry_metadata = metadata_resp.json()
            
            # Extract imageCorners from Pictometry metadata
            if "response" in pictometry_metadata and "imageCorners" in pictometry_metadata["response"]:
                corners = pictometry_metadata["response"]["imageCorners"]
                
                # Convert Pictometry corners to GeoJSON Polygon coordinates
                # GeoJSON uses [lon, lat] format and polygon must be closed
                coordinates = [
                    [corners["upperLeft"]["longitude"], corners["upperLeft"]["latitude"]],
                    [corners["upperRight"]["longitude"], corners["upperRight"]["latitude"]],
                    [corners["lowerRight"]["longitude"], corners["lowerRight"]["latitude"]],
                    [corners["lowerLeft"]["longitude"], corners["lowerLeft"]["latitude"]],
                    [corners["upperLeft"]["longitude"], corners["upperLeft"]["latitude"]]  # Close the polygon
                ]
                
                # Load existing gserve_entire_response.json
                gserve_file = os.path.join(os.getcwd(), "final_data", f"{latitude}_{longitude}", "gserve_entire_response.json")
                if os.path.exists(gserve_file):
                    with open(gserve_file, "r") as f:
                        gserve_data = json.load(f)
                    
                    # Update geometry for this orientation (use lowercase)
                    orientation_key = orientation.lower()
                    if orientation_key in gserve_data:
                        gserve_data[orientation_key]["geometry"] = {
                            "type": "Polygon",
                            "coordinates": [coordinates]
                        }
                        
                        # Save updated gserve_entire_response.json
                        with open(gserve_file, "w") as f:
                            json.dump(gserve_data, f, indent=2)
                        print(f"✅ Updated geometry in gserve_entire_response.json for {orientation}")
        
        except Exception as metadata_exc:
            print(f"⚠️ Warning: Could not update geometry from Pictometry metadata: {metadata_exc}")

    except Exception as exc:  # noqa: BLE001 - broad to continue others
        print(f"Error cropping image: {exc}")
        return None

    
__all__ = [
    "getpictometryresponse",
    "parseresponse",
    "saveimagesfrompictometry",
]