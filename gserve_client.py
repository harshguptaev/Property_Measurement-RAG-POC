import requests
from typing import Dict, Any

# Encode urn and pass as special character in urn variable
GSERVE_BASE_URL = "https://gserve-internal.cmh.prod.evinternal.net/v1/image"

def get_image_metadata(urn: str) -> Dict[str, Any]:
    """Get metadata for an image from GServe."""
    response = requests.get(GSERVE_BASE_URL+f"/{urn}")
    return response.json()