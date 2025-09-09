import base64
import io
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image


EXPECTED_ORDER = ["top", "east", "west", "north", "south"]
EXPECTED_SIZE = (1024, 1024)


def _load_and_normalize_image(path: str, expected_size: Tuple[int, int] = EXPECTED_SIZE) -> Optional[Image.Image]:
    if not os.path.isfile(path):
        return None
    img = Image.open(path)
    if img.size != expected_size:
        img = img.resize(expected_size, Image.LANCZOS)
    return img.convert("RGB")


def stitch_pictometry_images_for_folder(folder_path: str, output_name: str = "stitched_image.png") -> Optional[str]:
    """
    Stitch five images (top, east, west, north, south) horizontally in that order
    and save as output_name within the same folder. Returns the saved path, or None
    if any of the required images are missing.
    """
    images: List[Image.Image] = []
    missing: List[str] = []
    for name in EXPECTED_ORDER:
        # try common extensions
        candidates = [
            os.path.join(folder_path, f"{name}.jpg"),
            os.path.join(folder_path, f"{name}.jpeg"),
            os.path.join(folder_path, f"{name}.png"),
            os.path.join(folder_path, f"{name}.webp"),
        ]
        loaded: Optional[Image.Image] = None
        for p in candidates:
            loaded = _load_and_normalize_image(p)
            if loaded is not None:
                break
        if loaded is None:
            missing.append(name)
        images.append(loaded)

    if any(img is None for img in images):
        return None

    total_width = EXPECTED_SIZE[0] * len(images)
    height = EXPECTED_SIZE[1]
    stitched = Image.new("RGB", (total_width, height), color=(0, 0, 0))

    x_offset = 0
    for img in images:
        stitched.paste(img, (x_offset, 0))
        x_offset += EXPECTED_SIZE[0]

    output_path = os.path.join(folder_path, output_name)
    stitched.save(output_path, format="PNG")

    # Save a compressed variant for storage/embedding (WEBP) and write base64 from it
    compressed_path = os.path.join(folder_path, "stitched_image.webp")
    try:
        stitched.save(compressed_path, format="WEBP", quality=80, method=6)
    except Exception:
        compressed_path = output_path

    # Also save base64 string alongside the stitched image (use compressed if available)
    try:
        b64 = image_file_to_base64(compressed_path)
        b64_path = os.path.join(folder_path, "stiched_image_base64")
        with open(b64_path, "w", encoding="utf-8") as f:
            f.write(b64)
    except Exception:
        # If base64 persistence fails, we still return the image path
        pass

    return output_path


def stitch_all_pictometry_directories(root: str = None) -> Dict[str, Optional[str]]:
    """
    Iterate through pictometry_images/* and stitch images for each lat_lon folder.
    Returns a mapping of folder path to stitched image path (or None if skipped).
    """
    if root is None:
        root = os.path.join(os.getcwd(), "pictometry_images")
    if not os.path.isdir(root):
        return {}

    results: Dict[str, Optional[str]] = {}
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        result = stitch_pictometry_images_for_folder(folder)
        results[folder] = result
    return results


def image_file_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


__all__ = [
    "stitch_pictometry_images_for_folder",
    "stitch_all_pictometry_directories",
    "image_file_to_base64",
    "image_to_base64",
]


