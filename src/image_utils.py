"""
Utility functions for handling extracted images stored in separate files.
"""
import os
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageManager:
    """
    Manages extracted images stored in the file system.
    """
    
    def __init__(self, images_base_dir: str = "extracted_images"):
        """Initialize image manager with base directory."""
        self.images_base_dir = Path(images_base_dir)
        self.images_base_dir.mkdir(exist_ok=True)
    
    def get_image_path(self, image_metadata: Dict[str, Any]) -> Optional[Path]:
        """Get the file path for an image from its metadata."""
        image_file_path = image_metadata.get('image_file_path')
        if image_file_path:
            path = Path(image_file_path)
            if path.exists():
                return path
        return None
    
    def load_image(self, image_metadata: Dict[str, Any]) -> Optional[Image.Image]:
        """Load PIL Image from file system."""
        image_path = self.get_image_path(image_metadata)
        if image_path and image_path.exists():
            try:
                return Image.open(image_path)
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
        return None
    
    def get_image_as_base64(self, image_metadata: Dict[str, Any]) -> Optional[str]:
        """Get image as base64 string for web display."""
        image_path = self.get_image_path(image_metadata)
        if image_path and image_path.exists():
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
            except Exception as e:
                logger.error(f"Error encoding image {image_path}: {e}")
        return None
    
    def get_image_info(self, image_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive image information."""
        image_path = self.get_image_path(image_metadata)
        info = {
            'exists': False,
            'path': str(image_path) if image_path else None,
            'size_bytes': None,
            'dimensions': None,
            'format': None
        }
        
        if image_path and image_path.exists():
            try:
                info['exists'] = True
                info['size_bytes'] = image_path.stat().st_size
                
                # Get image dimensions and format
                with Image.open(image_path) as img:
                    info['dimensions'] = img.size
                    info['format'] = img.format
                    
            except Exception as e:
                logger.error(f"Error getting image info {image_path}: {e}")
        
        return info
    
    def list_images_for_report(self, report_id: str) -> List[Path]:
        """List all images for a specific report."""
        report_dir = self.images_base_dir / f"report_{report_id}"
        if report_dir.exists():
            return list(report_dir.glob("*.png"))
        return []
    
    def get_all_reports(self) -> List[str]:
        """Get list of all report IDs that have images."""
        reports = []
        for item in self.images_base_dir.iterdir():
            if item.is_dir() and item.name.startswith("report_"):
                report_id = item.name.replace("report_", "")
                reports.append(report_id)
        return sorted(reports)
    
    def cleanup_orphaned_images(self, valid_image_paths: List[str]) -> int:
        """Clean up image files that are no longer referenced in the vector store."""
        removed_count = 0
        valid_paths_set = set(Path(p) for p in valid_image_paths)
        
        for image_file in self.images_base_dir.rglob("*.png"):
            if image_file not in valid_paths_set:
                try:
                    image_file.unlink()
                    removed_count += 1
                    logger.info(f"Removed orphaned image: {image_file}")
                except Exception as e:
                    logger.error(f"Error removing orphaned image {image_file}: {e}")
        
        return removed_count


def create_image_serving_url(image_metadata: Dict[str, Any], base_url: str = "") -> Optional[str]:
    """
    Create a URL for serving an image via a web server.
    
    Args:
        image_metadata: Image metadata containing file path
        base_url: Base URL for the image server
    
    Returns:
        URL string or None if image not available
    """
    image_file_path = image_metadata.get('image_file_path')
    if image_file_path:
        # Convert file path to URL-safe format
        path = Path(image_file_path)
        if path.exists():
            # Create URL from path (relative to extracted_images)
            relative_path = path.relative_to("extracted_images")
            url_path = str(relative_path).replace("\\", "/")
            return f"{base_url}/images/{url_path}"
    return None


def get_image_display_html(image_metadata: Dict[str, Any]) -> str:
    """
    Generate HTML for displaying an image.
    
    Args:
        image_metadata: Image metadata
    
    Returns:
        HTML string for image display
    """
    manager = ImageManager()
    base64_data = manager.get_image_as_base64(image_metadata)
    
    if base64_data:
        report_id = image_metadata.get('report_id', 'Unknown')
        page_num = image_metadata.get('page_number', 'Unknown')
        
        return f"""
        <div class="image-container" style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
            <h4>Report {report_id} - Page {page_num}</h4>
            <img src="data:image/png;base64,{base64_data}" 
                 style="max-width: 100%; height: auto; border: 1px solid #ccc;" 
                 alt="Report {report_id} Image" />
            <p><small>File: {image_metadata.get('image_filename', 'Unknown')}</small></p>
        </div>
        """
    else:
        return f"""
        <div class="image-placeholder" style="margin: 10px 0; padding: 20px; border: 1px dashed #ccc; text-align: center;">
            <p>📷 Image file not found</p>
            <small>Expected: {image_metadata.get('image_file_path', 'Unknown')}</small>
        </div>
        """
