"""
Utility functions for handling extracted images stored in separate files.
Enhanced with Gemini Vision integration for image analysis and captioning.
"""
import os
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
import logging

# Import Gemini client with fallback
try:
    from .gemini_client import GeminiVisionClient, create_gemini_client, is_gemini_available
    GEMINI_INTEGRATION = True
except ImportError:
    GEMINI_INTEGRATION = False
    GeminiVisionClient = None

logger = logging.getLogger(__name__)


class ImageManager:
    """
    Manages extracted images stored in the file system.
    Enhanced with Gemini Vision capabilities for image analysis.
    """
    
    def __init__(self, images_base_dir: str = "extracted_images", enable_gemini: bool = True):
        """Initialize image manager with base directory and optional Gemini integration."""
        self.images_base_dir = Path(images_base_dir)
        self.images_base_dir.mkdir(exist_ok=True)
        
        # Initialize Gemini client if available and enabled
        self.gemini_client = None
        if enable_gemini and GEMINI_INTEGRATION and is_gemini_available():
            try:
                from .config import config
                gemini_config = config.get_gemini_config()
                if gemini_config.get("enable_image_captioning", True):
                    self.gemini_client = create_gemini_client(gemini_config)
                    logger.info("Gemini Vision integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
        
        if self.gemini_client is None:
            logger.info("Gemini Vision integration disabled or unavailable")
    
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
    
    """ So this is important function taht help us to determine which type of analysis to perform based on image metadata """
    def analyze_image_with_gemini(self, image_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an image using Gemini Vision API.
        
        Args:
            image_metadata: Image metadata containing file path
            
        Returns:
            Analysis results from Gemini
        """
        if not self.gemini_client:
            return {"error": "Gemini Vision not available"}
        
        image_path = self.get_image_path(image_metadata)
        if not image_path or not image_path.exists():
            return {"error": "Image file not found"}
        
        try:
            # Determine analysis type based on image metadata
            image_label = image_metadata.get('image_label', '').lower()
            
            if any(keyword in image_label for keyword in ['roof', 'side', 'aerial', 'top', 'north', 'south', 'east', 'west']):
                # Use specialized roof analysis
                analysis = self.gemini_client.analyze_roof_image(image_path)
            elif any(keyword in image_label for keyword in ['length', 'pitch', 'area', 'azimuth']):
                # Use measurement extraction
                analysis = self.gemini_client.extract_measurements(image_path)
            else:
                # Use general caption
                caption = self.gemini_client.caption_image(image_path, detailed=True)
                analysis = {"caption": caption, "analysis_type": "general"}
            
            analysis["gemini_analysis"] = True
            analysis["image_path"] = str(image_path)
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
            return {"error": str(e)}
    
    def batch_analyze_report_images(self, report_id: str) -> Dict[str, Any]:
        """
        Analyze all images for a specific report using Gemini Vision.
        
        Args:
            report_id: Report ID to analyze
            
        Returns:
            Batch analysis results
        """
        if not self.gemini_client:
            return {"error": "Gemini Vision not available"}
        
        image_paths = self.list_images_for_report(report_id)
        if not image_paths:
            return {"error": f"No images found for report {report_id}"}
        
        try:
            # Analyze roof-specific images
            roof_images = [p for p in image_paths if any(keyword in p.name.lower() 
                          for keyword in ['roof', 'side', 'aerial', 'top', 'north', 'south', 'east', 'west'])]
            
            measurement_images = [p for p in image_paths if any(keyword in p.name.lower()
                                for keyword in ['length', 'pitch', 'area', 'azimuth', 'rafter'])]
            
            results = {
                "report_id": report_id,
                "total_images": len(image_paths),
                "roof_analysis": [],
                "measurement_analysis": [],
                "general_analysis": []
            }
            
            # Process roof images
            if roof_images:
                roof_results = self.gemini_client.batch_process_images(roof_images, "roof")
                results["roof_analysis"] = roof_results
            
            # Process measurement images
            if measurement_images:
                measurement_results = self.gemini_client.batch_process_images(measurement_images, "measurements")
                results["measurement_analysis"] = measurement_results
            
            # Process remaining images
            other_images = [p for p in image_paths if p not in roof_images and p not in measurement_images]
            if other_images:
                general_results = self.gemini_client.batch_process_images(other_images, "caption")
                results["general_analysis"] = general_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch analysis for report {report_id}: {e}")
            return {"error": str(e), "report_id": report_id}
    
    def enhance_image_metadata_with_gemini(self, image_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance existing image metadata with Gemini Vision analysis.
        
        Args:
            image_metadata: Existing image metadata
            
        Returns:
            Enhanced metadata with Gemini analysis
        """
        enhanced_metadata = image_metadata.copy()
        
        if self.gemini_client:
            try:
                analysis = self.analyze_image_with_gemini(image_metadata)
                if "error" not in analysis:
                    enhanced_metadata["gemini_analysis"] = analysis
                    
                    # Add searchable text from analysis
                    searchable_text = enhanced_metadata.get("searchable_keywords", [])
                    
                    if "full_analysis" in analysis:
                        # Extract keywords from roof analysis
                        text = analysis["full_analysis"].lower()
                        searchable_text.extend([
                            analysis.get("roof_type", ""),
                            analysis.get("material", ""),
                            analysis.get("condition", ""),
                            analysis.get("orientation", "")
                        ])
                    elif "caption" in analysis:
                        # Extract keywords from caption
                        text = analysis["caption"].lower()
                    elif "measurements_analysis" in analysis:
                        # Extract measurement-related keywords
                        text = analysis["measurements_analysis"].lower()
                        if analysis.get("has_measurements"):
                            searchable_text.append("measurements")
                    
                    # Filter out empty strings and update metadata
                    enhanced_metadata["searchable_keywords"] = [k for k in searchable_text if k]
                    enhanced_metadata["gemini_enhanced"] = True
                    
            except Exception as e:
                logger.warning(f"Failed to enhance metadata with Gemini: {e}")
        
        return enhanced_metadata


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
