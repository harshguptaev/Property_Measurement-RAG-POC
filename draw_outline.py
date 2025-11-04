#!/usr/bin/env python3
"""
Draw Outline Module
Creates simplified roof outline diagrams from SageMaker results
Based on roof_generator.py but simplified for Top and North images only
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os
import logging
import shutil
from typing import List, Dict, Tuple
import boto3
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrawOutline:
    def __init__(self, lat: float, lon: float, s3_bucket: str = "evtech-us-east-2-pg-test-sunsitecomplete", 
                 s3_base_path: str = "property-data/LatLongData", region: str = "us-east-2"):
        """Initialize the draw outline handler"""
        self.lat = lat
        self.lon = lon
        self.folder_name = f"{lat}_{lon}"
        self.s3_bucket = s3_bucket
        self.s3_base_path = s3_base_path
        self.region = region
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Constants
        self.gsd_factor = 0.06765248465718485  # meters per pixel
        self.feet_conversion = 3.28084  # meters to feet
        
        # Target dimensions for output
        self.target_width = 512
        self.target_height = 512
    
    def download_image_from_s3(self, orientation: str, local_path: str) -> bool:
        """
        Download image from S3
        
        Args:
            orientation: Image orientation (Top or North)
            local_path: Local path to save the image
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            s3_key = f"{self.s3_base_path}/{self.folder_name}/Top_cropped.png"
            logger.info(f"🔄 Downloading {orientation} image from S3")
            
            self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
            logger.info(f"✅ Downloaded {orientation} image to {local_path}")
            return True
            
        except ClientError as e:
            logger.error(f"❌ S3 download failed for {orientation}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Error downloading {orientation} image: {str(e)}")
            return False
    
    def download_outline_json_from_s3(self, local_path: str) -> bool:
        """
        Download image_outline.json from S3
        
        Args:
            local_path: Local path to save the JSON file
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            s3_key = f"{self.s3_base_path}/{self.folder_name}/image_outline.json"
            logger.info(f"🔄 Downloading outline JSON from S3")
            
            self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
            logger.info(f"✅ Downloaded outline JSON to {local_path}")
            return True
            
        except ClientError as e:
            logger.error(f"❌ S3 download failed for outline JSON: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Error downloading outline JSON: {str(e)}")
            return False
    
    def load_outline_data(self, json_file: str) -> Dict:
        """
        Load outline data from JSON file
        
        Args:
            json_file: Path to the JSON file
            
        Returns:
            Dictionary with outline data
        """
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"✅ Loaded outline data from {json_file}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading outline data: {str(e)}")
            return {}
    
    def calculate_line_length_feet(self, line: List[int], image_width: int, image_height: int) -> float:
        """Calculate line length in feet"""
        x1, y1, x2, y2 = line
        pixel_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        meters = pixel_length * self.gsd_factor
        feet = meters * self.feet_conversion
        return feet
    
    def create_simplified_outline(self, outline_data: Dict, orientation: str, image_path: str, output_dir: str) -> bool:
        """
        Create simplified roof outline for a specific orientation
        
        Args:
            outline_data: Dictionary with outline data
            orientation: Image orientation (Top or North)
            image_path: Path to the original image
            output_dir: Directory to save the output
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🔄 Creating simplified outline for {orientation}")

            # Get predictions from the root level (since top_outline.json has predictions directly)
            predictions = outline_data.get('predictions', [])
            if not predictions:
                logger.warning(f"⚠️ No predictions found in outline data")
                return False
            
            # Load original image to get dimensions
            original_img = Image.open(image_path)
            img_array = np.array(original_img)
            image_height, image_width = img_array.shape[:2]
            
            # Create figure with target dimensions
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            # Set up the plot
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            # Remove axes and title
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Draw lines in grayscale
            for prediction in predictions:
                line_coords = prediction.get('line', [])
                if len(line_coords) != 4:
                    continue
                
                # Normalize coordinates
                x1, y1, x2, y2 = (line_coords[0] / image_width, 
                                 line_coords[1] / image_height,
                                 line_coords[2] / image_width, 
                                 line_coords[3] / image_height)
                
                # Draw black line
                ax.plot([x1, x2], [y1, y2], 
                       color='black', linewidth=2, alpha=1.0)
            
            # Save as grayscale image with target dimensions and "top_" prefix
            output_path = os.path.join(output_dir, f"roof_outline_simplified_{orientation.lower()}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()

            # Convert to grayscale and resize to 512x512
            self.convert_to_grayscale_512(output_path)
            
            logger.info(f"✅ Simplified outline saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating simplified outline for {orientation}: {str(e)}")
            return False
    
    def convert_to_grayscale_512(self, image_path: str) -> bool:
        """
        Convert image to grayscale and resize to 512x512 using OpenCV (consistent with image_preprocessor.py)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"❌ Could not read image: {image_path}")
                return False
            
            # Convert to grayscale
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img.copy()
            
            # Resize to 512x512 using INTER_AREA (same as image_preprocessor.py)
            resized_img = cv2.resize(gray_img, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
            
            # Save using OpenCV
            cv2.imwrite(image_path, resized_img)
            logger.info(f"✅ Converted to grayscale 512x512 using OpenCV: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error converting image: {str(e)}")
            return False
    
    def process_top(self, input_source_dir: str = "final_data") -> Dict:
        """
        Process Top image to create simplified outlines
        Saves top_roof_outline_simplified.png in final_data lat_lon folders

        Args:
            input_source_dir: Directory containing the final_data folders (default: "final_data")

        Returns:
            Dictionary with processing results
        """
        results = {
            'success': False,
            'lat': self.lat,
            'lon': self.lon,
            'orientations_processed': 0,
            'orientations_failed': 0,
            'output_files': [],
            'errors': []
        }

        try:
            logger.info(f"🔄 Processing Top and North images for {self.folder_name}")

            # Path to the folder in final_data
            folder_path = os.path.join(input_source_dir, self.folder_name)

            # Look for outline JSON directly in the lat_lon folder
            json_path = os.path.join(folder_path, "top_outline.json")
            if not os.path.exists(json_path):
                results['errors'].append(f"Failed to find top_outline.json in {folder_path}")
                return results

            # Load outline data
            outline_data = self.load_outline_data(json_path)
            if not outline_data:
                results['errors'].append("Failed to load outline data")
                return results

            # Create a temporary work directory
            local_work_dir = os.path.join(folder_path, "temp_work")
            os.makedirs(local_work_dir, exist_ok=True)

            # Process Top and North orientations
            orientations = ['Top']

            for orientation in orientations:
                # Try to use local image first (from final_data structure)
                image_path = os.path.join(folder_path, f"{orientation}_cropped.png")
                if not os.path.exists(image_path):
                    # Fallback: download from S3
                    image_path = os.path.join(local_work_dir, f"{orientation.lower()}_image.png")
                    if not self.download_image_from_s3(orientation, image_path):
                        results['orientations_failed'] += 1
                        results['errors'].append(f"Failed to find or download {orientation} image")
                        continue

                # Create simplified outline
                if self.create_simplified_outline(outline_data, orientation, image_path, folder_path):
                    # The function already saves to the correct location with prefix "roof_outline_simplified_"
                    local_output_file = os.path.join(folder_path, f"roof_outline_simplified_{orientation.lower()}.png")
                    logger.info(f"✅ Saved {orientation} outline: {local_output_file}")

                    results['orientations_processed'] += 1
                    results['output_files'].append(local_output_file)

                    # Upload to S3 with "top_" prefix
                    s3_key = f"{self.s3_base_path}/{self.folder_name}/roof_outline_simplified_{orientation.lower()}.png"
                    if self.upload_to_s3(local_output_file, s3_key):
                        logger.info(f"✅ Uploaded {orientation} outline to S3")
                    else:
                        logger.warning(f"⚠️ Failed to upload {orientation} outline to S3")
                else:
                    results['orientations_failed'] += 1
                    results['errors'].append(f"Failed to create outline for {orientation}")

            # Clean up temp directory
            if os.path.exists(local_work_dir):
                import shutil
                shutil.rmtree(local_work_dir)

            results['success'] = results['orientations_failed'] == 0
            logger.info(f"✅ Processing complete: {results['orientations_processed']} successful, {results['orientations_failed']} failed")

        except Exception as e:
            logger.error(f"❌ Error processing Top and North images: {str(e)}")
            results['errors'].append(str(e))

        return results
    
    def upload_to_s3(self, local_file_path: str, s3_key: str) -> bool:
        """
        Upload file to S3
        
        Args:
            local_file_path: Path to local file
            s3_key: S3 key for the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.upload_file(local_file_path, self.s3_bucket, s3_key)
            logger.info(f"✅ Uploaded to S3: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"❌ S3 upload failed: {str(e)}")
            return False

