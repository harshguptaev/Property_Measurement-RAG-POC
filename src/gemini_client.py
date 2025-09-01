"""
Google Gemini Pro Vision client for image processing and captioning.
Provides image analysis capabilities for property inspection documents.
"""

import os
import logging
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None


class GeminiVisionClient:
    """
    Client for Google Gemini Pro Vision model for image analysis and captioning.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        max_tokens: int = 4096
    ):
        """
        Initialize Gemini Vision client.
        
        Args:
            api_key: Gemini API key (if None, reads from environment)
            model_name: Model name to use
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google GenerativeAI library not available. "
                "Install with: pip install google-generativeai"
            )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure API key
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                "environment variable or pass api_key parameter."
            )
        
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        logging.info(f"Initialized Gemini Vision client with model: {self.model_name}")
    
    def caption_image(
        self,
        image: Union[str, Path, Image.Image, bytes],
        prompt: Optional[str] = None,
        detailed: bool = True
    ) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: Image path, PIL Image, or image bytes
            prompt: Custom prompt (if None, uses default)
            detailed: Whether to generate detailed captions
            
        Returns:
            Generated caption text
        """
        if prompt is None:
            if detailed:
                prompt = (
                    "Analyze this property inspection image in detail. "
                    "Describe what you see including any structural elements, "
                    "measurements, conditions, or notable features. "
                    "Focus on information relevant to property assessment."
                )
            else:
                prompt = "Provide a concise caption for this property inspection image."
        
        try:
            # Load and process image
            pil_image = self._load_image(image)
            
            # Generate content
            response = self.model.generate_content([prompt, pil_image])
            
            if response.text:
                return response.text.strip()
            else:
                logging.warning("Empty response from Gemini Vision")
                return "No caption generated"
                
        except Exception as e:
            logging.error(f"Error generating image caption: {e}")
            return f"Error generating caption: {str(e)}"
    
    def analyze_roof_image(self, image: Union[str, Path, Image.Image, bytes]) -> Dict[str, Any]:
        """
        Specialized analysis for roof inspection images.
        
        Args:
            image: Image to analyze
            
        Returns:
            Analysis results with structured data
        """
        prompt = """
        Analyze this roof inspection image. Provide a detailed analysis including:
        
        1. Roof Type: (gable, hip, shed, flat, etc.)
        2. Roofing Material: (asphalt shingles, metal, tile, etc.)
        3. Condition Assessment: (excellent, good, fair, poor)
        4. Visible Issues: (any damage, wear, or concerns)
        5. Structural Elements: (gutters, vents, chimneys, etc.)
        6. Measurements or Dimensions: (if visible)
        7. Orientation: (if determinable - north, south, east, west side)
        8. Overall Description: (comprehensive summary)
        
        Format your response as a detailed analysis suitable for a property inspection report.
        """
        
        try:
            caption = self.caption_image(image, prompt, detailed=True)
            
            # Parse response into structured format
            analysis = {
                "full_analysis": caption,
                "roof_type": self._extract_field(caption, ["roof type", "type"]),
                "material": self._extract_field(caption, ["material", "roofing material"]),
                "condition": self._extract_field(caption, ["condition", "assessment"]),
                "issues": self._extract_field(caption, ["issues", "damage", "problems"]),
                "elements": self._extract_field(caption, ["elements", "features", "components"]),
                "orientation": self._extract_field(caption, ["orientation", "side", "direction"])
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing roof image: {e}")
            return {
                "full_analysis": f"Error analyzing image: {str(e)}",
                "error": True
            }
    
    def extract_measurements(self, image: Union[str, Path, Image.Image, bytes]) -> Dict[str, Any]:
        """
        Extract measurements and dimensions from property images.
        
        Args:
            image: Image to analyze
            
        Returns:
            Extracted measurements and dimensions
        """
        prompt = """
        Analyze this property inspection image and extract any visible measurements, 
        dimensions, or numerical data. Look for:
        
        1. Length measurements
        2. Width measurements  
        3. Area calculations
        4. Angles or pitch measurements
        5. Heights or elevations
        6. Any scale indicators
        7. Coordinate or grid references
        
        List all visible numbers, measurements, and their units clearly.
        If no measurements are visible, state that clearly.
        """
        
        try:
            response = self.caption_image(image, prompt, detailed=True)
            
            return {
                "measurements_analysis": response,
                "has_measurements": "no measurements" not in response.lower(),
                "extracted_data": self._parse_measurements(response)
            }
            
        except Exception as e:
            logging.error(f"Error extracting measurements: {e}")
            return {
                "measurements_analysis": f"Error extracting measurements: {str(e)}",
                "error": True
            }
    
    def batch_process_images(
        self,
        image_paths: List[Union[str, Path]],
        analysis_type: str = "caption"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths to process
            analysis_type: Type of analysis ("caption", "roof", "measurements")
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logging.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                if analysis_type == "roof":
                    result = self.analyze_roof_image(image_path)
                elif analysis_type == "measurements":
                    result = self.extract_measurements(image_path)
                else:  # caption
                    result = {"caption": self.caption_image(image_path)}
                
                result["image_path"] = str(image_path)
                result["image_index"] = i
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error processing image {image_path}: {e}")
                results.append({
                    "image_path": str(image_path),
                    "image_index": i,
                    "error": str(e)
                })
        
        return results
    
    def _load_image(self, image: Union[str, Path, Image.Image, bytes]) -> Image.Image:
        """Load image from various input formats."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, (str, Path)):
            return Image.open(image)
        elif isinstance(image, bytes):
            return Image.open(BytesIO(image))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _extract_field(self, text: str, keywords: List[str]) -> str:
        """Extract specific field from analysis text."""
        text_lower = text.lower()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Look for lines containing the keyword
            for line in text.split('\n'):
                line_lower = line.lower()
                if keyword_lower in line_lower:
                    # Extract text after colon or keyword
                    if ':' in line:
                        return line.split(':', 1)[1].strip()
                    else:
                        return line.strip()
        return "Not specified"
    
    def _parse_measurements(self, text: str) -> List[Dict[str, str]]:
        """Parse measurements from text response."""
        import re
        
        # Regex patterns for common measurements
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(feet|ft|foot|inches|in|meters|m|mm|cm)',
            r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(feet|ft|inches|in|meters|m)',
            r'(\d+(?:\.\d+)?)\s*degrees?',
            r'(\d+(?:\.\d+)?)\s*(?:square\s*)?(feet|ft²|sqft|meters|m²)',
        ]
        
        measurements = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                measurements.append({
                    "value": match.group(1),
                    "unit": match.group(2) if len(match.groups()) > 1 else "degrees",
                    "full_match": match.group(0)
                })
        
        return measurements


def create_gemini_client(config: Optional[Dict[str, Any]] = None) -> GeminiVisionClient:
    """
    Create Gemini Vision client from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Gemini client
    """
    if config is None:
        config = {}
    
    return GeminiVisionClient(
        api_key=config.get("api_key"),
        model_name=config.get("model_name", "gemini-2.5-flash"),
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 4096)
    )


def is_gemini_available() -> bool:
    """Check if Gemini API is available."""
    return GEMINI_AVAILABLE and (
        os.getenv("GEMINI_API_KEY") is not None or 
        os.getenv("GOOGLE_API_KEY") is not None
    )
