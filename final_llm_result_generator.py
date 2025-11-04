#!/usr/bin/env python3
"""
Final LLM Result Generator
Compares Property 1 (from similarity_search_results.json top_result) 
with Property 2 (from house2_result_summary.json)
"""

import os
import json
import logging
import re
import math
import boto3
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinalLLMResultGenerator:
    def __init__(self, similarity_results_file: str = "similarity_search_results.json",
                 house2_summary_file: str = "house2_result_summary.json",
                 bedrock_region: str = "us-east-1",
                 bedrock_model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"):
        """
        Initialize the result generator
        
        Args:
            similarity_results_file: Path to similarity_search_results.json
            house2_summary_file: Path to house2_result_summary.json
            bedrock_region: AWS Bedrock region
            bedrock_model: Bedrock model ID for LLM
        """
        self.similarity_results_file = similarity_results_file
        self.house2_summary_file = house2_summary_file
        self.bedrock_region = bedrock_region
        self.bedrock_model = bedrock_model
        
        # Initialize Bedrock client
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=bedrock_region)
    
    def extract_numeric_value(self, text: str) -> Optional[float]:
        """
        Extract numeric value from text (e.g., "5,715 sq ft" -> 5715.0)
        
        Args:
            text: Text containing numeric value
            
        Returns:
            Numeric value or None if not found
        """
        try:
            # Remove commas and extract numbers
            cleaned = re.sub(r'[^\d.]', '', text.replace(',', ''))
            if cleaned:
                return float(cleaned)
            return None
        except Exception as e:
            logger.warning(f"Could not extract numeric value from '{text}': {str(e)}")
            return None
    
    def parse_pitch(self, pitch_str: str) -> Optional[float]:
        """
        Convert pitch string (e.g., "7/12") to degrees
        
        Args:
            pitch_str: Pitch string in format "X/12"
            
        Returns:
            Pitch in degrees or None if invalid
        """
        try:
            # Extract numerator and denominator
            match = re.match(r'(\d+)/(\d+)', pitch_str)
            if match:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
                # Convert to degrees: arctan(numerator/denominator) * 180/π
                pitch_radians = math.atan(numerator / denominator)
                pitch_degrees = math.degrees(pitch_radians)
                return pitch_degrees
            return None
        except Exception as e:
            logger.warning(f"Could not parse pitch '{pitch_str}': {str(e)}")
            return None
    
    def extract_length(self, length_str: str) -> Optional[float]:
        """
        Extract length value from string (e.g., "103 ft" -> 103.0)
        
        Args:
            length_str: Length string
            
        Returns:
            Numeric length value or None
        """
        try:
            cleaned = re.sub(r'[^\d.]', '', length_str.replace(',', ''))
            if cleaned:
                return float(cleaned)
            return None
        except Exception:
            return None
    
    def extract_property1_data(self, top_result: Dict[str, Any], 
                               final_results: list = None) -> Dict[str, Any]:
        """
        Extract and normalize Property 1 data from top_result
        Gets address and similarity_score from re-ranked Level 2 results (rank 1)
        
        Args:
            top_result: Top result dictionary from similarity_search_results.json
            final_results: Final results list from similarity_search_results.json (re-ranked Level 2)
            
        Returns:
            Normalized property data
        """
        property1 = {
            'property_id': top_result.get('property_id', 'Unknown'),
            'address': None,
            'report_id': None,
            'similarity_score': None,  # Will get from final_results rank 1
            'rank': None,
            'closeness_score': None,
            'total_area': None,
            'total_area_sqft': None,
            'predominant_pitch': None,
            'predominant_pitch_degrees': None,
            'num_of_facets': None,
            'num_of_ridges': None,
            'num_of_eaves': None,
            'num_of_rakes': None,
            'num_of_valleys': None,
            'num_of_hips': None,
            'num_of_flashing': None,
            'number_of_stories': None,
            'structure_complexity': None,
            'estimated_attic': None,
            'total_roof_obstructions': None
        }
        
        # Get address and similarity_score from final_results rank 1 (re-ranked Level 2)
        if final_results:
            rank_1_result = None
            for result in final_results:
                if result.get('rank') == 1:
                    rank_1_result = result
                    break
            
            if rank_1_result:
                # Get similarity_score from re-ranked Level 2 result
                property1['similarity_score'] = rank_1_result.get('similarity_score')
                property1['rank'] = rank_1_result.get('rank')
                property1['closeness_score'] = rank_1_result.get('closeness_score')
                
                # Try to get address from metadata or level1 as fallback
                level1 = top_result.get('level1_details', {})
                property1['address'] = level1.get('address', 'Unknown')
                property1['report_id'] = level1.get('report_id', 'Unknown')
            else:
                # Fallback to level1 if no rank 1 found
                level1 = top_result.get('level1_details', {})
                property1['address'] = level1.get('address', 'Unknown')
                property1['report_id'] = level1.get('report_id', 'Unknown')
                property1['similarity_score'] = level1.get('similarity_score', 0.0)
        else:
            # Fallback to level1 if no final_results
            level1 = top_result.get('level1_details', {})
            property1['address'] = level1.get('address', 'Unknown')
            property1['report_id'] = level1.get('report_id', 'Unknown')
            property1['similarity_score'] = level1.get('similarity_score', 0.0)
        
        # Extract level2 details - prioritize Roof Measurements, fallback to House Measurements
        level2_details = top_result.get('level2_details', [])
        
        roof_measurements = None
        house_measurements = None
        
        for detail in level2_details:
            section = detail.get('section', '')
            if section == 'Roof Measurements - All Structures':
                roof_measurements = detail.get('metadata', {}).get('data', {})
            elif section == 'House Measurements':
                house_measurements = detail.get('metadata', {}).get('data', {})
        
        # Extract roof measurements
        if roof_measurements:
            property1['total_area'] = roof_measurements.get('total_area', '')
            property1['total_area_sqft'] = self.extract_numeric_value(property1['total_area'])
            
            pitch_str = roof_measurements.get('predominant_pitch', '')
            property1['predominant_pitch'] = pitch_str
            property1['predominant_pitch_degrees'] = self.parse_pitch(pitch_str)
            
            property1['num_of_facets'] = roof_measurements.get('total_roof_facets')
            
            # Extract lengths and estimate segment counts (divide by average segment length)
            # This converts total linear measurements to estimated segment counts for comparison
            # Typical segment lengths: ridges/hips/valleys ~20-30ft, eaves/rakes ~10-15ft, flashing ~5-10ft
            # Note: Property 2 has actual measured segment counts, Property 1 gets estimates

            ridges_length = self.extract_length(roof_measurements.get('ridges', ''))
            property1['num_of_ridges'] = max(1, round(ridges_length / 25)) if ridges_length else None  # ~25ft per ridge segment

            eaves_length = self.extract_length(roof_measurements.get('eaves_starter', ''))
            property1['num_of_eaves'] = max(1, round(eaves_length / 12)) if eaves_length else None  # ~12ft per eave segment

            rakes_length = self.extract_length(roof_measurements.get('rakes', ''))
            property1['num_of_rakes'] = max(1, round(rakes_length / 12)) if rakes_length else None  # ~12ft per rake segment

            valleys_length = self.extract_length(roof_measurements.get('valleys', ''))
            property1['num_of_valleys'] = max(1, round(valleys_length / 25)) if valleys_length else None  # ~25ft per valley segment

            hips_length = self.extract_length(roof_measurements.get('hips', ''))
            property1['num_of_hips'] = max(1, round(hips_length / 25)) if hips_length else None  # ~25ft per hip segment

            flashing_length = self.extract_length(roof_measurements.get('flashing', ''))
            property1['num_of_flashing'] = max(1, round(flashing_length / 8)) if flashing_length else None  # ~8ft per flashing segment
        
        # Extract house measurements
        if house_measurements:
            property1['number_of_stories'] = house_measurements.get('number_of_stories')
            property1['structure_complexity'] = house_measurements.get('structure_complexity')
            property1['estimated_attic'] = house_measurements.get('estimated_attic')
            property1['total_roof_obstructions'] = house_measurements.get('total_roof_obstructions')
        
        return property1
    
    def extract_property2_data(self, house2_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and normalize Property 2 data from house2_result_summary.json
        
        Args:
            house2_data: House 2 summary data
            
        Returns:
            Normalized property data
        """
        property2 = {
            'address': house2_data.get('address', 'Unknown'),
            'latitude': house2_data.get('latitude'),
            'longitude': house2_data.get('longitude'),
            'total_area_sqft': house2_data.get('area'),
            'predominant_pitch_degrees': house2_data.get('predominant_pitch'),
            'num_of_facets': house2_data.get('num_of_facets'),
            'num_of_ridges': house2_data.get('num_of_ridges'),
            'num_of_eaves': house2_data.get('num_of_eaves'),
            'num_of_rakes': house2_data.get('num_of_rakes'),
            'num_of_valleys': house2_data.get('num_of_valleys'),
            'num_of_hips': house2_data.get('num_of_hips'),
            'num_of_flashing': house2_data.get('num_of_flashing'),
            'area_json': house2_data.get('area_json', {})
        }
        
        return property2
    
    def calculate_differences(self, property1: Dict[str, Any], property2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate differences between Property 1 and Property 2
        
        Args:
            property1: Property 1 data
            property2: Property 2 data
            
        Returns:
            Dictionary with difference calculations
        """
        differences = {}
        
        # Area difference
        area1 = property1.get('total_area_sqft')
        area2 = property2.get('total_area_sqft')
        if area1 and area2:
            area_diff = area2 - area1
            area_diff_percent = (area_diff / area1) * 100 if area1 > 0 else 0
            differences['area'] = {
                'property1': area1,
                'property2': area2,
                'difference': area_diff,
                'difference_percent': round(area_diff_percent, 2)
            }
        
        # Pitch difference
        pitch1 = property1.get('predominant_pitch_degrees')
        pitch2 = property2.get('predominant_pitch_degrees')
        if pitch1 and pitch2:
            pitch_diff = pitch2 - pitch1
            differences['pitch'] = {
                'property1': round(pitch1, 4),
                'property2': round(pitch2, 4),
                'difference_degrees': round(pitch_diff, 4)
            }
        
        # Facet count difference
        facets1 = property1.get('num_of_facets')
        facets2 = property2.get('num_of_facets')
        if facets1 is not None and facets2 is not None:
            facets_diff = facets2 - facets1
            differences['facets'] = {
                'property1': facets1,
                'property2': facets2,
                'difference': facets_diff
            }
        
        # Ridge count difference
        ridges1 = property1.get('num_of_ridges')
        ridges2 = property2.get('num_of_ridges')
        if ridges1 is not None and ridges2 is not None:
            ridges_diff = ridges2 - ridges1
            differences['ridges'] = {
                'property1': ridges1,
                'property2': ridges2,
                'difference': ridges_diff
            }
        
        # Eave count difference
        eaves1 = property1.get('num_of_eaves')
        eaves2 = property2.get('num_of_eaves')
        if eaves1 is not None and eaves2 is not None:
            eaves_diff = eaves2 - eaves1
            differences['eaves'] = {
                'property1': eaves1,
                'property2': eaves2,
                'difference': eaves_diff
            }
        
        # Rake count difference
        rakes1 = property1.get('num_of_rakes')
        rakes2 = property2.get('num_of_rakes')
        if rakes1 is not None and rakes2 is not None:
            rakes_diff = rakes2 - rakes1
            differences['rakes'] = {
                'property1': rakes1,
                'property2': rakes2,
                'difference': rakes_diff
            }
        
        # Valley count difference
        valleys1 = property1.get('num_of_valleys')
        valleys2 = property2.get('num_of_valleys')
        if valleys1 is not None and valleys2 is not None:
            valleys_diff = valleys2 - valleys1
            differences['valleys'] = {
                'property1': valleys1,
                'property2': valleys2,
                'difference': valleys_diff
            }
        
        # Hip count difference
        hips1 = property1.get('num_of_hips')
        hips2 = property2.get('num_of_hips')
        if hips1 is not None and hips2 is not None:
            hips_diff = hips2 - hips1
            differences['hips'] = {
                'property1': hips1,
                'property2': hips2,
                'difference': hips_diff
            }
        
        # Flashing count difference
        flashing1 = property1.get('num_of_flashing')
        flashing2 = property2.get('num_of_flashing')
        if flashing1 is not None and flashing2 is not None:
            flashing_diff = flashing2 - flashing1
            differences['flashing'] = {
                'property1': flashing1,
                'property2': flashing2,
                'difference': flashing_diff
            }
        
        return differences
    
    def generate_comparison(self) -> Dict[str, Any]:
        """
        Generate final comparison between Property 1 and Property 2
        
        Returns:
            Dictionary with complete comparison results
        """
        try:
            # Load similarity search results
            logger.info(f"📖 Loading similarity search results from: {self.similarity_results_file}")
            if not os.path.exists(self.similarity_results_file):
                raise FileNotFoundError(f"Similarity results file not found: {self.similarity_results_file}")
            
            with open(self.similarity_results_file, 'r') as f:
                similarity_data = json.load(f)
            
            top_result = similarity_data.get('top_result')
            if not top_result:
                raise ValueError("No 'top_result' found in similarity_search_results.json")
            
            logger.info(f"✅ Found top result for property: {top_result.get('property_id')}")
            
            # Get final_results (re-ranked Level 2 results) for address and similarity_score
            final_results = similarity_data.get('final_results', [])
            logger.info(f"✅ Found {len(final_results)} final results (re-ranked Level 2)")
            
            # Load house2 summary
            logger.info(f"📖 Loading house2 summary from: {self.house2_summary_file}")
            if not os.path.exists(self.house2_summary_file):
                raise FileNotFoundError(f"House2 summary file not found: {self.house2_summary_file}")
            
            with open(self.house2_summary_file, 'r') as f:
                house2_data = json.load(f)
            
            logger.info(f"✅ Found house2 data for address: {house2_data.get('address')}")
            
            # Extract property data (pass final_results to get re-ranked similarity_score and address)
            logger.info("🔄 Extracting property data...")
            property1 = self.extract_property1_data(top_result, final_results)
            property2 = self.extract_property2_data(house2_data)
            
            # Calculate differences
            logger.info("🔄 Calculating differences...")
            differences = self.calculate_differences(property1, property2)
            
            # Build final comparison result
            comparison_result = {
                'property1': {
                    'label': 'Similar Property (Rank 1)',
                    'property_id': property1.get('property_id'),
                    'address': property1.get('address'),
                    'report_id': property1.get('report_id'),
                    'similarity_score': property1.get('similarity_score'),
                    'measurements': {
                        'total_area_sqft': property1.get('total_area_sqft'),
                        'total_area': property1.get('total_area'),
                        'predominant_pitch': property1.get('predominant_pitch'),
                        'predominant_pitch_degrees': property1.get('predominant_pitch_degrees'),
                        'num_of_facets': property1.get('num_of_facets'),
                        'num_of_ridges': property1.get('num_of_ridges'),
                        'num_of_eaves': property1.get('num_of_eaves'),
                        'num_of_rakes': property1.get('num_of_rakes'),
                        'num_of_valleys': property1.get('num_of_valleys'),
                        'num_of_hips': property1.get('num_of_hips'),
                        'num_of_flashing': property1.get('num_of_flashing'),
                        'number_of_stories': property1.get('number_of_stories'),
                        'structure_complexity': property1.get('structure_complexity'),
                        'estimated_attic': property1.get('estimated_attic'),
                        'total_roof_obstructions': property1.get('total_roof_obstructions')
                    }
                },
                'property2': {
                    'label': 'Query Property',
                    'address': property2.get('address'),
                    'latitude': property2.get('latitude'),
                    'longitude': property2.get('longitude'),
                    'measurements': {
                        'total_area_sqft': property2.get('total_area_sqft'),
                        'predominant_pitch_degrees': property2.get('predominant_pitch_degrees'),
                        'num_of_facets': property2.get('num_of_facets'),
                        'num_of_ridges': property2.get('num_of_ridges'),
                        'num_of_eaves': property2.get('num_of_eaves'),
                        'num_of_rakes': property2.get('num_of_rakes'),
                        'num_of_valleys': property2.get('num_of_valleys'),
                        'num_of_hips': property2.get('num_of_hips'),
                        'num_of_flashing': property2.get('num_of_flashing'),
                        'area_json': property2.get('area_json')
                    }
                },
                'differences': differences,
                'summary': {
                    'property1_address': property1.get('address'),
                    'property2_address': property2.get('address'),
                    'similarity_score': property1.get('similarity_score'),
                    'total_comparisons': len(differences)
                }
            }
            
            logger.info(f"✅ Comparison generated successfully")
            logger.info(f"   Property 1: {property1.get('address')}")
            logger.info(f"   Property 2: {property2.get('address')}")
            logger.info(f"   Comparisons: {len(differences)} metrics")
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"❌ Error generating comparison: {str(e)}")
            raise
    
    def save_comparison(self, comparison_result: Dict[str, Any], 
                       output_file: str = "final_comparison_result.json") -> str:
        """
        Save comparison result to JSON file
        
        Args:
            comparison_result: Comparison result dictionary
            output_file: Output file path
            
        Returns:
            Path to saved file
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(comparison_result, f, indent=2, default=str)
            logger.info(f"💾 Comparison saved to: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"❌ Error saving comparison: {str(e)}")
            raise
    
    def generate_llm_prompt(self, comparison_result: Dict[str, Any]) -> str:
        """
        Generate LLM prompt with address, similarity_score, and comparison data
        
        Args:
            comparison_result: Comparison result dictionary
            
        Returns:
            Formatted prompt string for LLM
        """
        property1 = comparison_result.get('property1', {})
        property2 = comparison_result.get('property2', {})
        differences = comparison_result.get('differences', {})
        
        # Build prompt
        prompt = f"""You are a roofing expert analyzing two similar properties. Please provide a comprehensive comparison summary.

PROPERTY 1 (Similar Property - Rank 1 from Search):
- Address: {property1.get('address', 'Unknown')}
- Similarity Score: {property1.get('similarity_score', 0):.4f}
- Rank: {property1.get('rank', 'N/A')}
- Property ID: {property1.get('property_id', 'Unknown')}
- Report ID: {property1.get('report_id', 'Unknown')}

PROPERTY 2 (Query Property):
- Address: {property2.get('address', 'Unknown')}
- Latitude: {property2.get('latitude', 'N/A')}
- Longitude: {property2.get('longitude', 'N/A')}

PROPERTY 1 MEASUREMENTS:
- Total Area: {property1.get('total_area_sqft', 'N/A')} sq ft ({property1.get('total_area', 'N/A')})
- Predominant Pitch: {property1.get('predominant_pitch', 'N/A')} ({property1.get('predominant_pitch_degrees', 'N/A')} degrees)
- Number of Facets: {property1.get('num_of_facets', 'N/A')}
- Estimated Ridge Segments: {property1.get('num_of_ridges', 'N/A')} (estimated from total length)
- Estimated Eave Segments: {property1.get('num_of_eaves', 'N/A')} (estimated from total length)
- Estimated Rake Segments: {property1.get('num_of_rakes', 'N/A')} (estimated from total length)
- Estimated Valley Segments: {property1.get('num_of_valleys', 'N/A')} (estimated from total length)
- Estimated Hip Segments: {property1.get('num_of_hips', 'N/A')} (estimated from total length)
- Estimated Flashing Segments: {property1.get('num_of_flashing', 'N/A')} (estimated from total length)
- Number of Stories: {property1.get('number_of_stories', 'N/A')}
- Structure Complexity: {property1.get('structure_complexity', 'N/A')}

PROPERTY 2 MEASUREMENTS:
- Total Area: {property2.get('total_area_sqft', 'N/A')} sq ft
- Predominant Pitch: {property2.get('predominant_pitch_degrees', 'N/A')} degrees
- Number of Facets: {property2.get('num_of_facets', 'N/A')}
- Ridge Segments: {property2.get('num_of_ridges', 'N/A')} (actual count)
- Eave Segments: {property2.get('num_of_eaves', 'N/A')} (actual count)
- Rake Segments: {property2.get('num_of_rakes', 'N/A')} (actual count)
- Valley Segments: {property2.get('num_of_valleys', 'N/A')} (actual count)
- Hip Segments: {property2.get('num_of_hips', 'N/A')} (actual count)
- Flashing Segments: {property2.get('num_of_flashing', 'N/A')} (actual count)

KEY DIFFERENCES:
"""
        
        # Add differences
        if differences.get('area'):
            diff = differences['area']
            prompt += f"- Area: Property 1 has {diff['property1']:.2f} sq ft, Property 2 has {diff['property2']:.2f} sq ft. "
            prompt += f"Difference: {diff['difference']:+.2f} sq ft ({diff['difference_percent']:+.2f}%)\n"
        
        if differences.get('pitch'):
            diff = differences['pitch']
            prompt += f"- Pitch: Property 1 has {diff['property1']:.2f}°, Property 2 has {diff['property2']:.2f}°. "
            prompt += f"Difference: {diff['difference_degrees']:+.2f}°\n"
        
        # Helper function to format difference values
        def format_diff(diff_val):
            if diff_val is None:
                return "N/A"
            if isinstance(diff_val, int):
                return f"{diff_val:+d}"
            elif isinstance(diff_val, float):
                if diff_val == int(diff_val):
                    return f"{int(diff_val):+d}"
                else:
                    return f"{diff_val:+.2f}"
            else:
                return str(diff_val)
        
        if differences.get('facets'):
            diff = differences['facets']
            prompt += f"- Facets: Property 1 has {diff['property1']}, Property 2 has {diff['property2']}. "
            prompt += f"Difference: {format_diff(diff['difference'])}\n"
        
        if differences.get('ridges'):
            diff = differences['ridges']
            prompt += f"- Ridges: Property 1 has {diff['property1']}, Property 2 has {diff['property2']}. "
            prompt += f"Difference: {format_diff(diff['difference'])}\n"
        
        if differences.get('eaves'):
            diff = differences['eaves']
            prompt += f"- Eaves: Property 1 has {diff['property1']}, Property 2 has {diff['property2']}. "
            prompt += f"Difference: {format_diff(diff['difference'])}\n"
        
        if differences.get('rakes'):
            diff = differences['rakes']
            prompt += f"- Rakes: Property 1 has {diff['property1']}, Property 2 has {diff['property2']}. "
            prompt += f"Difference: {format_diff(diff['difference'])}\n"
        
        if differences.get('valleys'):
            diff = differences['valleys']
            prompt += f"- Valleys: Property 1 has {diff['property1']}, Property 2 has {diff['property2']}. "
            prompt += f"Difference: {format_diff(diff['difference'])}\n"
        
        if differences.get('hips'):
            diff = differences['hips']
            prompt += f"- Hips: Property 1 has {diff['property1']}, Property 2 has {diff['property2']}. "
            prompt += f"Difference: {format_diff(diff['difference'])}\n"
        
        if differences.get('flashing'):
            diff = differences['flashing']
            prompt += f"- Flashing: Property 1 has {diff['property1']}, Property 2 has {diff['property2']}. "
            prompt += f"Difference: {format_diff(diff['difference'])}\n"
        
        prompt += f"""
TASK:
Please provide a comprehensive summary comparing these two properties. Note that Property 1 segment counts are ESTIMATES derived from total linear measurements, while Property 2 segment counts are ACTUAL measured values. Include:

1. Overall similarity assessment (based on similarity score of {property1.get('similarity_score', 0):.4f})
2. Key structural similarities and differences (focusing on estimated vs actual segment counts)
3. Roofing complexity comparison
4. Material estimation implications based on the segment count differences
5. Any notable observations about the roof structures and the estimation methodology

IMPORTANT: Treat Property 1 segment counts as rough estimates and Property 2 counts as precise measurements. Adjust material estimates accordingly.

Provide the summary in a clear, professional format suitable for roofing contractors and material estimators.
"""
        
        return prompt
    
    def call_llm(self, prompt: str) -> str:
        """
        Call AWS Bedrock LLM to generate summary
        
        Args:
            prompt: Prompt string for LLM
            
        Returns:
            LLM generated summary text
        """
        try:
            logger.info("🤖 Calling AWS Bedrock LLM...")
            
            # Prepare message content
            message_content = [{"type": "text", "text": prompt}]
            
            # Prepare request body for Claude
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8000,
                "messages": [
                    {
                        "role": "user",
                        "content": message_content
                    }
                ]
            }
            
            # Invoke Bedrock model (matching working pattern)
            response = self.bedrock_client.invoke_model(
                modelId=self.bedrock_model,
                body=json.dumps(body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract content from Claude response
            return response_body['content'][0]['text']
                
        except ClientError as e:
            logger.error(f"❌ AWS Bedrock error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Error calling LLM: {str(e)}")
            raise
    
    def save_llm_summary(self, summary: str, output_file: str = "llm_summary_result.txt") -> str:
        """
        Save LLM summary to text file
        
        Args:
            summary: LLM generated summary text
            output_file: Output file path
            
        Returns:
            Path to saved file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            logger.info(f"💾 LLM summary saved to: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"❌ Error saving LLM summary: {str(e)}")
            raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate final LLM comparison between Property 1 (top result) and Property 2 (house2)"
    )
    parser.add_argument(
        "--similarity-results",
        default="similarity_search_results.json",
        help="Path to similarity_search_results.json (default: similarity_search_results.json)"
    )
    parser.add_argument(
        "--house2-summary",
        default="house2_result_summary.json",
        help="Path to house2_result_summary.json (default: house2_result_summary.json)"
    )
    parser.add_argument(
        "--output",
        default="final_comparison_result.json",
        help="Output file path (default: final_comparison_result.json)"
    )
    parser.add_argument(
        "--llm-summary-output",
        default="llm_summary_result.txt",
        help="LLM summary output file path (default: llm_summary_result.txt)"
    )
    parser.add_argument(
        "--bedrock-region",
        default="us-east-1",
        help="AWS Bedrock region (default: us-east-1)"
    )
    parser.add_argument(
        "--bedrock-model",
        default="anthropic.claude-3-5-sonnet-20240620-v1:0",
        help="Bedrock model ID (default: anthropic.claude-3-5-sonnet-20240620-v1:0)"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM call and only generate comparison JSON"
    )
    
    args = parser.parse_args()
    
    print("\n🔍 Final LLM Result Generator")
    print("=" * 60)
    print(f"Similarity Results: {args.similarity_results}")
    print(f"House2 Summary: {args.house2_summary}")
    print(f"Output: {args.output}")
    if not args.skip_llm:
        print(f"LLM Summary Output: {args.llm_summary_output}")
        print(f"Bedrock Region: {args.bedrock_region}")
        print(f"Bedrock Model: {args.bedrock_model}")
    else:
        print("LLM: Skipped")
    print()
    
    try:
        # Initialize generator
        generator = FinalLLMResultGenerator(
            similarity_results_file=args.similarity_results,
            house2_summary_file=args.house2_summary,
            bedrock_region=args.bedrock_region,
            bedrock_model=args.bedrock_model
        )
        
        # Generate comparison
        comparison_result = generator.generate_comparison()
        
        # Save result
        output_path = generator.save_comparison(comparison_result, args.output)
        
        # Display summary
        print("\n📊 Comparison Summary")
        print("=" * 60)
        summary = comparison_result.get('summary', {})
        print(f"Property 1 (Similar): {summary.get('property1_address')}")
        print(f"Property 2 (Query): {summary.get('property2_address')}")
        print(f"Similarity Score: {summary.get('similarity_score', 0):.4f}")
        print(f"Comparisons Made: {summary.get('total_comparisons', 0)} metrics")
        
        if comparison_result.get('differences'):
            print("\n📈 Key Differences:")
            print("-" * 60)
            
            if 'area' in comparison_result['differences']:
                diff = comparison_result['differences']['area']
                print(f"Area: {diff['property1']:.2f} sqft → {diff['property2']:.2f} sqft "
                      f"({diff['difference']:+.2f} sqft, {diff['difference_percent']:+.2f}%)")
            
            if 'pitch' in comparison_result['differences']:
                diff = comparison_result['differences']['pitch']
                print(f"Pitch: {diff['property1']:.2f}° → {diff['property2']:.2f}° "
                      f"({diff['difference_degrees']:+.2f}°)")
            
            if 'facets' in comparison_result['differences']:
                diff = comparison_result['differences']['facets']
                print(f"Facets: {diff['property1']} → {diff['property2']} ({diff['difference']:+d})")
        
        print(f"\n💾 Full comparison saved to: {output_path}")
        
        # Generate LLM summary if not skipped
        if not args.skip_llm:
            print("\n🤖 Generating LLM Summary...")
            print("=" * 60)
            
            # Generate prompt
            prompt = generator.generate_llm_prompt(comparison_result)
            
            # Call LLM
            llm_summary = generator.call_llm(prompt)
            
            # Save LLM summary
            llm_output_path = generator.save_llm_summary(llm_summary, args.llm_summary_output)
            
            print(f"\n✅ LLM Summary Generated")
            print(f"💾 LLM summary saved to: {llm_output_path}")
            print(f"\n📝 Summary Preview (first 500 characters):")
            print("-" * 60)
            print(llm_summary[:500] + "..." if len(llm_summary) > 500 else llm_summary)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

