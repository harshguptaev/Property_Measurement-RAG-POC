#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import base64
from datetime import datetime
import math

class RoofDataExtractor:
    def __init__(self):
        self.supported_image_formats = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    
    def extract_all_data(self, data_folder: str) -> Dict[str, Any]:
        data_path = Path(data_folder)
        
        extracted_data = {
            'property_info': self._extract_property_info(data_path),
            'measurements': self._extract_measurements(data_path),
            'facets': self._extract_facet_data(data_path),
            'pitch_data': self._extract_pitch_data(data_path),
            'area_data': self._extract_area_data(data_path),
            'images': self._extract_images(data_path),
            'coordinates': self._extract_coordinates(data_path),
            'metadata': self._extract_metadata(data_path)
        }
        
        return extracted_data
    
    def _extract_property_info(self, data_path: Path) -> Dict[str, Any]:
        property_info = {
            'address': 'N/A',
            'latitude': 0.0,
            'longitude': 0.0,
            'report_date': datetime.now().strftime('%B %d, %Y'),
            'report_time': datetime.now().strftime('%I:%M %p'),
            'report_id': self._generate_report_id(data_path)
        }
        
        final_data_file = data_path / "final_data.json"
        if final_data_file.exists():
            try:
                with open(final_data_file, 'r') as f:
                    data = json.load(f)
                    property_info.update({
                        'address': data.get('address', 'N/A'),
                        'latitude': data.get('latitude', 0.0),
                        'longitude': data.get('longitude', 0.0)
                    })
            except Exception as e:
                print(f"Error reading final_data.json: {e}")
        
        return property_info
    
    def _extract_measurements(self, data_path: Path) -> Dict[str, Any]:
        measurements = {
            'total_area': 0.0,
            'predominant_pitch': 0.0,
            'num_facets': 0,
            'num_ridges': 0,
            'num_eaves': 0,
            'num_rakes': 0,
            'num_valleys': 0,
            'num_hips': 0,
            'num_flashing': 0,
            'perimeter': 0.0
        }
        
        final_data_file = data_path / "final_data.json"
        if final_data_file.exists():
            try:
                with open(final_data_file, 'r') as f:
                    data = json.load(f)
                    measurements.update({
                        'total_area': round(data.get('area', 0.0), 2),
                        'predominant_pitch': round(data.get('predominant_pitch', 0.0), 2),
                        'num_facets': data.get('num_of_facets', 0),
                        'num_ridges': data.get('num_of_ridges', 0),
                        'num_eaves': data.get('num_of_eaves', 0),
                        'num_rakes': data.get('num_of_rakes', 0),
                        'num_valleys': data.get('num_of_valleys', 0),
                        'num_hips': data.get('num_of_hips', 0),
                        'num_flashing': data.get('num_of_flashing', 0)
                    })
            except Exception as e:
                print(f"Error reading measurements: {e}")
        
        return measurements
    
    def _extract_facet_data(self, data_path: Path) -> List[Dict[str, Any]]:
        facets = []
        
        response_file = data_path / "response.json"
        area_file = data_path / "area.json"
        
        area_data = {}
        if area_file.exists():
            try:
                with open(area_file, 'r') as f:
                    area_data = json.load(f)
            except Exception as e:
                print(f"Error reading area data: {e}")
        
        if response_file.exists():
            try:
                with open(response_file, 'r') as f:
                    response_data = json.load(f)
                    predictions = response_data.get('predictions', [])
                    
                    for i, prediction in enumerate(predictions, 1):
                        facet_key = f"facet{i}"
                        area_value = area_data.get(facet_key, 0.0)
                        total_area = area_data.get('total_sqft', 1.0)
                        
                        facet_info = {
                            'facet_number': i,
                            'area_sqft': round(area_value, 2),
                            'percentage': round((area_value / total_area) * 100, 1) if total_area > 0 else 0,
                            'bbox': prediction.get('bbox', []),
                            'contour': prediction.get('contour', []),
                            'confidence_score': round(prediction.get('score', 0), 3),
                            'class': prediction.get('class', 'facets')
                        }
                        facets.append(facet_info)
            except Exception as e:
                print(f"Error reading facet data: {e}")
        
        return facets
    
    def _extract_pitch_data(self, data_path: Path) -> Dict[str, Any]:
        pitch_data = {
            'directions': {},
            'average_pitch': 0.0,
            'pitch_variations': []
        }
        
        pitch_file = data_path / "pitch_response.json"
        if pitch_file.exists():
            try:
                with open(pitch_file, 'r') as f:
                    data = json.load(f)
                    
                    pitch_values = []
                    for direction, direction_data in data.items():
                        if isinstance(direction_data, dict) and 'predictions' in direction_data:
                            predictions = direction_data['predictions']
                            if predictions and len(predictions) > 0:
                                pitch_value = predictions[0].get('value', 0)
                                pitch_values.append(pitch_value)
                                
                                pitch_data['directions'][direction] = {
                                    'pitch_value': round(pitch_value, 2),
                                    'pitch_ratio': self._calculate_pitch_ratio(pitch_value),
                                    'pitch_angle': self._calculate_pitch_angle(pitch_value),
                                    'job_id': direction_data.get('jobId', ''),
                                    'model_version': direction_data.get('version', '')
                                }
                    
                    if pitch_values:
                        pitch_data['average_pitch'] = round(sum(pitch_values) / len(pitch_values), 2)
                        pitch_data['pitch_variations'] = [
                            abs(p - pitch_data['average_pitch']) for p in pitch_values
                        ]
                        
            except Exception as e:
                print(f"Error reading pitch data: {e}")
        
        return pitch_data
    
    def _extract_area_data(self, data_path: Path) -> Dict[str, Any]:
        area_data = {
            'total_area': 0.0,
            'facet_areas': {},
            'largest_facet': {},
            'smallest_facet': {},
            'area_distribution': []
        }
        
        area_file = data_path / "area.json"
        if area_file.exists():
            try:
                with open(area_file, 'r') as f:
                    data = json.load(f)
                    
                    area_data['total_area'] = round(data.get('total_sqft', 0.0), 2)
                    
                    facet_areas = {}
                    for key, value in data.items():
                        if key.startswith('facet') and key != 'total_sqft':
                            facet_num = int(key.replace('facet', ''))
                            facet_areas[facet_num] = round(value, 2)
                    
                    area_data['facet_areas'] = facet_areas
                    
                    if facet_areas:
                        largest = max(facet_areas.items(), key=lambda x: x[1])
                        smallest = min(facet_areas.items(), key=lambda x: x[1])
                        
                        area_data['largest_facet'] = {
                            'facet_number': largest[0],
                            'area': largest[1],
                            'percentage': round((largest[1] / area_data['total_area']) * 100, 1)
                        }
                        
                        area_data['smallest_facet'] = {
                            'facet_number': smallest[0],
                            'area': smallest[1],
                            'percentage': round((smallest[1] / area_data['total_area']) * 100, 1)
                        }
                        
                        area_data['area_distribution'] = [
                            {
                                'facet': facet_num,
                                'area': area,
                                'percentage': round((area / area_data['total_area']) * 100, 1)
                            }
                            for facet_num, area in sorted(facet_areas.items())
                        ]
                        
            except Exception as e:
                print(f"Error reading area data: {e}")
        
        return area_data
    
    def _extract_images(self, data_path: Path) -> Dict[str, Any]:
        images = {
            'main_images': {},
            'directional_images': {},
            'processed_images': {},
            'image_metadata': {}
        }
        
        main_image_files = {
            'top_view': 'top.jpg',
            'response_overview': 'response.png',
            'response_top': 'response-top.png',
            'response_top_lengths': 'response-top-lengths.png',
            'top_cropped': 'Top_cropped.png',
            'south_cropped': 'South_cropped.png',
            'north_cropped': 'North_cropped.png',
            'east_cropped': 'East_cropped.png',
            'west_cropped': 'West_cropped.png'
        }
        
        for key, filename in main_image_files.items():
            image_path = data_path / filename
            if image_path.exists():
                images['main_images'][key] = self._encode_image(image_path)
        
        for direction in ['top', 'south', 'north', 'east', 'west']:
            dir_path = data_path / direction
            if dir_path.exists():
                direction_images = {}
                for img_file in dir_path.glob('*.png'):
                    direction_images[img_file.stem] = self._encode_image(img_file)
                if direction_images:
                    images['directional_images'][direction] = direction_images
        
        return images
    
    def _extract_coordinates(self, data_path: Path) -> Dict[str, Any]:
        coordinates = {
            'pictometry_data': {},
            'gserve_data': {},
            'geometry': {}
        }
        
        pictometry_file = data_path / "pictometry_response.json"
        if pictometry_file.exists():
            try:
                with open(pictometry_file, 'r') as f:
                    coordinates['pictometry_data'] = json.load(f)
            except Exception as e:
                print(f"Error reading pictometry data: {e}")
        
        gserve_file = data_path / "gserve_response.json"
        if gserve_file.exists():
            try:
                with open(gserve_file, 'r') as f:
                    coordinates['gserve_data'] = json.load(f)
            except Exception as e:
                print(f"Error reading gserve data: {e}")
        
        return coordinates
    
    def _extract_metadata(self, data_path: Path) -> Dict[str, Any]:
        metadata = {
            'processing_info': {},
            'model_versions': {},
            'job_ids': {},
            'timestamps': {}
        }
        
        for json_file in data_path.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    if 'jobId' in data:
                        metadata['job_ids'][json_file.stem] = data['jobId']
                    
                    if 'version' in data:
                        metadata['model_versions'][json_file.stem] = data['version']
                    
                    if 'model_name' in data:
                        metadata['processing_info'][json_file.stem] = {
                            'model': data['model_name'],
                            'status': data.get('status', 'unknown')
                        }
                        
            except Exception as e:
                print(f"Error reading metadata from {json_file}: {e}")
        
        return metadata
    
    def _encode_image(self, image_path: Path) -> Optional[str]:
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
                file_ext = image_path.suffix.lower()
                
                if file_ext == '.jpg' or file_ext == '.jpeg':
                    mime_type = 'image/jpeg'
                elif file_ext == '.png':
                    mime_type = 'image/png'
                elif file_ext == '.gif':
                    mime_type = 'image/gif'
                elif file_ext == '.webp':
                    mime_type = 'image/webp'
                else:
                    mime_type = 'image/png'
                
                return f"data:{mime_type};base64,{image_data}"
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def _calculate_pitch_ratio(self, pitch_value: float) -> str:
        if pitch_value <= 0:
            return "N/A"
        try:
            ratio = 12 / pitch_value
            return f"1:{ratio:.1f}"
        except:
            return "N/A"
    
    def _calculate_pitch_angle(self, pitch_value: float) -> float:
        if pitch_value <= 0:
            return 0.0
        try:
            angle_rad = math.atan(pitch_value / 12)
            angle_deg = math.degrees(angle_rad)
            return round(angle_deg, 1)
        except:
            return 0.0
    
    def _generate_report_id(self, data_path: Path) -> str:
        folder_name = data_path.name
        timestamp = datetime.now().strftime('%Y%m%d')
        
        if '_' in folder_name:
            coords = folder_name.replace('_', '')
            return f"RR-{timestamp}-{coords[:8]}"
        else:
            return f"RR-{timestamp}-{folder_name[:8]}"
    
    def save_extracted_data(self, data_folder: str, output_file: str = None) -> str:
        if output_file is None:
            folder_name = Path(data_folder).name
            output_file = f"extracted_data_{folder_name}.json"
        
        extracted_data = self.extract_all_data(data_folder)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(extracted_data, f, indent=2, default=str)
            print(f"Extracted data saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving extracted data: {e}")
            return ""

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract roof measurement data from folder')
    parser.add_argument('data_folder', help='Path to folder containing roof measurement data')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_folder):
        print(f"Error: Data folder '{args.data_folder}' does not exist")
        return
    
    extractor = RoofDataExtractor()
    output_file = extractor.save_extracted_data(args.data_folder, args.output)
    
    if output_file:
        print(f"Data extraction completed successfully!")
    else:
        print("Data extraction failed!")

if __name__ == "__main__":
    main()
