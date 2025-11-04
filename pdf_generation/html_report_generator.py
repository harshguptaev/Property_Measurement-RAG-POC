#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import argparse

try:
    from jinja2 import Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from data_extractor import RoofDataExtractor

class HTMLReportGenerator:
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.data_extractor = RoofDataExtractor()
        
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
        
        if JINJA2_AVAILABLE:
            self.env = Environment(
                loader=FileSystemLoader(self.templates_dir),
                trim_blocks=True,
                lstrip_blocks=True
            )
            def safe_format(value, fmt):
                try:
                    if value is None:
                        return 'N/A'
                    return fmt % value
                except (TypeError, ValueError):
                    return str(value)
            self.env.filters['format'] = safe_format
        else:
            print("Warning: Jinja2 not available. Using simple template substitution.")
            self.env = None
    
    def generate_html_report(self, data_folder: str, output_dir: str = "output", 
                           template_name: str = "PROFESSIONAL_ROOF_REPORT.html") -> Dict[str, Any]:
        
        if not os.path.exists(data_folder):
            return {'error': f"Data folder '{data_folder}' does not exist"}
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            print(f"Extracting data from: {data_folder}")
            extracted_data = self.data_extractor.extract_all_data(data_folder)
            
            print("Processing template data...")
            template_data = self._prepare_template_data(extracted_data, data_folder)
            
            folder_name = Path(data_folder).name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            html_output = os.path.join(output_dir, f"roof_report_{folder_name}_{timestamp}.html")
            
            print("Generating HTML report...")
            if JINJA2_AVAILABLE:
                html_content = self._generate_html_with_jinja2(template_data, template_name)
            else:
                html_content = self._generate_html_simple(template_data)
            
            if not html_content:
                return {'error': 'Failed to generate HTML content'}
            
            with open(html_output, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML report saved: {html_output}")
            
            result = {
                'html_path': html_output,
                'data_summary': self._create_data_summary(extracted_data),
                'success': True
            }
            
            print("HTML report generation completed successfully!")
            return result
            
        except Exception as e:
            print(f"Error during report generation: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'success': False}
    
    def _prepare_template_data(self, extracted_data: Dict[str, Any], data_folder: str) -> Dict[str, Any]:
        template_data = {
            'property_info': extracted_data.get('property_info', {}),
            'measurements': extracted_data.get('measurements', {}),
            'facets': extracted_data.get('facets', []),
            'pitch_data': extracted_data.get('pitch_data', {}),
            'area_data': extracted_data.get('area_data', {}),
            'images': extracted_data.get('images', {}),
            'coordinates': extracted_data.get('coordinates', {}),
            'metadata': extracted_data.get('metadata', {})
        }
        
        abs_data_folder = Path(data_folder).resolve()
        template_data['property_info']['data_folder_path'] = abs_data_folder
        
        if not template_data['property_info'].get('report_date'):
            template_data['property_info']['report_date'] = datetime.now().strftime('%B %d, %Y')
        
        if not template_data['property_info'].get('report_id'):
            template_data['property_info']['report_id'] = f"RR-{datetime.now().strftime('%Y%m%d')}-001"
        
        return template_data
    
    def _generate_html_with_jinja2(self, template_data: Dict[str, Any], template_name: str) -> str:
        try:
            template = self.env.get_template(template_name)
            html_content = template.render(**template_data)
            return html_content
        except Exception as e:
            print(f"Error generating HTML with Jinja2: {e}")
            return ""
    
    def _generate_html_simple(self, template_data: Dict[str, Any]) -> str:
        property_info = template_data['property_info']
        measurements = template_data['measurements']
        facets = template_data['facets']
        pitch_data = template_data['pitch_data']
        images = template_data['images']
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Roof Measurement Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #32507C; color: white; padding: 30px; text-align: center; margin: -40px -40px 30px -40px; }}
        .header h1 {{ font-size: 32px; margin: 0; }}
        .section {{ margin-bottom: 30px; }}
        .section-title {{ color: #32507C; font-size: 24px; border-bottom: 2px solid #32507C; padding-bottom: 10px; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 20px 0; }}
        .info-item {{ display: flex; justify-content: space-between; padding: 8px; background: #f8f9fa; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: white; border: 2px solid #e9ecef; padding: 20px; text-align: center; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #32507C; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #32507C; color: white; }}
        .image-container {{ text-align: center; margin: 20px 0; }}
        .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .image-caption {{ margin-top: 10px; font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Roof Measurement Report</h1>
        <div>Professional Property Assessment</div>
    </div>

    <div class="section">
        <h2 class="section-title">Property Information</h2>
        <div class="info-grid">
            <div class="info-item">
                <span><strong>Address:</strong></span>
                <span>{property_info.get('address', 'N/A')}</span>
            </div>
            <div class="info-item">
                <span><strong>Report ID:</strong></span>
                <span>{property_info.get('report_id', 'N/A')}</span>
            </div>
            <div class="info-item">
                <span><strong>Coordinates:</strong></span>
                <span>{property_info.get('latitude', 0):.6f}, {property_info.get('longitude', 0):.6f}</span>
            </div>
            <div class="info-item">
                <span><strong>Report Date:</strong></span>
                <span>{property_info.get('report_date', 'N/A')}</span>
            </div>
        </div>
    </div>

    <div class="section">
        <h2 class="section-title">Roof Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="metric-value">{measurements.get('total_area', 0):.0f}</div>
                <div class="metric-label">Total Area (sq ft)</div>
            </div>
            <div class="summary-card">
                <div class="metric-value">{measurements.get('predominant_pitch', 0):.1f}</div>
                <div class="metric-label">Predominant Pitch</div>
            </div>
            <div class="summary-card">
                <div class="metric-value">{measurements.get('num_facets', 0)}</div>
                <div class="metric-label">Number of Facets</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2 class="section-title">Facet Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Facet #</th>
                    <th>Area (sq ft)</th>
                    <th>Percentage</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for facet in facets:
            html_content += f"""
                <tr>
                    <td>{facet['facet_number']}</td>
                    <td>{facet['area_sqft']:.2f}</td>
                    <td>{facet['percentage']:.1f}%</td>
                    <td>{facet['confidence_score']*100:.1f}%</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2 class="section-title">Pitch Analysis</h2>
        <div class="summary-grid">
"""
        
        directions = pitch_data.get('directions', {})
        for direction, data in directions.items():
            html_content += f"""
            <div class="summary-card">
                <h4>{direction.title()} Side</h4>
                <div><strong>Pitch:</strong> {data['pitch_value']:.2f}</div>
                <div><strong>Ratio:</strong> {data['pitch_ratio']}</div>
                <div><strong>Angle:</strong> {data['pitch_angle']:.1f}°</div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
        
        main_images = images.get('main_images', {})
        if main_images:
            html_content += """
    <div class="section">
        <h2 class="section-title">Property Images</h2>
"""
            
            if 'top_view' in main_images:
                html_content += f"""
        <div class="image-container">
            <img src="{main_images['top_view']}" alt="Top View">
            <div class="image-caption">Aerial Top View</div>
        </div>
"""
            
            if 'response_top' in main_images:
                html_content += f"""
        <div class="image-container">
            <img src="{main_images['response_top']}" alt="Roof Analysis">
            <div class="image-caption">Roof Facet Analysis</div>
        </div>
"""
            
            html_content += """
    </div>
"""
        
        html_content += """
    <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #32507C; text-align: center; color: #666; font-size: 12px;">
        <p>This report was generated using automated roof measurement technology.</p>
        <p>Report generated on """ + property_info.get('report_date', 'N/A') + """ | ID: """ + property_info.get('report_id', 'N/A') + """</p>
    </div>
</body>
</html>
"""
        
        return html_content
    
    def _create_data_summary(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            'total_facets': len(extracted_data.get('facets', [])),
            'total_area': extracted_data.get('measurements', {}).get('total_area', 0),
            'pitch_directions': len(extracted_data.get('pitch_data', {}).get('directions', {})),
            'images_found': len(extracted_data.get('images', {}).get('main_images', {})),
            'has_coordinates': bool(extracted_data.get('coordinates', {}).get('pictometry_data')),
            'processing_models': list(extracted_data.get('metadata', {}).get('model_versions', {}).keys())
        }
        return summary

def main():
    parser = argparse.ArgumentParser(description='Generate HTML roof measurement reports')
    parser.add_argument('data_folder', help='Path to folder containing roof measurement data')
    parser.add_argument('--output', '-o', default='output', help='Output directory for reports')
    parser.add_argument('--template', '-t', default='PROFESSIONAL_ROOF_REPORT.html', help='Template file to use')
    parser.add_argument('--templates-dir', default='templates', help='Templates directory')
    
    args = parser.parse_args()
    
    try:
        generator = HTMLReportGenerator(args.templates_dir)
        result = generator.generate_html_report(args.data_folder, args.output, args.template)
        
        if result.get('success'):
            print(f"\n=== HTML REPORT GENERATED SUCCESSFULLY ===")
            print(f"HTML: {result['html_path']}")
            
            summary = result['data_summary']
            print(f"\n=== DATA SUMMARY ===")
            print(f"Total Area: {summary['total_area']:.2f} sq ft")
            print(f"Facets Found: {summary['total_facets']}")
            print(f"Pitch Directions: {summary['pitch_directions']}")
            print(f"Images Found: {summary['images_found']}")
            print(f"Has Coordinates: {summary['has_coordinates']}")
            
        else:
            print(f"Error: {result.get('error', 'Unknown error occurred')}")
            sys.exit(1)
    
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
