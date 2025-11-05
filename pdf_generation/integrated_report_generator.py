#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import argparse

from pdf_generation.data_extractor import RoofDataExtractor

class IntegratedRoofReportGenerator:
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.data_extractor = RoofDataExtractor()
        
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
        
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        self.env.filters['format'] = lambda value, fmt: fmt % value if value is not None else 'N/A'
    
    def generate_complete_report(self, data_folder: str, output_dir: str = "output", 
                              template_name: str = "ROOF_REPORT_TEMPLATE.html") -> Dict[str, Any]:
        
        if not os.path.exists(data_folder):
            return {'error': f"Data folder '{data_folder}' does not exist", 'success': False}
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            print(f"📊 Extracting data from: {data_folder}")
            extracted_data = self.data_extractor.extract_all_data(data_folder)
            
            print("🔄 Processing template data...")
            template_data = self._prepare_template_data(extracted_data)
            
            folder_name = Path(data_folder).name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            html_output = os.path.join(output_dir, f"roof_report_{folder_name}_{timestamp}.html")
            pdf_output = os.path.join(output_dir, f"roof_report_{folder_name}_{timestamp}.pdf")
            
            print("📝 Generating HTML report...")
            html_content = self._generate_html(template_data, template_name)
            
            if not html_content:
                return {'error': 'Failed to generate HTML content', 'success': False}
            
            with open(html_output, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ HTML report saved: {html_output}")
            
            print("📄 Generating PDF report...")
            pdf_success = self._generate_pdf(html_content, pdf_output)
            
            result = {
                'html_path': html_output,
                'pdf_path': pdf_output if pdf_success else None,
                'data_summary': self._create_data_summary(extracted_data),
                'success': True
            }
            
            if pdf_success:
                print(f"✅ PDF report generated: {pdf_output}")
            else:
                print("⚠️  PDF generation failed - HTML report is available")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during report generation: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'success': False}
    
    def _prepare_template_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
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
        
        if not template_data['property_info'].get('report_date'):
            template_data['property_info']['report_date'] = datetime.now().strftime('%B %d, %Y')
        
        if not template_data['property_info'].get('report_id'):
            template_data['property_info']['report_id'] = f"RR-{datetime.now().strftime('%Y%m%d')}-001"
        
        return template_data
    
    def _generate_html(self, template_data: Dict[str, Any], template_name: str) -> str:
        try:
            template = self.env.get_template(template_name)
            html_content = template.render(**template_data)
            return html_content
        except Exception as e:
            print(f"❌ Error generating HTML: {e}")
            return ""
    
    def _generate_pdf(self, html_content: str, output_path: str) -> bool:
        try:
            from weasyprint import HTML
            html_doc = HTML(string=html_content, base_url=str(self.templates_dir))
            html_doc.write_pdf(output_path)
            return True
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            return False
    
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

