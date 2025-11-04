#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import weasyprint
from jinja2 import Environment, FileSystemLoader
import argparse

from data_extractor import RoofDataExtractor

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
            return {'error': f"Data folder '{data_folder}' does not exist"}
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            print(f"Extracting data from: {data_folder}")
            extracted_data = self.data_extractor.extract_all_data(data_folder)
            
            print("Processing template data...")
            template_data = self._prepare_template_data(extracted_data)
            
            folder_name = Path(data_folder).name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            html_output = os.path.join(output_dir, f"roof_report_{folder_name}_{timestamp}.html")
            pdf_output = os.path.join(output_dir, f"roof_report_{folder_name}_{timestamp}.pdf")
            
            print("Generating HTML report...")
            html_content = self._generate_html(template_data, template_name)
            
            if not html_content:
                return {'error': 'Failed to generate HTML content'}
            
            with open(html_output, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML report saved: {html_output}")
            
            print("Generating PDF report...")
            pdf_success = self._generate_pdf(html_content, pdf_output)
            
            result = {
                'html_path': html_output,
                'pdf_path': pdf_output if pdf_success else None,
                'data_summary': self._create_data_summary(extracted_data),
                'success': True
            }
            
            print("Report generation completed successfully!")
            return result
            
        except Exception as e:
            print(f"Error during report generation: {e}")
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
            print(f"Error generating HTML: {e}")
            return ""
    
    def _generate_pdf(self, html_content: str, output_path: str) -> bool:
        try:
            html_doc = weasyprint.HTML(string=html_content, base_url=str(self.templates_dir))
            html_doc.write_pdf(output_path)
            print(f"PDF report generated: {output_path}")
            return True
        except Exception as e:
            print(f"Error generating PDF: {e}")
            print("Note: Make sure weasyprint and its dependencies are properly installed")
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
    
    def batch_generate_reports(self, data_folders: list, output_dir: str = "batch_output", 
                             template_name: str = "ROOF_REPORT_TEMPLATE.html") -> Dict[str, Any]:
        
        os.makedirs(output_dir, exist_ok=True)
        results = {
            'successful': [],
            'failed': [],
            'summary': {}
        }
        
        for folder in data_folders:
            print(f"\n--- Processing: {folder} ---")
            
            if not os.path.exists(folder):
                print(f"Skipping non-existent folder: {folder}")
                results['failed'].append({'folder': folder, 'error': 'Folder does not exist'})
                continue
            
            folder_output_dir = os.path.join(output_dir, Path(folder).name)
            result = self.generate_complete_report(folder, folder_output_dir, template_name)
            
            if result.get('success'):
                results['successful'].append({
                    'folder': folder,
                    'html_path': result['html_path'],
                    'pdf_path': result['pdf_path'],
                    'data_summary': result['data_summary']
                })
            else:
                results['failed'].append({
                    'folder': folder,
                    'error': result.get('error', 'Unknown error')
                })
        
        results['summary'] = {
            'total_processed': len(data_folders),
            'successful_count': len(results['successful']),
            'failed_count': len(results['failed']),
            'success_rate': f"{(len(results['successful']) / len(data_folders)) * 100:.1f}%" if data_folders else "0%"
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive roof measurement reports')
    parser.add_argument('data_folder', nargs='?', help='Path to folder containing roof measurement data')
    parser.add_argument('--batch', '-b', nargs='+', help='Process multiple data folders')
    parser.add_argument('--output', '-o', default='output', help='Output directory for reports')
    parser.add_argument('--template', '-t', default='ROOF_REPORT_TEMPLATE.html', help='Template file to use')
    parser.add_argument('--templates-dir', default='templates', help='Templates directory')
    
    args = parser.parse_args()
    
    if not args.data_folder and not args.batch:
        print("Error: Please provide either a data folder or use --batch with multiple folders")
        parser.print_help()
        return
    
    try:
        generator = IntegratedRoofReportGenerator(args.templates_dir)
        
        if args.batch:
            print(f"Starting batch processing of {len(args.batch)} folders...")
            results = generator.batch_generate_reports(args.batch, args.output, args.template)
            
            print(f"\n=== BATCH PROCESSING SUMMARY ===")
            print(f"Total folders processed: {results['summary']['total_processed']}")
            print(f"Successful: {results['summary']['successful_count']}")
            print(f"Failed: {results['summary']['failed_count']}")
            print(f"Success rate: {results['summary']['success_rate']}")
            
            if results['failed']:
                print(f"\nFailed folders:")
                for failed in results['failed']:
                    print(f"  - {failed['folder']}: {failed['error']}")
        
        else:
            print(f"Processing single folder: {args.data_folder}")
            result = generator.generate_complete_report(args.data_folder, args.output, args.template)
            
            if result.get('success'):
                print(f"\n=== REPORT GENERATED SUCCESSFULLY ===")
                print(f"HTML: {result['html_path']}")
                if result['pdf_path']:
                    print(f"PDF: {result['pdf_path']}")
                
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
        sys.exit(1)

if __name__ == "__main__":
    main()
