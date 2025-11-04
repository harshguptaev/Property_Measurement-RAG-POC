#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate roof measurement PDF reports')
    parser.add_argument('data_folder', help='Path to folder containing roof measurement data')
    parser.add_argument('--output', '-o', default='output', help='Output directory for reports')
    parser.add_argument('--html', action='store_true', help='Generate HTML instead of PDF')
    parser.add_argument('--template', '-t', default='EAGLEVIEW_STYLE_REPORT.html', help='Template file to use')
    parser.add_argument('--templates-dir', default='templates', help='Templates directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_folder):
        print(f"Error: Data folder '{args.data_folder}' does not exist")
        sys.exit(1)
    
    if not args.html:
        print("Generating professional HTML report (PDF conversion available)...")
        print("Note: PDF generation requires system dependencies. HTML reports can be converted to PDF using:")
        print("  - Browser: Open HTML file and 'Print to PDF'")
        print("  - Online tools: Convert HTML to PDF online")
        print("  - wkhtmltopdf: brew install wkhtmltopdf && wkhtmltopdf input.html output.pdf")
        print()

        # Generate HTML first
        args.html = True  # Force HTML generation
        html_result = None

        if args.html:
            from html_report_generator import HTMLReportGenerator
            print("Generating HTML report...")
            generator = HTMLReportGenerator(args.templates_dir)
            html_result = generator.generate_html_report(args.data_folder, args.output, args.template)

        if html_result and html_result.get('success'):
            html_path = html_result['html_path']
            pdf_path = html_path.replace('.html', '.pdf')

            print(f"\n=== HTML REPORT GENERATED SUCCESSFULLY ===")
            print(f"HTML: {html_path}")
            print(f"📄 To convert to PDF, use one of these methods:")
            print(f"   1. Browser: Open {html_path} → Print → Save as PDF")
            print(f"   2. Command: wkhtmltopdf '{html_path}' '{pdf_path}'")
            print(f"   3. Online: Upload HTML file to any HTML-to-PDF converter")

            summary = html_result['data_summary']
            print(f"\n=== DATA SUMMARY ===")
            print(f"Total Area: {summary['total_area']:.2f} sq ft")
            print(f"Facets Found: {summary['total_facets']}")
            print(f"Pitch Directions: {summary['pitch_directions']}")
            print(f"Images Found: {summary['images_found']}")

            # Try basic PDF conversion methods
            print("\nAttempting PDF conversion...")
            try:
                # Method 1: Try wkhtmltopdf if available
                import subprocess
                try:
                    result = subprocess.run(['wkhtmltopdf', '--enable-local-file-access',
                                          '--footer-center', '© EagleView Technologies 2025 All rights reserved. ',
                                          '--footer-font-size', '8',
                                          '--footer-spacing', '5',
                                          html_path, pdf_path],
                                          capture_output=True, timeout=30)
                    if result.returncode == 0:
                        print(f"✓ PDF created using wkhtmltopdf: {pdf_path}")
                        return
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

                # Method 2: Try Chrome if available
                try:
                    chrome_cmds = ['chromium-browser', 'google-chrome', 'chrome']
                    for cmd in chrome_cmds:
                        try:
                            result = subprocess.run([
                                cmd, '--headless', '--disable-gpu',
                                '--print-to-pdf-no-header',
                                f'--print-to-pdf={pdf_path}',
                                f'file://{html_path}'
                            ], capture_output=True, timeout=30)
                            if result.returncode == 0:
                                print(f"✓ PDF created using {cmd}: {pdf_path}")
                                return
                        except (subprocess.TimeoutExpired, FileNotFoundError):
                            continue
                except Exception:
                    pass

                print("ℹ️  PDF conversion methods not available, but HTML report is ready")
                print("   Use browser 'Print to PDF' or install wkhtmltopdf for automatic conversion")

            except Exception as e:
                print(f"ℹ️  PDF conversion failed: {e}")
                print("   HTML report is ready - convert manually using browser Print → PDF")

        else:
            print(f"Error generating HTML report: {html_result.get('error', 'Unknown error') if html_result else 'Failed to generate'}")
            sys.exit(1)

    if args.html:
        from html_report_generator import HTMLReportGenerator
        print("Using HTML-only generator...")
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
            
        else:
            print(f"Error: {result.get('error', 'Unknown error occurred')}")
            sys.exit(1)

if __name__ == "__main__":
    main()
