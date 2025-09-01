#!/usr/bin/env python3
"""
Demo script showing how to use Gemini Pro Vision for image analysis
in the Property Measurement RAG system.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.gemini_client import GeminiVisionClient, is_gemini_available
from src.image_utils import ImageManager
from src.config import config

def main():
    """Demonstrate Gemini Vision capabilities."""
    
    print("🔍 Property Measurement RAG - Gemini Vision Demo")
    print("=" * 50)
    
    # Check if Gemini is available
    if not is_gemini_available():
        print("❌ Gemini API not available. Please set GEMINI_API_KEY environment variable.")
        print("\nTo get started:")
        print("1. Get API key from: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable: export GEMINI_API_KEY='your-api-key'")
        print("3. Install requirements: pip install google-generativeai")
        return
    
    print("✅ Gemini API available")
    
    try:
        # Initialize Gemini client
        gemini_config = config.get_gemini_config()
        client = GeminiVisionClient(
            api_key=gemini_config["api_key"],
            model_name=gemini_config["model_name"],
            temperature=gemini_config["temperature"]
        )
        
        print(f"✅ Initialized Gemini client with model: {gemini_config['model_name']}")
        
        # Initialize image manager
        image_manager = ImageManager(enable_gemini=True)
        
        # Find sample images
        images_dir = Path("extracted_images")
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            print("Run the document processing pipeline first to extract images.")
            return
        
        # Get all available reports
        reports = image_manager.get_all_reports()
        if not reports:
            print("❌ No report images found.")
            print("Process some PDF documents first to extract images.")
            return
        
        print(f"📂 Found {len(reports)} reports with images")
        
        # Demo 1: Analyze a single image
        print("\n🔍 Demo 1: Single Image Analysis")
        print("-" * 30)
        
        # Get first available image
        first_report = reports[0]
        image_paths = image_manager.list_images_for_report(first_report)
        
        if image_paths:
            sample_image = image_paths[0]
            print(f"Analyzing: {sample_image.name}")
            
            # Basic caption
            caption = client.caption_image(sample_image)
            print(f"Caption: {caption}")
            
            # Roof analysis
            roof_analysis = client.analyze_roof_image(sample_image)
            print(f"\nRoof Analysis:")
            for key, value in roof_analysis.items():
                if key != "full_analysis":
                    print(f"  {key}: {value}")
            
            # Measurement extraction
            measurements = client.extract_measurements(sample_image)
            print(f"\nMeasurements: {measurements.get('has_measurements', False)}")
            if measurements.get("extracted_data"):
                for measurement in measurements["extracted_data"]:
                    print(f"  {measurement['full_match']}")
        
        # Demo 2: Batch analysis of a report
        print(f"\n🔍 Demo 2: Batch Analysis of Report {first_report}")
        print("-" * 40)
        
        batch_results = image_manager.batch_analyze_report_images(first_report)
        
        if "error" not in batch_results:
            print(f"Total images analyzed: {batch_results['total_images']}")
            print(f"Roof images: {len(batch_results.get('roof_analysis', []))}")
            print(f"Measurement images: {len(batch_results.get('measurement_analysis', []))}")
            print(f"General images: {len(batch_results.get('general_analysis', []))}")
            
            # Show sample roof analysis
            if batch_results.get('roof_analysis'):
                sample_roof = batch_results['roof_analysis'][0]
                print(f"\nSample roof analysis:")
                print(f"  Image: {Path(sample_roof['image_path']).name}")
                if 'roof_type' in sample_roof:
                    print(f"  Roof Type: {sample_roof.get('roof_type', 'Unknown')}")
                    print(f"  Material: {sample_roof.get('material', 'Unknown')}")
                    print(f"  Condition: {sample_roof.get('condition', 'Unknown')}")
        
        # Demo 3: Image enhancement example
        print(f"\n🔍 Demo 3: Metadata Enhancement")
        print("-" * 35)
        
        # Create sample metadata
        sample_metadata = {
            'image_file_path': str(sample_image),
            'image_label': sample_image.stem,
            'report_id': first_report,
            'searchable_keywords': ['roof', 'inspection']
        }
        
        enhanced_metadata = image_manager.enhance_image_metadata_with_gemini(sample_metadata)
        
        if enhanced_metadata.get('gemini_enhanced'):
            print("✅ Metadata enhanced with Gemini analysis")
            print(f"Keywords: {len(enhanced_metadata.get('searchable_keywords', []))}")
            if 'gemini_analysis' in enhanced_metadata:
                analysis = enhanced_metadata['gemini_analysis']
                print(f"Analysis type: {analysis.get('analysis_type', 'roof')}")
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Process your PDF documents with: python main.py")
        print("2. The system will automatically use Gemini for image analysis")
        print("3. Search queries will benefit from enhanced image descriptions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
