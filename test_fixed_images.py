#!/usr/bin/env python3
"""
Test the fixed image extraction
"""
import sys
sys.path.append('src')

from pathlib import Path
from docling.document_converter import DocumentConverter

def test_fixed_extraction():
    """Test the corrected image extraction method."""
    
    converter = DocumentConverter()
    test_file = Path("input_files/RoofReport-44995431.pdf")
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
        
    print(f"Testing fixed extraction with: {test_file}")
    
    try:
        result = converter.convert(test_file)
        converted_doc = result.document
        
        if hasattr(converted_doc, 'pictures') and converted_doc.pictures:
            print(f"Found {len(converted_doc.pictures)} pictures")
            
            # Test extraction for first few images
            for i, picture in enumerate(converted_doc.pictures[:3]):
                print(f"\n--- Testing Picture {i} ---")
                
                try:
                    # Use get_image with doc parameter
                    pil_image = picture.get_image(converted_doc)
                    if pil_image:
                        print(f"✅ Successfully extracted image {i}: {type(pil_image)}, size: {pil_image.size}")
                        
                        # Test saving
                        test_dir = Path("test_extracted")
                        test_dir.mkdir(exist_ok=True)
                        test_path = test_dir / f"test_image_{i}.png"
                        pil_image.save(test_path, format='PNG')
                        print(f"✅ Saved test image to: {test_path}")
                    else:
                        print(f"❌ get_image() returned None for picture {i}")
                        
                except Exception as e:
                    print(f"❌ Error extracting picture {i}: {e}")
        else:
            print("No pictures found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_extraction()
