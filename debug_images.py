#!/usr/bin/env python3
"""
Debug script to understand Docling picture structure
"""
import sys
sys.path.append('src')

from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions

def debug_docling_images():
    """Debug what image data is available from Docling."""
    
    # Setup converter (simplified initialization)
    converter = DocumentConverter()
    
    # Test with one PDF
    test_files = list(Path("input_files").glob("*.pdf"))
    if not test_files:
        print("No PDF files found in input_files/")
        return
        
    test_file = test_files[0]
    print(f"Testing with: {test_file}")
    
    # Convert document
    try:
        result = converter.convert(test_file)
        converted_doc = result.document
        print(f"Document converted successfully")
        
        # Check document-level pictures
        if hasattr(converted_doc, 'pictures') and converted_doc.pictures:
            print(f"\n📊 Found {len(converted_doc.pictures)} document-level pictures")
            
            for i, picture in enumerate(converted_doc.pictures[:3]):  # Check first 3
                print(f"\n--- Picture {i} ---")
                print(f"Type: {type(picture)}")
                print(f"Attributes: {[attr for attr in dir(picture) if not attr.startswith('_')]}")
                
                # Test different ways to access image data
                if hasattr(picture, 'data'):
                    print(f"Has .data: {type(picture.data)}, size: {len(picture.data) if picture.data else 'None'}")
                else:
                    print("No .data attribute")
                    
                if hasattr(picture, 'image'):
                    print(f"Has .image: {type(picture.image)}")
                else:
                    print("No .image attribute")
                    
                if hasattr(picture, 'pil_image'):
                    print(f"Has .pil_image: {type(picture.pil_image)}")
                else:
                    print("No .pil_image attribute")
                    
                if hasattr(picture, 'get_image'):
                    try:
                        img = picture.get_image()
                        print(f"picture.get_image() returns: {type(img)}")
                    except Exception as e:
                        print(f"picture.get_image() failed: {e}")
                else:
                    print("No .get_image() method")
                    
                # Check all attributes for image-related ones
                image_attrs = [attr for attr in dir(picture) if 'image' in attr.lower() or 'data' in attr.lower()]
                if image_attrs:
                    print(f"Image-related attributes: {image_attrs}")
        else:
            print("No document-level pictures found")
            
        # Check page-level images
        if hasattr(converted_doc, 'pages') and converted_doc.pages:
            print(f"\n📄 Found {len(converted_doc.pages)} pages")
            
            for page_num, page in enumerate(converted_doc.pages[:2]):  # Check first 2 pages
                if hasattr(page, 'images') and page.images:
                    print(f"\n--- Page {page_num + 1} Images ---")
                    print(f"Found {len(page.images)} images on page")
                    
                    for img_idx, image_info in enumerate(page.images[:2]):  # Check first 2 images
                        print(f"\nPage {page_num + 1}, Image {img_idx}:")
                        print(f"Type: {type(image_info)}")
                        print(f"Attributes: {[attr for attr in dir(image_info) if not attr.startswith('_')]}")
                else:
                    print(f"Page {page_num + 1}: No images")
        else:
            print("No pages found")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from docling.datamodel.base_models import InputFormat
    debug_docling_images()
