#!/usr/bin/env python3
"""
Test script for Docling-based document processing.
"""
import os
import sys
from pathlib import Path
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_docling_import():
    """Test if Docling can be imported."""
    print("🧪 Testing Docling import...")
    try:
        from src.docling_index import DOCLING_AVAILABLE, DoclingProcessor
        if DOCLING_AVAILABLE:
            print("✅ Docling is available and ready!")
            return True
        else:
            print("❌ Docling is not available. Please install with: pip install docling")
            return False
    except ImportError as e:
        print(f"❌ Failed to import Docling modules: {e}")
        return False


def test_aws_setup():
    """Test AWS setup."""
    print("\n🧪 Testing AWS configuration...")
    try:
        import boto3
        # Try to create a client
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
        print("✅ AWS Bedrock client created successfully")
        return True
    except Exception as e:
        print(f"❌ AWS setup issue: {e}")
        return False


def test_document_processing():
    """Test document processing capabilities."""
    print("\n🧪 Testing document processing...")
    
    try:
        from src.docling_index import DoclingProcessor, DOCLING_AVAILABLE
        
        if not DOCLING_AVAILABLE:
            print("❌ Docling not available, skipping document processing test")
            return False
        
        # Create processor
        processor = DoclingProcessor()
        print("✅ DoclingProcessor created successfully")
        
        # Test supported file types
        print(f"📄 Supported file types: {processor.supported_extensions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False


def test_vector_store():
    """Test vector store functionality."""
    print("\n🧪 Testing vector store...")
    
    try:
        from src.vector_store import VectorStoreManager
        from src.config import config
        
        # Test FAISS
        try:
            import faiss
            print("✅ FAISS is available")
        except ImportError:
            print("❌ FAISS not available")
            return False
        
        # Test configuration
        vector_config = config.get_vector_store_config()
        print(f"📊 Vector store config: {vector_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False


def create_sample_content():
    """Create sample content for testing."""
    print("\n📝 Creating sample content...")
    
    input_dir = Path("input_files")
    input_dir.mkdir(exist_ok=True)
    
    # Create a sample text file for testing
    sample_file = input_dir / "sample_test.txt"
    if not sample_file.exists():
        with open(sample_file, "w") as f:
            f.write("""Sample Property Document

This is a test document for the Property Retrieval System with Docling.

Property Details:
- Address: 123 Main Street
- Type: Residential
- Size: 2,500 sq ft
- Bedrooms: 3
- Bathrooms: 2
- Year Built: 2010

Features:
- Modern kitchen with granite countertops
- Hardwood floors throughout
- Two-car garage
- Landscaped backyard
- Energy-efficient appliances

This document tests the text processing capabilities of the system.
""")
        print(f"✅ Created sample file: {sample_file}")
    else:
        print(f"📄 Sample file already exists: {sample_file}")
    
    return sample_file


def test_end_to_end():
    """Test end-to-end processing with sample content."""
    print("\n🧪 Testing end-to-end processing...")
    
    try:
        # Create sample content
        sample_file = create_sample_content()
        
        # Test processing
        from src.docling_index import DoclingProcessor, DOCLING_AVAILABLE
        
        if not DOCLING_AVAILABLE:
            print("❌ Docling not available for end-to-end test")
            return False
        
        processor = DoclingProcessor()
        
        # Process the sample file
        documents = processor.process_file(str(sample_file), extract_images=False)
        
        print(f"✅ Processed {len(documents)} document chunks")
        
        if documents:
            first_doc = documents[0]
            print(f"📄 First chunk preview: {first_doc.page_content[:100]}...")
            print(f"🏷️  Metadata: {first_doc.metadata}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🏠 Property Retrieval with Docling - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Docling Import", test_docling_import),
        ("AWS Setup", test_aws_setup),
        ("Document Processing", test_document_processing),
        ("Vector Store", test_vector_store),
        ("End-to-End Processing", test_end_to_end)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Add your PDF files to the 'input_files' directory")
        print("2. Run: python main.py")
        print("3. Access the web interface at the provided URL")
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed. Please review the errors above.")
        
        if not results.get("Docling Import", False):
            print("\n💡 To install Docling, run:")
            print("   python setup_docling.py")
        
        if not results.get("AWS Setup", False):
            print("\n💡 To fix AWS setup, ensure your credentials are set correctly.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
