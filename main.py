#!/usr/bin/env python3
"""
Main entry point for Property Document Processing.

This script processes property documents and extracts images from PDFs.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import config
from src.docling_index import process_directory_with_docling, DOCLING_AVAILABLE
from src.image_similarity_poc.pictometry_client import saveimagesfrompictometry
from src.image_similarity_poc.poc_image_similarity import stitch_all_pictometry_directories, push_image_embedings_todb, find_similar_images
from src.image_similarity_poc.image_milvus import ImageMilvus, print_matches
from src.image_similarity_poc.s3_client import batch_download_from_reports

def setup_logging():
    """Setup logging configuration."""
    log_level = config.get("logging", "level", "INFO")
    log_format = config.get("logging", "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def setup_aws_credentials():
    """Setup AWS credentials from environment variables."""
    # Set default region if not set
    if not os.getenv('AWS_DEFAULT_REGION'):
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'


def main():
    """Main function to process property documents."""

    # # Image processing setup
    # batch_download_from_reports()
    # im = ImageMilvus()
    # im.index_top_embeddings()
    # matches = im.search_similar_by_image("pictometry_images/29.482943_-98.456349/top_image.webp")
    # print_matches(matches)

    # Setup AWS credentials
    setup_aws_credentials()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🏠 Starting Property Document Processing with Docling...")

    try:
        # Check if input directory exists
        input_dir = Path("premium_reports")
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            print(f"\n❌ Error: Input directory '{input_dir}' not found.")
            print("Please make sure your PDF files are in the 'premium_reports' directory.")
            return

        # Check for documents (PDFs and other files)
        all_files = list(input_dir.glob("*"))
        doc_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.md', '.docx', '.doc']]
        pdf_files = [f for f in doc_files if f.suffix.lower() == '.pdf']

        if not doc_files:
            logger.warning(f"No document files found in {input_dir}")
            print(f"\n⚠️  Warning: No document files found in '{input_dir}'.")
            return
        else:
            logger.info(f"Found {len(doc_files)} document files ({len(pdf_files)} PDFs)")
            print(f"\n📄 Found {len(doc_files)} document files:")
            for doc_file in doc_files:
                print(f"  - {doc_file.name} ({doc_file.suffix})")

        # Process documents
        print("\n🔄 Processing documents with Docling...")
        try:
            documents = process_directory_with_docling(
                directory_path=str(input_dir),
                extract_images=True
            )

            print(f"✅ Successfully processed {len(documents)} documents!")

        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            print(f"\n❌ Error processing documents: {e}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n👋 Processing stopped by user")
        logger.info("Application stopped by user")

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the logs for more details.")
        sys.exit(1)


def test_setup():
    """Test if the system is properly configured."""
    
    print("🔧 Testing system setup...")
    
    # Test AWS credentials
    try:
        import boto3
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
        print("✅ AWS credentials configured")
    except Exception as e:
        print(f"❌ AWS credentials issue: {e}")
        return False
    
    # Test required packages
    required_packages = ['langchain', 'gradio', 'faiss', 'PyPDF2']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            return False
    
    print("✅ System setup looks good!")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_setup()
    else:
        main()
