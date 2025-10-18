#!/usr/bin/env python3
"""
Docling Document Processor.

This script processes property documents using Docling for advanced PDF parsing
and image extraction, saving the results as structured JSON chunks.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.docling_index import DoclingProcessor, DOCLING_AVAILABLE

def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def save_chunks_to_json(documents: List, output_dir: Path, report_id: str):
    """
    Save processed documents as JSON chunks in the expected format.

    Args:
        documents: List of processed Document objects
        output_dir: Directory to save JSON files
        report_id: Report identifier for the chunks
    """
    output_dir.mkdir(exist_ok=True)

    # Convert documents to chunk format
    chunks = []
    for i, doc in enumerate(documents, 1):
        chunk = {
            "chunk_id": f"{report_id}_chunk_{i}",
            "section": doc.metadata.get("section", "Unknown"),
            "type": "text",
            "data": doc.page_content
        }
        chunks.append(chunk)

    # Save as JSON
    output_file = output_dir / f"RoofReport-{report_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"text": chunks}, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved {len(chunks)} chunks to {output_file}")


def main():
    """Main function to process documents with Docling."""

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    if not DOCLING_AVAILABLE:
        logger.error("Docling is not available. Please install with: pip install docling")
        print("❌ Docling is required but not installed.")
        sys.exit(1)

    logger.info("🔄 Starting Docling document processing...")

    try:
        # Check if input directory exists
        input_dir = Path("input_files")
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            print(f"❌ Error: Input directory '{input_dir}' not found.")
            return

        # Check for documents (PDFs and other files)
        all_files = list(input_dir.glob("*"))
        doc_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.md', '.docx', '.doc']]
        pdf_files = [f for f in doc_files if f.suffix.lower() == '.pdf']

        if not doc_files:
            logger.warning(f"No document files found in {input_dir}")
            print(f"⚠️ Warning: No document files found in '{input_dir}'.")
            return

        logger.info(f"Found {len(doc_files)} document files ({len(pdf_files)} PDFs)")
        print(f"📄 Found {len(doc_files)} document files:")
        for doc_file in doc_files:
            print(f"  - {doc_file.name} ({doc_file.suffix})")

        # Create output directory
        output_dir = Path("Final_Chunks")
        output_dir.mkdir(exist_ok=True)

        print("\n📊 Processing documents with Docling...")
        # Initialize Docling processor
        processor = DoclingProcessor(extract_images=True)

        total_chunks = 0

        # Process each file individually to create separate JSON outputs
        for doc_file in doc_files:
            try:
                logger.info(f"Processing {doc_file.name}...")
                print(f"  Processing {doc_file.name}...")

                # Extract report ID from filename (remove extension and prefix)
                report_id = doc_file.stem.replace("RoofReport-", "").replace("report_", "")

                # Process the file
                documents = processor.process_file(str(doc_file), extract_images=True)

                if documents:
                    # Save chunks to JSON
                    save_chunks_to_json(documents, output_dir, report_id)
                    total_chunks += len(documents)
                    print(f"  ✅ Processed {doc_file.name}: {len(documents)} chunks")
                else:
                    print(f"  ⚠️ No content extracted from {doc_file.name}")

            except Exception as e:
                logger.error(f"Error processing {doc_file.name}: {e}")
                print(f"  ❌ Error processing {doc_file.name}: {e}")
                continue

        print(f"\n✅ Processing complete! Created {total_chunks} total chunks in {output_dir}/")
        print(f"📁 Output files saved to: {output_dir}/")

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

