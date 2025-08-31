#!/usr/bin/env python3
"""
Test script for the enhanced UI with image capabilities.
This script demonstrates the new image gallery and multimodal chat features.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ui import create_ui
from src.config import config
from src.bedrock_client import create_multimodal_bedrock_llm
from src.agent import AgenticRAG
from src.vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to launch the enhanced UI."""
    print("🚀 Starting Property Data RAG System with Image Support")
    print("=" * 60)
    
    # Check if extracted images exist
    extracted_images_path = Path("extracted_images")
    if extracted_images_path.exists():
        reports = list(extracted_images_path.iterdir())
        print(f"📁 Found {len(reports)} report directories with extracted images:")
        for report in reports:
            if report.is_dir():
                image_count = len([f for f in report.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
                print(f"   • {report.name}: {image_count} images")
    else:
        print("⚠️  No extracted images found. Run document processing first to extract images.")
    
    print("\n🖼️  New Features Available:")
    print("   • Image Gallery: Browse and select extracted images")
    print("   • Multimodal Chat: Upload images or select from gallery for analysis")
    print("   • Enhanced AI: Claude 3 Sonnet with vision capabilities")
    print("   • Image Context: Ask questions about specific images")
    
    print("\n💡 Example Questions to Try:")
    print("   • 'Analyze this roof image for damage'")
    print("   • 'What measurements can you see in these diagrams?'")
    print("   • 'Compare the condition across these property photos'")
    print("   • 'Identify structural issues in the uploaded image'")
    
    try:
        # Create enhanced UI with multimodal support
        ui = create_ui(
            rag_agent=None,  # Will be created when documents are loaded
            config_instance=config
        )
        
        print(f"\n🌐 Launching UI on http://localhost:7860")
        print("   Use the Image Gallery tab to browse extracted images")
        print("   Use the Chat tab to analyze images and ask questions")
        print("\n" + "=" * 60)
        
        # Launch the interface
        ui.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            show_tips=True
        )
        
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        logging.error(f"Error in main: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
