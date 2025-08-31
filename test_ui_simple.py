#!/usr/bin/env python3
"""
Simple test script for the enhanced UI with image support.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ui import create_ui
from src.config import config

def main():
    """Test the enhanced UI."""
    print("🚀 Starting Enhanced Property RAG UI Test...")
    
    # Create UI without RAG agent (for basic testing)
    ui = create_ui(
        rag_agent=None,
        config_instance=config
    )
    
    print("✅ UI created successfully!")
    print("🖼️ Image gallery features enabled")
    print("💬 Multimodal chat interface ready")
    print("📁 Document management available")
    
    # Launch the interface
    print("\n🌐 Launching Gradio interface...")
    print("📱 Open your browser to interact with the UI")
    print("🖼️ You can:")
    print("   • Browse extracted images in the Image Gallery tab")
    print("   • Upload images directly in the Chat tab")
    print("   • Select multiple images for analysis")
    print("   • Upload documents in the Document Management tab")
    
    try:
        ui.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            debug=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
