#!/usr/bin/env python3
"""
Main entry point for the Property Data RAG System.

This script provides a simple way to run the system with your property documents.
It will process the PDFs in the input_file directory and launch the web interface.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import config
from src.index import process_and_index_directory
from src.docling_index import process_and_index_directory_with_docling, DOCLING_AVAILABLE
from src.agent import AgenticRAG
from src.ui import create_ui
from src.bedrock_client import create_bedrock_llm, create_bedrock_embeddings


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
    """Setup AWS credentials from environment or provided values."""
    # Set AWS credentials if provided
    # Only set if not already set in environment
    if not os.getenv('AWS_ACCESS_KEY_ID'):
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    if not os.getenv('AWS_SECRET_ACCESS_KEY'):
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
    if not os.getenv('AWS_SESSION_TOKEN'):
        os.environ['AWS_SESSION_TOKEN'] = aws_session_token
    if not os.getenv('AWS_DEFAULT_REGION'):
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'


def main():
    """Main function to run the Property Data RAG System."""
    
    # Setup AWS credentials
    setup_aws_credentials()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🏠 Starting Property Data RAG System with Docling...")
    
    try:
        # Check if input directory exists
        input_dir = Path("input_files")
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            print(f"\n❌ Error: Input directory '{input_dir}' not found.")
            print("Please make sure your PDF files are in the 'input_files' directory.")
            return
        
        # Check for documents (PDFs and other files)
        all_files = list(input_dir.glob("*"))
        doc_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.md', '.docx', '.doc']]
        pdf_files = [f for f in doc_files if f.suffix.lower() == '.pdf']
        
        if not doc_files:
            logger.warning(f"No document files found in {input_dir}")
            print(f"\n⚠️  Warning: No document files found in '{input_dir}'.")
            print("You can still use the web interface to upload documents.")
        else:
            logger.info(f"Found {len(doc_files)} document files ({len(pdf_files)} PDFs)")
            print(f"\n📄 Found {len(doc_files)} document files:")
            for doc_file in doc_files:
                print(f"  - {doc_file.name} ({doc_file.suffix})")
        
        # Check if we should process documents (add command line argument support)
        import sys
        should_process = '--process' in sys.argv or '--reprocess' in sys.argv
        force_reprocess = '--reprocess' in sys.argv
        
        # Check if vector store already exists
        vector_store_path = Path("vectorstore_faiss")
        has_existing_index = vector_store_path.exists() and any(vector_store_path.glob("*.faiss"))
        
        # Process documents if needed
        rag_agent = None
        if doc_files and (should_process or not has_existing_index):
            print(f"\n🔄 Processing documents with {'Docling (enhanced)' if DOCLING_AVAILABLE else 'standard processor'}...")
            try:
                # Use Docling processor if available, otherwise fallback to standard
                if DOCLING_AVAILABLE:
                    print("📊 Using Docling for advanced PDF parsing and image extraction...")
                    vector_store_manager = process_and_index_directory_with_docling(
                        directory_path=str(input_dir),
                        drop_existing=force_reprocess,  # Only clear if forced reprocess
                        extract_images=True  # Enable image extraction
                    )
                else:
                    print("📄 Using standard PDF processor...")
                    vector_store_manager = process_and_index_directory(
                        directory_path=str(input_dir),
                        drop_existing=force_reprocess  # Only clear if forced reprocess
                    )
                
                # Create RAG agent
                bedrock_config = config.get_bedrock_config()
                llm = create_bedrock_llm(bedrock_config)
                
                rag_agent = AgenticRAG(
                    vector_store_manager=vector_store_manager,
                    llm=llm
                )
                
                doc_count = vector_store_manager.get_count()
                print(f"✅ Successfully processed documents! Created {doc_count} document chunks.")
                
            except Exception as e:
                logger.error(f"Error processing documents: {e}")
                print(f"\n❌ Error processing documents: {e}")
                print("\nYou can still use the web interface to upload and process documents.")
        
        elif doc_files and has_existing_index:
            print(f"\n📚 Found existing vector store with processed documents")
            print("💡 Use --process to reprocess documents or --reprocess to force full reprocessing")
            
            # Load existing vector store
            try:
                bedrock_config = config.get_bedrock_config()
                vector_config = config.get_vector_store_config()
                embeddings = create_bedrock_embeddings(bedrock_config)
                
                from src.vector_store import VectorStoreManager
                vector_store_manager = VectorStoreManager(
                    store_type=vector_config["store_type"],
                    collection_name=vector_config["collection_name"],
                    embeddings=embeddings
                )
                
                # Vector store is automatically loaded in _setup_faiss if it exists
                
                # Create RAG agent with existing vector store
                llm = create_bedrock_llm(bedrock_config)
                rag_agent = AgenticRAG(
                    vector_store_manager=vector_store_manager,
                    llm=llm
                )
                
                doc_count = vector_store_manager.get_count()
                print(f"✅ Loaded existing index with {doc_count} document chunks")
                
            except Exception as e:
                logger.error(f"Error loading existing vector store: {e}")
                print(f"❌ Error loading existing documents: {e}")
                print("💡 Try running with --reprocess to rebuild the index")
        
        # Create and launch web interface
        print(f"\n🌐 Launching web interface...")
        
        # Ensure we have a RAG agent
        if rag_agent is None and has_existing_index:
            print("🔧 Creating RAG agent for existing documents...")
            try:
                bedrock_config = config.get_bedrock_config()
                vector_config = config.get_vector_store_config()
                embeddings = create_bedrock_embeddings(bedrock_config)
                
                from src.vector_store import VectorStoreManager
                vector_store_manager = VectorStoreManager(
                    store_type=vector_config["store_type"],
                    collection_name=vector_config["collection_name"],
                    embeddings=embeddings
                )
                
                if vector_store_manager.vector_store is not None:
                    llm = create_bedrock_llm(bedrock_config)
                    rag_agent = AgenticRAG(
                        vector_store_manager=vector_store_manager,
                        llm=llm
                    )
                    doc_count = vector_store_manager.get_count()
                    print(f"✅ RAG agent created with {doc_count} documents")
                
            except Exception as e:
                logger.error(f"Error creating RAG agent: {e}")
                print(f"❌ Could not create RAG agent: {e}")
        
        ui = create_ui(rag_agent=rag_agent)
        
        # Get UI configuration
        ui_config = config.get("ui") or {}
        port = ui_config.get("port", 7860)
        share = ui_config.get("share", False)
        
        print(f"\n🚀 Property Data RAG System is ready!")
        print(f"📱 Open your browser to: http://localhost:{port}")
        if share:
            print("🌍 Public link will be generated...")
        
        print(f"\n💡 You can now:")
        print(f"   • Ask questions about your property documents")
        print(f"   • Upload additional documents through the web interface")
        print(f"   • Adjust settings in the Settings tab")
        
        print(f"\n📋 Command line options:")
        print(f"   • python main.py --process    (reprocess documents)")
        print(f"   • python main.py --reprocess  (force full reprocessing)")
        
        print(f"\n🛑 Press Ctrl+C to stop the server")
        
        # Launch the interface
        ui.launch(
            share=share,
            server_port=port,
            server_name="0.0.0.0"
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
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
