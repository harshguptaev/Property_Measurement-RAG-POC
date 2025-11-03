#!/usr/bin/env python3
"""
Run Create Embeddings
Main script to create embeddings for processed roof images and store them in Milvus
"""

import os
import sys
import json
import logging
from typing import List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from milvus_embeddings import MilvusEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingsController:
    def __init__(self):
        """Initialize the embeddings controller"""
        self.milvus_embeddings = MilvusEmbeddings()
        self.processed_dir = "source_data/processed"
    
    def run_embeddings_pipeline(self) -> bool:
        """
        Run the complete embeddings pipeline
        
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        try:
            print("🚀 Starting Milvus Embeddings Pipeline")
            print("=" * 60)
            
            # Step 1: Check Milvus connection
            print("\n🔗 Step 1: Checking Milvus Connection")
            if not self.milvus_embeddings.check_milvus_connection():
                print("❌ Cannot connect to Milvus server. Please ensure it's running on localhost:19530")
                return False
            print("✅ Milvus connection successful!")
            
            # Step 2: Create collection
            print("\n📊 Step 2: Setting up Collection")
            if not self.milvus_embeddings.create_collection():
                print("❌ Failed to create collection. Exiting.")
                return False
            print("✅ Collection setup complete!")
            
            # Step 3: Process images and create embeddings
            print("\n🖼️  Step 3: Processing Images and Creating Embeddings")
            results = self.milvus_embeddings.process_all_images(self.processed_dir)
            
            # Step 4: Load collection into memory
            print("\n🔄 Step 4: Loading Collection into Memory")
            if self.milvus_embeddings.load_collection():
                print("✅ Collection loaded successfully!")
            else:
                print("⚠️ Collection loading failed, but data is stored.")
            
            # Step 5: Display results
            print("\n📈 Step 5: Processing Results")
            self.print_results(results)
            
            # Step 6: Collection info
            print("\n📋 Step 6: Collection Information")
            collection_info = self.milvus_embeddings.get_collection_info()
            self.print_collection_info(collection_info)
            
            print("\n🎉 Embeddings pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed with error: {str(e)}")
            return False
    
    def print_results(self, results: dict):
        """Print processing results"""
        print(f"📊 Total files processed: {results['total_files']}")
        print(f"✅ Successful: {results['successful']}")
        print(f"❌ Failed: {results['failed']}")
        
        if results['errors']:
            print(f"\n⚠️  Failed files:")
            for error_file in results['errors']:
                print(f"   - {error_file}")
    
    def print_collection_info(self, info: dict):
        """Print collection information"""
        if 'error' in info:
            print(f"❌ Error: {info['error']}")
        else:
            print(f"📁 Collection: {info['collection_name']}")
            print(f"📝 Description: {info['description']}")
            print(f"📊 Total entities: {info['num_entities']}")
            print(f"🔢 Vector dimension: {info['vector_dim']}")

def create_embeddings_for_report(report_id: str) -> bool:
    """
    Create embeddings for a specific report's processed image

    Args:
        report_id: The report ID to process

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🎯 Creating embeddings for report: {report_id}")

        # Check if processed image exists
        processed_image_path = f"input_data/{report_id}/processed_DDD.png"
        if not os.path.exists(processed_image_path):
            print(f"❌ Processed image not found: {processed_image_path}")
            return False

        # Check if JSON exists for metadata
        json_path = f"input_data/{report_id}/{report_id}.json"
        address = "Unknown Address"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # Get address from the first item (property info)
                    if data and isinstance(data, list) and len(data) > 0:
                        address = data[0].get("address", "Unknown Address")
            except Exception as e:
                print(f"⚠️ Could not read address from JSON: {e}")
        else:
            print(f"❌ JSON file not found: {json_path}")
            return False

        print(f"📁 Found processed image: {processed_image_path}")
        print(f"🏠 Address: {address}")

        # Initialize Milvus embeddings handler
        milvus_embeddings = MilvusEmbeddings()

        # Check Milvus connection
        if not milvus_embeddings.check_milvus_connection():
            print("❌ Cannot connect to Milvus server")
            return False

        # Create collection if needed
        if not milvus_embeddings.create_collection():
            print("❌ Failed to create/setup collection")
            return False

        # Process the specific image
        success = milvus_embeddings.process_image_file(processed_image_path, address)

        if success:
            print(f"✅ Successfully created embeddings for report {report_id}")

            # Load collection into memory
            milvus_embeddings.load_collection()

            return True
        else:
            print(f"❌ Failed to create embeddings for report {report_id}")
            return False

    except Exception as e:
        print(f"❌ Error creating embeddings for report {report_id}: {e}")
        return False

def main():
    """Main entry point"""
    print("🏠 Roof Imagery RAG - Milvus Embeddings Creator")
    print("=" * 60)

    # Check if processed images exist
    processed_dir = "source_data/processed"
    if not os.path.exists(processed_dir):
        print(f"❌ Processed directory not found: {processed_dir}")
        print("Please run the main pipeline first to process images.")
        return

    # Count processed images
    image_files = [f for f in os.listdir(processed_dir) if f.lower().endswith('.png')]
    if not image_files:
        print(f"❌ No processed images found in: {processed_dir}")
        print("Please run the main pipeline first to process images.")
        return

    allowOnlyReportIds = [65287819,65167520,65145525,65143839,65088612,65072336,65033234,65033227,65033221,65030115,65025964,65008550,64975447,64951635,64920454,64920436,64903555,64895822,64892357,64891144];
    #allowOnlyReportIds = [];
    if allowOnlyReportIds:
        image_files = [f for f in image_files if f.startswith("DDD_") and f.endswith(".png") and f.replace("DDD_", "").replace(".png", "") in allowOnlyReportIds]

    print(f"📁 Found {len(image_files)} processed images to create embeddings for")
    print(f"🔗 Milvus server: localhost:19530")
    print(f"🤖 Embedding model: amazon.titan-embed-image-v1")
    print()

    # Initialize and run pipeline
    controller = EmbeddingsController()
    success = controller.run_embeddings_pipeline()

    if success:
        print("\n✅ All embeddings created and stored in Milvus successfully!")
    else:
        print("\n❌ Embeddings pipeline failed. Please check the logs for details.")

if __name__ == "__main__":
    main()