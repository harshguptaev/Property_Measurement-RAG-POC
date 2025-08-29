"""
Document processing and indexing pipeline for various file formats.
Supports multimodal processing of text and images from documents.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import base64
from io import BytesIO

# Document processing
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Image processing
try:
    from PIL import Image
    import fitz  # PyMuPDF
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    logging.warning("Multimodal processing dependencies not available. Install with: pip install pillow pymupdf")

from .config import config
from .vector_store import VectorStoreManager, create_text_splitter
from .bedrock_client import create_bedrock_embeddings


class DocumentProcessor:
    """
    Processes documents and extracts text and images for indexing.
    """
    
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        config_instance: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize document processor.
        
        Args:
            vector_store_manager: Vector store manager instance
            config_instance: Configuration instance
            **kwargs: Additional arguments
        """
        self.config = config_instance or config
        self.vector_store_manager = vector_store_manager
        self.text_splitter = None
        self.supported_extensions = {'.pdf', '.txt', '.md', '.docx', '.doc'}
        
        self._setup_text_splitter()
    
    def _setup_text_splitter(self):
        """Setup text splitter for chunking documents."""
        vector_config = self.config.get_vector_store_config()
        self.text_splitter = create_text_splitter(
            chunk_size=vector_config.get("chunk_size", 1000),
            chunk_overlap=vector_config.get("chunk_overlap", 200)
        )
    
    def process_file(self, file_path: str, extract_images: bool = True) -> List[Document]:
        """
        Process a single file and return documents.
        
        Args:
            file_path: Path to the file
            extract_images: Whether to extract images from the document
            
        Returns:
            List of processed documents
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_extensions:
            logging.warning(f"Unsupported file type: {file_path.suffix}")
            return []
        
        try:
            documents = []
            
            if file_path.suffix.lower() == '.pdf':
                documents = self._process_pdf(file_path, extract_images)
            else:
                documents = self._process_text_file(file_path)
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_name': file_path.name,
                    'file_type': file_path.suffix,
                    'file_size': file_path.stat().st_size
                })
            
            logging.info(f"Processed {len(documents)} documents from {file_path.name}")
            return documents
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            raise
    
    def _process_pdf(self, file_path: Path, extract_images: bool = True) -> List[Document]:
        """Process PDF file and extract text and images."""
        documents = []
        
        try:
            # Extract text using PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            text_documents = loader.load()
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_documents(text_documents)
            documents.extend(text_chunks)
            
            # Extract images if enabled and dependencies are available
            if extract_images and MULTIMODAL_AVAILABLE:
                image_documents = self._extract_images_from_pdf(file_path)
                documents.extend(image_documents)
            
            return documents
            
        except Exception as e:
            logging.error(f"Error processing PDF {file_path}: {e}")
            # Fallback to UnstructuredPDFLoader
            try:
                loader = UnstructuredPDFLoader(str(file_path))
                fallback_documents = loader.load()
                return self.text_splitter.split_documents(fallback_documents)
            except Exception as e2:
                logging.error(f"Fallback PDF processing also failed: {e2}")
                raise
    
    def _extract_images_from_pdf(self, file_path: Path) -> List[Document]:
        """Extract images from PDF using PyMuPDF."""
        if not MULTIMODAL_AVAILABLE:
            return []
        
        image_documents = []
        
        try:
            pdf_document = fitz.open(str(file_path))
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            # Convert to PIL Image
                            img_data = pix.tobytes("png")
                            image = Image.open(BytesIO(img_data))
                            
                            # Convert to base64 for storage
                            buffered = BytesIO()
                            image.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode()
                            
                            # Create document with image data
                            img_doc = Document(
                                page_content=f"Image from page {page_num + 1}",
                                metadata={
                                    'type': 'image',
                                    'page_number': page_num + 1,
                                    'image_index': img_index,
                                    'image_data': img_base64,
                                    'image_format': 'png',
                                    'image_size': image.size
                                }
                            )
                            image_documents.append(img_doc)
                        
                        pix = None  # Clean up
                        
                    except Exception as e:
                        logging.warning(f"Error extracting image {img_index} from page {page_num}: {e}")
                        continue
            
            pdf_document.close()
            logging.info(f"Extracted {len(image_documents)} images from {file_path.name}")
            
        except Exception as e:
            logging.error(f"Error extracting images from PDF {file_path}: {e}")
        
        return image_documents
    
    def _process_text_file(self, file_path: Path) -> List[Document]:
        """Process text-based files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create document
            document = Document(
                page_content=content,
                metadata={'type': 'text'}
            )
            
            # Split into chunks
            return self.text_splitter.split_documents([document])
            
        except Exception as e:
            logging.error(f"Error processing text file {file_path}: {e}")
            raise
    
    def process_directory(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
        extract_images: bool = True
    ) -> List[Document]:
        """
        Process all files in a directory.
        
        Args:
            directory_path: Path to the directory
            file_extensions: List of file extensions to process
            recursive: Whether to process subdirectories
            extract_images: Whether to extract images from documents
            
        Returns:
            List of all processed documents
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        file_extensions = file_extensions or list(self.supported_extensions)
        file_extensions = [ext.lower() for ext in file_extensions]
        
        all_documents = []
        
        # Get all files
        if recursive:
            files = [f for f in directory_path.rglob("*") if f.is_file()]
        else:
            files = [f for f in directory_path.iterdir() if f.is_file()]
        
        # Filter by extensions
        target_files = [f for f in files if f.suffix.lower() in file_extensions]
        
        logging.info(f"Found {len(target_files)} files to process in {directory_path}")
        
        for file_path in target_files:
            try:
                documents = self.process_file(file_path, extract_images)
                all_documents.extend(documents)
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                continue
        
        logging.info(f"Processed {len(all_documents)} total documents from {len(target_files)} files")
        return all_documents


def process_and_index_directory(
    directory_path: str,
    vector_store_manager: Optional[VectorStoreManager] = None,
    drop_existing: bool = False,
    file_extensions: Optional[List[str]] = None,
    extract_images: bool = True,
    config_instance: Optional[Any] = None
) -> VectorStoreManager:
    """
    Process and index all documents in a directory.
    
    Args:
        directory_path: Path to the directory containing documents
        vector_store_manager: Existing vector store manager (optional)
        drop_existing: Whether to drop existing collection
        file_extensions: List of file extensions to process
        extract_images: Whether to extract images from documents
        config_instance: Configuration instance
        
    Returns:
        Vector store manager with indexed documents
    """
    config_instance = config_instance or config
    
    # Create vector store manager if not provided
    if vector_store_manager is None:
        bedrock_config = config_instance.get_bedrock_config()
        vector_config = config_instance.get_vector_store_config()
        
        embeddings = create_bedrock_embeddings(bedrock_config)
        
        vector_store_manager = VectorStoreManager(
            store_type=vector_config["store_type"],
            collection_name=vector_config["collection_name"],
            embeddings=embeddings
        )
    
    # Drop existing collection if requested
    if drop_existing:
        vector_store_manager.delete_collection()
        vector_store_manager._setup_vector_store()
    
    # Process documents
    processor = DocumentProcessor(
        vector_store_manager=vector_store_manager,
        config_instance=config_instance
    )
    
    documents = processor.process_directory(
        directory_path=directory_path,
        file_extensions=file_extensions,
        extract_images=extract_images
    )
    
    if documents:
        # Add documents to vector store
        logging.info(f"Adding {len(documents)} documents to vector store...")
        vector_store_manager.add_documents(documents)
        
        # Save vector store
        vector_store_manager.save()
        
        logging.info(f"Successfully indexed {len(documents)} documents")
    else:
        logging.warning("No documents found to index")
    
    return vector_store_manager


def get_document_processor(config_instance: Optional[Any] = None) -> DocumentProcessor:
    """
    Get a document processor instance.
    
    Args:
        config_instance: Configuration instance
        
    Returns:
        Document processor instance
    """
    return DocumentProcessor(config_instance=config_instance)


# Set up logging for this module
def set_log_level(level: int = logging.INFO, log_file: Optional[str] = None):
    """Set logging level for this module."""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    if log_file:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
