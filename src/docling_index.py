"""
Enhanced document processing and indexing pipeline using Docling.
Supports advanced multimodal processing of text and images from documents.
Based on the reference implementation with improved PDF and image parsing.
"""
import os
import logging
import base64
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from io import BytesIO
import hashlib
import json

# Docling imports for advanced document processing
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    # Try different import paths for ConvertedDocument
    try:
        from docling.datamodel.document import ConvertedDocument
    except ImportError:
        try:
            from docling.datamodel.base_models import ConvertedDocument
        except ImportError:
            # Use Any as fallback type
            ConvertedDocument = Any
    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    ConvertedDocument = Any
    DocumentConverter = None
    InputFormat = None
    PdfPipelineOptions = None
    logging.warning(f"Docling not available: {e}. Install with: pip install docling")

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Image processing
try:
    from PIL import Image
    import cv2
    import numpy as np
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logging.warning("Vision processing dependencies not available. Install with: pip install pillow opencv-python")

from .config import config
from .vector_store import VectorStoreManager, create_text_splitter
from .bedrock_client import create_bedrock_embeddings


class DoclingProcessor:
    """
    Advanced document processor using Docling for superior PDF and image handling.
    """
    
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        config_instance: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize Docling document processor.
        
        Args:
            vector_store_manager: Vector store manager instance
            config_instance: Configuration instance
            **kwargs: Additional arguments
        """
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is required for this processor. Install with: pip install docling")
            
        self.config = config_instance or config
        self.vector_store_manager = vector_store_manager
        self.text_splitter = None
        self.supported_extensions = {'.pdf', '.docx', '.pptx', '.html', '.md', '.txt'}
        
        # Initialize Docling converter with enhanced options
        self._setup_docling_converter()
        self._setup_text_splitter()
        
        # Setup directories
        self.temp_dir = Path("temp_docling")
        self.temp_dir.mkdir(exist_ok=True)
    
    def _setup_docling_converter(self):
        """Setup Docling converter with simplified PDF processing options."""
        try:
            # Configure pipeline options for better PDF processing
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Enable OCR for scanned PDFs
            pipeline_options.do_table_structure = True  # Extract table structure
            
            # Initialize converter with simplified options
            self.converter = DocumentConverter()
            
            logging.info("Docling converter initialized with simplified PDF processing")
        except Exception as e:
            logging.warning(f"Error setting up advanced options, using default converter: {e}")
            # Fallback to basic converter
            self.converter = DocumentConverter()
            logging.info("Docling converter initialized with default settings")
    
    def _setup_text_splitter(self):
        """Setup text splitter for chunking documents."""
        vector_config = self.config.get_vector_store_config()
        self.text_splitter = create_text_splitter(
            chunk_size=vector_config.get("chunk_size", 1000),
            chunk_overlap=vector_config.get("chunk_overlap", 200)
        )
    
    def process_file(self, file_path: str, extract_images: bool = True) -> List[Document]:
        """
        Process a single file using Docling and return documents.
        
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
                documents = self._process_pdf_with_docling(file_path, extract_images)
            elif file_path.suffix.lower() in ['.docx', '.pptx']:
                documents = self._process_office_document(file_path, extract_images)
            else:
                documents = self._process_text_file(file_path)
            
            # Add metadata with enhanced indexing information
            for i, doc in enumerate(documents):
                # Extract report ID from filename if it's a roof report
                report_id = None
                if 'RoofReport-' in file_path.name:
                    try:
                        report_id = file_path.name.split('RoofReport-')[1].split('.')[0]
                    except:
                        pass
                
                doc.metadata.update({
                    'source': str(file_path),
                    'file_name': file_path.name,
                    'file_type': file_path.suffix,
                    'file_size': file_path.stat().st_size,
                    'processor': 'docling',
                    'document_index': i,
                    'report_id': report_id,
                    'content_type': doc.metadata.get('type', 'text'),
                    'searchable_text': doc.page_content.lower()  # For better search matching
                })
            
            logging.info(f"Processed {len(documents)} documents from {file_path.name}")
            return documents
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            raise
    
    def _process_pdf_with_docling(self, file_path: Path, extract_images: bool = True) -> List[Document]:
        """Process PDF using Docling's advanced capabilities."""
        documents = []
        
        try:
            # Convert document with Docling
            result = self.converter.convert(str(file_path))
            converted_doc = result.document  # Remove type hint to avoid import issues
            
            # Extract main document text
            main_text = converted_doc.export_to_markdown()
            # Persist Docling exports (Markdown and JSON)
            try:
                out_dir = Path("docling_exports")
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(file_path).stem
                out_report_dir = out_dir / stem
                out_report_dir.mkdir(parents=True, exist_ok=True)
                # Save Markdown
                (out_report_dir / f"{stem}.md").write_text(main_text or "", encoding="utf-8")
                # Build JSON using Docling's model if available; otherwise fallback
                try:
                    doc_json = converted_doc.model_dump()
                except Exception:
                    try:
                        doc_json = converted_doc.to_dict()
                    except Exception:
                        tables_md: List[Union[str, Dict[str, Any]]] = []
                        if hasattr(converted_doc, "tables") and converted_doc.tables:
                            for t in converted_doc.tables:
                                try:
                                    if hasattr(t, "export_to_markdown"):
                                        tables_md.append(t.export_to_markdown())
                                    elif hasattr(t, "to_dict"):
                                        tables_md.append(t.to_dict())
                                    else:
                                        tables_md.append(str(t))
                                except Exception:
                                    tables_md.append(str(t))
                        doc_json = {
                            "markdown": main_text,
                            "tables": tables_md,
                            "pictures_count": len(getattr(converted_doc, "pictures", []) or []),
                            "meta": {"file_name": Path(file_path).name},
                        }
                (out_report_dir / f"{stem}.json").write_text(
                    json.dumps(doc_json, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as save_err:
                logging.warning(f"Error saving Docling exports: {save_err}")
            
            if main_text.strip():
                text_doc = Document(
                    page_content=main_text,
                    metadata={
                        'type': 'text',
                        'extraction_method': 'docling_markdown'
                    }
                )
                # Split into chunks
                text_chunks = self.text_splitter.split_documents([text_doc])
                documents.extend(text_chunks)
            
            # Extract page-level content with images
            if extract_images:
                page_documents = self._extract_pages_and_images(converted_doc, file_path)
                documents.extend(page_documents)
            
            # Extract tables if present
            table_documents = self._extract_tables(converted_doc)
            documents.extend(table_documents)
            
            logging.info(f"Docling extracted {len(documents)} elements from {file_path.name}")
            return documents
            
        except Exception as e:
            logging.error(f"Error processing PDF with Docling {file_path}: {e}")
            raise
    
    def _extract_pages_and_images(self, converted_doc: Any, file_path: Path) -> List[Document]:
        """Extract page content and images from converted document."""
        documents = []

        try:
            # Since Docling image extraction is not working, fall back to PyMuPDF for actual image extraction
            # but keep Docling metadata for organization
            image_documents = self._extract_images_with_pymupdf(file_path)
            documents.extend(image_documents)
            
            # Also process Docling picture metadata for additional context
            if hasattr(converted_doc, 'pictures') and converted_doc.pictures:
                logging.info(f"Docling detected {len(converted_doc.pictures)} pictures (using PyMuPDF for extraction)")
                
        except Exception as e:
            logging.error(f"Error extracting pages and images: {e}")

        return documents
    
    def _extract_images_with_pymupdf(self, file_path: Path) -> List[Document]:
        """Extract images using PyMuPDF as fallback since Docling image extraction isn't working."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logging.warning("PyMuPDF not available for image extraction")
            return []
            
        documents = []
        
        try:
            # Extract report ID for better organization
            report_id = None
            if 'RoofReport-' in file_path.name:
                try:
                    report_id = file_path.name.split('RoofReport-')[1].split('.')[0]
                except:
                    pass
            
            # Open PDF with PyMuPDF
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
                            # Create images directory structure
                            images_dir = Path("extracted_images")
                            if report_id:
                                report_dir = images_dir / f"report_{report_id}"
                            else:
                                report_dir = images_dir / file_path.stem
                            report_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Generate image filename
                            image_filename = f"page_{page_num + 1}_image_{img_index}.png"
                            image_file_path = report_dir / image_filename
                            
                            # Save image to file
                            img_data = pix.tobytes("png")
                            with open(image_file_path, 'wb') as f:
                                f.write(img_data)
                            
                            logging.info(f"Saved image to: {image_file_path}")
                            
                            # Convert to PIL for size info
                            from PIL import Image
                            image = Image.open(BytesIO(img_data))
                            
                            # Add location-specific keywords based on page
                            location_keywords = ["roof", "inspection", f"page{page_num + 1}"]
                            if report_id:
                                location_keywords.extend([report_id, f"report{report_id}"])
                            
                            # Infer location from page position
                            if page_num <= 2:
                                location_keywords.extend(["overview", "aerial", "top"])
                            elif page_num % 4 == 1:
                                location_keywords.extend(["north", "side", "north side"])
                            elif page_num % 4 == 2:
                                location_keywords.extend(["south", "side", "south side"])
                            elif page_num % 4 == 3:
                                location_keywords.extend(["east", "side", "east side"])
                            elif page_num % 4 == 0:
                                location_keywords.extend(["west", "side", "west side"])
                            
                            # Create enhanced image content
                            image_content = f"Image {img_index + 1} from page {page_num + 1} of {file_path.name}"
                            if report_id:
                                image_content += f" Report ID: {report_id}"
                            image_content += f" Keywords: {', '.join(location_keywords)}"
                            
                            # Create document with image metadata
                            img_doc = Document(
                                page_content=image_content,
                                metadata={
                                    'type': 'image',
                                    'content_type': 'image',
                                    'page_number': page_num + 1,
                                    'image_index': img_index,
                                    'source_file': file_path.name,
                                    'report_id': report_id,
                                    'extraction_method': 'pymupdf_fallback',
                                    'image_description': f"Image {img_index + 1} from page {page_num + 1}",
                                    'searchable_keywords': location_keywords,
                                    'image_type': 'roof_page_image',
                                    'has_raw_data': True,
                                    'image_file_path': str(image_file_path),
                                    'image_filename': image_filename,
                                    'image_size': image.size
                                }
                            )
                            documents.append(img_doc)
                        
                        pix = None  # Cleanup
                    except Exception as e:
                        logging.warning(f"Error processing image {img_index} on page {page_num + 1}: {e}")
            
            pdf_document.close()
            logging.info(f"Extracted {len(documents)} images from {file_path.name} using PyMuPDF")
            
        except Exception as e:
            logging.error(f"Error extracting images with PyMuPDF: {e}")
            
        return documents
    
    def _process_page_image(self, image_info: Any, page_num: int, img_idx: int, file_path: Path) -> Optional[Document]:
        """Process an image from a page."""
        if not VISION_AVAILABLE:
            return None
        
        try:
            # Get image data
            if hasattr(image_info, 'image') and image_info.image:
                # Convert to PIL Image
                if isinstance(image_info.image, np.ndarray):
                    image = Image.fromarray(image_info.image)
                else:
                    image = image_info.image
                
                # Convert to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Create image hash for deduplication
                img_hash = hashlib.md5(buffered.getvalue()).hexdigest()
                
                # Extract text from image if possible (OCR)
                image_text = self._extract_text_from_image(image)
                
                # Create document
                content = f"Image from page {page_num}"
                if image_text:
                    content += f"\nExtracted text: {image_text}"
                
                return Document(
                    page_content=content,
                    metadata={
                        'type': 'image',
                        'page_number': page_num,
                        'image_index': img_idx,
                        'image_data': img_base64,
                        'image_format': 'png',
                        'image_size': image.size,
                        'image_hash': img_hash,
                        'extraction_method': 'docling_page_image',
                        'has_text': bool(image_text)
                    }
                )
        
        except Exception as e:
            logging.warning(f"Error processing page image: {e}")
        
        return None
    
    def _process_standalone_image(self, picture: Any, img_idx: int, file_path: Path) -> Optional[Document]:
        """Process a standalone image from the document."""
        if not VISION_AVAILABLE:
            return None
        
        try:
            # Similar processing as page images
            if hasattr(picture, 'image') and picture.image:
                if isinstance(picture.image, np.ndarray):
                    image = Image.fromarray(picture.image)
                else:
                    image = picture.image
                
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                img_hash = hashlib.md5(buffered.getvalue()).hexdigest()
                
                image_text = self._extract_text_from_image(image)
                
                content = f"Standalone image {img_idx + 1}"
                if image_text:
                    content += f"\nExtracted text: {image_text}"
                
                return Document(
                    page_content=content,
                    metadata={
                        'type': 'image',
                        'image_index': img_idx,
                        'image_data': img_base64,
                        'image_format': 'png',
                        'image_size': image.size,
                        'image_hash': img_hash,
                        'extraction_method': 'docling_standalone_image',
                        'has_text': bool(image_text)
                    }
                )
        
        except Exception as e:
            logging.warning(f"Error processing standalone image: {e}")
        
        return None
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            
            # Basic image preprocessing for better OCR
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply threshold to get better OCR results
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Note: For production, you might want to use pytesseract or AWS Textract
            # For now, we'll return empty string as OCR is not implemented
            return ""
            
        except Exception as e:
            logging.warning(f"Error extracting text from image: {e}")
            return ""
    
    def _extract_tables(self, converted_doc: Any) -> List[Document]:
        """Extract tables from the converted document."""
        documents = []
        
        try:
            # Check if document has tables
            if hasattr(converted_doc, 'tables') and converted_doc.tables:
                for table_idx, table in enumerate(converted_doc.tables):
                    try:
                        # Convert table to markdown or text format
                        if hasattr(table, 'export_to_markdown'):
                            table_content = table.export_to_markdown()
                        elif hasattr(table, 'to_dict'):
                            table_dict = table.to_dict()
                            table_content = str(table_dict)
                        else:
                            table_content = str(table)
                        
                        if table_content.strip():
                            table_doc = Document(
                                page_content=table_content,
                                metadata={
                                    'type': 'table',
                                    'table_index': table_idx,
                                    'extraction_method': 'docling_table'
                                }
                            )
                            documents.append(table_doc)
                    
                    except Exception as e:
                        logging.warning(f"Error processing table {table_idx}: {e}")
        
        except Exception as e:
            logging.error(f"Error extracting tables: {e}")
        
        return documents
    
    def _process_office_document(self, file_path: Path, extract_images: bool = True) -> List[Document]:
        """Process Office documents (DOCX, PPTX) using Docling."""
        documents = []
        
        try:
            result = self.converter.convert(str(file_path))
            converted_doc = result.document  # Remove type hint
            
            # Extract main text
            main_text = converted_doc.export_to_markdown()
            if main_text.strip():
                text_doc = Document(
                    page_content=main_text,
                    metadata={
                        'type': 'text',
                        'extraction_method': 'docling_office'
                    }
                )
                text_chunks = self.text_splitter.split_documents([text_doc])
                documents.extend(text_chunks)
            
            # Extract images if enabled
            if extract_images:
                image_documents = self._extract_pages_and_images(converted_doc, file_path)
                documents.extend(image_documents)
        
        except Exception as e:
            logging.error(f"Error processing Office document {file_path}: {e}")
            raise
        
        return documents
    
    def _process_text_file(self, file_path: Path) -> List[Document]:
        """Process text-based files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            document = Document(
                page_content=content,
                metadata={
                    'type': 'text',
                    'extraction_method': 'direct_text'
                }
            )
            
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
        Process all files in a directory using Docling.
        
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


def process_and_index_directory_with_docling(
    directory_path: str,
    vector_store_manager: Optional[VectorStoreManager] = None,
    drop_existing: bool = False,
    file_extensions: Optional[List[str]] = None,
    extract_images: bool = True,
    config_instance: Optional[Any] = None
) -> VectorStoreManager:
    """
    Process and index all documents in a directory using Docling.
    
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
    
    # Process documents with Docling
    processor = DoclingProcessor(
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
        
        logging.info(f"Successfully indexed {len(documents)} documents with Docling")
    else:
        logging.warning("No documents found to index")
    
    return vector_store_manager


def get_docling_processor(config_instance: Optional[Any] = None) -> DoclingProcessor:
    """
    Get a Docling document processor instance.
    
    Args:
        config_instance: Configuration instance
        
    Returns:
        Docling document processor instance
    """
    return DoclingProcessor(config_instance=config_instance)


# Compatibility function to replace the original index processor
def create_enhanced_index_processor(use_docling: bool = True, config_instance: Optional[Any] = None):
    """
    Create an enhanced document processor.
    
    Args:
        use_docling: Whether to use Docling (recommended)
        config_instance: Configuration instance
        
    Returns:
        Document processor instance
    """
    if use_docling and DOCLING_AVAILABLE:
        return DoclingProcessor(config_instance=config_instance)
    else:
        # Fallback to original processor
        from .index import DocumentProcessor
        return DocumentProcessor(config_instance=config_instance)
