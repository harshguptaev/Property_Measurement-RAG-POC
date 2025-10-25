"""
Enhanced document processing and indexing pipeline using Docling.
Supports advanced multimodal processing of text and images from documents.
Based on the reference implementation with improved PDF and image parsing.
"""
import os
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from io import BytesIO
from PIL import Image

# Docling imports for advanced document processing
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import TextItem
    from docling.datamodel.document import ConversionResult
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
from PIL import Image  # still required for PyMuPDF image size handling

import json
from .config import config
from .vector_store import VectorStoreManager, create_text_splitter
from .bedrock_client import create_bedrock_embeddings
from .image_utils import ImageManager


class DoclingProcessor:
    """
    Advanced document processor using Docling for superior PDF and image handling.
    """
    
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        config_instance: Optional[Any] = None,
        enable_gemini: bool = True,
        **kwargs
    ):
        """
        Initialize Docling document processor.
        
        Args:
            vector_store_manager: Vector store manager instance
            config_instance: Configuration instance
            enable_gemini: Whether to enable Gemini Vision for image analysis
            **kwargs: Additional arguments
        """
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is required for this processor. Install with: pip install docling")
            
        self.config = config_instance or config
        self.vector_store_manager = vector_store_manager
        self.text_splitter = None
        self.supported_extensions = {'.pdf', '.docx', '.pptx', '.html', '.md', '.txt'}
        
        # Initialize image manager with Gemini support
        self.image_manager = ImageManager(enable_gemini=enable_gemini)
        
        # Initialize Docling converter with enhanced options
        self._setup_docling_converter()
        self._setup_text_splitter()
    
    def _setup_docling_converter(self):
        """Setup Docling converter with simplified PDF processing options."""
        try:
            # Configure pipeline options for better PDF processing
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Enable OCR for scanned PDFs
            pipeline_options.images_scale = 4
            pipeline_options.generate_page_images = True

            # Initialize converter with simplified options

            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
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
            
            
            # Add metadata with enhanced indexing information
            for i, doc in enumerate(documents):
                # Extract report ID from filename if it's a roof report
                report_id = None
                filename = file_path.name
                if 'RoofReport-' in filename:
                    try:
                        report_id = filename.split('RoofReport-')[1].split('.')[0]
                    except:
                        pass
                elif 'report_' in filename:
                    try:
                        report_id = filename.split('report_')[1].split('.')[0]
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
    
    def save_docling_exports(self, main_text: str, converted_doc: ConversionResult, file_path: Path):
        """Persist Docling exports (Markdown and JSON)"""
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
                    doc_json = converted_doc.export_to_dict()
                except Exception:
                    doc_json = {
                        "markdown": main_text,
                        "pictures_count": len(getattr(converted_doc, "pictures", []) or []),
                        "meta": {"file_name": Path(file_path).name},
                    }
            (out_report_dir / f"{stem}.json").write_text(
                json.dumps(doc_json, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as save_err:
            logging.warning(f"Error saving Docling exports: {save_err}")

    def _process_pdf_with_docling(self, file_path: Path, extract_images: bool = True) -> List[Document]:
        """Process PDF using Docling's advanced capabilities."""
        documents = []
        
        try:
            # Convert document with Docling
            result = self.converter.convert(str(file_path))
            converted_doc = result.document  # Remove type hint to avoid import issues

            # Extract main document text
            main_text = converted_doc.export_to_markdown()
            self.save_docling_exports(main_text, converted_doc, file_path)
            
            for text_doc in converted_doc.iterate_items():
                if isinstance(text_doc, TextItem):  
                    text_doc = Document(
                        page_content=text_doc.text,
                        metadata={
                            'type': 'text',
                            'extraction_method': 'docling_markdown'
                        }
                    )
                    # Split into chunks
                    text_chunks = self.text_splitter.split_documents([text_doc])
                    documents.extend(text_chunks)
                
            
            # Extract important chunks
            important_chunks = self._extract_and_save_important_chunks(file_path)
            documents.extend(important_chunks)

            # Extract page-level content with images
            if extract_images:
                page_documents = self._extract_pages_and_images(converted_doc, file_path)
                documents.extend(page_documents)
            

            # Add image chunks to the existing final chunks structure
            self._add_image_chunks_to_final(final_chunks_file, file_path)
            
            

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
    

    # Extract images using PyMuPDF as fallback since Docling image extraction isn't working.
    def _extract_images_with_pymupdf(self, file_path: Path) -> List[Document]:
        """Extract images using PyMuPDF as fallback since Docling image extraction isn't working."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logging.warning("PyMuPDF not available for image extraction")
            return []

        documents = []
        extracted_images = {}  # Track extracted images for chunk creation

        try:
            # Extract report ID for better organization
            report_id = None
            filename = file_path.name
            if 'RoofReport-' in filename:
                try:
                    report_id = filename.split('RoofReport-')[1].split('.')[0]
                except:
                    pass
            elif 'report_' in filename:
                try:
                    report_id = filename.split('report_')[1].split('.')[0]
                except:
                    pass
            
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(str(file_path))
            
            for page_num in range(len(pdf_document)):
                if page_num == 0 or page_num > 10:  # Skip page 1 and any pages after west side (page 11+)
                    continue

                page = pdf_document.load_page(page_num)

                # For pages 2 and 8, get full page image
                if page_num in [1, 7]:  # Page 2 and 8
                    try:
                        # Render full page as image
                        pix = page.get_pixmap(dpi=150)  # Higher DPI for better quality

                        # Create images directory structure
                        images_dir = Path("extracted_images")
                        if report_id:
                            report_dir = images_dir / f"report_{report_id}"
                        else:
                            report_dir = images_dir / file_path.stem
                        report_dir.mkdir(parents=True, exist_ok=True)

                        # Name full page images
                        if page_num == 1:
                            image_label = "Lengths"
                        else:  # page_num == 7
                            image_label = "Roof_Penetrations"

                        image_filename = f"{image_label}.png"
                        image_file_path = report_dir / image_filename

                        # Save full page image
                        img_data = pix.tobytes("png")
                        with open(image_file_path, 'wb') as f:
                            f.write(img_data)

                        logging.info(f"Saved full page image to: {image_file_path}")

                        # Track extracted image for chunk creation
                        extracted_images[image_label] = str(image_file_path)

                        # Convert to PIL for size info
                        from PIL import Image
                        image = Image.open(BytesIO(img_data))

                        # Create document with image metadata
                        location_keywords = ["roof", "inspection", f"page{page_num + 1}", "full_page"]
                        if report_id:
                            location_keywords.extend([report_id, f"report{report_id}"])

                        if page_num == 1:
                            location_keywords.extend(["lengths", "diagram", "measurements", "length_diagram"])
                        else:
                            location_keywords.extend(["roof_penetrations", "penetrations", "roof_plan"])

                        image_content = f"{image_label} from page {page_num + 1} of {file_path.name}"
                        if report_id:
                            image_content += f" Report ID: {report_id}"
                        image_content += f" Keywords: {', '.join(location_keywords)}"

                        img_doc = Document(
                            page_content=image_content,
                            metadata={
                                'type': 'image',
                                'content_type': 'image',
                                'page_number': page_num + 1,
                                'source_file': file_path.name,
                                'report_id': report_id,
                                'extraction_method': 'full_page_render',
                                'image_description': image_label,
                                'searchable_keywords': location_keywords,
                                'image_type': 'full_page_image',
                                'has_raw_data': True,
                                'image_file_path': str(image_file_path),
                                'image_filename': image_filename,
                                'image_label': image_label,
                                'image_size': image.size
                            }
                        )

                        documents.append(img_doc)
                        continue  # Skip individual image extraction for these pages

                    except Exception as e:
                        logging.error(f"Error rendering full page {page_num + 1}: {e}")
                        continue

                # Extract individual images for other pages
                image_list = page.get_images()
                num_images_on_page = len(image_list)

                for img_index, img in enumerate(image_list):
                    # Apply page-specific image extraction rules
                    if page_num in [2, 3, 4, 5, 6] and img_index != 1:  # Pages 3-7: take second image only
                        continue
                    elif page_num in [8, 9] and img_index not in [1, 2]:  # Pages 9-10: take 2nd and 3rd images
                        continue
                    elif page_num == 10 and img_index != 1:  # Page 11: take second image only
                        continue

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
                            
                            def _label_for(p, i, n):
                                if p == 2:  # Page 3
                                    return "Pitch_Degrees"
                                elif p == 3:  # Page 4
                                    return "Pitch_on_12"
                                elif p == 4:  # Page 5
                                    return "Rafters"
                                elif p == 5:  # Page 6
                                    return "Azimuth"
                                elif p == 6:  # Page 7
                                    return "Area"
                                elif p == 8:  # Page 9
                                    if i == 1:
                                        return "Top_View"
                                    elif i == 2:
                                        return "North_Side"
                                elif p == 9:  # Page 10
                                    if i == 1:
                                        return "South_Side"
                                    elif i == 2:
                                        return "East_Side"
                                elif p == 10:  # Page 11
                                    return "West_Side"
                                return f"Page_{p + 1}_Image_{i + 1}"

                            image_label = _label_for(page_num, img_index, num_images_on_page)
                            image_filename = f"{image_label}.png"
                            image_file_path = report_dir / image_filename
                            
                            # Save image to file
                            img_data = pix.tobytes("png")
                            with open(image_file_path, 'wb') as f:
                                f.write(img_data)

                            logging.info(f"Saved image to: {image_file_path}")

                            # Track extracted image for chunk creation
                            extracted_images[image_label] = str(image_file_path)
                            
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
                            location_keywords.append(image_label.replace("_", " ").lower())
                            
                            # Create enhanced image content
                            image_content = f"{image_label} from page {page_num + 1} of {file_path.name}"
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
                                    'image_description': image_label,
                                    'searchable_keywords': location_keywords,
                                    'image_type': 'roof_page_image',
                                    'has_raw_data': True,
                                    'image_file_path': str(image_file_path),
                                    'image_filename': image_filename,
                                    'image_label': image_label,
                                    'image_size': image.size
                                }
                            )
                            
                            # Enhance with Gemini analysis if available
                            if self.image_manager.gemini_client:
                                try:
                                    enhanced_metadata = self.image_manager.enhance_image_metadata_with_gemini(img_doc.metadata)
                                    img_doc.metadata.update(enhanced_metadata)
                                    
                                    # Update page content with Gemini analysis
                                    if "gemini_analysis" in enhanced_metadata:
                                        analysis = enhanced_metadata["gemini_analysis"]
                                        if "full_analysis" in analysis:
                                            img_doc.page_content += f"\n\nGemini Analysis: {analysis['full_analysis']}"
                                        elif "caption" in analysis:
                                            img_doc.page_content += f"\n\nGemini Caption: {analysis['caption']}"
                                        elif "measurements_analysis" in analysis:
                                            img_doc.page_content += f"\n\nMeasurement Analysis: {analysis['measurements_analysis']}"
                                except Exception as e:
                                    logging.warning(f"Failed to enhance image with Gemini: {e}")
                            documents.append(img_doc)
                        
                        pix = None  # Cleanup
                    except Exception as e:
                        logging.warning(f"Error processing image {img_index} on page {page_num + 1}: {e}")
            
            pdf_document.close()
            logging.info(f"Extracted {len(documents)} images from {file_path.name} using PyMuPDF")

            # Create structured chunks for diagrams and imagery
            self._create_structured_chunks(extracted_images, report_id, file_path)

        except Exception as e:
            logging.error(f"Error extracting images with PyMuPDF: {e}")

        return documents

    def _create_structured_chunks(self, extracted_images, report_id, file_path):
        """Create structured chunks C003 (Diagrams) and C004 (Imagery) with image paths"""
        try:
            import json
            from pathlib import Path

            # Prepare property_id from report_id
            property_id = f"PROP_{report_id}" 

            # Create C003 - Diagrams chunk
            diagrams_chunk = {
                "chunk_id": "C003",
                "property_id": property_id,
                "section": "Diagrams",
                "type": "diagram",
                "data": {
                    "Lengths": extracted_images.get("Lengths", ""),
                    "Pitch (Degrees)": extracted_images.get("Pitch_Degrees", ""),
                    "Pitch (on 12)": extracted_images.get("Pitch_on_12", ""),
                    "Rafters": extracted_images.get("Rafters", ""),
                    "Azimuth": extracted_images.get("Azimuth", ""),
                    "Area": extracted_images.get("Area", ""),
                    "Roof Obstructions": extracted_images.get("Roof_Penetrations", "")
                }
            }

            # Create C004 - Imagery chunk
            imagery_chunk = {
                "chunk_id": "C004",
                "property_id": property_id,
                "section": "Imagery",
                "type": "image",
                "data": {
                    "North": extracted_images.get("North_Side", ""),
                    "South": extracted_images.get("South_Side", ""),
                    "East": extracted_images.get("East_Side", ""),
                    "West": extracted_images.get("West_Side", ""),
                    "Top": extracted_images.get("Top_View", "")
                }
            }

            # Save chunks to Final_Chunks directory
            final_chunks_dir = Path("Final_Chunks")
            final_chunks_dir.mkdir(exist_ok=True)

            chunk_filename = f"report_{report_id}.json" if report_id else f"{file_path.stem}.json"
            chunk_file_path = final_chunks_dir / chunk_filename

            # Load existing chunks if file exists
            existing_chunks = []
            if chunk_file_path.exists():
                try:
                    with open(chunk_file_path, 'r') as f:
                        existing_chunks = json.load(f)
                        if not isinstance(existing_chunks, list):
                            existing_chunks = [existing_chunks]
                except:
                    existing_chunks = []

            # Update or add chunks
            chunk_ids = {chunk.get("chunk_id") for chunk in existing_chunks}
            if "C003" not in chunk_ids:
                existing_chunks.append(diagrams_chunk)
            else:
                # Update existing C003
                for chunk in existing_chunks:
                    if chunk.get("chunk_id") == "C003":
                        chunk.update(diagrams_chunk)
                        break

            if "C004" not in chunk_ids:
                existing_chunks.append(imagery_chunk)
            else:
                # Update existing C004
                for chunk in existing_chunks:
                    if chunk.get("chunk_id") == "C004":
                        chunk.update(imagery_chunk)
                        break

            # Save updated chunks
            with open(chunk_file_path, 'w') as f:
                json.dump(existing_chunks, f, indent=2)

            logging.info(f"Created/updated structured chunks in {chunk_file_path}")

        except Exception as e:
            logging.error(f"Error creating structured chunks: {e}")

    # Removed unused image OCR helper methods (_process_page_image, _process_standalone_image, _extract_text_from_image)

    # Extract important chunks Text Chunks and save them
    def _extract_and_save_important_chunks(self, file_path: Path) -> None:
        """Extract important chunks and save them organized by report ID."""

        documents = []
        try:
            logging.info(f"Starting important chunk extraction for {file_path.name}")
            
            from src.premium_chunk_extractor import extract_premium_chunks
            
            # Extract report ID from filename
            report_id = None
            filename = file_path.name
            if 'RoofReport-' in filename:
                try:
                    report_id = filename.split('RoofReport-')[1].split('.')[0]
                except:
                    pass
            elif 'report_' in filename:
                try:
                    report_id = filename.split('report_')[1].split('.')[0]
                except:
                    pass

            if not report_id:
                report_id = file_path.stem
            
            logging.info(f"Extracting chunks for report ID: {report_id}")
            
            # Extract important chunks
            chunks_data = extract_premium_chunks(str(file_path))
        
            
            if chunks_data:
                # New flattened format: list of chunk dicts
                if isinstance(chunks_data, list):
                    if not chunks_data:
                        logging.warning(f"No content found in chunks for {file_path.name}")
                        return
                    chunks_dir = Path("Final_Chunks")
                    chunks_dir.mkdir(parents=True, exist_ok=True)
                    chunks_file = chunks_dir / f"{file_path.stem}.json"
                    with open(chunks_file, 'w', encoding='utf-8') as f:
                        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

                    all_chunks = []
                    for chunk_dict in chunks_data:
                        if not isinstance(chunk_dict, dict):
                            continue
                        # Build a textual representation for vector index (for text chunks) or minimal for others
                        if chunk_dict.get('type') == 'text':
                            page_content = json.dumps({"section": chunk_dict.get('section'), **chunk_dict.get('data', {})}, ensure_ascii=False)
                        else:  # image
                            page_content = f"Image Section: {chunk_dict.get('section')} {chunk_dict.get('data', {}).get('description','')}"

                        metadata = {
                            'type': chunk_dict.get('type'),
                            'extraction_method': 'important_chunks_flat',
                            'report_id': report_id,
                            'chunk_id': chunk_dict.get('chunk_id'),
                            'section': chunk_dict.get('section'),
                            'source_file': file_path.name
                        }
                        metadata.update(chunk_dict.get('data', {}))
                        if 'image_file' in chunk_dict.get('data', {}):
                            metadata['image_file'] = chunk_dict['data']['image_file']
                        if 'images' in chunk_dict.get('data', {}):
                            metadata['images'] = chunk_dict['data']['images']
                        doc = Document(page_content=page_content, metadata=metadata)
                        all_chunks.append(doc)

                   
                    
                    documents.extend(all_chunks)
                    logging.info(f"✓ Saved {len(all_chunks)} important flattened chunks to {chunks_file}")
                else:
                    logging.warning("Unexpected chunks_data format (expected list). Skipping save.")
            else:
                logging.warning(f"No important chunks extracted from {file_path.name}")
                
        except Exception as e:
            logging.error(f"Error extracting important chunks from {file_path}: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
        return documents


    # Main function to process a directory with Docling
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


# Main function to process a directory with Docling
def process_directory_with_docling(
    directory_path: str,
    file_extensions: Optional[List[str]] = None,
    extract_images: bool = True,
    config_instance: Optional[Any] = None
) -> List[Document]:
    """
    Process all documents in a directory using Docling without indexing.

    Args:
        directory_path: Path to the directory containing documents
        file_extensions: List of file extensions to process
        extract_images: Whether to extract images from documents
        config_instance: Configuration instance

    Returns:
        List of processed documents
    """
    config_instance = config_instance or config

    # Create processor without vector store
    processor = DoclingProcessor(
        vector_store_manager=None,  # No vector store needed
        config_instance=config_instance
    )

    documents = processor.process_directory(
        directory_path=directory_path,
        file_extensions=file_extensions,
        extract_images=extract_images
    )

    if documents:
        logging.info(f"Successfully processed {len(documents)} documents with Docling")
    else:
        logging.warning("No documents found to process")

    return documents


## Removed unused public helper functions get_docling_processor and create_enhanced_index_processor (not referenced in codebase)