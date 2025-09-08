"""
Enhanced document processing and indexing pipeline using Docling.
Supports advanced multimodal processing of text and images from documents.
Based on the reference implementation with improved PDF and image parsing.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from io import BytesIO
from PIL import Image
import pandas as pd

# Docling imports for advanced document processing
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import TableItem, TextItem
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
from .vector_store import VectorStoreManager, create_text_splitter, create_table_splitter
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
        self.table_splitter = None
        self.supported_extensions = {'.pdf', '.docx', '.pptx', '.html', '.md', '.txt'}
        
        # Initialize image manager with Gemini support
        self.image_manager = ImageManager(enable_gemini=enable_gemini)
        
        # Initialize Docling converter with enhanced options
        self._setup_docling_converter()
        self._setup_text_splitter()
        self._setup_table_splitter()
    
    def _setup_docling_converter(self):
        """Setup Docling converter with simplified PDF processing options."""
        try:
            # Configure pipeline options for better PDF processing
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Enable OCR for scanned PDFs
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
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
    
    def _setup_table_splitter(self):
        """Setup table splitter for chunking tables."""
        vector_config = self.config.get_vector_store_config()
        self.table_splitter = create_table_splitter(
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
            try:
                tables_list = list(getattr(converted_doc, "tables", []) or [])
            except Exception:
                tables_list = getattr(converted_doc, "tables", []) or []
            logging.info(f"Docling tables count: {len(tables_list)}")

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
                
            
            # Extract page-level content with images
            if extract_images:
                page_documents = self._extract_pages_and_images(converted_doc, file_path)
                documents.extend(page_documents)
            
            # Extract tables if present
            table_documents = self._extract_tables(converted_doc, tables_list, file_path)
            documents.extend(table_documents)
            
            # Extract important chunks and save them
            important_chunks = self._extract_and_save_important_chunks(file_path)
            documents.extend(important_chunks)
            

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
                num_images_on_page = len(image_list)
                
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
                            
                            def _label_for(p, i, n):
                                if p == 0:
                                    return "Cover_Image" if i == 0 else f"Cover_Image_{i + 1}"
                                if p == 1:
                                    return "Lengthsimage" if n == 1 else f"Lengthsimage_{i + 1}"
                                if p == 2:
                                    return "Pitch_Degrees" if i == 0 else f"Pitch_Degrees_{i + 1}"
                                if p == 3:
                                    return "Pitch_on_12" if i == 0 else f"Pitch_on_12_{i + 1}"
                                if p == 4:
                                    return "Rafters" if i == 0 else f"Rafters_{i + 1}"
                                if p == 5:
                                    return "Azimuth" if i == 0 else f"Azimuth_{i + 1}"
                                if p == 6:
                                    return "Area" if i == 0 else f"Area_{i + 1}"
                                if p == 7:
                                    return "Roof_Penetrations" if i == 0 else f"Roof_Penetrations_{i + 1}"
                                if p == 8:
                                    return "Top_View" if i == 0 else ("North_Side" if i == 1 else f"Page_8_Image_{i + 1}")
                                if p == 9:
                                    return "South_Side" if i == 0 else ("East_Side" if i == 1 else f"Page_9_Image_{i + 1}")
                                if p == 10:
                                    return "West_Side" if i == 0 else f"West_Side_{i + 1}"
                                if p == 11:
                                    return "Structure_Summary" if i == 0 else f"Structure_Summary_{i + 1}"
                                return f"Page_{p}_Image_{i + 1}"

                            image_label = _label_for(page_num, img_index, num_images_on_page)
                            image_filename = f"{image_label}.png"
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
                            print("Enhanced image metadata with Gemini.", img_doc.metadata)
                            documents.append(img_doc)
                        
                        pix = None  # Cleanup
                    except Exception as e:
                        logging.warning(f"Error processing image {img_index} on page {page_num + 1}: {e}")
            
            pdf_document.close()
            logging.info(f"Extracted {len(documents)} images from {file_path.name} using PyMuPDF")
            
        except Exception as e:
            logging.error(f"Error extracting images with PyMuPDF: {e}")
            
        return documents
    
    # Removed unused image OCR helper methods (_process_page_image, _process_standalone_image, _extract_text_from_image)
        
    def _extract_tables(self, converted_doc, tables_list, file_path: Path) -> List[Document]:
        """
        For each Docling table:
        - Export to DataFrame for analytic correctness and downstream SQL/DF pipelines.
        - Export Areas per pitch to json.
        - Export Waste Calculation to images.
        - Chunk tables.
        """
        
        out_docs: List[Document] = []
        # Ensure we can iterate all tables reliably across versions
        for idx, table in enumerate(tables_list):
            # 1) DataFrame export (safe structure)
            df = None
            try:
                df = table.export_to_dataframe()
            except Exception as e:
                logging.warning(f"Error exporting table {idx} to dataframe: {e}")
                df = None

            self.export_table_data(df, file_path, idx, tables_list)

        self.export_table_images(converted_doc, file_path)
        self.export_table_data_chunks(file_path)

        areas_per_pitch_path = Path("docling_exports") / file_path.stem / "tables" / "json"
        waste_calculation_path = Path("docling_exports") / file_path.stem / "tables" / "images"
        report_id = file_path.stem.split('RoofReport-')[1].split('.')[0]
        for area_per_pitch_file in areas_per_pitch_path.glob("*.json"):
            print("area_per_pitch_file", area_per_pitch_file)
            table_doc = Document(
                page_content=f"Report ID: {report_id}\n{area_per_pitch_file.read_text()}",
                metadata={
                    'type': 'table',
                    'file_name': area_per_pitch_file.name,
                    'source_path': str(area_per_pitch_file),
                    'file_type': 'json',
                    'report_id': report_id,
                    'name': 'Areas Per Pitch',
                    'description': 'This table comes under ROOFING REPORT SUMMARY in pdf. The table lists each pitch on this roof and the total area and percent of the roof with that pitch. and the suffix of the file name tells which structure it belongs to. If suffix is AllStructures, then it is the total of the roofs for all structures.',
                    'extraction_method': 'docling_table'
                }
            )
            # Chunk tables
            chunks = self.text_splitter.split_documents([table_doc])
            out_docs.extend(chunks)

        for waste_calculation_file in waste_calculation_path.glob("*.png"):
            print("waste_calculation_file", waste_calculation_file.name)
            if waste_calculation_file.name.startswith("Structure_Complexity"):
                name = "Structure_Complexity"
            else:
                name = "Waste_Calculation"
            table_doc = Document(
                page_content=f"Report ID: {report_id}\n{waste_calculation_file}",
                metadata={
                    'type': 'image',
                    'file_name': waste_calculation_file.name,
                    'file_type': 'png',
                    'report_id': file_path.stem.split('RoofReport-')[1].split('.')[0],
                    'name': name,
                    'description':'''These are basically the Structure Complexity and Waste Calculation tables in the pdf and we are storing them as images. 
                                    This Table comes under ROOFING REPORT SUMMARY in pdf. *Squares are rounded up to the 1/3 of a square
                                    Additional materials needed for ridge, hip, and starter lengths are not included in the above table. The provided suggested waste
                                    factor is intended to serve as a guide–actual waste percentages may differ based upon several variables that EagleView does not
                                    control. These waste factor variables include, but are not limited to, individual installation techniques, crew experiences, asphalt
                                    shingle material subtleties, and potential salvage from the site. Individual results may vary from suggested waste factor that
                                    EagleView has provided. The suggested waste is not to replace or substitute for experience or judgement as to any given
                                    replacement or repair work''',
                    'extraction_method': 'docling_image'
                }
            )
            out_docs.append(table_doc)
        return out_docs
    
    def export_table_data(self, df, file_path, idx, tables_list):

        """Export table data to JSON files."""
        if idx==0:
            return
        name = ""

        if idx%2==0:
            name = "Waste_Calculation_" + str(idx//2)
        if idx%2==1:
            name = "Areas_per_Pitch_Structure_" + str(idx//2 + 1)
        if len(tables_list)>3 and idx == len(tables_list)-1:
            name = "Areas_per_Pitch_AllStructures"

        out_dir = Path("docling_exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file_path).stem
        out_report_dir = out_dir / stem
        out_report_dir.mkdir(parents=True, exist_ok=True)
        out_report_table_dir = out_report_dir / "tables"
        out_report_table_dir.mkdir(parents=True, exist_ok=True)
        out_json_report_table_dir = out_report_table_dir / "json"
        out_json_report_table_dir.mkdir(parents=True, exist_ok=True)

        if idx%2==0:
            json_file = out_json_report_table_dir / f"{name}.json"
            # data = self.df_to_waste_json(df)
            df_dict = {
                "columns": df.columns.tolist(),
                "data": df.values.tolist()
            }
            data = self.df_to_waste_json(df)
            json_str = json.dumps(data, ensure_ascii=False, indent=4)
            json_file.write_text(json_str, encoding="utf-8")
            return

        df = df.T
        df.columns = df.iloc[0]   # first row becomes column names
        df = df.drop(0)           # drop the old header row
        (out_json_report_table_dir / f"{name}.json").write_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=4),encoding="utf-8")

    def df_to_waste_json(self, df):

        # Initialize indices
        waste_row_idx = None
        area_row_idx = None
        squares_row_idx = None
        
        # Search for rows starting with specific labels (case-insensitive)
        for idx in range(len(df)):
            first_cell = str(df.iloc[idx, 0]).strip().lower()
            if 'waste%' in first_cell:
                waste_row_idx = idx
            elif 'area' in first_cell:
                area_row_idx = idx
            elif 'squares' in first_cell:
                squares_row_idx = idx
        
        # Extract lists
        if waste_row_idx is not None:
            waste_list = df.iloc[waste_row_idx, 1:].astype(str).tolist()
        else:
            # Parse waste from columns (last part after '.')
            waste_list = []
            for col in df.columns[1:]:
                parts = str(col).split('.')
                last_part = parts[-1].strip() if parts else ''
                waste_list.append(last_part)
        
        if area_row_idx is not None:
            area_list = df.iloc[area_row_idx, 1:].astype(str).tolist()
        else:
            # Assume first row is area if not found
            area_list = df.iloc[0, 1:].astype(str).tolist()
        
        if squares_row_idx is not None:
            squares_list = df.iloc[squares_row_idx, 1:].astype(str).tolist()
        else:
            # Assume second row is squares if not found
            squares_list = df.iloc[1, 1:].astype(str).tolist()
        
        # Determine the minimum length to align lists
        min_len = min(len(waste_list), len(area_list), len(squares_list))
        
        # Build the result list
        result = []
        for i in range(min_len):
            result.append({
                "waste": waste_list[i],
                "area": area_list[i],
                "squares": squares_list[i]
            })
        
        return result
    def export_table_images(self, converted_doc, file_path):

        """Export table images to PNG files."""
        table_counter = 0
        out_dir = Path("docling_exports")
        stem = Path(file_path).stem
        out_report_dir = out_dir / stem
        out_report_table_dir = out_report_dir / "tables"
        out_report_table_images_dir = out_report_table_dir / "images"
        out_report_table_images_dir.mkdir(parents=True, exist_ok=True)

        for element, _ in converted_doc.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                if table_counter == 1:
                    continue
                
                # Even tables -> Areas_per_Pitch_Structure
                if table_counter % 2 == 0:
                    name = f"Areas_per_Pitch_Structure_{table_counter // 2}.png"
                    if table_counter == len(converted_doc.tables):
                        name = "Areas_per_Pitch_AllStructures.png"
                    img = element.get_image(converted_doc)
                    img.save(out_report_table_images_dir / name)
                    continue

                img = element.get_image(converted_doc)
                width, height = img.size
                # Y positions in pixels
                y26 = int(height * 0.26)
                y28 = int(height * 0.28)
                # Top part (0% → 26%)
                top_img = img.crop((0, 0, width, y26))

                # Bottom part (28% → 100%)
                bottom_img = img.crop((0, y28, width, height))

                # Save results
                top_img.save(f"{out_report_table_images_dir}/Structure_Complexity_{table_counter//2}.png")
                bottom_img.save(f"{out_report_table_images_dir}/Waste_Calculation_{table_counter//2}.png")

    def export_table_data_chunks(self, file_path):
        """Export table data chunks to JSON files.
          "table":[{
                "section":"Hardcoded",
                "raw_text":"JSON or CSV",
                "id":"autogenerated",
                "metadata":{},
                "src_image_path":"reference path to the image"
        }}]"""
        out_dir = Path("docling_exports")
        stem = Path(file_path).stem
        out_report_dir = out_dir / stem
        out_report_table_dir = out_report_dir / "tables"
        out_report_table_json_dir = out_report_table_dir / "json"
        out_report_table_images_dir = out_report_table_dir / "images"

        table_chunks = []
        chunk_counter = 1

        # Iterate over all JSON files
        for json_file in sorted(out_report_table_json_dir.glob("*.json")):
            try:
                json_content = json.loads(json_file.read_text(encoding="utf-8"))
                raw_text_str = json.dumps(json_content, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.warning(f"Skipping {json_file.name}, failed to read JSON: {e}")
                continue

            # Determine section
            section = ""
            if "Areas_per_Pitch" in json_file.name:
                section = f"This table named {json_file.stem } lists each pitch on this roof and the total area and percent of the roof with that pitch."
            elif "Waste_Calculation" in json_file.name:
                section = f"""NOTE: This waste calculation table named {json_file.stem} is for asphalt shingle roofing applications. All values in the table below 
                            only include roof areas of 3/12 pitch or greater. *Squares are rounded up to the 1/3 of a square
                            Additional materials needed for ridge, hip, and starter lengths are not included in the above table. The provided suggested waste
                            factor is intended to serve as a guide–actual waste percentages may differ based upon several variables that EagleView does not
                            control. These waste factor variables include, but are not limited to, individual installation techniques, crew experiences, asphalt
                            shingle material subtleties, and potential salvage from the site. Individual results may vary from suggested waste factor that
                            EagleView has provided. The suggested waste is not to replace or substitute for experience or judgement as to any given
                            replacement or repair work."""
            # Corresponding image path (same filename but .png)
            image_name = json_file.stem + ".png"
            image_path = out_report_table_images_dir / image_name
            if not image_path.exists():
                logging.warning(f"Image not found for {json_file.name}: {image_path}")
                image_path_str = ""
            else:
                image_path_str = str(image_path)

            # Append chunk
            table_chunks.append({
                "section": section,
                "raw_text": raw_text_str,
                "id": f"chunk{chunk_counter}",
                "metadata": {},
                "src_image_path": image_path_str
            })
            chunk_counter += 1

        for png_file in sorted(out_report_table_images_dir.glob("*.png")):
            if "Structure_Complexity" in png_file.name:
                section = f"This table named {png_file.stem} lists the structure complexity of the roof."
                table_chunks.append({
                    "section": section,
                    "raw_text": "Dummy Text for now.",
                    "id": f"chunk{chunk_counter}",
                    "metadata": {},
                    "src_image_path": str(png_file)
                })
                chunk_counter += 1

        # Save consolidated table_chunks.json
        chunks_file = out_report_dir / "table_chunks.json"
        chunks_data = {"tables": table_chunks}
        chunks_file.write_text(json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8")
    
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
    
    def _extract_and_save_important_chunks(self, file_path: Path) -> None:
        """Extract important chunks and save them organized by report ID."""

        documents = []
        try:
            logging.info(f"Starting important chunk extraction for {file_path.name}")
            
            # Try to import the extractor
            try:
                from .important_chunk_extractor import extract_important_chunks
                logging.info("Successfully imported important_chunk_extractor")
            except ImportError as ie:
                logging.error(f"Failed to import important_chunk_extractor: {ie}")
                return
            
            # Extract report ID from filename
            report_id = None
            if 'RoofReport-' in file_path.name:
                try:
                    report_id = file_path.name.split('RoofReport-')[1].split('.')[0]
                except:
                    pass
            
            if not report_id:
                report_id = file_path.stem
            
            logging.info(f"Extracting chunks for report ID: {report_id}")
            
            # Extract important chunks
            chunks_data = extract_important_chunks(str(file_path))
            
            # Debug logging
            logging.info(f"Chunks data type: {type(chunks_data)}")
            logging.info(f"Chunks data truthy: {bool(chunks_data)}")
            if isinstance(chunks_data, dict):
                logging.info(f"Chunks data keys: {list(chunks_data.keys())}")
                for key, value in chunks_data.items():
                    if isinstance(value, list):
                        logging.info(f"  {key}: {len(value)} items")
                    else:
                        logging.info(f"  {key}: {type(value)}")
            
            if chunks_data:
                # New flattened format: list of chunk dicts
                if isinstance(chunks_data, list):
                    if not chunks_data:
                        logging.warning(f"No content found in chunks for {file_path.name}")
                        return
                    chunks_dir = Path("docling_exports") / file_path.stem
                    chunks_dir.mkdir(parents=True, exist_ok=True)
                    chunks_file = chunks_dir / "important_chunks.json"
                    with open(chunks_file, 'w', encoding='utf-8') as f:
                        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

                    all_chunks = []
                    for chunk_dict in chunks_data:
                        if not isinstance(chunk_dict, dict):
                            continue
                        # Build a textual representation for vector index (for text chunks) or minimal for others
                        if chunk_dict.get('type') == 'text':
                            page_content = json.dumps({"section": chunk_dict.get('section'), **chunk_dict.get('data', {})}, ensure_ascii=False)
                        elif chunk_dict.get('type') == 'table':
                            page_content = f"Table Section: {chunk_dict.get('section')}"
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
                        if chunk_dict.get('image_placeholder'):
                            metadata['image_placeholder'] = chunk_dict['image_placeholder']
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


## Removed unused public helper functions get_docling_processor and create_enhanced_index_processor (not referenced in codebase)
