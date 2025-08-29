# Property Retrieval with Docling Integration

This enhanced version of the Property Retrieval system uses **Docling** for superior PDF parsing and multimodal document processing.

## 🌟 New Features with Docling

### Enhanced PDF Processing
- **Advanced Layout Detection**: Better understanding of document structure
- **Table Extraction**: Automatic detection and extraction of tables
- **Image Extraction**: Extract embedded images from PDFs with high quality
- **OCR Support**: Handle scanned PDFs and image-based text
- **Multimodal Understanding**: Process both text and visual elements

### Supported File Types
- **PDF**: Enhanced processing with layout analysis
- **DOCX**: Microsoft Word documents
- **PPTX**: PowerPoint presentations  
- **HTML**: Web pages and HTML documents
- **Markdown**: .md files
- **Text**: Plain text files

## 🚀 Quick Start

### 1. Installation

#### Option A: Automatic Setup (Recommended)
```bash
python setup_docling.py
```

#### Option B: Manual Setup
```bash
# Install core dependencies
pip install -r requirements.txt

# Install Docling
pip install docling docling-core docling-ibm-models

# Install vision processing (optional)
pip install opencv-python pytesseract easyocr
```

### 2. Configuration

The system will automatically use your provided AWS credentials:
- AWS_ACCESS_KEY_ID: ``
- AWS_SECRET_ACCESS_KEY: `[Your provided key]`
- AWS_SESSION_TOKEN: `[Your provided token]`

### 3. Usage

```bash
# Test the setup
python test_docling.py

# Run the main application
python main.py
```

## 📊 Features Comparison

| Feature | Standard Processor | Docling Processor |
|---------|-------------------|-------------------|
| PDF Text Extraction | ✅ Basic | ✅ Advanced Layout |
| Image Extraction | ✅ Basic | ✅ High Quality |
| Table Detection | ❌ | ✅ Structure Preserved |
| OCR Support | ❌ | ✅ Built-in |
| Scanned PDFs | ❌ | ✅ Full Support |
| Office Documents | ❌ | ✅ DOCX, PPTX |
| Multimodal RAG | ❌ | ✅ Text + Images |

## 🔧 Configuration Options

### Vector Store Options
The system supports multiple vector stores. If you need **Milvus** instead of FAISS, update `config.yaml`:

```yaml
database:
  vector_store_type: "milvus"  # Change from "faiss"
  collection_name: "property_documents"
  # Add Milvus-specific configuration
  milvus:
    host: "localhost"
    port: 19530
    user: ""
    password: ""
```

### Processing Configuration
```yaml
processing:
  batch_size: 10
  max_workers: 4
  extract_images: true  # Enable image extraction
  supported_extensions: [".pdf", ".txt", ".md", ".docx", ".doc", ".pptx"]
```

## 📁 Directory Structure

```
property-retrieval/
├── src/
│   ├── docling_index.py      # New Docling processor
│   ├── index.py              # Original processor (fallback)
│   └── ...
├── input_files/              # Place your documents here
├── temp_docling/             # Temporary processing files
├── vector_store/             # Vector database storage
├── setup_docling.py          # Setup script
├── test_docling.py           # Test script
└── main.py                   # Main application
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_docling.py
```

This will test:
- ✅ Docling installation and import
- ✅ AWS credentials and Bedrock access
- ✅ Document processing capabilities
- ✅ Vector store functionality
- ✅ End-to-end processing pipeline

## 💡 Usage Examples

### Processing PDFs with Images
```python
from src.docling_index import DoclingProcessor

processor = DoclingProcessor()
documents = processor.process_file("property_report.pdf", extract_images=True)

# Documents will include:
# - Text chunks from the PDF
# - Extracted images with metadata
# - Table data (if present)
# - OCR text from images
```

### Batch Processing
```python
from src.docling_index import process_and_index_directory_with_docling

vector_store = process_and_index_directory_with_docling(
    directory_path="input_files/",
    extract_images=True,
    drop_existing=True
)
```

## 🔍 Document Types and Processing

### Property Documents
- **Inspection Reports**: Extract text, images, and diagrams
- **Appraisal Reports**: Process comparable sales data and photos
- **Floor Plans**: Extract images and analyze layout
- **Legal Documents**: Process complex formatting and tables
- **Financial Reports**: Extract financial tables and charts

### Multimodal Capabilities
- **Image Analysis**: Extract and describe property images
- **Chart Processing**: Understand graphs and financial data
- **Table Extraction**: Preserve table structure and relationships
- **Layout Understanding**: Maintain document hierarchy

## 🛠️ Troubleshooting

### Common Issues

1. **Docling Import Error**
   ```bash
   pip install --upgrade docling docling-core
   ```

2. **AWS Credentials**
   - Credentials are embedded in `main.py`
   - Verify region is set to `us-east-1`

3. **Memory Issues with Large PDFs**
   - Reduce batch size in `config.yaml`
   - Process files individually

4. **Image Processing Errors**
   ```bash
   pip install opencv-python pillow
   ```

### Performance Tips

- **Large Documents**: Enable multiprocessing in configuration
- **High-Resolution Images**: Adjust `images_scale` in Docling options
- **Memory Usage**: Monitor and adjust chunk sizes

## 🔄 Migration from Standard Processor

The system automatically detects Docling availability and uses it when possible. No code changes needed - just install Docling!

```python
# Automatic fallback logic in main.py
if DOCLING_AVAILABLE:
    # Use Docling processor
    vector_store_manager = process_and_index_directory_with_docling(...)
else:
    # Fallback to standard processor
    vector_store_manager = process_and_index_directory(...)
```

## 📞 Support

For issues specific to Docling integration:
1. Run `python test_docling.py` to diagnose problems
2. Check the logs for detailed error messages
3. Verify all dependencies are installed correctly

## 🚀 Next Steps

1. **Add Your Documents**: Place PDFs in `input_files/`
2. **Run Setup**: Execute `python setup_docling.py`
3. **Test System**: Run `python test_docling.py`
4. **Launch Application**: Execute `python main.py`
5. **Access Interface**: Open browser to provided URL

Enjoy enhanced document processing with Docling! 🎉
