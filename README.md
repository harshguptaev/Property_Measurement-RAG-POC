# Property Retrieval RAG System



### RUN IT LOCALLY USING DOCKERFILE
-   export credentials
-   run docker build
 ```
 docker build -t property-retrieval-poc:latest .
 ```
 -  docker run 
 ```
 docker run --rm -it \
    -p 7860:7860 \
    -v "$(pwd)/input_files:/app/input_files" \
    -v "$(pwd)/input_data:/app/input_data" \
    -v "$(pwd)/vectorstore_faiss:/app/vectorstore_faiss" \
    -v "$(pwd)/extracted_images:/app/extracted_images" \
    -v "$(pwd)/config.yaml:/app/config.yaml" \
    -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -e AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
    -e AWS_DEFAULT_REGION="us-east-1" \
    --name property-retrieval-app \
    property-retrieval-poc
 ```

 
## 🏠 **Property Document Analysis with Diagram Retrieval**

A multimodal Retrieval-Augmented Generation (RAG) system that processes property documents (PDFs) and enables intelligent question-answering with **diagram and image retrieval capabilities**.

---

## ✨ **Features**

- **📄 PDF Processing**: Extract text and images from property reports
- **🖼️ Diagram Retrieval**: Find and reference diagrams, charts, and photos
- **🧠 AI-Powered Q&A**: Claude 3 Sonnet for intelligent responses
- **🔍 Vector Search**: FAISS-based semantic search
- **🌐 Web Interface**: User-friendly Gradio interface
- **☁️ AWS Integration**: Bedrock for LLMs and embeddings

---

## 🛠️ **Tech Stack**

| Component | Technology |
|-----------|------------|
| **PDF Processing** | PyPDFLoader, PyMuPDF (fitz) |
| **Vector Store** | FAISS |
| **Embeddings** | AWS Bedrock Titan Embeddings |
| **LLM** | AWS Bedrock Claude 3 Sonnet |
| **Web Interface** | Gradio |
| **Image Processing** | PIL (Pillow) |
| **Workflow** | LangGraph (optional) |

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- AWS Account with Bedrock access
- AWS credentials configured

### **1. Setup Environment**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Configure AWS Credentials**
```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_SESSION_TOKEN="your_session_token"  # if using temporary credentials

# Option 2: AWS CLI
aws configure
```

### **3. Add Documents**
Place your PDF files in the `input_files/` directory:
```bash
cp your_property_reports.pdf input_files/
```

### **4. Run the Application**
```bash
python main.py
```

### **5. Access Web Interface**
Open your browser to: `http://localhost:7860`

---

## 📊 **Usage Examples**

### **Document Analysis Queries:**
- "What is the condition of the roof?"
- "Summarize the key findings from the property report"
- "What repairs are recommended?"

### **Diagram/Image Queries:**
- "Show me diagrams from the roof report"
- "What images are available on page 1?"
- "Are there any photos of roof damage?"
- "Display visual evidence from the documents"

---

## 📁 **Project Structure**

```
property-retrieval/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── bedrock_client.py    # AWS Bedrock integration
│   ├── vector_store.py      # FAISS vector store management
│   ├── index.py             # Document processing & indexing
│   ├── agent.py             # RAG agent with image retrieval
│   └── ui.py                # Gradio web interface
├── input_files/             # Place your PDF documents here
├── config.yaml              # System configuration
├── requirements.txt         # Python dependencies
├── main.py                  # Application entry point
└── README.md               # This file
```

---

## ⚙️ **Configuration**

Edit `config.yaml` to customize:

```yaml
bedrock:
  region: "us-east-1"
  model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
  embedding_model_id: "amazon.titan-embed-text-v1"

retrieval:
  vector_store_type: "faiss"
  collection_name: "property_documents"
  k: 10

ui:
  port: 7860
  share: false
```

---

## 🖼️ **Image/Diagram Processing**

The system automatically:
1. **Extracts images** from PDFs using PyMuPDF
2. **Stores metadata** (page numbers, sizes, sources)
3. **Creates embeddings** for both text and image references
4. **Retrieves relevant visuals** during Q&A
5. **Displays image info** in chat responses

### **Example Response with Images:**
```
User: "Show me diagrams from the roof report"
Assistant: "Based on the provided context, I found several visual elements that may contain important information about the roof condition..."

📊 Retrieved 3 diagram(s)/image(s):
• Image 1: Page 1 of RoofReport-44995436.pdf (1852x1078 pixels)
• Image 2: Page 2 of RoofReport-44995436.pdf (2460x2176 pixels)
• Image 3: Page 3 of RoofReport-44995436.pdf (2460x2176 pixels)

💡 Note: The above images/diagrams contain visual information that supports this response.
```

---

## 🔧 **Troubleshooting**

### **Common Issues:**

**AWS Credentials Error:**
```bash
# Check credentials
aws sts get-caller-identity

# Refresh temporary credentials
aws sso login
```

**No Images Retrieved:**
- Ensure PDFs contain images/diagrams
- Check AWS Bedrock permissions
- Verify documents are properly processed

**Dependencies Issues:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## 🎯 **Features Overview**

### **✅ Core Capabilities:**
- Multimodal PDF processing (text + images)
- Intelligent document question-answering
- Diagram and image retrieval
- Web-based chat interface
- Vector similarity search
- AWS Bedrock integration

### **🔜 Future Enhancements:**
- Direct image display in chat
- Image content description AI
- Multi-document comparison
- Batch document processing
- Advanced search filters

---

## 📞 **Support**

For issues or questions:
1. Check the troubleshooting section above
2. Verify AWS credentials and permissions
3. Ensure all dependencies are installed
4. Check that input PDFs contain processable content

---

## 📄 **License**

This project is for internal use. All rights reserved.

