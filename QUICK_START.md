# 🚀 Quick Start Guide

## **Property Retrieval RAG System**

Get your property document analysis system running in **5 minutes**!

---

## ⚡ **Quick Setup**

### **1. Run Setup Script**
```bash
cd property-retrieval
python setup.py
```

### **2. Activate Environment**
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### **3. Set AWS Credentials**
```bash
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_SESSION_TOKEN="your_token"  # if using temporary credentials
```

### **4. Add Documents**
```bash
# Copy your PDF files to:
cp your_property_reports.pdf input_files/
```

### **5. Start Application**
```bash
python main.py
```

### **6. Open Web Interface**
Navigate to: **http://localhost:7860**

---

## 💬 **Test Questions**

Once running, try these questions:

### **Basic Property Analysis:**
- "What is the condition of the roof?"
- "Summarize the key findings"
- "What repairs are recommended?"

### **Image/Diagram Retrieval:**
- "Show me diagrams from the roof report"
- "What images are on page 1?"
- "Are there any photos of damage?"

---

## 🔧 **Troubleshooting**

**No PDFs found?**
- Add PDF files to `input_files/` directory

**AWS errors?**
- Check credentials: `aws sts get-caller-identity`
- Ensure Bedrock access is enabled

**Dependencies missing?**
- Run: `pip install -r requirements.txt`

---

## 📊 **What You Get**

✅ **Multimodal PDF Processing** - Text + Images  
✅ **AI-Powered Q&A** - Claude 3 Sonnet  
✅ **Diagram Retrieval** - Find visual content  
✅ **Web Interface** - Easy to use chat  
✅ **Vector Search** - Semantic document search  

Ready to analyze your property documents! 🏠
