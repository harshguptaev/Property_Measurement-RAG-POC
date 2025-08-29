#!/usr/bin/env python3
"""
Setup script for installing Docling and dependencies for enhanced PDF processing.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required for Docling")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install all required dependencies."""
    print("🚀 Installing enhanced dependencies for Property Retrieval with Docling...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install core requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing core requirements"):
        return False
    
    # Install Docling specifically (latest version)
    docling_packages = [
        "docling",
        "docling-core", 
        "docling-ibm-models"
    ]
    
    for package in docling_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}")
    
    # Install additional vision processing packages
    vision_packages = [
        "opencv-python",
        "pytesseract",  # For OCR
        "easyocr"       # Alternative OCR
    ]
    
    for package in vision_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package} (optional)")
    
    return True


def test_installation():
    """Test if all packages are properly installed."""
    print("\n🧪 Testing installation...")
    
    test_results = {}
    
    # Test core packages
    core_packages = [
        "langchain",
        "boto3",
        "gradio",
        "faiss",
        "PIL",
        "cv2"
    ]
    
    for package in core_packages:
        try:
            __import__(package)
            test_results[package] = True
            print(f"✅ {package} imported successfully")
        except ImportError:
            test_results[package] = False
            print(f"❌ {package} import failed")
    
    # Test Docling specifically
    try:
        from docling.document_converter import DocumentConverter
        test_results["docling"] = True
        print("✅ Docling imported successfully")
    except ImportError:
        test_results["docling"] = False
        print("❌ Docling import failed")
    
    # Summary
    passed = sum(test_results.values())
    total = len(test_results)
    
    print(f"\n📊 Test Results: {passed}/{total} packages working")
    
    if test_results.get("docling", False):
        print("🎉 Docling is ready for enhanced PDF processing!")
        return True
    else:
        print("⚠️  Docling installation needs attention")
        return False


def setup_environment():
    """Setup environment and directories."""
    print("\n📁 Setting up environment...")
    
    # Create necessary directories
    directories = [
        "input_files",
        "temp_docling", 
        "vector_store",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")
    
    # Create example files
    example_readme = Path("input_files/README.txt")
    if not example_readme.exists():
        with open(example_readme, "w") as f:
            f.write("""Welcome to Property Retrieval with Docling!

Place your PDF files in this directory to process them.

Docling Features:
- Advanced PDF parsing
- Image extraction from PDFs
- Table detection and extraction
- OCR for scanned documents
- Multimodal document understanding

Simply place your PDFs here and run:
python main.py
""")
        print("✅ Created example README in input_files/")


def main():
    """Main setup function."""
    print("🏠 Property Retrieval Setup with Docling")
    print("=" * 50)
    
    try:
        # Install dependencies
        if not install_dependencies():
            print("\n❌ Dependency installation failed")
            return False
        
        # Test installation
        if not test_installation():
            print("\n⚠️  Some packages may need manual installation")
        
        # Setup environment
        setup_environment()
        
        print("\n" + "=" * 50)
        print("🎉 Setup completed!")
        print("\nNext steps:")
        print("1. Place your PDF files in the 'input_files' directory")
        print("2. Run: python main.py")
        print("3. Open your browser to the provided URL")
        print("\nFor AWS credentials, make sure they are set in your environment or")
        print("the script will use the provided temporary credentials.")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
