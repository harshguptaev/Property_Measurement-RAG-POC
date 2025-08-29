#!/usr/bin/env python3
"""
Setup script for Property Retrieval RAG System.
Run this script to set up the environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    
    print("🚀 Property Retrieval RAG System Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create virtual environment if it doesn't exist
    venv_path = Path("venv")
    if not venv_path.exists():
        print("🏗️  Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv venv", "Virtual environment creation"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation script
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        python_cmd = "venv\\Scripts\\python"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_script = "source venv/bin/activate"
        python_cmd = "venv/bin/python"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    print("📦 Installing dependencies...")
    install_cmd = f"{pip_cmd} install -r requirements.txt"
    if not run_command(install_cmd, "Dependencies installation"):
        return False
    
    # Create input_files directory
    input_dir = Path("input_files")
    if not input_dir.exists():
        input_dir.mkdir()
        print("✅ Created input_files directory")
    else:
        print("✅ input_files directory already exists")
    
    # Check AWS CLI (optional)
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
        print("✅ AWS CLI detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  AWS CLI not found (optional but recommended)")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print(f"   1. Activate virtual environment: {activate_script}")
    print("   2. Set AWS credentials (see README.md)")
    print("   3. Add PDF files to input_files/ directory")
    print("   4. Run: python main.py")
    print("\n📚 See README.md for detailed instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
