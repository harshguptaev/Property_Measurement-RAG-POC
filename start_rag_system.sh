#!/bin/bash

echo "🚀 Starting Property Measurement RAG System with Milvus DB"
echo "========================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

if ! command_exists python3; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Start Milvus services
echo ""
echo "🐳 Starting Milvus database services..."
./start_milvus.sh

# Check if Milvus is ready
echo ""
echo "⏳ Waiting for Milvus to be fully ready..."
sleep 10

# Test Milvus connection
echo "🔗 Testing Milvus connection..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from pymilvus import MilvusClient
    client = MilvusClient(uri='http://localhost:19530')
    collections = client.list_collections()
    print(f'✅ Milvus connection successful! Found {len(collections)} collections.')
except Exception as e:
    print(f'❌ Milvus connection failed: {e}')
    print('Please check if Docker services are running with: docker-compose ps')
    exit(1)
"

# Start the RAG server
echo ""
echo "🧠 Starting Hierarchical RAG Server..."
echo "Using virtual environment: venv_rag"

if [ -d "venv_rag" ]; then
    echo "✅ Found venv_rag virtual environment"
    source venv_rag/bin/activate
    echo "✅ Activated virtual environment"
else
    echo "⚠️  venv_rag not found, using system Python"
fi

# Install/update requirements if needed
echo "📦 Ensuring dependencies are installed..."
pip install -q -r requirements.txt

echo ""
echo "🎉 Starting RAG API Server..."
echo "📡 Server will be available at: http://localhost:8001"
echo "🔍 API documentation at: http://localhost:8001/docs"
echo ""
echo "To stop the system:"
echo "1. Press Ctrl+C to stop the server"
echo "2. Run: docker-compose down"
echo ""

# Start the server
python hierarchical_rag_server.py
