#!/bin/bash

# Start Milvus using Docker Compose
echo "🚀 Starting Milvus database services..."

# Create volumes directory if it doesn't exist
mkdir -p volumes

# Start services in detached mode
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for Milvus services to start..."
sleep 30

# Check if services are running
echo "📊 Checking service status..."
docker-compose ps

# Test connection to Milvus
echo "🔗 Testing connection to Milvus..."
python3 -c "
import sys
sys.path.insert(0, '.')
from pymilvus import MilvusClient
try:
    client = MilvusClient(uri='http://localhost:19530')
    collections = client.list_collections()
    print(f'✅ Connected to Milvus successfully! Found {len(collections)} collections.')
except Exception as e:
    print(f'❌ Failed to connect to Milvus: {e}')
    print('Please wait a bit longer for services to fully start.')
"

echo "🎉 Milvus is ready! You can now run your RAG system."
