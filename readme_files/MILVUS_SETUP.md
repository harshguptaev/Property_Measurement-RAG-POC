# Milvus Database Setup for Property Measurement RAG

This system now uses **Milvus DB** (via Docker) instead of Milvus Lite for better performance and scalability.

## Quick Start

### 1. Start the Complete System
```bash
./start_backend.sh
```

**Note:** Volumes are now managed by Docker (not in project directory)

This will:
- Start Milvus services (etcd, minio, milvus)
- Test the connection
- Activate the venv_rag virtual environment
- Start the RAG API server

### 2. Manual Startup (Alternative)

#### Start Milvus Only
```bash
./start_milvus.sh
```

#### Start RAG Server Only (after Milvus is running)
```bash
source venv_rag/bin/activate  # Activate virtual environment
python hierarchical_rag_server.py
```

## System Architecture

- **Milvus Standalone**: Vector database on port 19530
- **etcd**: Coordination service on port 2379
- **MinIO**: Object storage on ports 9000/9001
- **RAG API Server**: FastAPI server on port 8001

## Key Changes from Milvus Lite

1. **Connection URI**: Changed from `./milvus_demo.db` to `http://localhost:19530`
2. **Docker Services**: Uses Docker Compose for reliable deployment
3. **Virtual Environment**: Uses `venv_rag` for dependency management
4. **Better Performance**: Full Milvus features for production use

## Troubleshooting

### Milvus Connection Issues
```bash
# Check if services are running
docker-compose ps

# View logs
docker-compose logs milvus-standalone

# Restart services
docker-compose restart
```

### Virtual Environment Issues
```bash
# Recreate venv_rag if needed
python3 -m venv venv_rag
source venv_rag/bin/activate
pip install -r requirements.txt
```

### Port Conflicts
- Milvus: 19530, 9091
- etcd: 2379
- MinIO: 9000, 9001
- RAG Server: 8001

## API Usage

Once running, access:
- **API**: http://localhost:8001
- **Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

## Volume Management

**Volumes are now managed by Docker** (not in your project directory). This keeps your codebase clean.

### Volume Locations:
- `milvus_data`: Milvus vector database data
- `milvus_etcd_data`: etcd coordination data
- `milvus_minio_data`: MinIO object storage data

### View Volume Info:
```bash
docker volume ls | grep milvus
docker volume inspect property_measurement-rag-poc_milvus_data
```

## Stopping the System

```bash
# Stop server (Ctrl+C in terminal)

# Stop Milvus services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data!)
docker-compose down -v
```

## Troubleshooting

### Clean Restart (if needed):
```bash
# Stop everything and remove volumes
docker-compose down -v

# Remove Docker volumes completely
docker volume prune

# Restart fresh
./start_backend.sh
```
