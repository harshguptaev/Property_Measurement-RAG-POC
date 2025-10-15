# 🚀 Simple Backend Startup

## One-Command Backend Launch

```bash
./start_backend.sh
```

## What It Does

1. **Prerequisites Check** - Ensures Docker, Docker Compose, Python are installed
2. **Cleanup** - Kills any existing processes on ports 8001 (RAG) and 19530 (Milvus)
3. **Milvus Database** - Starts the vector database (etcd, minio, standalone, attu)
4. **Connection Test** - Verifies Milvus is working
5. **Python Setup** - Activates virtual environment, installs dependencies
6. **RAG Server** - Starts the Hierarchical RAG system on port 8001

## Ports Used

- **8001** - Hierarchical RAG API server
- **19530** - Milvus vector database
- **9091** - Milvus metrics
- **9000/9001** - MinIO storage
- **8000** - Milvus Attu (web UI)

## To Stop

```bash
# Stop RAG server
Ctrl+C

# Stop Milvus database
docker-compose down
```

## Connect Assistant UI

To use with the web interface, set in `assistant-ui/.env.local`:

```env
RAG_BACKEND_URL=http://localhost:8001
```

Then run the assistant UI separately:

```bash
cd assistant-ui
npm run dev
```
