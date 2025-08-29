#!/bin/bash

# Docker management script for Property Retrieval System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     Build the Docker image"
    echo "  run       Run the container (interactive)"
    echo "  start     Start the container (detached)"
    echo "  stop      Stop the container"
    echo "  restart   Restart the container"
    echo "  logs      Show container logs"
    echo "  shell     Open shell in running container"
    echo "  clean     Remove container and image"
    echo "  status    Show container status"
    echo "  help      Show this help message"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
        exit 1
    fi
}

check_env() {
    if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
        echo -e "${YELLOW}Warning: AWS credentials not set in environment${NC}"
        echo "Make sure to export your AWS credentials before running the container"
    else
        echo -e "${GREEN}✓ AWS credentials found in environment${NC}"
    fi
}

IMAGE_NAME="property-retrieval"
CONTAINER_NAME="property-retrieval-app"

case "${1:-help}" in
    build)
        echo -e "${BLUE}Building Docker image...${NC}"
        check_docker
        docker build -t $IMAGE_NAME .
        echo -e "${GREEN}✓ Image built successfully${NC}"
        ;;
    
    run)
        echo -e "${BLUE}Running container interactively...${NC}"
        check_docker
        check_env
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
            -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
            --name $CONTAINER_NAME \
            $IMAGE_NAME
        ;;
    
    start)
        echo -e "${BLUE}Starting container in detached mode...${NC}"
        check_docker
        check_env
        docker run -d \
            -p 7860:7860 \
            -v "$(pwd)/input_files:/app/input_files" \
            -v "$(pwd)/input_data:/app/input_data" \
            -v "$(pwd)/vectorstore_faiss:/app/vectorstore_faiss" \
            -v "$(pwd)/extracted_images:/app/extracted_images" \
            -v "$(pwd)/config.yaml:/app/config.yaml" \
            -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
            -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
            -e AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
            -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
            --name $CONTAINER_NAME \
            --restart unless-stopped \
            $IMAGE_NAME
        echo -e "${GREEN}✓ Container started${NC}"
        echo -e "${BLUE}Access the app at: http://localhost:7860${NC}"
        ;;
    
    stop)
        echo -e "${BLUE}Stopping container...${NC}"
        check_docker
        docker stop $CONTAINER_NAME 2>/dev/null || echo "Container not running"
        docker rm $CONTAINER_NAME 2>/dev/null || echo "Container not found"
        echo -e "${GREEN}✓ Container stopped${NC}"
        ;;
    
    restart)
        echo -e "${BLUE}Restarting container...${NC}"
        $0 stop
        $0 start
        ;;
    
    logs)
        echo -e "${BLUE}Showing container logs...${NC}"
        check_docker
        docker logs -f $CONTAINER_NAME
        ;;
    
    shell)
        echo -e "${BLUE}Opening shell in container...${NC}"
        check_docker
        docker exec -it $CONTAINER_NAME /bin/bash
        ;;
    
    clean)
        echo -e "${BLUE}Cleaning up container and image...${NC}"
        check_docker
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        docker rmi $IMAGE_NAME 2>/dev/null || true
        echo -e "${GREEN}✓ Cleanup complete${NC}"
        ;;
    
    status)
        echo -e "${BLUE}Container status:${NC}"
        check_docker
        docker ps -a --filter name=$CONTAINER_NAME --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        ;;
    
    compose-up)
        echo -e "${BLUE}Starting with docker-compose...${NC}"
        check_docker
        check_env
        docker-compose up -d
        echo -e "${GREEN}✓ Services started${NC}"
        echo -e "${BLUE}Access the app at: http://localhost:7860${NC}"
        ;;
    
    compose-down)
        echo -e "${BLUE}Stopping docker-compose services...${NC}"
        check_docker
        docker-compose down
        echo -e "${GREEN}✓ Services stopped${NC}"
        ;;
    
    help)
        print_usage
        ;;
    
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        print_usage
        exit 1
        ;;
esac
