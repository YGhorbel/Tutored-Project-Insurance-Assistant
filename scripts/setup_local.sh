#!/bin/bash
set -e

echo "=== Insurance RAG System - Local Setup ==="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not found. Please install Docker."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose not found. Please install Docker Compose."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

# Create data directory
mkdir -p data

# Build Docker images
echo "Building Docker images..."
docker-compose build

# Start services
echo "Starting services..."
docker-compose up -d zookeeper kafka minio minio-init

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check Kafka
echo "Checking Kafka..."
docker-compose exec -T kafka kafka-topics --bootstrap-server localhost:9092 --list || {
    echo "Kafka not ready, waiting more..."
    sleep 30
}

# Start remaining services
echo "Starting worker services..."
docker-compose up -d spark-master spark-worker ingestor enricher indexer api

echo ""
echo "=== Setup Complete ==="
echo "Services running:"
echo "  - Kafka UI: http://localhost:8080"
echo "  - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
echo "  - Spark Master: http://localhost:8080"
echo "  - API: http://localhost:8000"
echo "  - Gradio UI: http://localhost:7860"
echo ""
echo "Next steps:"
echo "  1. Upload PDFs to MinIO bucket 'insurance-docs'"
echo "  2. Trigger ingestion: docker-compose exec ingestor python /app/workers/ingestor.py"
echo "  3. Wait for processing (check logs)"
echo "  4. Build index: docker-compose exec indexer python /app/workers/indexer.py"
echo "  5. Query API: curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"question\":\"test\"}'"
echo ""
echo "To view logs: docker-compose logs -f [service-name]"
echo "To stop: docker-compose down"
