#!/bin/bash
set -e

echo "=== Insurance RAG System - Kubernetes Deployment ==="

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl not found. Please install kubectl."
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Cannot connect to Kubernetes cluster."
    exit 1
fi

# Install Strimzi Kafka operator
echo "Installing Strimzi Kafka operator..."
kubectl create namespace kafka --dry-run=client -o yaml | kubectl apply -f -
kubectl create -f 'https://strimzi.io/install/latest?namespace=kafka' -n kafka || echo "Strimzi already installed"

# Wait for Strimzi
echo "Waiting for Strimzi operator..."
kubectl wait --for=condition=ready pod -l name=strimzi-cluster-operator -n kafka --timeout=300s

# Create namespace
echo "Creating insurance-rag namespace..."
kubectl apply -f k8s/namespace.yaml

# Apply configurations
echo "Applying configurations..."
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml

# Deploy infrastructure
echo "Deploying infrastructure..."
kubectl apply -f k8s/minio-deployment.yaml
kubectl apply -f k8s/kafka-cluster.yaml

# Wait for infrastructure
echo "Waiting for infrastructure to be ready..."
kubectl wait --for=condition=ready pod -l app=minio -n insurance-rag --timeout=300s
kubectl wait --for=condition=ready kafka/kafka-cluster -n insurance-rag --timeout=600s

# Build and push Docker images (assumes you have a registry)
echo "Building Docker images..."
docker build -t insurance-rag/ingestor:latest -f docker/ingestor.Dockerfile .
docker build -t insurance-rag/enricher:latest -f docker/enricher.Dockerfile .
docker build -t insurance-rag/api:latest -f docker/api.Dockerfile .
docker build -t insurance-rag/spark-base:latest -f docker/spark-base.Dockerfile .

# For minikube, load images
if command -v minikube &> /dev/null; then
    echo "Loading images into minikube..."
    minikube image load insurance-rag/ingestor:latest
    minikube image load insurance-rag/enricher:latest
    minikube image load insurance-rag/api:latest
    minikube image load insurance-rag/spark-base:latest
fi

# Deploy workers
echo "Deploying workers..."
kubectl apply -f k8s/ingestor-deployment.yaml
kubectl apply -f k8s/enricher-deployment.yaml

# Wait for workers
echo "Waiting for workers..."
kubectl wait --for=condition=ready pod -l app=ingestor -n insurance-rag --timeout=300s
kubectl wait --for=condition=ready pod -l app=enricher -n insurance-rag --timeout=300s

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo "  1. Upload documents to MinIO:"
echo "     kubectl port-forward -n insurance-rag svc/minio-service 9001:9001"
echo "     Open http://localhost:9001"
echo ""
echo "  2. Run Spark embedding job:"
echo "     kubectl apply -f k8s/embedder-spark-job.yaml"
echo ""
echo "  3. Build FAISS index:"
echo "     kubectl apply -f k8s/indexer-job.yaml"
echo ""
echo "  4. Deploy API:"
echo "     kubectl apply -f k8s/api-deployment.yaml"
echo ""
echo "  5. Access API:"
echo "     kubectl port-forward -n insurance-rag svc/api-service 8000:8000"
echo "     Open http://localhost:8000"
echo ""
echo "Monitor pods: kubectl get pods -n insurance-rag -w"
echo "View logs: kubectl logs -n insurance-rag <pod-name>"