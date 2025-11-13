#!/bin/bash

# RAG Education Assistant - Docker Run Script
# This script starts all Docker containers for the RAG system

set -e

echo "🚀 Starting RAG Education Assistant..."
echo "===================================="

# Set environment variables
export COMPOSE_PROJECT_NAME=rag-education-assistant

# Load environment variables
if [ -f .env ]; then
    echo "📋 Loading environment variables from .env"
    source .env
else
    echo "⚠️  Warning: .env file not found, using defaults"
fi

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker first."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to stop existing containers
cleanup_containers() {
    echo ""
    echo "🧹 Cleaning up existing containers..."
    docker-compose down --remove-orphans || true
    echo "✅ Cleanup completed"
}

# Function to start services
start_services() {
    echo ""
    echo "🏁 Starting all services..."
    echo "----------------------------"
    
    # Start all services
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        echo "✅ All services started successfully"
    else
        echo "❌ Failed to start services"
        exit 1
    fi
}

# Function to check service health
check_health() {
    echo ""
    echo "🏥 Checking service health..."
    echo "-----------------------------"
    
    # Wait a moment for containers to start
    sleep 10
    
    # Check each service
    services=("auth-service" "chromadb-service" "docstrange-service" "document-processing-service" "model-inference-service" "api-gateway" "rag3-frontend")
    
    for service in "${services[@]}"; do
        if docker-compose ps $service | grep -q "Up"; then
            echo "✅ $service: Running"
        else
            echo "❌ $service: Not running"
        fi
    done
}

# Function to show service URLs
show_urls() {
    echo ""
    echo "🌐 Service URLs"
    echo "==============="
    echo "Frontend (React):           http://localhost:3000"
    echo "API Gateway:                http://localhost:8000"
    echo "Auth Service:               http://localhost:8006"
    echo "  - API Docs:               http://localhost:8006/docs"
    echo "  - Health Check:           http://localhost:8006/health"
    echo "ChromaDB:                   http://localhost:8004"
    echo "DocStrange Service:         http://localhost:8005"
    echo "Document Processing:        http://localhost:8003"
    echo "Model Inference:            http://localhost:8002"
    echo ""
    echo "🔑 Default Login Credentials:"
    echo "Username: admin"
    echo "Password: admin123"
    echo "(Please change the password after first login)"
    echo ""
}

# Function to show logs
show_logs() {
    echo "📋 Showing real-time logs (Ctrl+C to stop)..."
    echo "=============================================="
    docker-compose logs -f
}

# Main execution
main() {
    # Check prerequisites
    check_docker
    
    # Cleanup existing containers
    cleanup_containers
    
    # Start services
    start_services
    
    # Check health
    check_health
    
    # Show URLs
    show_urls
    
    # Ask if user wants to see logs
    echo "Would you like to see real-time logs? (y/N): "
    read -n 1 response
    echo
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        show_logs
    else
        echo ""
        echo "✅ RAG Education Assistant is running!"
        echo ""
        echo "Useful commands:"
        echo "  docker-compose logs -f              # View logs"
        echo "  docker-compose ps                   # Check status"
        echo "  docker-compose down                 # Stop all services"
        echo "  docker-compose restart <service>    # Restart specific service"
        echo ""
    fi
}

# Handle script arguments
case "${1:-}" in
    --logs|-l)
        docker-compose logs -f
        ;;
    --status|-s)
        docker-compose ps
        ;;
    --stop)
        echo "🛑 Stopping all services..."
        docker-compose down
        ;;
    --restart|-r)
        echo "🔄 Restarting all services..."
        docker-compose restart
        ;;
    --help|-h)
        echo "RAG Education Assistant Docker Runner"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --logs, -l      Show real-time logs"
        echo "  --status, -s    Show service status"
        echo "  --stop          Stop all services"
        echo "  --restart, -r   Restart all services"
        echo "  --help, -h      Show this help"
        echo ""
        ;;
    *)
        main
        ;;
esac