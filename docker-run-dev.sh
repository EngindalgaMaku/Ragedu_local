#!/bin/bash

# RAG Education Assistant - Development Mode Docker Run Script
# This script starts all Docker containers with development configurations

set -e

echo "🚀 Starting RAG Education Assistant (Development Mode)..."
echo "======================================================="

# Set environment variables for development
export COMPOSE_PROJECT_NAME=rag-education-assistant-dev
export NODE_ENV=development
export DEBUG=true

# Load environment variables
if [ -f .env ]; then
    echo "📋 Loading environment variables from .env"
    source .env
else
    echo "⚠️  Warning: .env file not found, using defaults"
fi

# Override some settings for development
export NEXT_PUBLIC_DEMO_MODE=true
export DEBUG=true
export CORS_ORIGINS="http://localhost:3000,http://localhost:8000,http://host.docker.internal:3000,http://frontend:3000,http://api-gateway:8080"

echo "🔧 Development mode settings:"
echo "  - Debug mode: enabled"
echo "  - Demo mode: enabled"
echo "  - CORS: relaxed for development"
echo "  - Hot reload: enabled where possible"

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

# Function to start services in development mode
start_services_dev() {
    echo ""
    echo "🏁 Starting all services in development mode..."
    echo "-----------------------------------------------"
    
    # Start with build for development
    docker-compose up -d --build
    
    if [ $? -eq 0 ]; then
        echo "✅ All services started successfully in development mode"
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
    
    # Wait a bit longer for development builds
    echo "⏳ Waiting for services to start (this may take a while for first build)..."
    sleep 15
    
    # Check each service
    services=("auth-service" "chromadb-service" "docstrange-service" "document-processing-service" "model-inference-service" "api-gateway" "rag3-frontend")
    
    for service in "${services[@]}"; do
        if docker-compose ps $service | grep -q "Up"; then
            echo "✅ $service: Running"
        else
            echo "❌ $service: Not running"
            echo "   Check logs: docker-compose logs $service"
        fi
    done
}

# Function to show development URLs and credentials
show_dev_info() {
    echo ""
    echo "🌐 Development Service URLs"
    echo "============================"
    echo "Frontend (React Dev):       http://localhost:3000"
    echo "  - Hot reload enabled"
    echo "  - React DevTools compatible"
    echo ""
    echo "API Gateway:                http://localhost:8000"
    echo "  - API Documentation:      http://localhost:8000/docs"
    echo "  - Debug endpoints enabled"
    echo ""
    echo "Auth Service:               http://localhost:8006"
    echo "  - API Docs:               http://localhost:8006/docs"
    echo "  - Health Check:           http://localhost:8006/health"
    echo "  - Service Info:           http://localhost:8006/info"
    echo ""
    echo "Other Services:"
    echo "  - ChromaDB:               http://localhost:8004"
    echo "  - DocStrange Service:     http://localhost:8005"
    echo "  - Document Processing:    http://localhost:8003"
    echo "  - Model Inference:        http://localhost:8002"
    echo ""
    echo "🔑 Development Login Credentials:"
    echo "================================="
    echo "Admin User:"
    echo "  Username: admin"
    echo "  Password: admin123"
    echo ""
    echo "Teacher Demo:"
    echo "  Username: teacher_demo"
    echo "  Password: admin123"
    echo ""
    echo "Student Demo:"
    echo "  Username: student_demo"
    echo "  Password: admin123"
    echo ""
    echo "⚠️  Note: Change passwords in production!"
    echo ""
}

# Function to show development logs with filtering
show_dev_logs() {
    echo "📋 Development Logs (filtered for readability)"
    echo "=============================================="
    echo "Press Ctrl+C to stop log viewing"
    echo ""
    
    # Show logs with some filtering for development
    docker-compose logs -f --tail=50
}

# Function to run development tests
run_tests() {
    echo "🧪 Running development tests..."
    echo "==============================="
    
    # Test auth service
    echo "Testing Auth Service..."
    curl -s http://localhost:8006/health | jq . || echo "Auth service not responding"
    
    # Test API gateway
    echo "Testing API Gateway..."
    curl -s http://localhost:8000/health | jq . || echo "API gateway not responding"
    
    # Test frontend
    echo "Testing Frontend..."
    curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 || echo "Frontend not responding"
    
    echo "✅ Basic connectivity tests completed"
}

# Main execution
main() {
    # Check prerequisites
    check_docker
    
    # Cleanup existing containers
    cleanup_containers
    
    # Start services in development mode
    start_services_dev
    
    # Check health
    check_health
    
    # Show development info
    show_dev_info
    
    # Ask what to do next
    echo "What would you like to do?"
    echo "1) View real-time logs"
    echo "2) Run basic tests"
    echo "3) Exit (services keep running)"
    echo ""
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            show_dev_logs
            ;;
        2)
            run_tests
            echo ""
            echo "Tests completed. Services are still running."
            ;;
        3)
            echo ""
            echo "✅ RAG Education Assistant is running in development mode!"
            ;;
        *)
            echo "Invalid choice. Services are running."
            ;;
    esac
    
    echo ""
    echo "Useful development commands:"
    echo "  docker-compose logs -f <service>     # View specific service logs"
    echo "  docker-compose ps                    # Check status"
    echo "  docker-compose down                  # Stop all services"
    echo "  docker-compose restart <service>     # Restart specific service"
    echo "  docker-compose exec <service> bash   # Enter container shell"
    echo ""
}

# Handle script arguments
case "${1:-}" in
    --logs|-l)
        docker-compose logs -f --tail=100
        ;;
    --test|-t)
        run_tests
        ;;
    --status|-s)
        docker-compose ps
        ;;
    --stop)
        echo "🛑 Stopping all development services..."
        docker-compose down
        ;;
    --help|-h)
        echo "RAG Education Assistant Development Runner"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --logs, -l      Show real-time logs"
        echo "  --test, -t      Run development tests"
        echo "  --status, -s    Show service status"
        echo "  --stop          Stop all services"
        echo "  --help, -h      Show this help"
        echo ""
        ;;
    *)
        main
        ;;
esac