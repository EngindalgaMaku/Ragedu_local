#!/bin/bash

# RAG Education Assistant - Docker Build Script
# This script builds all Docker containers for the RAG system

set -e

echo "🚀 Building RAG Education Assistant Docker Containers..."
echo "================================================"

# Set environment variables
export COMPOSE_PROJECT_NAME=rag-education-assistant
export DOCKER_BUILDKIT=1

# Load environment variables
if [ -f .env ]; then
    echo "📋 Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  Warning: .env file not found, using defaults"
fi

# Function to build individual service
build_service() {
    local service_name=$1
    echo ""
    echo "🔧 Building $service_name..."
    echo "--------------------------------"
    
    docker-compose build --no-cache $service_name
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully built $service_name"
    else
        echo "❌ Failed to build $service_name"
        exit 1
    fi
}

# Function to show build summary
show_summary() {
    echo ""
    echo "📊 Build Summary"
    echo "================================"
    echo "Built images:"
    docker images | grep -E "(rag-|auth-service|frontend|api-gateway|chromadb|docstrange|document-processing|model-inference)"
    echo ""
    echo "🎉 All containers built successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run: ./docker-run.sh           # Start all services"
    echo "2. Run: ./docker-run-dev.sh       # Start in development mode"
    echo "3. Run: docker-compose logs -f    # View logs"
    echo ""
}

# Build services in dependency order
echo "🏗️  Building services in dependency order..."

# 1. Build database-independent services first
build_service "chromadb-service"
build_service "docstrange-service"

# 2. Build auth service (needs database)
build_service "auth-service"

# 3. Build processing services
build_service "model-inference-service"
build_service "document-processing-service"

# 4. Build API gateway (depends on other services)
build_service "api-gateway"

# 5. Build frontend (depends on API gateway and auth service)
build_service "frontend"

# Show build summary
show_summary

echo "🏁 Build process completed successfully!"
echo "Total build time: $SECONDS seconds"