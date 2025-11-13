#!/bin/bash
echo "🔄 Stopping and rebuilding services with similarity fix..."

# Stop all services
docker-compose down

# Remove old images to force rebuild
docker-compose rm -f
docker system prune -f

# Rebuild and start services
echo "🚀 Rebuilding document processing service..."
docker-compose build document-processing-service

echo "🚀 Starting all services..."
docker-compose up -d

# Wait a bit for services to start
sleep 10

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

echo "✅ Services restarted with similarity fix!"
echo "📌 Now you can test your RAG system - similarity scores should show properly."