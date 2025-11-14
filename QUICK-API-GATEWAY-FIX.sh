#!/bin/bash

echo "🚀 Quick API Gateway CORS Fix"
echo "================================"

# Stop and rebuild ONLY api-gateway service
echo "⏹️  Stopping API Gateway..."
docker-compose stop api-gateway

echo "🗑️  Removing API Gateway container..."
docker-compose rm -f api-gateway

echo "🔨 Force rebuilding API Gateway (no cache)..."
docker-compose build --no-cache api-gateway

echo "🚀 Starting API Gateway..."
docker-compose up -d api-gateway

echo "⏳ Waiting for service to start..."
sleep 15

echo "🧪 Testing CORS fix..."
echo ""
echo "Testing /documents/list-markdown endpoint:"
curl -H "Origin: http://46.62.254.131:3000" \
     -v "http://46.62.254.131:8000/documents/list-markdown" 2>&1 | grep -i "access-control-allow-origin"

echo ""
echo "✅ Done! Check browser console now."