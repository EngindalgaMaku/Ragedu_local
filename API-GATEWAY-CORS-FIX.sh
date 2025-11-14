#!/bin/bash

# API Gateway CORS Fix Script
# Bu script sadece API Gateway container'ını yeniden oluşturur

echo "🔧 API Gateway CORS Fix Script"
echo "========================================="

# Stop only API Gateway service
echo "⏹️  Stopping API Gateway service..."
docker-compose stop api-gateway

# Remove API Gateway container to force rebuild
echo "🗑️  Removing API Gateway container..."
docker-compose rm -f api-gateway

# Rebuild and start API Gateway with no-cache
echo "🔨 Rebuilding API Gateway with updated CORS configuration..."
docker-compose build --no-cache api-gateway

# Start API Gateway
echo "🚀 Starting API Gateway..."
docker-compose up -d api-gateway

# Wait for service to be ready
echo "⏳ Waiting for API Gateway to be ready..."
sleep 10

# Test CORS fix
echo "🧪 Testing CORS fix..."
echo "Testing preflight request to /health endpoint..."

curl -X OPTIONS -H "Origin: http://46.62.254.131:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: authorization,content-type" \
     -v http://46.62.254.131:8000/health

echo ""
echo "Testing actual GET request to /health endpoint..."

curl -H "Origin: http://46.62.254.131:3000" \
     -v http://46.62.254.131:8000/health

echo ""
echo "✅ API Gateway CORS fix completed!"
echo "🌐 Frontend should now be able to access http://46.62.254.131:8000/health"
echo ""
echo "If CORS errors persist, check the browser console for specific error messages."