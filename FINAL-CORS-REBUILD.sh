#!/bin/bash

echo "🚀 FINAL CORS REBUILD - API Gateway"
echo "===================================="

echo ""
echo "⏹️  Stopping API Gateway..."
docker-compose stop api-gateway

echo "🗑️  Removing container..."
docker-compose rm -f api-gateway

echo "🔨 Force rebuilding with simplified CORS (no cache)..."
docker-compose build --no-cache api-gateway

echo "🚀 Starting API Gateway..."
docker-compose up -d api-gateway

echo "⏳ Waiting for startup..."
sleep 20

echo ""
echo "🧪 Testing CORS configuration..."
echo "================================"

echo ""
echo "1. Testing preflight OPTIONS request:"
curl -X OPTIONS \
     -H "Origin: http://46.62.254.131:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: authorization,content-type" \
     -i "http://46.62.254.131:8000/documents/list-markdown"

echo ""
echo ""
echo "2. Testing actual GET request:"
curl -H "Origin: http://46.62.254.131:3000" \
     -i "http://46.62.254.131:8000/documents/list-markdown"

echo ""
echo ""
echo "✅ CORS rebuild completed!"
echo "Check above output for 'Access-Control-Allow-Origin: *' headers"
echo "Browser should now work without CORS errors!"