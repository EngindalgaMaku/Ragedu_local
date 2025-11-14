#!/bin/bash

echo "🔐 Credentials-Compatible CORS Fix"
echo "=================================="

echo "⏹️  Stopping API Gateway..."
docker-compose stop api-gateway

echo "🔨 Quick rebuild (no cache)..."
docker-compose build --no-cache api-gateway

echo "🚀 Starting API Gateway..."
docker-compose up -d api-gateway

echo "⏳ Waiting..."
sleep 15

echo ""
echo "🧪 Testing credentials-compatible CORS..."
echo "Testing /sessions endpoint with credentials:"
curl -H "Origin: http://46.62.254.131:3000" \
     -H "Authorization: Bearer test-token" \
     -i "http://46.62.254.131:8000/sessions"

echo ""
echo "✅ Credentials CORS fix completed!"
echo "Browser should now work with /sessions endpoint!"