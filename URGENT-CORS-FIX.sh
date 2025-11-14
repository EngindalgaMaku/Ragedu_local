#!/bin/bash

echo "🚨 URGENT CORS FIX - Backend Services Restart"
echo "=============================================="

cd /path/to/your/rag3_for_local

echo "1. Stopping backend services..."
docker-compose stop api-gateway auth-service aprag-service

echo "2. Removing old containers to force environment reload..."
docker-compose rm -f api-gateway auth-service aprag-service

echo "3. Rebuilding with new CORS configuration..."
docker-compose build --no-cache api-gateway auth-service aprag-service

echo "4. Starting services with new CORS origins..."
docker-compose up -d auth-service
sleep 5
docker-compose up -d aprag-service  
sleep 5
docker-compose up -d api-gateway
sleep 10

echo "5. Checking service status..."
docker-compose ps api-gateway auth-service aprag-service

echo "6. Checking CORS environment variables..."
echo "--- API GATEWAY CORS ---"
docker-compose exec api-gateway printenv CORS_ORIGINS
echo "--- AUTH SERVICE CORS ---"  
docker-compose exec auth-service printenv CORS_ORIGINS
echo "--- APRAG SERVICE CORS ---"
docker-compose exec aprag-service printenv CORS_ORIGINS

echo "7. Testing API Gateway CORS headers..."
curl -H "Origin: http://46.62.254.131:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS \
     http://46.62.254.131:8000/health -v

echo ""
echo "🎯 CORS Fix Complete!"
echo "Test URL: http://46.62.254.131:3000"
echo "Expected: No more CORS errors in browser console"