#!/bin/bash

echo "🔍 CORS Headers Debug"
echo "===================="

echo ""
echo "1️⃣ Testing API Gateway CORS headers:"
curl -H "Origin: http://46.62.254.131:3000" \
     -v "http://46.62.254.131:8000/documents/list-markdown" 2>&1 | head -20

echo ""
echo "2️⃣ Testing with preflight OPTIONS request:"
curl -X OPTIONS \
     -H "Origin: http://46.62.254.131:3000" \
     -H "Access-Control-Request-Method: GET" \
     -v "http://46.62.254.131:8000/documents/list-markdown" 2>&1 | head -20

echo ""
echo "3️⃣ Checking container status:"
docker-compose ps api-gateway

echo ""
echo "4️⃣ Checking API Gateway logs for CORS info:"
docker-compose logs --tail=50 api-gateway | grep -i cors

echo ""
echo "5️⃣ Quick rebuild check:"
echo "Last API Gateway build time:"
docker images | grep api-gateway