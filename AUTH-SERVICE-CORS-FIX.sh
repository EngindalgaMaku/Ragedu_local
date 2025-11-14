#!/bin/bash

echo "🔧 AUTH SERVICE CORS FIX - Credentials compatible rebuild"
echo "=================================================="
echo

# Stop auth service
echo "⏹️  Stopping auth service..."
docker-compose stop auth-service
docker-compose rm -f auth-service

echo "🔄 Rebuilding auth service with updated CORS..."
docker-compose build --no-cache auth-service

echo "▶️  Starting auth service..."
docker-compose up -d auth-service

echo "⏳ Waiting for auth service to be ready..."
sleep 10

echo "🏥 Health checking auth service..."
curl -s "http://46.62.254.131:8006/health" || echo "Health check failed"

echo
echo "🧪 Testing CORS preflight (OPTIONS request)..."
curl -v -X OPTIONS "http://46.62.254.131:8006/admin/users" \
  -H "Origin: http://46.62.254.131:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type,Authorization" \
  2>&1 | grep -E "(HTTP|Access-Control|Origin)"

echo
echo "🧪 Testing actual password change endpoint..."
curl -v -X OPTIONS "http://46.62.254.131:8006/admin/users/2/password" \
  -H "Origin: http://46.62.254.131:3000" \
  -H "Access-Control-Request-Method: PUT" \
  -H "Access-Control-Request-Headers: Content-Type,Authorization" \
  2>&1 | grep -E "(HTTP|Access-Control|Origin)"

echo
echo "✅ Auth Service CORS fix completed!"
echo "🌐 Try the password change again from frontend"