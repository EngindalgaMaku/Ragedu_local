#!/bin/bash

echo "🔧 PATCH CORS FIX - Şifre değiştirme sorunu"
echo "========================================"

# Quick rebuild
echo "⏹️  Stopping auth service..."
docker-compose stop auth-service

echo "🔄 Rebuilding with PATCH method support..."
docker-compose build --no-cache auth-service
docker-compose up -d auth-service

echo "⏳ Waiting 8 seconds..."
sleep 8

echo "🧪 Testing PATCH CORS preflight:"
curl -v -X OPTIONS "http://46.62.254.131:8006/admin/users/2/password" \
  -H "Origin: http://46.62.254.131:3000" \
  -H "Access-Control-Request-Method: PATCH" \
  -H "Access-Control-Request-Headers: Content-Type,Authorization" \
  2>&1 | grep -E "(HTTP|Access-Control)"

echo
echo "✅ PATCH CORS fix applied!"
echo "🧪 Test şifre değiştirmeyi şimdi"