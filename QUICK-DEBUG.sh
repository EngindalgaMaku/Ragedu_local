#!/bin/bash

echo "🔍 HIZLI DEBUG - Auth Service durumu"
echo "=================================="

echo "📊 Container durumu:"
docker-compose ps auth-service

echo
echo "📝 Auth service logları:"
docker-compose logs --tail=10 auth-service

echo
echo "🧪 Service test:"
curl -v "http://46.62.254.131:8006/health" 2>&1 | head -20

echo
echo "🧪 OPTIONS test:"
curl -v -X OPTIONS "http://46.62.254.131:8006/admin/users/2/password" \
  -H "Origin: http://46.62.254.131:3000" 2>&1 | head -15