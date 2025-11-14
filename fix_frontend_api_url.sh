#!/bin/bash

echo "🔍 Frontend API URL Problemi - Diagnose ve Fix"
echo "=============================================="

echo "1. Mevcut container environment'ları kontrol et:"
echo "------------------------------------------------"
echo "Frontend container env variables:"
docker exec rag3-frontend env | grep -E "(NEXT_PUBLIC|API_URL|NODE_ENV)"

echo ""
echo "API Gateway container env variables:"
docker exec api-gateway env | grep -E "(CORS|HOST|PORT)"

echo ""
echo "2. Network bağlantı testi:"
echo "-------------------------"
echo "API Gateway localhost erişimi test:"
curl -s -I http://localhost:8000/health || echo "❌ localhost:8000 erişilemez"

echo ""
echo "API Gateway container internal erişimi test:"
docker exec rag3-frontend curl -s -I http://api-gateway:8000/health || echo "❌ Container içinden api-gateway erişilemez"

echo ""
echo "3. Server IP ile erişim test:"
echo "----------------------------"
SERVER_IP="46.62.254.131"
echo "Server IP ($SERVER_IP) ile erişim test:"
curl -s -I http://$SERVER_IP:8000/health || echo "❌ Server IP ile erişilemez"

echo ""
echo "4. Port listening kontrolü:"
echo "---------------------------"
netstat -tlnp | grep :8000 || echo "Port 8000 listening değil"

echo ""
echo "🔧 ÖNERİLEN ÇÖZÜMLER:"
echo "==================="
echo ""
echo "ÇÖZÜM 1: Frontend environment'ı düzelt"
echo "--------------------------------------"
echo "docker exec -it rag3-frontend sh -c 'echo NEXT_PUBLIC_API_URL=http://46.62.254.131:8000 >> /app/.env.local'"
echo "docker-compose restart frontend"
echo ""
echo "ÇÖZÜM 2: .env dosyasını güncelle ve restart"
echo "------------------------------------------"  
echo "# .env dosyasında şunları kontrol et:"
echo 'NEXT_PUBLIC_API_URL=http://46.62.254.131:8000'
echo 'CORS_ORIGINS=http://46.62.254.131:3000,http://46.62.254.131:8000,http://localhost:3000,http://localhost:8000'
echo ""
echo "# Sonra restart:"
echo "docker-compose restart frontend api-gateway"
echo ""
echo "ÇÖZÜM 3: Network binding kontrolü" 
echo "---------------------------------"
echo "# API Gateway'in 0.0.0.0:8000'de dinlediğini kontrol et"
echo 'docker exec api-gateway netstat -tlnp | grep :8000'
echo ""
echo "ÇÖZÜM 4: Firewall kontrolü"
echo "--------------------------"
echo "# Ubuntu firewall 8000 portunu açık tutmalı"
echo "ufw status"
echo "ufw allow 8000/tcp"