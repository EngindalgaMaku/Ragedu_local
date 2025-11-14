#!/bin/bash

echo "🔍 Docker Container Durumları Kontrol Ediliyor..."
echo "================================================"

echo "📊 Çalışan Containerlar:"
docker-compose ps

echo ""
echo "📋 Tüm Container Durumları:"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🚨 Son 20 Log Satırı (API Gateway):"
echo "-----------------------------------"
docker-compose logs api-gateway --tail=20

echo ""
echo "🚨 Son 20 Log Satırı (Model Inference):"
echo "--------------------------------------"
docker-compose logs model-inference-service --tail=20

echo ""
echo "🚨 Son 20 Log Satırı (Auth Service):"
echo "------------------------------------"
docker-compose logs auth-service --tail=20

echo ""
echo "🔧 Port Kontrolü:"
echo "-----------------"
echo "8000 portu (API Gateway):"
netstat -an | grep :8000 || echo "Port 8000 boş"

echo "8002 portu (Model Inference):"
netstat -an | grep :8002 || echo "Port 8002 boş"

echo "8006 portu (Auth Service):"
netstat -an | grep :8006 || echo "Port 8006 boş"

echo ""
echo "🚀 Önerilen Çözüm Adımları:"
echo "============================"
echo "1. Tüm servisleri durdur:"
echo "   docker-compose down"
echo ""
echo "2. Tüm servisleri yeniden başlat:"
echo "   docker-compose up -d"
echo ""
echo "3. Eğer hala sorun varsa, imajları yeniden build et:"
echo "   docker-compose down && docker-compose build && docker-compose up -d"