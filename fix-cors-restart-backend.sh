#!/bin/bash

# CORS Fix: Backend Servislerini Restart Et
# CORS ayarları güncellendikten sonra çalıştır

echo "🔧 CORS Düzeltmesi: Backend servisler restart ediliyor..."

cd /path/to/your/rag3_for_local

# Backend servislerini sırayla restart et
echo "🛑 Backend servislerini durduruyor..."
docker-compose stop api-gateway auth-service aprag-service

echo "🗑️ Eski container'ları siliyor..."
docker-compose rm -f api-gateway auth-service aprag-service

echo "🏗️ Backend servislerini yeniden build ediyor..."
docker-compose build --no-cache api-gateway auth-service aprag-service

echo "🚀 Backend servislerini başlatıyor..."
docker-compose up -d auth-service
sleep 10
docker-compose up -d aprag-service  
sleep 10
docker-compose up -d api-gateway

echo "⏳ Servislerin hazır olmasını bekliyor..."
sleep 20

echo "✅ Servis durumlarını kontrol ediyor..."
docker-compose ps api-gateway auth-service aprag-service

echo "🔍 Backend loglarını kontrol ediyor..."
echo "--- API GATEWAY LOGS ---"
docker-compose logs --tail=20 api-gateway | grep -i cors
echo "--- AUTH SERVICE LOGS ---"  
docker-compose logs --tail=20 auth-service | grep -i cors
echo "--- APRAG SERVICE LOGS ---"
docker-compose logs --tail=20 aprag-service | grep -i cors

echo "🎉 Backend CORS düzeltmesi tamamlandı!"
echo "🌐 Test URL: http://46.62.254.131:3000"
echo "📋 Tüm logları görmek için: docker-compose logs -f api-gateway auth-service aprag-service"