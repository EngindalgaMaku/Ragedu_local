#!/bin/bash

# Frontend Docker Build ve Restart Komutları
# Server'da çalıştırılacak komutlar

echo "🏗️ Frontend Docker Build ve Restart İşlemi Başlatılıyor..."

# Mevcut dizine git
cd /path/to/your/rag3_for_local

# Mevcut frontend container'ı durdur ve sil
echo "📦 Mevcut frontend container'ı durduruyor..."
docker-compose stop frontend
docker-compose rm -f frontend

# Frontend image'ını yeniden build et
echo "🔨 Frontend image'ını build ediyor..."
docker-compose build --no-cache frontend

# Tüm servisleri restart et (dependency sıralaması için)
echo "🚀 Frontend container'ı başlatıyor..."
docker-compose up -d frontend

# Container durumunu kontrol et
echo "✅ Container durumu kontrol ediliyor..."
docker-compose ps frontend

# Frontend loglarını göster
echo "📋 Frontend logları (son 50 satır):"
docker-compose logs --tail=50 frontend

echo "🎉 Frontend deployment tamamlandı!"
echo "🌐 Frontend URL: http://46.62.254.131:3000"
echo "🔧 Logları izlemek için: docker-compose logs -f frontend"