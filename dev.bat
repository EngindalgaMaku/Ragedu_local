@echo off
REM Development Mode - Hot Reload ile başlat
echo 🚀 Development mode başlatılıyor (Hot Reload aktif)...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up



