# Development Mode - Hot Reload Kullanım Kılavuzu

Bu dosya, development modunda hot reload ile çalışmak için hazırlanmıştır. Kod değişiklikleriniz anında yansır, build gerekmez!

## 🚀 Hızlı Başlangıç

### ⚡ En Kolay Yol: Cursor Task'ları (Önerilen!)

Cursor'da `Ctrl+Shift+P` (veya `Cmd+Shift+P` Mac'te) → "Tasks: Run Task" → **"🚀 Dev: Development Mode Başlat"**

Veya:
- `Ctrl+Shift+B` → Task seç
- Command Palette'de "task" yaz → Task seç

**Mevcut Task'lar:**
- 🚀 **Dev: Development Mode Başlat** - Hot reload ile başlat
- 🚀 **Dev: Development Mode Başlat (Arka Plan)** - Arka planda başlat
- 🛑 **Dev: Development Mode Durdur** - Durdur
- 📋 **Dev: Logları Göster** - Tüm loglar
- 📋 **Dev: Document Processing Service Logları** - Sadece DPS logları
- 📋 **Dev: Frontend Logları** - Sadece frontend logları
- 🔄 **Dev: Restart Document Processing Service** - DPS'i yeniden başlat
- 🔄 **Dev: Restart Frontend** - Frontend'i yeniden başlat
- 🏭 **Production: Normal Mode Başlat** - Production mode
- 🏭 **Production: Build ve Başlat** - Production build

### ⚡ Script'ler ile (Alternatif)

Windows PowerShell'de:

```powershell
# Development mode'da başlat (hot reload aktif)
.\dev.ps1

# Arka planda başlat
.\dev-up.ps1

# Durdur
.\dev-down.ps1

# Logları göster
.\dev-logs.ps1

# Belirli bir servisin loglarını göster
.\dev-logs.ps1 document-processing-service
```

Windows CMD'de:

```cmd
# Development mode'da başlat
dev.bat
```

### 🔧 Manuel Yol

```bash
# Tüm servisleri development mode'da başlat
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Sadece belirli servisleri (örnek: document-processing-service ve frontend)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up document-processing-service frontend

# Arka planda çalıştır
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Production Mode (Normal)

```bash
# Production mode (build gerekir)
docker-compose up --build
```

## 📋 Hot Reload Özellikleri

### ✅ Hot Reload Destekleyen Servisler

1. **API Gateway** (`api-gateway`)
   - Python kod değişiklikleri anında yansır
   - `src/api/` ve `src/database/` mount edilmiş

2. **Document Processing Service** (`document-processing-service`)
   - Python kod değişiklikleri anında yansır
   - `services/document_processing_service/` ve `src/` mount edilmiş

3. **Auth Service** (`auth-service`)
   - Python kod değişiklikleri anında yansır
   - `services/auth_service/` mount edilmiş

4. **APRAG Service** (`aprag-service`)
   - Python kod değişiklikleri anında yansır
   - `services/aprag_service/` mount edilmiş

5. **Model Inference Service** (`model-inference-service`)
   - Python kod değişiklikleri anında yansır
   - `services/model_inference_service/` mount edilmiş

6. **Evaluation Service** (`evaluation-service`)
   - Python kod değişiklikleri anında yansır
   - `services/evaluation_service/` mount edilmiş

7. **Frontend** (`frontend`)
   - Next.js hot reload aktif
   - `frontend/` mount edilmiş
   - `node_modules` ve `.next` cache container'da kalır (performans için)

### ❌ Hot Reload Desteklemeyen Servisler

Bu servisler harici image'lar veya özel yapılandırmalar gerektirir:

- `chromadb-service` - Harici image
- `marker-api` - Harici image
- `docstrange-service` - Harici image
- `reranker-service` - Model cache gerektirir

## 🔧 Nasıl Çalışır?

### Python Servisleri

`docker-compose.dev.yml` dosyası, Python servislerine `--reload` flag'i ekler:

```yaml
command: python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Bu sayede:
- Kod değişikliklerinde uvicorn otomatik restart yapar
- Build gerekmez
- Değişiklikler anında yansır

### Frontend

Frontend için Next.js development server kullanılır:

```yaml
command: npm run dev
```

Bu sayede:
- React/Next.js hot reload aktif
- CSS değişiklikleri anında yansır
- Build gerekmez

## 📝 Örnek Kullanım Senaryoları

### Senaryo 1: Sadece Backend Geliştirme

```bash
# Sadece backend servisleri
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up \
  api-gateway \
  document-processing-service \
  auth-service \
  aprag-service
```

### Senaryo 2: Sadece Frontend Geliştirme

```bash
# Sadece frontend (backend'ler production'da çalışıyor)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up frontend
```

### Senaryo 3: Belirli Bir Servis

```bash
# Sadece document-processing-service
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up document-processing-service
```

## ⚠️ Önemli Notlar

1. **İlk Çalıştırmada Build Gerekebilir**
   - Development Dockerfile'lar ilk kez build edilmeli
   - Sonraki çalıştırmalarda build gerekmez

2. **Volume Mount Performansı**
   - Windows'ta volume mount performansı düşük olabilir
   - WSL2 kullanıyorsanız daha iyi performans alırsınız

3. **node_modules ve .next Cache**
   - Frontend'de `node_modules` ve `.next` container'da kalır
   - Bu sayede npm install her seferinde çalışmaz
   - Eğer dependency değiştiyse container'ı rebuild edin

4. **Database ve Session Data**
   - Database ve session data volume'ları korunur
   - Development ve production aynı data'yı kullanır

## 🐛 Sorun Giderme

### Hot Reload Çalışmıyor

1. Container'ın restart olduğunu kontrol edin:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f document-processing-service
   ```

2. Volume mount'un çalıştığını kontrol edin:
   ```bash
   docker exec document-processing-service ls -la /app
   ```

3. Uvicorn'un `--reload` flag'i ile çalıştığını kontrol edin:
   ```bash
   docker exec document-processing-service ps aux | grep uvicorn
   ```

### Frontend Hot Reload Çalışmıyor

1. Next.js dev server'ın çalıştığını kontrol edin:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f frontend
   ```

2. Port 3000'in açık olduğunu kontrol edin:
   ```bash
   curl http://localhost:3000
   ```

## 📚 Daha Fazla Bilgi

- `docker-compose.yml` - Production konfigürasyonu
- `docker-compose.dev.yml` - Development override'ları
- `frontend/Dockerfile.dev` - Frontend development Dockerfile

