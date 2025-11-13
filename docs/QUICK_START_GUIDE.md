# 🚀 RAG3 Eğitim Sistemi - Hızlı Başlangıç Rehberi

## ⚡ 5 Dakikada Başlayın!

### 📋 Gereksinimler
- 🐳 Docker Desktop (çalışır durumda)
- 💾 8GB+ RAM
- 🌐 İnternet bağlantısı

### 🏃‍♂️ Hızlı Kurulum

#### 1. 📥 Projeyi İndirin
```bash
git clone <repository-url>
cd rag3_for_colab
```

#### 2. 🔑 API Key'leri Ayarlayın
`docker-compose.yml` dosyasında şu satırları bulun ve güncelleyin:
```yaml
- GROQ_API_KEY=your_groq_api_key_here
- DOCSTRANGE_API_KEY=5f7583ed-b5d8-11f0-9225-2efa885dd201
```

> 💡 **Groq API Key**: [console.groq.com](https://console.groq.com) adresinden ücretsiz alabilirsiniz.

#### 3. 🚀 Sistemi Başlatın
```bash
# Tüm servisleri build edin ve başlatın
docker-compose up -d

# Durumu kontrol edin (tüm servisler "Up" olmalı)
docker-compose ps
```

#### 4. ✅ Test Edin
- 🌐 **Frontend**: http://localhost:3000
- 🔧 **API Health**: http://localhost:8000/health

### 🎯 İlk Kullanım

#### 1. 🔐 Giriş Yapın
- Frontend'e gidin: http://localhost:3000
- "Giriş Yap" butonuna tıklayın (otomatik giriş)

#### 2. 📚 Ders Oturumu Oluşturun
- "📚 Ders Oturumları" sekmesine gidin
- "➕ Yeni Ders Oturumu" butonuna tıklayın
- Bilgileri doldurun ve oluşturun

#### 3. 📄 Belge Yükleyin
- "📄 Belge Merkezi" sekmesine gidin
- PDF, DOCX, PPTX, XLSX veya MD dosyası yükleyin
- Dönüştürme işlemini bekleyin

#### 4. 🤖 Soru Sorun
- "🎓 Eğitim Asistanı" sekmesine gidin
- Ders oturumunu seçin
- Belgeleriniz hakkında soru sorun!

### 🔧 Sorun Giderme

#### 🚨 Servis Başlamıyor?
```bash
# Logları kontrol edin
docker-compose logs [service-name]

# Yeniden başlatın
docker-compose restart [service-name]
```

#### 🤖 Ollama Modeli Yüklenmiyor?
```bash
# Model inference servisine bağlanın
docker exec -it model-inference-service bash

# Modeli manuel yükleyin
ollama pull llama3:8b
ollama pull nomic-embed-text
```

#### 💾 ChromaDB Bağlantı Hatası?
```bash
# ChromaDB durumunu kontrol edin
curl http://localhost:8004/api/v1/heartbeat

# Servisi yeniden başlatın
docker-compose restart chromadb-service
```

### 📞 Yardım

Sorun yaşıyorsanız:
1. 📖 Ana dokümantasyonu okuyun: `SYSTEM_ARCHITECTURE_AND_DEPLOYMENT_GUIDE.md`
2. 📝 Logları kontrol edin: `docker-compose logs -f`
3. 🔄 Sistemi yeniden başlatın: `docker-compose restart`

---

**🎉 Başarılar! Artık RAG3 Eğitim Sistemi'ni kullanmaya hazırsınız!**
