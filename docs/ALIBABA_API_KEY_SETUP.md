# Alibaba DashScope API Anahtarı Kurulumu

## DashScope Nedir?

**DashScope**, Alibaba Cloud'un AI servis platformudur. Qwen modelleri (qwen-plus, qwen-turbo, qwen-max vb.) DashScope üzerinden çalışır.

## API Anahtarı Nasıl Alınır?

1. **Alibaba Cloud Console**'a giriş yapın: https://dashscope.console.aliyun.com/
2. **API Keys** bölümüne gidin
3. Yeni bir API anahtarı oluşturun veya mevcut anahtarı kopyalayın

## API Anahtarını Ayarlama

### Windows PowerShell (Docker için)

```powershell
# Geçici olarak (sadece bu terminal için)
$env:ALIBABA_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxx"

# Kalıcı olarak (sistem genelinde)
[System.Environment]::SetEnvironmentVariable('ALIBABA_API_KEY', 'sk-xxxxxxxxxxxxxxxxxxxxx', 'User')
```

### Linux/Mac (Docker için)

```bash
# Geçici olarak (sadece bu terminal için)
export ALIBABA_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxx"

# Kalıcı olarak (.bashrc veya .zshrc dosyasına ekleyin)
echo 'export ALIBABA_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

### .env Dosyası Oluşturma (Önerilen)

Proje kök dizininde `.env` dosyası oluşturun:

```env
ALIBABA_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
```

Docker Compose otomatik olarak `.env` dosyasını okur.

## Kontrol Etme

API anahtarının doğru ayarlandığını kontrol etmek için:

```bash
# Windows PowerShell
echo $env:ALIBABA_API_KEY

# Linux/Mac
echo $ALIBABA_API_KEY
```

## Servisleri Yeniden Başlatma

API anahtarını ayarladıktan sonra servisleri yeniden başlatın:

```bash
docker-compose restart model-inference-service aprag-service reranker-service
```

Veya tüm servisleri:

```bash
docker-compose restart
```

## Not

- Kod hem `ALIBABA_API_KEY` hem de `DASHSCOPE_API_KEY` environment variable'larını destekler
- İkisi de aynı API anahtarını kullanır
- `ALIBABA_API_KEY` önceliklidir, yoksa `DASHSCOPE_API_KEY` kullanılır


