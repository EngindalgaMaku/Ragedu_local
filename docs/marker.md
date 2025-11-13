# Marker API - Kullanım Notları

## Kullanılan Docker İmajı
- İmaj: `wirawan/marker-api:latest`
- Kaynak: Docker Hub – [wirawan/marker-api](https://hub.docker.com/r/wirawan/marker-api)

## docker-compose Servis Tanımı
Aşağıdaki servis `rag3_for_colab/docker-compose.yml` içinde tanımlıdır:

```yaml
marker-api:
  image: wirawan/marker-api:latest
  container_name: marker-api
  ports:
    - "8090:8080"
  environment:
    - OMP_NUM_THREADS=1
    - TOKENIZERS_PARALLELISM=false
    - NVIDIA_VISIBLE_DEVICES=all
    - NVIDIA_DRIVER_CAPABILITIES=compute,utility
  mem_limit: 12g
  cpus: "4"
  gpus: all
  restart: unless-stopped
  networks:
    - rag-network
```

Notlar:
- GPU kullanımı için Docker Desktop’ta GPU entegrasyonu ve host’ta NVIDIA driver olması gerekir (`nvidia-smi` ile doğrulanabilir).
- CPU ortamında da çalışır ama büyük PDF’lerde yavaş ve RAM yoğun olabilir.

## HTTP Uç Noktaları (Marker)
- Sağlık kontrolü: `GET /health` → 200 OK
- Dönüştürme: `POST /convert`
  - Form alanı: `pdf_file` (dosya)
  - Yanıt: JSON (genellikle `markdown` veya `text` alanında çıktı)

Örnek `curl`:
```bash
curl -F "pdf_file=@example.pdf" http://localhost:8090/convert
```

## API Gateway Entegrasyonu (Özet)
- Servis adresi: `MARKER_API_URL=http://marker-api:8080`
- Readiness kontrolü: `GET /health` ve `/` ile 10 sn’lik bekleme + 3 deneme backoff
- Boş içerik kontrolü: Kaydetmeden önce içerik `trim()` ile doğrulanır; boş ise 502 döner ve `.md` oluşturulmaz
- Sayfa parçalama (büyük PDF’ler): varsayılan olarak 3 sayfalık chunk’lar
  - Chunk başarısız olursa tek sayfaya kadar düşürüp yeniden dener
  - Parça çıktıları `\n\n` ile birleştirilir
- Timeout: 15 dk (`/convert` çağrısı)

## Gateway Üzerinden Kullanım Akışı
1. PDF dosyası yüklenir.
2. Büyük/karmaşık PDF’te gateway PDF’i 3 sayfalık bölümlere ayırır, her parçayı `POST /convert` ile işler.
3. Parça boş/başarısız ise tek sayfa moduna geçer, yine başarısızsa hataya düşer (ve boş içerik kaydedilmez).
4. Başarılı parça çıktıları birleştirilir ve `data/markdown/` altına kaydedilir.

## Manuel Çalıştırma (docker run)
```bash
# CPU (hızlı deneme)
docker run --rm -p 8090:8080 wirawan/marker-api:latest

# GPU (NVIDIA Container Toolkit gerekir)
docker run --rm -p 8090:8080 \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e OMP_NUM_THREADS=1 \
  -e TOKENIZERS_PARALLELISM=false \
  wirawan/marker-api:latest
```

## Sorun Giderme
- “NVIDIA Driver not detected” uyarısı:
  - Docker Desktop GPU entegrasyonu açık mı? Host’ta `nvidia-smi` çalışıyor mu?
- `Killed`/OOM:
  - `mem_limit` değerini yükseltin (ör. 12–16 GB), chunk boyutunu küçük tutun (3 → 2/1 sayfa).
- 200 OK ama içerik boş:
  - PDF tamamen görsel olabilir (metin katmanı yok); gateway boş içeriği kaydetmez, 502 döndürür.
  - Alternatif OCR (Marker) kullanın veya dosyayı OCR’lı PDF’e dönüştürüp tekrar deneyin.

## İlgili Dosyalar
- docker-compose: `rag3_for_colab/docker-compose.yml` (`marker-api` servisi)
- Gateway: `rag3_for_colab/src/api/main.py` (Marker entegrasyonu, chunk/retry/guard mantığı)

## Değişiklik Geçmişi
- GPU ve 12 GB RAM sınırı eklendi.
- Readiness + retry/backoff eklendi.
- 3 sayfalık chunk + tek sayfa fallback eklendi.
- Boş içerik kaydını engelleyen guard eklendi.








