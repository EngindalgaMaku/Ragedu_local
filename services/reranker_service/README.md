# Reranker Service

Ayrı bir mikroservis olarak çalışan reranker servisi. BGE-Reranker-V2-M3 ve MS-MARCO desteği ile seçimli reranking sağlar.

## Özellikler

- ✅ **BGE-Reranker-V2-M3**: Türkçe optimize, 100+ dil desteği
- ✅ **MS-MARCO**: Hafif, hızlı, geriye dönük uyumlu
- ✅ **Seçimli Kullanım**: Environment variable ile model seçimi
- ✅ **Ayrı Container**: İzolasyon ve ölçeklenebilirlik
- ✅ **Mevcut Reranker Korundu**: Fallback desteği

## Hızlı Başlangıç

```bash
# Servisi başlat
docker-compose up -d reranker-service

# Health check
curl http://localhost:8008/health

# Info
curl http://localhost:8008/info
```

## Environment Variables

```bash
RERANKER_TYPE=bge  # "bge" or "ms-marco"
BGE_MODEL_NAME=BAAI/bge-reranker-v2-m3
MS_MARCO_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## API Kullanımı

```python
import requests

response = requests.post("http://localhost:8008/rerank", json={
    "query": "Türkçe sorgu",
    "documents": ["doc1", "doc2", "doc3"],
    "top_k": 5
})

results = response.json()
```

## Detaylı Dokümantasyon

Bkz: `RERANKER_SERVICE_KULLANIM_KILAVUZU.md`



