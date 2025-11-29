# Debug Raporları Klasörü

Bu klasör, KBRAG ve Kişiselleştirme sisteminin debug raporlarını içerir.

## Klasör Yapısı

```
debug_raporlari/
├── README.md                    # Bu dosya
├── raporlar/                    # Ham debug raporları (text formatında)
├── analizler/                   # Rapor analizleri ve özetler
└── ornekler/                    # Örnek raporlar
```

## Rapor Formatı

Her rapor şu bilgileri içerir:

1. **İstek Parametreleri**: Session ID, Query, Model, vb.
2. **Session RAG Settings**: Model ve embedding ayarları
3. **Retrieval Aşamaları**: 
   - Topic Classification
   - Chunk Retrieval
   - QA Pairs Matching
   - Knowledge Base Retrieval
   - Merged Results
   - Context Built
4. **Rerank Scores**: Döküman sıralama skorları
5. **LLM Generation**: Prompt ve yanıt detayları
6. **Response**: Final yanıt
7. **Timing**: İşlem süreleri
8. **Sources Summary**: Kullanılan kaynaklar
9. **APRAG Personalization**: 
   - Feature Flags
   - Student Profile
   - CACS Scoring
   - Pedagogical Analysis (ZPD, Bloom, Cognitive Load)
   - Personalization Factors
   - Pedagogical Instructions
   - Response Comparison

## Kullanım

Raporları buraya kopyalayıp analiz edebilirsiniz. Her rapor için:

1. Raporu `raporlar/` klasörüne kaydedin
2. Önemli bulguları `analizler/` klasörüne not edin
3. Örnek raporları `ornekler/` klasöründe saklayın

## Notlar

- Raporlar text formatında saklanmalıdır
- Her rapor için tarih ve sorgu bilgisi eklenmelidir
- Önemli bulgular README'ye eklenebilir

