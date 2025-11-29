# Profaz Nedir - Rapor Analizi

**Tarih:** 27.11.2025 20:23:51  
**Sorgu:** profaz nedir  
**Session ID:** 6f3318202dd81b5fcab7b6621a6f4807

## Özet

Bu rapor, "profaz nedir" sorusu için KBRAG ve kişiselleştirme sisteminin nasıl çalıştığını gösterir.

## KBRAG Retrieval Analizi

### Topic Classification
- **3 konu** eşleşti
- **Güven:** 92.0%
- En yüksek güven: Mitoz (95.0%)

### Chunk Retrieval
- **5 chunk** çekildi
- En yüksek skor: 54.2% (MITOTİK EVRE konusu)
- Skorlar: 45.2%, 54.2%, 45.5%, 50.8%, 49.6%

### Knowledge Base
- **3 KB öğesi** çekildi
- Mitoz: 95.0% ilgililik
- Mayoz: 85.0% ilgililik
- Karşılaştırma: 80.0% ilgililik

## CACS Puanlama Analizi

### Döküman Skorları

| Döküman | Base Score | Personal Score | Global Score | Context Score | Final Score | Sıra |
|---------|------------|----------------|--------------|---------------|-------------|------|
| Doc 1   | 95.0%      | 35.0%          | 50.0%        | 50.0%         | **59.75%**  | 1    |
| Doc 2   | 85.0%      | 35.0%          | 50.0%        | 50.0%         | **56.75%**  | 2    |
| Doc 3   | 80.0%      | 35.0%          | 50.0%        | 50.0%         | **55.25%**  | 3    |

### Gözlemler
- **Base Score** yüksek (80-95%) → Dökümanlar soruya çok ilgili
- **Personal Score** düşük (35%) → Öğrencinin geçmiş etkileşimleriyle uyum düşük
- **Global Score** ve **Context Score** orta (50%) → Genel başarı ve bağlam uyumu normal
- **Final Score** Base Score'dan düşük → Personal Score final skoru düşürüyor

## Pedagojik Analiz

### ZPD (Zone of Proximal Development)
- **Mevcut Seviye:** intermediate
- **Önerilen Seviye:** elementary
- **Başarı Oranı:** 0% (başarısız etkileşimler var)
- **Sonuç:** Öğrenci için daha basit seviye öneriliyor

### Bloom Taksonomisi
- **Seviye:** remember (Seviye 1)
- **Güven:** 100%
- **Tespit:** "nedir" kelimesi → Hatırlama seviyesi
- **Sonuç:** Basit tanım gerekiyor

### Cognitive Load
- **Toplam Yük:** 0.32 (düşük)
- **Sadeleştirme Gerekli:** Hayır
- **Sonuç:** Yanıt öğrenci için uygun karmaşıklıkta

## Kişiselleştirme Analizi

### Parametreler
- **Anlama Seviyesi:** intermediate
- **Zorluk Seviyesi:** elementary (ZPD'den)
- **Açıklama Stili:** balanced
- **Örnekler Gerekli:** Hayır

### Prompt'a Yansıma
1. **ZPD →** "Temel kavramları önce açıkla" talimatı eklendi
2. **Bloom →** "Kısa, net ve doğrudan tanım ver" talimatı eklendi
3. **Cognitive Load →** Simplification talimatı eklenmedi (yük düşük)

## Yanıt Karşılaştırması

### Orijinal Yanıt
- **Uzunluk:** 959 karakter
- **Özellikler:** 
  - Detaylı açıklama
  - Kalın vurgular (**mitoz**, **kromatin iplikler**, vb.)
  - Teknik terimler

### Kişiselleştirilmiş Yanıt
- **Uzunluk:** 649 karakter (-310 karakter, %32 kısalma)
- **Özellikler:**
  - Daha basit dil
  - Kalın vurgular kaldırıldı
  - Daha kısa cümleler
  - Teknik detaylar azaltıldı

### Benzerlik
- **Benzerlik Oranı:** 53.76%
- **Farklılık:** Evet (önemli ölçüde farklı)
- **Sonuç:** Kişiselleştirme başarılı, yanıt öğrenci seviyesine uyarlandı

## Sonuçlar ve Öneriler

### Başarılı Yönler
1. ✅ Topic classification doğru çalıştı (92% güven)
2. ✅ İlgili dökümanlar çekildi (yüksek base score)
3. ✅ Kişiselleştirme uygulandı (yanıt kısaltıldı ve basitleştirildi)
4. ✅ Bloom seviyesi doğru tespit edildi (remember)

### İyileştirme Alanları
1. ⚠️ Personal Score düşük (35%) → Öğrenci profili güncellenebilir
2. ⚠️ Success rate 0% → ZPD hesaplaması gözden geçirilebilir
3. ⚠️ Rerank skorları düşük (max 17.2%) → Reranker ayarları kontrol edilebilir

### Öneriler
- Öğrencinin daha fazla feedback vermesi teşvik edilmeli
- Personal Score hesaplaması gözden geçirilmeli
- Rerank skorları düşük olduğu için reranker ayarları optimize edilebilir





