# Kişiselleştirme Sistemi Test Senaryoları

## Test Amacı
Pedagogical prompt sisteminin etkinliğini ölçmek ve kişiselleştirme kalitesini değerlendirmek.

## Test Metodolojisi

### 1. Test Senaryosu: Bloom Seviye 1 (Remember) - Elementary Öğrenci

**Öğrenci Profili:**
- ZPD Seviyesi: elementary
- Bloom Seviyesi: remember (nedir soruları)
- Cognitive Load: Düşük olmalı

**Test Soruları:**
1. "profaz nedir"
2. "mitoz nedir"
3. "kromozom nedir"
4. "interfaz nedir"
5. "sitokinez nedir"

**Beklenen Sonuçlar:**
- ✅ Anahtar kelimeler **kalın** formatında vurgulanmalı
- ✅ Hafıza ipuçları eklenmeli (ör: "Profaz = pro (ön) + faz")
- ✅ Kısa ve net tanım (2-3 paragraf, max)
- ✅ Teknik terimler basitleştirilmeli
- ✅ Görsel benzetmeler eklenmeli

**Değerlendirme Kriterleri:**
- [ ] Anahtar kelimeler vurgulandı mı? (0-5 puan)
- [ ] Hafıza ipuçları eklendi mi? (0-5 puan)
- [ ] Uzunluk uygun mu? (0-5 puan)
- [ ] Teknik terimler basitleştirildi mi? (0-5 puan)
- [ ] Öğrenci seviyesine uygun mu? (0-5 puan)

**Toplam Puan:** /25

---

### 2. Test Senaryosu: Bloom Seviye 2 (Understand) - Intermediate Öğrenci

**Öğrenci Profili:**
- ZPD Seviyesi: intermediate
- Bloom Seviyesi: understand (açıkla, anlat soruları)
- Cognitive Load: Orta

**Test Soruları:**
1. "mitoz ve mayoz arasındaki farkı açıkla"
2. "hücre bölünmesi nasıl gerçekleşir"
3. "profaz evresini detaylı anlat"
4. "kromozom sayısı nasıl değişir"
5. "interfaz evresinde neler olur"

**Beklenen Sonuçlar:**
- ✅ Açıklayıcı ve anlaşılır dil
- ✅ Örneklerle desteklenmeli
- ✅ Karşılaştırmalar yapılmalı
- ✅ Adım adım açıklama
- ✅ Orta seviye detay

**Değerlendirme Kriterleri:**
- [ ] Açıklama net mi? (0-5 puan)
- [ ] Örnekler var mı? (0-5 puan)
- [ ] Karşılaştırma yapıldı mı? (0-5 puan)
- [ ] Adım adım açıklama var mı? (0-5 puan)
- [ ] Seviyeye uygun mu? (0-5 puan)

**Toplam Puan:** /25

---

### 3. Test Senaryosu: Bloom Seviye 3 (Apply) - Advanced Öğrenci

**Öğrenci Profili:**
- ZPD Seviyesi: advanced
- Bloom Seviyesi: apply (uygula, çöz soruları)
- Cognitive Load: Yüksek olabilir

**Test Soruları:**
1. "bir hücrenin mitoz geçirdiğini nasıl anlarsın"
2. "kromozom sayısı değişimini hesapla"
3. "hücre döngüsünde hangi evrede ne olur"
4. "mayoz sonucu kaç hücre oluşur"
5. "profaz evresini mikroskopta nasıl görürsün"

**Beklenen Sonuçlar:**
- ✅ Pratik uygulama örnekleri
- ✅ Adım adım çözüm
- ✅ Gerçek hayat senaryoları
- ✅ İleri seviye detay
- ✅ Problem çözme yaklaşımı

**Değerlendirme Kriterleri:**
- [ ] Pratik örnekler var mı? (0-5 puan)
- [ ] Adım adım çözüm var mı? (0-5 puan)
- [ ] Gerçek hayat senaryoları var mı? (0-5 puan)
- [ ] İleri seviye detay var mı? (0-5 puan)
- [ ] Problem çözme yaklaşımı uygun mu? (0-5 puan)

**Toplam Puan:** /25

---

## Test Süreci

### Adım 1: Başlangıç Profili
1. Öğrenci chat ekranına gir
2. Debug kartında mevcut parametreleri gözlemle:
   - ZPD Seviyesi
   - Bloom Seviyesi
   - Cognitive Load
   - Personalization Factors

### Adım 2: İlk Soru
1. Test senaryosundan bir soru seç
2. Soruyu gönder
3. Cevabı al
4. Debug kartında parametreleri kontrol et:
   - ZPD değişti mi?
   - Bloom doğru tespit edildi mi?
   - Cognitive Load hesaplandı mı?

### Adım 3: Cevap Analizi
1. Orijinal vs Kişiselleştirilmiş cevabı karşılaştır
2. Değerlendirme kriterlerini kontrol et
3. Puan ver (0-5 her kriter için)

### Adım 4: Emoji Feedback
1. Cevaba emoji feedback ver:
   - 😊 = Çok iyi (1.0)
   - 👍 = İyi (0.7)
   - 😐 = Orta (0.3)
   - ❌ = Kötü (0.0)

### Adım 5: İkinci Soru
1. Aynı test senaryosundan başka bir soru sor
2. Profil değişimini gözlemle:
   - ZPD seviyesi değişti mi?
   - Başarı oranı güncellendi mi?
3. Cevabın kalitesi değişti mi?

### Adım 6: Tekrar Test
1. 3-5 soru daha sor
2. Her soruda:
   - Profil değişimini kaydet
   - Cevap kalitesini değerlendir
   - Emoji feedback ver

### Adım 7: Raporlama
1. Tüm test sonuçlarını kaydet
2. Ortalama puanları hesapla
3. Başarı oranını belirle:
   - 20-25 puan: Mükemmel (✅)
   - 15-19 puan: İyi (⚠️)
   - 10-14 puan: Orta (❌)
   - 0-9 puan: Kötü (❌❌)

---

## Test Raporu Formatı

```markdown
# Test Raporu: [Test Senaryosu Adı]

**Tarih:** [Tarih]
**Öğrenci ID:** [ID]
**Session ID:** [ID]

## Başlangıç Profili
- ZPD: [seviye]
- Bloom: [seviye]
- Cognitive Load: [değer]

## Test Sonuçları

### Soru 1: "[soru]"
- **ZPD:** [başlangıç] → [son]
- **Bloom:** [tespit edilen]
- **Cognitive Load:** [değer]
- **Emoji Feedback:** [emoji]
- **Puanlar:**
  - Kriter 1: [puan]/5
  - Kriter 2: [puan]/5
  - Kriter 3: [puan]/5
  - Kriter 4: [puan]/5
  - Kriter 5: [puan]/5
- **Toplam:** [puan]/25
- **Başarı:** [✅/⚠️/❌]

### Soru 2: "[soru]"
[...]

## Genel Değerlendirme
- **Ortalama Puan:** [puan]/25
- **Başarı Oranı:** [oran]%
- **ZPD Adaptasyonu:** [iyi/kötü]
- **Bloom Tespiti:** [doğru/yanlış]
- **Prompt Etkinliği:** [etkili/etkisiz]

## Öneriler
- [Öneri 1]
- [Öneri 2]
```

---

## Hızlı Test Checklist

Her test için şunları kontrol et:

- [ ] Debug kartı görünüyor mu?
- [ ] ZPD seviyesi doğru mu?
- [ ] Bloom seviyesi doğru tespit edildi mi?
- [ ] Cognitive Load hesaplandı mı?
- [ ] Anahtar kelimeler vurgulandı mı? (Remember için)
- [ ] Hafıza ipuçları eklendi mi? (Remember için)
- [ ] Örnekler var mı? (Understand için)
- [ ] Pratik uygulama var mı? (Apply için)
- [ ] Emoji feedback çalışıyor mu?
- [ ] Profil güncelleniyor mu?

---

## Notlar

- Her test senaryosu için en az 5 soru sorulmalı
- Emoji feedback mutlaka verilmeli (profil güncellemesi için)
- Test sonuçları `debug_raporlari/test_sonuclari/` klasörüne kaydedilmeli
- Başarı oranı %80'in üzerinde olmalı

