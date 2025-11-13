# Araştırma Raporu Taslağı: Eğitimde RAG Sistemlerinin Kullanımı

Bu döküman, "Eğitimde RAG (Retrieval-Augmented Generation) Sistemlerinin Etkinliği ve Uygulamaları" üzerine hazırlanacak olan araştırma raporu için bir şablon ve yol haritası sunmaktadır.

---

## 1. Problem ve Motivasyon

### 1.1. Problem Tanımı

- **Problem nedir?** Öğrencilerin ve eğitimcilerin, ders materyalleri, akademik makaleler ve kurumsal belgeler gibi büyük ve yapılandırılmamış bilgi yığınları içinden doğru, güvenilir ve bağlama uygun bilgilere hızlı bir şekilde erişmesindeki zorluklar.
- **Kimin için önemlidir?** Bu problem, bilgiye anında erişim ihtiyacı duyan öğrenciler, ders materyallerini yöneten ve soru cevaplayan eğitimciler ve kurum içi bilgi yönetimini optimize etmek isteyen eğitim kurumları için önemlidir.

### 1.2. Mevcut Pratikteki Darboğazlar

- Geleneksel arama motorları (keyword-based) anlamsal ve bağlamsal sorguları anlamada yetersiz kalır.
- Öğrenciler, aradıkları spesifik cevabı bulmak için uzun dökümanları manuel olarak taramak zorunda kalırlar.
- Eğitimciler, sıkça sorulan ve cevapları dökümanlarda mevcut olan soruları tekrar tekrar yanıtlamak için önemli zaman harcarlar.
- Standart LLM'ler (ChatGPT gibi), kuruma özel veya güncel olmayan bilgiler hakkında "halüsinasyon" görmeye (yanlış bilgi uydurmaya) eğilimlidir ve kaynak gösteremezler.

---

## 2. Literatür Taraması ve Boşluk Analizi

### 2.1. Benzer Çalışmaların Özeti

Aşağıdaki tablo, eğitimde RAG veya benzeri bilgi erişim sistemlerini inceleyen temel çalışmaları özetlemektedir.

| Çalışma (Yazar, Yıl)     | Veri Seti                 | Yöntem     | Değerlendirme Metrikleri | Güçlü Yönleri                                   | Zayıf Yönleri / Boşluk                                                         |
| :----------------------- | :------------------------ | :--------- | :----------------------- | :---------------------------------------------- | :----------------------------------------------------------------------------- |
| Örnek: Gao et al. (2023) | Wikipedia, Open-domain QA | RAG, FAISS | R-Precision, Recall@K    | Geniş ölçekli veri üzerinde etkinlik gösterimi. | Eğitime özel dökümanların (ders notları, sunumlar) karmaşıklığını ele almıyor. |
| (Çalışma 2)              |                           |            |                          |                                                 |                                                                                |
| (Çalışma 3)              |                           |            |                          |                                                 |                                                                                |
| (Çalışma 4)              |                           |            |                          |                                                 |                                                                                |
| (Çalışma 5)              |                           |            |                          |                                                 |                                                                                |
| (Çalışma 6)              |                           |            |                          |                                                 |                                                                                |
| (Çalışma 7)              |                           |            |                          |                                                 |                                                                                |
| (Çalışma 8)              |                           |            |                          |                                                 |                                                                                |

### 2.2. Literatürdeki Boşluk

Literatürdeki mevcut çalışmalar genellikle geniş ölçekli, genel amaçlı veri setleri üzerinde yoğunlaşmıştır. Örneğin, _[Örnek: Gao et al. (2023)]_ RAG'in genel bilgi erişimindeki başarısını kanıtlamış olsa da, **bu çalışma, PDF, DOCX ve PPTX gibi farklı formatlardaki ders materyallerinin heterojen yapısını ve bu materyallerden tutarlı bir bilgi tabanı oluşturmanın zorluklarını ele almamaktadır.** Ayrıca, çoğu çalışma, farklı embedding ve LLM kombinasyonlarının, özellikle Türkçe gibi morfolojik olarak zengin dillerdeki performansını karşılaştırmalı olarak analiz etmemektedir.

**Bizim çalışmamız bu boşluğu şu şekilde adreslemektedir:**

1.  Farklı formatlardaki (PDF, PPTX, DOCX) akademik ve kurumsal dökümanları işleyebilen modüler bir RAG mimarisi sunuyoruz.
2.  Farklı embedding (örn: `mxbai-embed-large`, `all-MiniLM-L6-v2`) ve LLM (örn: `Qwen`, `Mistral`, `Llama3`) kombinasyonlarının performansını, hem İngilizce hem de Türkçe sorgular için karşılaştırmalı olarak analiz ediyoruz.
3.  Sistemin performansını, hem geleneksel metrikler (örn: R-Precision) hem de kullanıcı deneyimine dayalı nitel metrikler (örn: cevap tutarlılığı, kaynak doğruluğu) üzerinden değerlendiriyoruz.

---

## 3. Yöntem

### 3.1. Sistem Mimarisi

Bu çalışmada geliştirilen RAG sistemi, aşağıdaki blok diyagramında gösterilen modüler bir pipeline'dan oluşmaktadır.

```mermaid
graph TD
    subgraph "1. Veri İşleme (Data Ingestion)"
        A[Kullanıcı Dökümanları<br>(PDF, DOCX, PPTX)] --> B{Döküman Ayrıştırıcı};
        B --> C[Metin Çıkarımı];
        C --> D[Metin Bölütleme<br>(Chunking)];
    end

    subgraph "2. Vektörleştirme (Indexing)"
        D --> E["Embedding Modeli<br>(örn: mxbai-embed-large)"];
        E --> F[(Vektör Veritabanı<br>FAISS)];
    end

    subgraph "3. Sorgu ve Cevap (Querying)"
        G[Kullanıcı Sorusu] --> H["Embedding Modeli"];
        H -- "Anlamsal Arama" --> F;
        F -- "İlgili Bölütleri Getir (Context)" --> I{Prompt Mühendisliği};
        G -- "Orijinal Soru" --> I;
        I --> J["<b>Üretim Modeli (LLM)</b><br>(örn: Qwen, Mistral)"];
        J --> K[Oluşturulan Cevap + Kaynaklar];
    end

    style J fill:#9f9,stroke:#333,stroke-width:2px
```

### 3.2. Tasarım Kararları ve Gerekçelendirme

- **Döküman İşleme:** `unstructured.io` kütüphanesi, farklı dosya formatlarını tek bir arayüz üzerinden işleme yeteneği ve zengin metadata çıkarma potansiyeli nedeniyle tercih edilmiştir.
- **Embedding Modeli:** Varsayılan olarak `mxbai-embed-large` seçilmiştir çünkü MTEB (Massive Text Embedding Benchmark) liderlik tablosunda yüksek sıralarda yer alması, anlamsal benzerliği yakalamada ne kadar başarılı olduğunun bir göstergesidir.
- **Vektör Veritabanı:** `FAISS` (Facebook AI Similarity Search), yerel olarak çalışabilmesi, yüksek hızda arama yapabilmesi ve bellek içi (in-memory) çalışarak küçük ve orta ölçekli projeler için ek bir sunucu maliyeti gerektirmemesi nedeniyle seçilmiştir.
- **LLM Seçenekleri:** `Qwen`, `Mistral` ve `Llama3` gibi farklı modellerin sisteme entegre edilmesi, farklı dillerdeki (özellikle Türkçe) yeteneklerini, hızlarını ve cevap kalitelerini karşılaştırma imkanı sunmak için stratejik bir karardır.

### 3.3. Hiperparametreler

- **Chunk Boyutu:** 1000 karakter
- **Chunk Çakışması (Overlap):** 200 karakter
- **Top-K (Getirilecek Bölüt Sayısı):** 5
- **LLM Temperature:** 0.3 (Daha tutarlı ve olgusal cevaplar için)

---

## 4. Veri ve Deney Tasarımı

### 4.1. Veri Kaynakları

- **Veri Toplama:** Burdur Mehmet Akif Ersoy Üniversitesi'nin çeşitli bölümlerine ait ders notları, sunumlar ve yönetmelikler gibi halka açık dökümanlar kullanılmıştır.
- **Veri Temizleme:** Dökümanlardan metin çıkarımı sırasında oluşan formatlama hataları (örn: gereksiz boşluklar, sayfa numaraları) temel düzeyde temizlenmiştir.
- **Veri Ayrımı:** Bu çalışmada, geleneksel bir train/val/test ayrımı yerine, bir "soru-cevap-bağlam" değerlendirme seti oluşturulmuştur. Bu set, dökümanlardan manuel olarak çıkarılan 50 adet soru, bu soruların ideal cevapları ve cevabın bulunduğu orijinal metin parçasını içermektedir.

### 4.2. Değerlendirme Metrikleri

- **Nicel Metrikler:**
  - **Retrieval (Getirme) Metrikleri:**
    - `Recall@K`: Getirilen ilk K döküman arasında doğru cevabı içeren dökümanın bulunma oranı.
    - `MRR (Mean Reciprocal Rank)`: Doğru cevabın getirildiği ortalama sıra.
  - **Generation (Üretim) Metrikleri (RAGAs kütüphanesi ile):**
    - `Faithfulness`: Üretilen cevabın, sağlanan bağlama ne kadar sadık kaldığı.
    - `Answer Relevancy`: Üretilen cevabın, orijinal soruyla ne kadar ilgili olduğu.
- **Nitel Metrikler:**
  - Üretilen cevapların 5 farklı uzman tarafından 1-5 arası bir ölçekte (doğruluk, akıcılık, faydalılık) puanlanması.

---

## 5. Sonuçlar

Bu bölümde, farklı model kombinasyonları ile yapılan deneylerin sonuçları sunulacaktır.

### 5.1. Nicel Sonuçlar Tablosu

_(Bu tablo, deneyler yapıldıktan sonra doldurulacaktır.)_

| LLM Modeli | Embedding Modeli  | Recall@5 | MRR | Faithfulness | Answer Relevancy | Ortalama Puan (Nitel) |
| :--------- | :---------------- | :------- | :-- | :----------- | :--------------- | :-------------------- |
| Qwen 2.5   | mxbai-embed-large |          |     |              |                  |                       |
| Mistral 7B | mxbai-embed-large |          |     |              |                  |                       |
| Llama 3 8B | mxbai-embed-large |          |     |              |                  |                       |
| Qwen 2.5   | all-MiniLM-L6-v2  |          |     |              |                  |                       |

### 5.2. Grafiksel Analizler

- _(Öğrenme eğrileri veya model karşılaştırma grafikleri burada yer alacak.)_
- **Örnek Nitel Vaka Analizi:**
  - **Soru:** "FAISS kütüphanesinin avantajları nelerdir?"
  - **Qwen Cevabı:** _(Cevap metni ve kaynak atfı)_
  - **Mistral Cevabı:** _(Cevap metni ve kaynak atfı)_
  - **Analiz:** Qwen'in daha detaylı ve yapılandırılmış bir cevap verdiği, Mistral'in ise daha özet ve hızlı bir yanıt ürettiği gözlemlenmiştir.

---

## 6. Sınırlılıklar

- **Veri Önyargıları:** Kullanılan dökümanlar belirli bir üniversite ve bölümle sınırlı olduğu için, sonuçların genellenebilirliği kısıtlıdır.
- **Genellenebilirlik:** Sistem, ağırlıklı olarak Türkçe ve İngilizce metinler üzerinde test edilmiştir. Diğer dillerdeki performansı bilinmemektedir.
- **Hesaplama Maliyeti:** `mxbai-embed-large` gibi büyük embedding modelleri ve 14B parametreli LLM'ler, önemli miktarda RAM ve işlem gücü gerektirmektedir. Bu, sistemin düşük kaynaklı cihazlarda çalışmasını zorlaştırmaktadır.
- **Hata Modları:** Sistem, dökümanlarda hiç bulunmayan veya çok dolaylı olarak ima edilen konularda hala yetersiz kalabilmekte veya alakasız cevaplar üretebilmektedir.

---

## 7. Etik ve Uyum

- **Veri Gizliliği:** Çalışmada kullanılan tüm dökümanlar halka açık kaynaklardan temin edilmiştir ve kişisel veri içermemektedir. Herhangi bir anonimleştirme işlemine gerek duyulmamıştır.
- **Telif ve Kaynak Atfı:** Sistem, ürettiği her cevabın sonunda, bilgiyi aldığı orijinal dökümanı ve ilgili bölümü kaynak olarak göstermektedir. Bu, hem telif haklarına saygı duymayı hem de kullanıcının bilgiyi doğrulamasına olanak tanımayı amaçlar.
- **Sentetik Veri Kullanımı:** Değerlendirme aşamasında sentetik veri üretilmemiş, tüm soru-cevap çiftleri gerçek dökümanlardan manuel olarak oluşturulmuştur.

---

## 8. Kaynakça

_(Bu bölüm, raporda atıf yapılan tüm makale, kitap ve diğer kaynakların listesini içerecektir. En az 10-20 güncel kaynak hedeflenmektedir.)_

1.  Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... & Wang, H. (2023). Retrieval-augmented generation for large language models: A survey. _arXiv preprint arXiv:2312.10997_.
2.  Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Nogueira, R., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems, 33_, 9459-9474.
3.  ...
