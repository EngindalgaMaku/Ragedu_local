## 9 Katmanlı Eğitsel RAG Uygulama Mimarisi – Görsel Prompt Taslakları

Bu dosya, mevcut APRAG / Eğitsel-KBRAG / DYSK tabanlı **9 katmanlı** uygulama mimarisini görsel olarak ifade etmek için kullanılabilecek Türkçe ve İngilizce image prompt örneklerini içerir. Amaç; kullanıcıdan LLM’e kadar olan tam veri akışını, RAG çekirdeği, DYSK katmanı, iş mantığı ve veri katmanlarını net bir akış diyagramı şeklinde gösterebilmektir.

Tanımlanan 9 katman:

1. Kullanıcı Katmanı (Öğrenci / Öğretmen)
2. Frontend Katmanı (Web UI – Next.js / Streamlit)
3. API ve Güvenlik Katmanı (FastAPI, Auth, Rate Limiting, CORS)
4. Eğitsel İş Mantığı Katmanı (CACS, ZPD, Bloom, Bilişsel Yük, feature flags)
5. Konuşma Belleği ve Öğrenci Profili Katmanı (conversation history, student_profiles, feedback)
6. Doküman İşleme ve Chunking Katmanı (upload, text extraction, cleaning, chunking)
7. Embedding ve Vektör Arama Katmanı (FAISS/Chroma, vector DB)
8. DYSK Doğrulayıcı Yeniden Sıralama Katmanı (ACCEPT/FILTER/REJECT karar mekanizması)
9. LLM Yanıt Üretimi ve Analitik Katmanı (LLM, response post-processing, analytics DB, monitoring)

---

### Türkçe Görsel Prompt Örnekleri

**Prompt 1 – Katmanlı Mimari Diyagramı (Türkçe etiketli)**

“Beyaz arka planlı, temiz ve modern bir sistem mimarisi diyagramı oluştur. 9 katmanlı, eğitim odaklı bir RAG uygulaması göster:  
1) **Kullanıcı Katmanı**: Öğrenci ve öğretmen ikonları.  
2) **Frontend Katmanı**: Web arayüzü kutusu (Next.js / Streamlit UI).  
3) **API ve Güvenlik Katmanı**: FastAPI kutusu, JWT kimlik doğrulama, rate limiting, CORS ve feature flag kontrolü.  
4) **Eğitsel İş Mantığı Katmanı**: CACS skorlaması, ZPD hesaplama, Bloom seviyesi ve Bilişsel Yük yöneticisi içeren bir kutu.  
5) **Konuşma Belleği ve Öğrenci Profili Katmanı**: conversation history, student_profiles ve student_feedback tablolarını temsil eden kutular.  
6) **Doküman İşleme ve Chunking Katmanı**: doküman yükleme, metin çıkarma, temizleme ve chunking aşamalarını içeren işlem hattı.  
7) **Embedding ve Vektör Arama Katmanı**: embedding modeli ve FAISS/Chroma vektör veritabanı, cosine similarity okları ile göster.  
8) **DYSK Katmanı (Doğrulayıcı Yeniden Sıralama Katmanı)**: Reranker modeli, ACCEPT / FILTER / REJECT karar şeması, skor barları.  
9) **LLM Yanıt ve Analitik Katmanı**: LLM kutusu (ör. Llama / GPT), yanıt post-processing, analytics database ve izleme panosu.  
Katmanları dikey olarak sırala, aralarındaki veri akışını oklarla göster. Tüm metin etiketleri Türkçe olsun, düz (flat) tasarım, yüksek çözünürlüklü, profesyonel teknik diagram tarzı kullan.”

**Prompt 2 – Akış Şeması + Zaman Ekseni (Türkçe)**  

“Tek bir geniş şemada, 9 katmanlı bir Eğitsel RAG sistemi için **uçtan uca veri akışını** gösteren Türkçe bir akış diyagramı çiz. Soldan sağa veya yukarıdan aşağıya doğru şu sırayı takip et:  
- Öğrenci/öğretmen → Web Arayüzü (soru sorma, doküman yükleme)  
- API + Auth + Rate Limiting katmanı →  
- Eğitsel İş Mantığı (CACS doküman skorlaması, ZPD, Bloom, Bilişsel Yük) →  
- Konuşma Belleği ve Öğrenci Profili (geçmiş etkileşimler, öğrenci profili güncelleme) →  
- Doküman İşleme Pipeline’ı (metin çıkarma, temizleme, Türkçe-aware chunking) →  
- Embedding + Vektör Arama (vektör DB’den top-k chunk arama) →  
- DYSK Katmanı (MS-MARCO MiniLM reranker, threshold tabanlı ACCEPT/FILTER/REJECT) →  
- LLM Yanıt Üretimi (bağlam ile prompt hazırlama, yanıt üretimi) →  
- Analitik ve Loglama (sorgu logları, kalite skorları, dashboard).  
Her kutuyu Türkçe başlıklarla etiketle (ör. ‘Eğitsel İş Mantığı Katmanı’, ‘Konuşma Belleği’, ‘DYSK Katmanı’). Oklarla veri akış yönünü belirt, önemli karar noktalarını elmas (decision) şekilleriyle göster. Flat design, sade renk paleti, okunabilir tipografi, teknik paper’a konulabilecek seviyede profesyonel stil.”

---

### English Visual Prompt Examples

**Prompt 3 – 9-Layer Architecture Overview (English labels)**  

“Create a clean, high-resolution system architecture diagram on a white background that shows a **9-layer educational RAG application**. The layers from top to bottom should be:  
1) **User Layer** – student and teacher users.  
2) **Frontend Layer** – web UI (Next.js / Streamlit) where users upload documents and ask questions.  
3) **API & Security Layer** – FastAPI gateway with authentication, rate limiting, CORS and feature flag checks.  
4) **Educational Business Logic Layer** – CACS scoring, ZPD calculator, Bloom level classifier, cognitive load manager.  
5) **Conversation Memory & Student Profile Layer** – conversation history, student profiles, feedback logs.  
6) **Document Processing & Chunking Layer** – document upload, text extraction, cleaning and Turkish-aware chunking pipeline.  
7) **Embedding & Vector Search Layer** – embedding model and FAISS/Chroma vector database with cosine similarity search.  
8) **DYSK Corrective Reranking Layer** – MS-MARCO cross-encoder reranker, ACCEPT/FILTER/REJECT decision box with thresholds.  
9) **LLM Answer & Analytics Layer** – LLM generation, answer post-processing, analytics database and monitoring dashboard.  
Use English labels on all boxes, arrange the 9 layers vertically with strong horizontal separators, and connect them with arrows showing data flow from the user down to the LLM and back. Flat design, subtle colors, professional technical illustration style.”

**Prompt 4 – End-to-End Flowchart with Swimlanes (English)**  

“Design an end-to-end **flowchart with swimlanes** that visualizes how a 9-layer educational RAG system processes a student question. Use horizontal swimlanes for: Frontend, API & Security, Educational Logic, RAG Core (document processing + embedding + vector DB), DYSK Layer, LLM & Analytics. Inside these lanes, show the following sequence:  
- User asks a question in the web UI (Frontend lane).  
- Request goes through FastAPI, authentication, rate limiting, CORS, feature flags (API & Security lane).  
- Educational business logic enriches the query with CACS scores, ZPD level, Bloom level, cognitive load constraints (Educational Logic lane).  
- RAG core retrieves relevant chunks via embedding + vector search (RAG Core lane).  
- DYSK corrective reranking layer reranks chunks, applies thresholds and outputs ACCEPT/FILTER/REJECT decision (DYSK lane).  
- LLM generates a final answer with cited sources, analytics tracker records query stats and quality metrics (LLM & Analytics lane).  
Label all steps in English, use clear arrows and decision diamonds, and visually emphasize the 9 conceptual layers of the application. Minimalist, flat UI diagram style suitable for inclusion in an academic paper or architecture document.”

---

Bu prompt’lar doğrudan bir görsel üretim aracına (ör. DALL·E, Midjourney, Stable Diffusion vb.) verilecek şekilde tasarlanmıştır; gerekirse katman adları veya teknoloji isimleri kolayca özelleştirilebilir.







