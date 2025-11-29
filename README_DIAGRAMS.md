# 📊 RAG Altyapısı Diagramları - Makale İçin Kullanım Kılavuzu

## 🎯 Makale İçin En Uygun Formatlar

### 1. **SVG (Önerilen)**
- ✅ Vektör formatı - ölçeklenebilir, kalite kaybı yok
- ✅ LaTeX ve Word'e kolay eklenir
- ✅ Küçük dosya boyutu
- ✅ Mermaid diagramlarından otomatik oluşturulabilir

### 2. **PDF (Vektör)**
- ✅ Akademik yayınlar için standart
- ✅ Yüksek kalite
- ✅ Python script ile oluşturulabilir

### 3. **PNG (Yüksek Çözünürlük)**
- ✅ 300 DPI veya daha yüksek
- ✅ Basit ekleme
- ✅ Tüm dergiler kabul eder

---

## 🚀 Hızlı Başlangıç

### Seçenek 1: Mermaid Diagramlarından SVG Oluşturma

```bash
# 1. Node.js yüklü olmalı
# 2. Mermaid CLI'yi yükleyin
npm install -g @mermaid-js/mermaid-cli

# 3. Script'i çalıştırın
python generate_architecture_diagrams.py
```

**Çıktı:** `diagrams/` klasöründe SVG dosyaları

### Seçenek 2: Python ile Profesyonel Diagramlar

```bash
# Gerekli kütüphaneleri yükleyin
pip install matplotlib numpy

# Script'i çalıştırın
python generate_professional_diagrams.py
```

**Çıktı:** 
- `diagrams/architecture_diagram.png` (300 DPI)
- `diagrams/architecture_diagram.pdf` (vektör)
- `diagrams/query_flow_diagram.png`
- `diagrams/hybrid_retrieval_diagram.png`

---

## 📝 Makaleye Ekleme

### LaTeX İçin:

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{diagrams/architecture_diagram.pdf}
    \caption{RAG Education Assistant Sistem Mimarisi}
    \label{fig:architecture}
\end{figure}
```

### Word İçin:

1. **Insert** → **Pictures** → Dosyayı seçin
2. SVG, PDF veya PNG formatında ekleyebilirsiniz
3. Gerekirse boyutlandırın (kalite kaybı olmaz)

### Overleaf/LaTeX Online:

1. SVG dosyalarını projeye yükleyin
2. `\usepackage{svg}` ekleyin
3. `\includesvg{diagrams/architecture_diagram}` kullanın

---

## 🎨 Diagram Önerileri

### Makale İçin En Önemli Diagramlar:

1. **Genel Sistem Mimarisi** ✅
   - Tüm servisleri gösterir
   - Makalenin "System Architecture" bölümü için ideal

2. **RAG Query Akış Diyagramı** ✅
   - Sequence diagram formatında
   - "Query Processing" bölümü için ideal

3. **Hybrid Retrieval Sistemi** ✅
   - Retrieval stratejisini detaylandırır
   - "Retrieval Method" bölümü için ideal

### Ek Diagramlar (Opsiyonel):

4. **Document Processing Pipeline**
5. **APRAG Mimarisi**
6. **Deployment Architecture**

---

## 🔧 Özelleştirme

### Renkleri Değiştirme:

`generate_professional_diagrams.py` dosyasındaki `colors` dictionary'sini düzenleyin:

```python
colors = {
    'frontend': '#E1F5FF',  # Açık mavi
    'gateway': '#FFF4E1',    # Açık sarı
    # ... diğer renkler
}
```

### Font Boyutunu Ayarlama:

```python
plt.rcParams['font.size'] = 12  # Varsayılan: 10
```

### Çözünürlüğü Artırma:

```python
plt.rcParams['figure.dpi'] = 300  # Varsayılan: 300 (yeterli)
# Daha yüksek için: 600 (ama dosya boyutu artar)
```

---

## 📦 Gerekli Paketler

### Mermaid CLI için:
```bash
npm install -g @mermaid-js/mermaid-cli
```

### Python script için:
```bash
pip install matplotlib numpy
# PNG export için (opsiyonel):
pip install cairosvg
```

---

## 💡 İpuçları

1. **Makale formatına uygun renkler kullanın:**
   - Siyah-beyaz yazdırma için: Gri tonları
   - Renkli yayın için: Yumuşak pastel renkler

2. **Font tutarlılığı:**
   - Tüm diagramlarda aynı font kullanın
   - Akademik standart: Times New Roman veya Arial

3. **Boyutlandırma:**
   - Tek sütun: 8-9 cm genişlik
   - Çift sütun: 17-18 cm genişlik

4. **Kalite kontrol:**
   - PDF/PNG dosyalarını açıp kontrol edin
   - Metinler okunabilir olmalı
   - Çizgiler net olmalı

---

## 🆘 Sorun Giderme

### Mermaid CLI bulunamıyor:
```bash
# Node.js yüklü mü kontrol edin
node --version

# Mermaid CLI'yi tekrar yükleyin
npm install -g @mermaid-js/mermaid-cli
```

### Python script hata veriyor:
```bash
# Gerekli paketleri yükleyin
pip install --upgrade matplotlib numpy
```

### SVG görünmüyor:
- LaTeX için: `\usepackage{svg}` ekleyin
- Word için: SVG yerine PDF kullanın

---

## 📚 Referanslar

- [Mermaid Documentation](https://mermaid.js.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [LaTeX Figure Guide](https://www.overleaf.com/learn/latex/Inserting_Images)

---

*Son Güncelleme: 26 Kasım 2025*



