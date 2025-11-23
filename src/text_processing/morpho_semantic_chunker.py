"""
Hybrid Morpho-Semantic Chunker for Turkish RAG Systems
------------------------------------------------------
Bu modül, Markdown tabanlı dökümanlar için geliştirilmiş hibrit bir parçalama (chunking) sistemidir.
Üç temel yaklaşımı birleştirir:

1. YAPISAL (Structural): Markdown başlık hiyerarşisini ve listeleri asla bozmaz.
2. SEMANTİK (Semantic): Uzun blokları cümleler arası anlamsal benzerliğe göre böler.
3. MORFOLOJİK (Morphological): Chunk içindeki kelimelerin köklerini (stem) çıkarır ve
   vektör araması için 'zenginleştirilmiş bağlam' (enriched context) oluşturur.

Akademik Değer:
"Sondan eklemeli dillerde (Türkçe) morfolojik köklerin RAG erişim başarımına etkisi."
"""

import re
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import hashlib

# Embedding için (Opsiyonel, yoksa sadece structural çalışır)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

# Yapısal analiz için mevcut modüllerden faydalanabiliriz veya temiz bir yapı kurabiliriz.
# Burada bağımsız ve taşınabilir bir sınıf oluşturuyoruz.

@dataclass
class EnrichedChunk:
    """Zenginleştirilmiş Chunk Yapısı"""
    content: str              # Chunk'ın ham metni (kullanıcıya gösterilen)
    search_content: str       # Aramada kullanılacak zengin metin (Kökler + Metin)
    metadata: Dict            # Başlık yolu, sayfa no vb.
    chunk_id: str
    token_count: int
    split_reason: str         # 'structure', 'semantic', 'size_limit'

class TurkishStemmer:
    """
    Hafifletilmiş Türkçe Gövdeleme (Stemming) Motoru.
    Zemberek gibi ağır kütüphaneler yerine, RAG için yeterli hassasiyette
    regex tabanlı kural seti kullanır.
    """
    def __init__(self):
        # Basit ve etkili ek atma kuralları (Order matters!)
        self.suffixes = [
            # Çekim Ekleri
            r'casına$', r'cesine$', r'çesine$',  # Eşitlik
            r'lardan$', r'lerden$', r'larda$', r'lerde$',  # Çokluk + Hal
            r'daki$', r'deki$', r'taki$', r'teki$', # İlgi
            r'ımız$', r'imiz$', r'umuz$', r'ümüz$', # İyelik
            r'nız$', r'niz$', r'nuz$', r'nüz$', 
            r'lar$', r'ler$',                       # Çokluk
            r'dan$', r'den$', r'tan$', r'ten$',     # Ayrılma
            r'da$', r'de$', r'ta$', r'te$',         # Bulunma
            r'ı$', r'i$', r'u$', r'ü$',             # Belirtme (dikkatli olmalı)
            r'a$', r'e$',                           # Yönelme
            r'nın$', r'nin$', r'nun$', r'nün$',     # Tamlayan
            r'le$', r'la$',                         # Vasıta
            # Fiil Çekimleri (Basit)
            r'yor$', r'mek$', r'mak$', r'dı$', r'di$', r'tı$', r'ti$'
        ]
        # Stop words (Etkisiz kelimeler)
        self.stop_words = {
            've', 'veya', 'ile', 'için', 'bir', 'bu', 'şu', 'o', 'da', 'de',
            'ki', 'mi', 'mı', 'ama', 'fakat', 'lakin', 'ancak', 'gibi', 'kadar'
        }

    def stem_word(self, word: str) -> str:
        word = word.lower().strip()
        if len(word) < 4 or word in self.stop_words: # Çok kısa kelimelere dokunma
            return word
        
        original = word
        for suffix in self.suffixes:
            if re.search(suffix, word):
                # Eki at, ancak kök çok kısalırsa (örn < 3 harf) iptal et
                potential_stem = re.sub(suffix, '', word)
                if len(potential_stem) >= 3:
                    word = potential_stem
                    # Bir kelimede birden fazla ek olabilir, döngü devam etsin mi?
                    # Basit stemming için genelde en dıştaki ekleri atmak yeterlidir,
                    # ama Türkçe'de "kitap-lar-ım-da" gibi zincirleme vardır.
                    # Şimdilik tek pas geçiyoruz (agresif olmamak için).
                    break 
        return word

    def extract_stems(self, text: str) -> str:
        """Metindeki kelimelerin köklerini çıkarır ve boşlukla birleştirir."""
        words = re.findall(r'\w+', text.lower())
        stems = [self.stem_word(w) for w in words if w not in self.stop_words]
        # Tekrar edenleri temizle (Set) ama sırayı koru
        unique_stems = []
        seen = set()
        for s in stems:
            if s not in seen:
                unique_stems.append(s)
                seen.add(s)
        return " ".join(unique_stems)

class MorphoSemanticChunker:
    def __init__(
        self, 
        embedding_model_name: str = "all-MiniLM-L6-v2",
        max_chunk_size: int = 1000,
        min_chunk_size: int = 200,
        semantic_threshold: float = 0.75
    ):
        self.stemmer = TurkishStemmer()
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.semantic_threshold = semantic_threshold
        self.logger = logging.getLogger(__name__)

        self.model = None
        if EMBEDDING_AVAILABLE:
            try:
                self.logger.info(f"Loading embedding model: {embedding_model_name}")
                self.model = SentenceTransformer(embedding_model_name)
            except Exception as e:
                self.logger.warning(f"Model yüklenemedi, sadece yapısal çalışacak: {e}")

    def process_markdown(self, markdown_text: str) -> List[EnrichedChunk]:
        """
        Markdown metnini alır, işler ve zenginleştirilmiş chunklar döner.
        Marker kütüphanesinden gelen çıktı ile uyumludur.
        """
        # 1. Adım: Yapısal Bölümleme (Structural Splitting)
        sections = self._split_by_headers(markdown_text)
        
        final_chunks = []
        
        for section in sections:
            header_path = section['header_path'] # Örn: "Giriş > Amaç"
            content = section['content']
            
            # 2. Adım: Boyut ve Semantik Kontrol
            if len(content) > self.max_chunk_size:
                # Büyük bloğu semantik olarak böl
                sub_chunks = self._semantic_split(content)
            else:
                # Küçükse olduğu gibi al
                sub_chunks = [content]
            
            # 3. Adım: Morfolojik Zenginleştirme ve Paketleme
            for sub_content in sub_chunks:
                if len(sub_content.strip()) < 20: continue # Çöp veriyi at
                
                stems = self.stemmer.extract_stems(sub_content)
                
                # Search content: Orjinal metin + [Kökler] + Başlık Bağlamı
                # Bu format Vektör DB için optimize edilmiştir.
                search_content = (
                    f"BAŞLIK: {header_path}\n"
                    f"İÇERİK: {sub_content}\n"
                    f"ANAHTAR KAVRAMLAR: {stems}"
                )
                
                chunk_id = hashlib.md5(search_content.encode()).hexdigest()
                
                final_chunks.append(EnrichedChunk(
                    content=sub_content, # Kullanıcıya gösterilecek temiz metin
                    search_content=search_content, # Embedding'e girecek kirli ama zengin metin
                    metadata={
                        "header_path": header_path,
                        "source": "morpho_semantic"
                    },
                    chunk_id=chunk_id,
                    token_count=len(sub_content.split()), # Yaklaşık token
                    split_reason="semantic" if len(sub_chunks) > 1 else "structure"
                ))
                
        return final_chunks

    def _split_by_headers(self, text: str) -> List[Dict]:
        """
        Markdown başlıklarına göre metni hiyerarşik olarak böler.
        Marker çıktısı genellikle '#' ile başlar.
        """
        lines = text.split('\n')
        sections = []
        current_path = []
        current_content = []
        
        for line in lines:
            if line.startswith('#'):
                # Mevcut bölümü kaydet
                if current_content:
                    sections.append({
                        'header_path': " > ".join(current_path) if current_path else "Root",
                        'content': "\n".join(current_content).strip()
                    })
                    current_content = []
                
                # Yeni başlık seviyesini algıla
                level = len(line.split()[0])
                title = line.strip('# ').strip()
                
                # Path güncelleme mantığı (Stack yapısı)
                if level > len(current_path):
                    current_path.append(title)
                else:
                    current_path = current_path[:level-1]
                    current_path.append(title)
            else:
                current_content.append(line)
                
        # Son bölümü ekle
        if current_content:
            sections.append({
                'header_path': " > ".join(current_path) if current_path else "Root",
                'content': "\n".join(current_content).strip()
            })
            
        return sections

    def _semantic_split(self, text: str) -> List[str]:
        """
        Uzun metni cümlelere böler ve anlamsal bütünlüğe göre gruplar.
        Eğer model yoksa, sadece karaktere göre böler (Fallback).
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not self.model or len(sentences) < 2:
            # Basit bölme (Recursive Character Splitter mantığına yakın)
            return self._simple_split(text)
            
        # Cümle embeddingleri
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk_sents = [sentences[0]]
        current_chunk_emb = [embeddings[0]]
        
        for i in range(1, len(sentences)):
            sent = sentences[i]
            emb = embeddings[i]
            
            # Önceki cümlelerin ortalaması ile şimdiki cümlenin benzerliği
            context_emb = np.mean(current_chunk_emb, axis=0).reshape(1, -1)
            curr_emb = emb.reshape(1, -1)
            sim = cosine_similarity(context_emb, curr_emb)[0][0]
            
            current_len = sum(len(s) for s in current_chunk_sents)
            
            # BÖLME KARARI:
            # 1. Anlamsal benzerlik düşükse (Konu değişti)
            # 2. VEYA Chunk çok büyüdüyse
            if (sim < self.semantic_threshold and current_len > self.min_chunk_size) or \
               (current_len > self.max_chunk_size):
                
                chunks.append(" ".join(current_chunk_sents))
                current_chunk_sents = [sent]
                current_chunk_emb = [emb]
            else:
                current_chunk_sents.append(sent)
                current_chunk_emb.append(emb)
                
        if current_chunk_sents:
            chunks.append(" ".join(current_chunk_sents))
            
        return chunks

    def _simple_split(self, text: str) -> List[str]:
        """Embedding olmadığında kullanılan basit bölücü."""
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        
        for w in words:
            current.append(w)
            current_len += len(w) + 1
            if current_len >= self.max_chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
        if current:
            chunks.append(" ".join(current))
        return chunks

if __name__ == "__main__":
    # Hızlı Test
    test_md = """
# Türkiye Coğrafyası
Türkiye, Asya ve Avrupa kıtaları arasında bir köprüdür.
## İklim
Ülkede üç ana iklim tipi görülür. Karadeniz iklimi boldur.
Akdeniz iklimi sıcak ve kuraktır.
### Bitki Örtüsü
Karadeniz'de ormanlar yaygındır. Akdeniz'de makiler bulunur.
    """
    
    chunker = MorphoSemanticChunker(max_chunk_size=100) # Test için küçük limit
    result = chunker.process_markdown(test_md)
    
    print(f"Toplam Chunk: {len(result)}")
    for i, c in enumerate(result):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Path: {c.metadata['header_path']}")
        print(f"Content: {c.content}")
        print(f"Search Context (Enriched): {c.search_content}")











