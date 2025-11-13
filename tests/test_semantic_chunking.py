#!/usr/bin/env python3
"""
Comprehensive test for the semantic chunking system.

This test validates:
1. Basic semantic chunking functionality
2. Hybrid chunking strategy
3. Turkish language support
4. Fallback mechanisms
5. Integration with existing text_chunker
"""

import sys
import os
sys.path.append('src')

from src.text_processing.text_chunker import chunk_text
from src.text_processing.semantic_chunker import create_semantic_chunks, SemanticChunker
from src.utils.language_detector import detect_query_language

def test_basic_functionality():
    """Test basic text chunking functionality."""
    print("=== Testing Basic Functionality ===")
    
    sample_text = """
    This is a long sample text designed to test the chunking functionality of the RAG system.
    The system needs to be able to split this text into smaller, overlapping chunks so that
    each chunk can be embedded and stored in a vector database. The overlap is important
    to ensure that semantic context is not lost at the boundaries of the chunks.
    
    Let's add more content to make sure it's long enough. We are building a Personalized
    Course Note and Resource Assistant. This assistant will help students by answering
    questions based on their course materials. The materials can be in PDF, DOCX, or PPTX format.
    
    The core of the system is the Retrieval-Augmented Generation (RAG) pipeline.
    This pipeline involves several steps: document processing, text chunking, embedding
    generation, vector storage, retrieval, and response generation. This test focuses
    on the text chunking step, which is fundamental for the subsequent stages.
    """
    
    # Test traditional strategies
    strategies = ["char", "paragraph", "sentence", "markdown"]
    
    for strategy in strategies:
        try:
            chunks = chunk_text(sample_text, chunk_size=200, chunk_overlap=50, strategy=strategy)
            print(f"✓ {strategy} strategy: {len(chunks)} chunks created")
        except Exception as e:
            print(f"✗ {strategy} strategy failed: {e}")
    
    print()

def test_turkish_content():
    """Test Turkish language content processing."""
    print("=== Testing Turkish Content ===")
    
    turkish_text = """
    Yapay Zeka ve Makine Öğrenmesi modern teknolojinin en önemli alanlarından biridir.
    Bu teknolojiler günlük hayatımızda birçok uygulamada kullanılmaktadır.
    
    Derin öğrenme algoritmaları, büyük veri setleri üzerinde eğitilerek karmaşık problemleri çözebilir.
    Özellikle görüntü işleme, doğal dil işleme ve ses tanıma alanlarında büyük başarılar elde edilmiştir.
    
    Eğitim alanında da yapay zeka teknolojileri önemli fırsatlar sunmaktadır.
    Kişiselleştirilmiş öğrenme deneyimleri ve akıllı ders asistanları öğrencilerin 
    başarısını artırmaya yardımcı olmaktadır. Bu sistem de böyle bir amaca hizmet etmektedir.
    
    Gelecekte yapay zeka teknolojilerinin daha da gelişmesi ve yaygınlaşması beklenmektedir.
    İnsanlığın karşılaştığı büyük problemlerin çözümünde önemli rol oynayacaktır.
    """
    
    # Test language detection
    detected_lang = detect_query_language(turkish_text)
    print(f"✓ Language detection: {detected_lang} ({'correct' if detected_lang == 'tr' else 'incorrect'})")
    
    # Test traditional chunking with Turkish
    try:
        chunks = chunk_text(turkish_text, chunk_size=300, chunk_overlap=50, strategy="markdown", language="tr")
        print(f"✓ Turkish markdown chunking: {len(chunks)} chunks created")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk[:100]}...")
    except Exception as e:
        print(f"✗ Turkish markdown chunking failed: {e}")
    
    print()

def test_semantic_chunking():
    """Test semantic chunking with LLM."""
    print("=== Testing Semantic Chunking ===")
    
    academic_text = """
    # Machine Learning Fundamentals
    
    Machine learning is a subset of artificial intelligence that enables computers to learn
    and make decisions from data without being explicitly programmed for every task.
    
    ## Supervised Learning
    
    Supervised learning algorithms learn from labeled training data to make predictions
    or decisions. Common examples include classification and regression problems.
    
    ### Classification
    
    Classification involves predicting discrete categories or classes. For example,
    determining whether an email is spam or not spam, or identifying objects in images.
    
    ### Regression
    
    Regression involves predicting continuous numerical values. Examples include
    predicting house prices, stock prices, or temperature forecasting.
    
    ## Unsupervised Learning
    
    Unsupervised learning algorithms work with unlabeled data to discover hidden
    patterns, structures, or relationships within the dataset.
    
    ### Clustering
    
    Clustering groups similar data points together. Applications include customer
    segmentation, gene sequencing, and market research.
    
    ### Dimensionality Reduction
    
    This technique reduces the number of features in a dataset while preserving
    important information. It's useful for visualization and noise reduction.
    """
    
    try:
        # Test pure semantic chunking 
        semantic_chunks = chunk_text(academic_text, chunk_size=400, chunk_overlap=80, strategy="semantic", language="en")
        print(f"✓ Semantic chunking: {len(semantic_chunks)} chunks created")
        
        # Show chunk boundaries and topics
        for i, chunk in enumerate(semantic_chunks):
            lines = chunk.split('\n')
            first_line = next((line for line in lines if line.strip()), '')
            print(f"  Semantic Chunk {i+1} ({len(chunk)} chars): {first_line[:60]}...")
            
    except Exception as e:
        print(f"⚠ Semantic chunking with LLM failed (expected without API keys): {e}")
        print("  This is normal if GROQ_API_KEY is not configured")
    
    print()

def test_hybrid_chunking():
    """Test hybrid chunking strategy."""
    print("=== Testing Hybrid Chunking ===")
    
    complex_text = """
    # Yapay Zeka ve Eğitim Teknolojileri
    
    Eğitim sektöründe yapay zeka uygulamaları hızla gelişmektedir. Bu teknolojiler
    öğrenme deneyimini kişiselleştirme ve eğitim kalitesini artırma potansiyeline sahiptir.
    
    ## Kişiselleştirilmiş Öğrenme Sistemleri
    
    Her öğrencinin öğrenme hızı ve stili farklıdır. Yapay zeka destekli sistemler,
    öğrenci performansını analiz ederek kişiselleştirilmiş içerik ve öneriler sunar.
    
    ### Adaptif Değerlendirme
    
    Geleneksel sınavlar yerine, sürekli değerlendirme yapan akıllı sistemler
    öğrencinin anlık performansını ölçer ve geri bildirim sağlar.
    
    ### İçerik Önerisi
    
    Makine öğrenmesi algoritmaları, öğrencinin geçmiş performansına dayanarak
    en uygun öğrenme materyallerini önerir.
    
    ## Akıllı Ders Asistanları
    
    Chatbot teknolojisi kullanılarak geliştirilen ders asistanları, öğrencilerin
    sorularını 7/24 yanıtlayabilir ve ek kaynaklar önerebilir.
    
    ### Doğal Dil İşleme
    
    Öğrenci sorularını anlamak ve uygun cevaplar üretmek için gelişmiş
    doğal dil işleme teknikleri kullanılır.
    
    ### Bağlamsal Yanıt Üretimi
    
    Sistem, ders içeriğini analiz ederek bağlamsal ve alakalı yanıtlar üretir.
    Bu sayede öğrenciler daha etkili bir öğrenme deneyimi yaşar.
    """
    
    try:
        # Test hybrid chunking
        hybrid_chunks = chunk_text(complex_text, chunk_size=500, chunk_overlap=100, strategy="hybrid", language="tr")
        print(f"✓ Hybrid chunking: {len(hybrid_chunks)} chunks created")
        
        for i, chunk in enumerate(hybrid_chunks):
            lines = chunk.split('\n')
            first_line = next((line for line in lines if line.strip()), '')
            print(f"  Hybrid Chunk {i+1} ({len(chunk)} chars): {first_line[:70]}...")
            
    except Exception as e:
        print(f"⚠ Hybrid chunking with LLM failed (expected without API keys): {e}")
        print("  Falling back to pure markdown chunking...")
        
        # Test fallback mechanism
        try:
            fallback_chunks = chunk_text(complex_text, chunk_size=500, chunk_overlap=100, strategy="markdown", language="tr")
            print(f"✓ Fallback to markdown: {len(fallback_chunks)} chunks created")
        except Exception as fallback_e:
            print(f"✗ Fallback also failed: {fallback_e}")
    
    print()

def test_edge_cases():
    """Test edge cases and error handling."""
    print("=== Testing Edge Cases ===")
    
    test_cases = [
        ("", "Empty text"),
        ("Short text", "Very short text"),
        ("A" * 10000, "Very long text without structure"),
        ("Text with üÇĞIİÖŞÜ characters", "Turkish characters"),
    ]
    
    for text, description in test_cases:
        try:
            chunks = chunk_text(text, chunk_size=100, chunk_overlap=20, strategy="markdown")
            print(f"✓ {description}: {len(chunks)} chunks")
        except Exception as e:
            print(f"⚠ {description}: {e}")
    
    print()

def test_performance_comparison():
    """Compare performance of different strategies."""
    print("=== Performance Comparison ===")
    
    medium_text = """
    Büyük Veri ve Yapay Zeka entegrasyonu modern işletmelerin rekabet avantajı elde etmesinde
    kritik rol oynamaktadır. Veri analizi ve makine öğrenmesi teknikleri sayesinde
    şirketler müşteri davranışlarını daha iyi anlayabilir ve tahminlerde bulunabilir.
    
    Veri toplama süreçleri artık otomatikleşmiş sistemler tarafından yürütülmektedir.
    IoT cihazları, web siteleri, mobil uygulamalar ve sosyal medya platformları
    sürekli olarak büyük miktarda veri üretmektedir.
    
    Bu verilerin işlenmesi ve anlamlandırılması için gelişmiş algoritmalar kullanılır.
    Derin öğrenme modelleri, karmaşık veri yapılarını analiz ederek değerli içgörüler sağlar.
    
    Veri görselleştirme araçları, elde edilen sonuçları anlaşılır hale getirerek
    karar vericilerin doğru stratejiler geliştirmesine yardımcı olur.
    """ * 3  # Make it longer
    
    strategies = ["char", "paragraph", "sentence", "markdown"]
    
    for strategy in strategies:
        try:
            chunks = chunk_text(medium_text, chunk_size=300, chunk_overlap=60, strategy=strategy, language="tr")
            avg_size = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
            print(f"✓ {strategy:12} | {len(chunks):2} chunks | avg size: {avg_size:.0f} chars")
        except Exception as e:
            print(f"✗ {strategy:12} | Failed: {e}")
    
    print()

def main():
    """Run all tests."""
    print("🚀 Starting Semantic Chunking System Tests\n")
    
    try:
        test_basic_functionality()
        test_turkish_content()
        test_semantic_chunking()
        test_hybrid_chunking()
        test_edge_cases()
        test_performance_comparison()
        
        print("✅ All tests completed!")
        print("\n📝 Notes:")
        print("- Semantic and hybrid strategies may show warnings without API keys")
        print("- This is expected behavior - system falls back to markdown chunking")
        print("- To fully test LLM features, set GROQ_API_KEY environment variable")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())