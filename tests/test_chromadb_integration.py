#!/usr/bin/env python3
"""
ChromaDB Integration Test

Bu test dosyası ChromaDB entegrasyonunun çalışıp çalışmadığını kontrol eder:
1. ChromaDB container'ına bağlantı testi
2. ChromaVectorStore sınıfının temel işlevlerini test
3. RAG pipeline ile entegrasyon testi
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

def test_chromadb_connection():
    """ChromaDB container'ına bağlantı testi"""
    print("🧪 ChromaDB Bağlantı Testi")
    print("=" * 50)
    
    try:
        from src.vector_store.chroma_store import ChromaVectorStore
        
        # Test basic connection
        store = ChromaVectorStore(collection_name="test_connection")
        print("✅ ChromaDB container'ına başarıyla bağlanıldı")
        
        # Get collection info
        info = store.get_collection_info()
        print(f"📊 Collection bilgisi: {info}")
        
        # Clean up test collection
        store.delete_collection()
        print("🧹 Test collection'ı temizlendi")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB bağlantı hatası: {e}")
        print("💡 ChromaDB container'ının çalıştığından emin olun (docker-compose up -d)")
        return False

def test_vector_store_operations():
    """ChromaVectorStore temel işlemleri testi"""
    print("\n🧪 Vector Store İşlemleri Testi")
    print("=" * 50)
    
    try:
        from src.vector_store.chroma_store import ChromaVectorStore
        
        # Create test store
        store = ChromaVectorStore(collection_name="test_operations")
        
        # Test data
        test_texts = [
            "Python programlama dili nesne yönelimli bir dildir.",
            "Machine Learning yapay zeka alanının önemli bir dalıdır.",
            "ChromaDB vektör veritabanı sistemidir."
        ]
        
        # Generate dummy embeddings (384 dimensions - ChromaDB default)
        test_embeddings = np.random.rand(len(test_texts), 384).astype('float32').tolist()
        
        # Test metadata
        test_metadata = [
            {"source": "test1.txt", "topic": "programming"},
            {"source": "test2.txt", "topic": "ai"},
            {"source": "test3.txt", "topic": "database"}
        ]
        
        print(f"📝 {len(test_texts)} adet test döküman ekleniyor...")
        
        # Add documents
        store.add_documents(test_texts, test_embeddings, test_metadata)
        
        print(f"📊 Toplam döküman sayısı: {store.ntotal}")
        
        # Test search
        print("🔍 Benzerlik araması test ediliyor...")
        query_embedding = np.random.rand(384).astype('float32').tolist()
        
        results = store.search(query_embedding, k=2)
        
        print(f"🔍 Arama sonuçları ({len(results)} adet):")
        for i, (text, score, metadata) in enumerate(results):
            print(f"  {i+1}. Skor: {score:.4f}")
            print(f"     Metin: {text[:50]}...")
            print(f"     Metadata: {metadata}")
        
        # Clean up
        store.delete_collection()
        print("🧹 Test collection'ı temizlendi")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store işlem hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_pipeline_integration():
    """RAG pipeline ile ChromaDB entegrasyon testi"""
    print("\n🧪 RAG Pipeline Entegrasyon Testi")
    print("=" * 50)
    
    try:
        # Test config
        test_config = {
            "ollama_embedding_model": "mixedbread-ai/mxbai-embed-large-v1",
            "ollama_generation_model": "llama-3.1-8b-instant",
            "temperature": 0.7,
            "top_k": 5,
            "enable_cache": False,  # Disable cache for testing
            "enable_reranking": False  # Disable reranking for simple test
        }
        
        from src.vector_store.chroma_store import ChromaVectorStore
        from src.rag.rag_pipeline import RAGPipeline
        
        # Create test store
        chroma_store = ChromaVectorStore(collection_name="test_rag_pipeline")
        
        # Test documents
        test_docs = [
            "Türkiye Cumhuriyeti Anayasası 1982 yılında kabul edilmiştir.",
            "Python programlama dili Guido van Rossum tarafından geliştirilmiştir.",
            "Yapay zeka teknolojileri günümüzde hızla gelişmektedir."
        ]
        
        # Generate embeddings with sentence transformers (more realistic)
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(test_docs).tolist()
            print("✅ SentenceTransformers ile embedding oluşturuldu")
        except ImportError:
            print("⚠️ SentenceTransformers bulunamadı, dummy embedding kullanılıyor")
            embeddings = np.random.rand(len(test_docs), 384).astype('float32').tolist()
        
        # Add documents to store
        chroma_store.add_documents(test_docs, embeddings)
        print(f"📚 {len(test_docs)} döküman ChromaDB'ye eklendi")
        
        # Create RAG pipeline
        rag_pipeline = RAGPipeline(test_config, chroma_store)
        print("🔧 RAG Pipeline oluşturuldu")
        
        # Test query
        test_query = "Python kimden geliştirdi?"
        print(f"❓ Test sorgusu: '{test_query}'")
        
        # Retrieve documents (test retrieval only, skip generation for now)
        retrieved_docs = rag_pipeline.retrieve(test_query, top_k=2)
        print(f"📖 {len(retrieved_docs)} döküman bulundu:")
        
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. Skor: {doc.get('score', 'N/A')}")
            print(f"     Metin: {doc.get('text', '')[:100]}...")
        
        # Clean up
        chroma_store.delete_collection()
        print("🧹 Test collection'ı temizlendi")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG pipeline entegrasyon hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_based_collections():
    """Session-based collection yönetimi testi"""
    print("\n🧪 Session-based Collection Testi")
    print("=" * 50)
    
    try:
        from src.vector_store.chroma_store import ChromaVectorStore
        
        # Test different session collections
        session_ids = ["test_session_1", "test_session_2"]
        
        for session_id in session_ids:
            collection_name = f"session_{session_id}"
            store = ChromaVectorStore(collection_name=collection_name)
            
            # Add unique data for each session
            texts = [f"Bu {session_id} için özel veri {i}" for i in range(2)]
            embeddings = np.random.rand(len(texts), 384).astype('float32').tolist()
            
            store.add_documents(texts, embeddings)
            print(f"📂 {collection_name}: {store.ntotal} döküman")
            
            # Verify isolation - search should only return docs from this session
            query_embedding = np.random.rand(384).astype('float32').tolist()
            results = store.search(query_embedding, k=5)
            
            session_specific = all(session_id in result[0] for result in results)
            if session_specific:
                print(f"✅ {collection_name} izolasyonu başarılı")
            else:
                print(f"⚠️ {collection_name} izolasyon problemi")
        
        # Clean up test collections
        for session_id in session_ids:
            collection_name = f"session_{session_id}"
            store = ChromaVectorStore(collection_name=collection_name)
            store.delete_collection()
        
        print("🧹 Tüm test collection'ları temizlendi")
        return True
        
    except Exception as e:
        print(f"❌ Session collection testi hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ana test fonksiyonu"""
    print("🚀 ChromaDB Entegrasyon Test Paketi")
    print("=" * 50)
    print("Bu test ChromaDB container'ının çalıştığını varsayar.")
    print("Önce şu komutu çalıştırın: docker-compose up -d chromadb")
    print()
    
    test_results = []
    
    # Test 1: Connection
    test_results.append(test_chromadb_connection())
    
    # Test 2: Vector Store Operations (only if connection successful)
    if test_results[-1]:
        test_results.append(test_vector_store_operations())
    else:
        print("\n⏭️  Vector store testleri atlanıyor (bağlantı başarısız)")
        test_results.append(False)
    
    # Test 3: RAG Pipeline Integration (only if previous tests successful)
    if all(test_results):
        test_results.append(test_rag_pipeline_integration())
    else:
        print("\n⏭️  RAG pipeline testleri atlanıyor (önceki testler başarısız)")
        test_results.append(False)
    
    # Test 4: Session Collections (only if basic operations work)
    if test_results[1]:  # Vector store operations successful
        test_results.append(test_session_based_collections())
    else:
        print("\n⏭️  Session collection testleri atlanıyor")
        test_results.append(False)
    
    # Summary
    print("\n📋 TEST SONUÇLARI")
    print("=" * 50)
    tests = [
        "ChromaDB Bağlantısı",
        "Vector Store İşlemleri", 
        "RAG Pipeline Entegrasyonu",
        "Session Collection Yönetimi"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (test_name, result) in enumerate(zip(tests, test_results)):
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{i+1}. {test_name}: {status}")
    
    print(f"\nTOPLAM: {passed}/{total} test başarılı")
    
    if passed == total:
        print("🎉 Tüm testler başarılı! ChromaDB entegrasyonu hazır.")
        return 0
    else:
        print("⚠️ Bazı testler başarısız. Lütfen hataları kontrol edin.")
        return 1

if __name__ == "__main__":
    sys.exit(main())