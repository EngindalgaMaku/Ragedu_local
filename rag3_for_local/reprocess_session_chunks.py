#!/usr/bin/env python3
"""
Session'daki chunk'ları silip yeniden oluştur
"""
import requests
import sys
import time
import chromadb
from chromadb.config import Settings

def delete_chunks_from_chromadb(session_id):
    """ChromaDB'den chunk'ları sil"""
    try:
        client = chromadb.HttpClient(
            host="localhost",
            port=8004,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        # Collection name formatları
        collection_names = [
            session_id,
            f"session_{session_id}",
            f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:]}"
        ]
        
        # Timestamped versions için pattern
        all_collections = client.list_collections()
        for coll in all_collections:
            for pattern in collection_names:
                if coll.name.startswith(pattern + "_") or coll.name == pattern:
                    try:
                        collection = client.get_collection(coll.name)
                        # Delete all chunks
                        results = collection.get()
                        if results['ids']:
                            collection.delete(ids=results['ids'])
                            print(f"✅ Deleted {len(results['ids'])} chunks from {coll.name}")
                    except Exception as e:
                        print(f"⚠️ Error deleting from {coll.name}: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Error deleting chunks: {e}")
        return False

def reprocess_via_api(session_id):
    """API üzerinden reprocess"""
    # Önce chunk'ları sil
    print(f"🗑️  Eski chunk'ları siliyorum...")
    delete_chunks_from_chromadb(session_id)
    
    # Sonra dosyaları yeniden işle (bu kısım için API endpoint gerekli)
    # Şimdilik sadece chunk'ları sildik, kullanıcı frontend'den yeniden yükleyebilir
    print(f"✅ Chunk'lar silindi. Lütfen frontend'den dosyaları yeniden yükleyin.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python reprocess_session_chunks.py <session_id>")
        sys.exit(1)
    
    session_id = sys.argv[1]
    reprocess_via_api(session_id)



