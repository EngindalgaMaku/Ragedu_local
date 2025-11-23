#!/usr/bin/env python3
"""
Gerçek chunk'ları çekip analiz et
"""
import requests
import sys
import json

def get_chunks(session_id):
    """Chunk'ları çek"""
    try:
        response = requests.get(
            f"http://localhost:8003/sessions/{session_id}/chunks",
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("chunks", [])
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return []
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python debug_chunks.py <session_id>")
        print("\nSession ID'yi frontend'den veya API'den alabilirsiniz.")
        sys.exit(1)
    
    session_id = sys.argv[1]
    print(f"🔍 Chunk'ları çekiyorum: {session_id}\n")
    
    chunks = get_chunks(session_id)
    
    if not chunks:
        print("❌ Chunk bulunamadı!")
        sys.exit(1)
    
    print(f"✅ {len(chunks)} chunk bulundu\n")
    print("="*80)
    print("İLK 5 CHUNK ANALİZİ:")
    print("="*80)
    
    for i in range(min(5, len(chunks))):
        chunk = chunks[i]
        text = chunk.get("chunk_text", "")
        idx = chunk.get("chunk_index", i + 1)
        
        print(f"\n--- Chunk {idx} (Length: {len(text)}) ---")
        print(f"BAŞLANGIÇ (ilk 150 karakter):")
        print(text[:150])
        print(f"\nBİTİŞ (son 150 karakter):")
        print(text[-150:])
        
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            next_text = next_chunk.get("chunk_text", "")
            print(f"\n--- Chunk {idx+1} BAŞLANGICI (ilk 150 karakter) ---")
            print(next_text[:150])
            
            # Overlap kontrolü
            current_end = text[-100:].strip()
            next_start = next_text[:100].strip()
            
            if current_end == next_start:
                print(f"\n❌ TAM EŞLEŞME! Chunk {idx} sonu ile Chunk {idx+1} başı aynı!")
            elif current_end in next_text or next_start in text:
                print(f"\n⚠️ KISMI EŞLEŞME tespit edildi")
                if len(current_end) > 50:
                    print(f"   Overlap: {current_end[:50]}...")



