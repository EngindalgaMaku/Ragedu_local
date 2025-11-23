#!/usr/bin/env python3
"""
Mevcut oturumdaki chunk'ları silip yeniden oluştur
"""
import requests
import sys
import json
import time

def get_sessions():
    """Tüm session'ları al"""
    try:
        response = requests.get("http://localhost:8000/sessions", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"❌ Error getting sessions: {e}")
        return []

def get_session_chunks(session_id):
    """Session'dan chunk'ları al"""
    try:
        response = requests.get(
            f"http://localhost:8003/sessions/{session_id}/chunks",
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("chunks", [])
        return []
    except Exception as e:
        print(f"❌ Error getting chunks: {e}")
        return []

def reprocess_session(session_id, chunk_size=1000, chunk_overlap=200):
    """Session'ı yeniden işle"""
    try:
        response = requests.post(
            f"http://localhost:8003/sessions/{session_id}/reprocess",
            json={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": "nomic-embed-text"
            },
            timeout=300
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Reprocess error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Error reprocessing: {e}")
        return None

def analyze_chunks(chunks):
    """Chunk'ları analiz et"""
    if len(chunks) < 2:
        print("⚠️ En az 2 chunk gerekli")
        return
    
    print(f"\n{'='*80}")
    print(f"CHUNK ANALİZİ - {len(chunks)} chunk")
    print(f"{'='*80}\n")
    
    issues = []
    
    for i in range(len(chunks) - 1):
        current = chunks[i]
        next_chunk = chunks[i + 1]
        
        current_text = current.get("chunk_text", "").strip()
        next_text = next_chunk.get("chunk_text", "").strip()
        
        current_idx = current.get("chunk_index", i + 1)
        next_idx = next_chunk.get("chunk_index", i + 2)
        
        # Son 100 karakter ile ilk 100 karakteri karşılaştır
        current_end = current_text[-100:] if len(current_text) > 100 else current_text
        next_start = next_text[:100] if len(next_text) > 100 else next_text
        
        # Tam eşleşme kontrolü
        if current_end.strip() == next_start.strip():
            issues.append({
                "type": "exact_duplicate",
                "chunk_pair": (current_idx, next_idx),
                "overlap": current_end[:50] + "..."
            })
        # Kısmi eşleşme kontrolü
        elif current_end.strip() in next_text or next_start.strip() in current_text:
            issues.append({
                "type": "partial_overlap",
                "chunk_pair": (current_idx, next_idx),
                "overlap": current_end[:50] + "..."
            })
    
    if issues:
        print(f"❌ {len(issues)} SORUN BULUNDU:\n")
        for issue in issues[:10]:  # İlk 10'unu göster
            print(f"   Chunk {issue['chunk_pair'][0]} -> {issue['chunk_pair'][1]}: {issue['type']}")
            print(f"   Overlap: {issue['overlap']}\n")
    else:
        print("✅ Chunk'lar düzgün - iç içe geçme yok!\n")
    
    # İlk 3 chunk'ı göster
    print(f"{'='*80}")
    print("İLK 3 CHUNK:")
    print(f"{'='*80}\n")
    
    for i in range(min(3, len(chunks))):
        chunk = chunks[i]
        text = chunk.get("chunk_text", "")
        idx = chunk.get("chunk_index", i + 1)
        
        print(f"--- Chunk {idx} (Length: {len(text)}) ---")
        print(f"Başlangıç: {text[:150]}...")
        print(f"Bitiş: ...{text[-150:]}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python reprocess_chunks.py <session_id>")
        print("\nSession ID'yi frontend'den veya şu komutla bulabilirsiniz:")
        print("  python -c \"import requests; print(requests.get('http://localhost:8000/sessions').json())\"")
        sys.exit(1)
    
    session_id = sys.argv[1]
    
    print(f"🔍 Session: {session_id}")
    print(f"📦 Mevcut chunk'ları kontrol ediyorum...")
    
    old_chunks = get_session_chunks(session_id)
    print(f"✅ {len(old_chunks)} eski chunk bulundu")
    
    print(f"\n🔄 Chunk'ları yeniden oluşturuyorum...")
    print("   (Bu işlem birkaç dakika sürebilir...)\n")
    
    result = reprocess_session(session_id, chunk_size=1000, chunk_overlap=200)
    
    if not result:
        print("❌ Reprocess başarısız!")
        sys.exit(1)
    
    print(f"✅ Reprocess tamamlandı!")
    print(f"   İşlenen chunk sayısı: {result.get('chunks_processed', 0)}")
    
    print(f"\n⏳ Yeni chunk'ların hazır olmasını bekliyorum (5 saniye)...")
    time.sleep(5)
    
    print(f"\n📊 Yeni chunk'ları analiz ediyorum...")
    new_chunks = get_session_chunks(session_id)
    
    if not new_chunks:
        print("❌ Yeni chunk bulunamadı!")
        sys.exit(1)
    
    print(f"✅ {len(new_chunks)} yeni chunk bulundu\n")
    
    analyze_chunks(new_chunks)



