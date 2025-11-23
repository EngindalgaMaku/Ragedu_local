#!/usr/bin/env python3
"""
Chunk Overlap Kontrol Scripti
Gerçek chunk'ları çekip iç içe geçme sorununu kontrol eder
"""
import requests
import sys
from typing import List, Dict

def get_session_chunks(session_id: str) -> List[Dict]:
    """Session'dan chunk'ları çek"""
    try:
        # Document Processing Service'den chunk'ları al
        response = requests.get(
            f"http://localhost:8003/sessions/{session_id}/chunks",
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("chunks", [])
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Error fetching chunks: {e}")
        return []

def analyze_chunk_overlap(chunks: List[Dict]):
    """Chunk'ların iç içe geçip geçmediğini analiz et"""
    if len(chunks) < 2:
        print("⚠️ En az 2 chunk gerekli")
        return
    
    print(f"\n{'='*80}")
    print(f"CHUNK OVERLAP ANALİZİ - {len(chunks)} chunk")
    print(f"{'='*80}\n")
    
    overlap_issues = []
    duplicate_issues = []
    
    for i in range(len(chunks) - 1):
        current = chunks[i]
        next_chunk = chunks[i + 1]
        
        current_text = current.get("chunk_text", "").strip()
        next_text = next_chunk.get("chunk_text", "").strip()
        
        current_idx = current.get("chunk_index", i + 1)
        next_idx = next_chunk.get("chunk_index", i + 2)
        
        # İlk 100 karakteri karşılaştır
        current_end = current_text[-100:] if len(current_text) > 100 else current_text
        next_start = next_text[:100] if len(next_text) > 100 else next_text
        
        # Tam eşleşme kontrolü
        if current_end == next_start:
            duplicate_issues.append({
                "chunk_pair": (current_idx, next_idx),
                "overlap_text": current_end[:50] + "...",
                "type": "exact_duplicate"
            })
        
        # Kısmi eşleşme kontrolü (50+ karakter)
        elif len(current_end) > 50 and len(next_start) > 50:
            # Son 50 karakter ile ilk 50 karakteri karşılaştır
            current_last_50 = current_end[-50:]
            next_first_50 = next_start[:50]
            
            if current_last_50 == next_first_50:
                overlap_issues.append({
                    "chunk_pair": (current_idx, next_idx),
                    "overlap_text": current_last_50,
                    "type": "50_char_overlap"
                })
            elif current_last_50 in next_start or next_first_50 in current_end:
                overlap_issues.append({
                    "chunk_pair": (current_idx, next_idx),
                    "overlap_text": current_last_50[:30] + "...",
                    "type": "partial_overlap"
                })
        
        # Cümle seviyesinde kontrol
        current_sentences = [s.strip() for s in current_text.split('.') if s.strip()]
        next_sentences = [s.strip() for s in next_text.split('.') if s.strip()]
        
        if current_sentences and next_sentences:
            # Son 3 cümle ile ilk 3 cümleyi karşılaştır
            current_last_3 = current_sentences[-3:]
            next_first_3 = next_sentences[:3]
            
            duplicate_sentences = []
            for sent in current_last_3:
                if sent in next_first_3:
                    duplicate_sentences.append(sent[:50] + "..." if len(sent) > 50 else sent)
            
            if duplicate_sentences:
                overlap_issues.append({
                    "chunk_pair": (current_idx, next_idx),
                    "duplicate_sentences": duplicate_sentences,
                    "type": "sentence_overlap"
                })
    
    # Sonuçları göster
    print(f"📊 ANALİZ SONUÇLARI:\n")
    
    if duplicate_issues:
        print(f"❌ TAM EŞLEŞME SORUNLARI: {len(duplicate_issues)}")
        for issue in duplicate_issues[:5]:  # İlk 5'ini göster
            print(f"   Chunk {issue['chunk_pair'][0]} -> {issue['chunk_pair'][1]}:")
            print(f"   Eşleşen: '{issue['overlap_text']}'")
        print()
    
    if overlap_issues:
        print(f"⚠️ İÇ İÇE GEÇME SORUNLARI: {len(overlap_issues)}")
        for issue in overlap_issues[:5]:  # İlk 5'ini göster
            print(f"   Chunk {issue['chunk_pair'][0]} -> {issue['chunk_pair'][1]}:")
            if 'overlap_text' in issue:
                print(f"   Overlap: '{issue['overlap_text']}'")
            if 'duplicate_sentences' in issue:
                print(f"   Tekrar eden cümleler: {len(issue['duplicate_sentences'])}")
                for sent in issue['duplicate_sentences'][:2]:
                    print(f"      - {sent}")
        print()
    
    if not duplicate_issues and not overlap_issues:
        print("✅ İç içe geçme sorunu yok!")
    else:
        print(f"\n❌ TOPLAM SORUN: {len(duplicate_issues) + len(overlap_issues)}")
    
    # İlk 3 chunk'ı detaylı göster
    print(f"\n{'='*80}")
    print("İLK 3 CHUNK DETAYI:")
    print(f"{'='*80}\n")
    
    for i in range(min(3, len(chunks))):
        chunk = chunks[i]
        text = chunk.get("chunk_text", "")
        idx = chunk.get("chunk_index", i + 1)
        
        print(f"--- Chunk {idx} (Length: {len(text)}) ---")
        print(f"Başlangıç: {text[:100]}...")
        print(f"Bitiş: ...{text[-100:]}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python check_chunks_overlap.py <session_id>")
        print("\nÖrnek:")
        print("  python check_chunks_overlap.py 48307ee1deb18ba8b56f85dc53385e1d")
        sys.exit(1)
    
    session_id = sys.argv[1]
    print(f"🔍 Chunk'ları çekiyorum: {session_id}...")
    
    chunks = get_session_chunks(session_id)
    
    if not chunks:
        print("❌ Chunk bulunamadı!")
        sys.exit(1)
    
    print(f"✅ {len(chunks)} chunk bulundu\n")
    
    analyze_chunk_overlap(chunks)



