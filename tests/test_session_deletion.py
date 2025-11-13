#!/usr/bin/env python3
"""
Session Silme Butonunu Test Et
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.session_manager import (
    professional_session_manager, SessionCategory, SessionStatus
)

def test_session_deletion():
    """Session silme işlemini test et"""
    print("🧪 Session silme işlemini test ediyoruz...")
    
    # Test session oluştur
    try:
        session_metadata = professional_session_manager.create_session(
            name="Test Silme Oturumu",
            description="Bu oturum silinecek",
            category=SessionCategory.GENERAL,
            created_by="test_user",
            tags=["test", "delete"]
        )
        
        session_id = session_metadata.session_id
        print(f"✅ Test session oluşturuldu: {session_id}")
        
        # Session'u listele
        sessions = professional_session_manager.list_sessions(created_by="test_user")
        print(f"📋 Toplam session sayısı: {len(sessions)}")
        
        # Test vector dosyaları oluştur (silme testi için)
        vector_dir = Path("data/vector_db/sessions")
        vector_dir.mkdir(parents=True, exist_ok=True)
        
        test_files = [
            f"{vector_dir}/{session_id}.index",
            f"{vector_dir}/{session_id}.chunks", 
            f"{vector_dir}/{session_id}.meta.jsonl"
        ]
        
        for file_path in test_files:
            Path(file_path).write_text("test data")
            
        print(f"📁 Test vector dosyaları oluşturuldu")
        
        # Session'u sil
        print(f"🗑️ Session siliniyor...")
        success = professional_session_manager.delete_session(
            session_id, create_backup=True, deleted_by="test_user"
        )
        
        if success:
            print("✅ Session başarıyla silindi!")
            
            # Dosyaların silindiğini kontrol et
            remaining_files = [f for f in test_files if Path(f).exists()]
            if not remaining_files:
                print("✅ Tüm vector dosyalar başarıyla silindi")
            else:
                print(f"⚠️ Bazı dosyalar silinemedi: {remaining_files}")
            
            # Database'den silindiğini kontrol et
            deleted_session = professional_session_manager.get_session_metadata(session_id)
            if deleted_session is None:
                print("✅ Session database'den başarıyla silindi")
            else:
                print("❌ Session hala database'de mevcut")
                
        else:
            print("❌ Session silme işlemi başarısız!")
            
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_session_deletion()