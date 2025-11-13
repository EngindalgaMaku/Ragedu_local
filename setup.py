#!/usr/bin/env python3
"""
Akıllı Kütüphane RAG Sistemi - Kurulum Script
Tüm gerekli paketleri yükler ve sistemi hazırlar
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Komut çalıştır ve sonucu göster"""
    print(f"\n🔧 {description}")
    print(f"Çalıştırılan komut: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ Başarılı: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Hata: {e}")
        if e.stderr:
            print(f"Hata detayı: {e.stderr.strip()}")
        return False

def main():
    print("🚀 Akıllı Kütüphane RAG Sistemi Kurulum Başlatılıyor...")
    
    # Python versiyonunu kontrol et
    python_version = sys.version_info
    print(f"🐍 Python sürümü: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ gerekli! Lütfen Python'ı güncelleyin.")
        sys.exit(1)
    
    # Pip'i güncelle
    print("\n📦 Pip güncelleniyor...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Pip güncelleme")
    
    # Requirements yükle
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        print("\n📋 Gereksinimler yükleniyor...")
        success = run_command(
            f"{sys.executable} -m pip install -r requirements.txt", 
            "Python paketleri yükleme"
        )
        if not success:
            print("❌ Paket yükleme başarısız! Manuel yükleme deneyin:")
            print(f"   pip install -r {requirements_file}")
            sys.exit(1)
    else:
        print("❌ requirements.txt bulunamadı!")
        sys.exit(1)
    
    # Ollama'nın yüklenip yüklenmediğini kontrol et
    print("\n🦙 Ollama kontrolü...")
    ollama_check = run_command("ollama --version", "Ollama versiyonu kontrolü")
    if not ollama_check:
        print("⚠️  Ollama yüklü değil veya PATH'de değil!")
        print("📖 Ollama kurulumu için:")
        print("   - Windows: https://ollama.com/download/windows")
        print("   - macOS: brew install ollama")
        print("   - Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("\n🔄 Ollama kurduktan sonra şu modelleri indirin:")
        print("   ollama pull qwen2.5:14b")
        print("   ollama pull mxbai-embed-large")
    else:
        # Gerekli modelleri kontrol et
        print("\n📥 Gerekli modeller kontrol ediliyor...")
        models_to_check = ["qwen2.5:14b", "mxbai-embed-large"]
        for model in models_to_check:
            model_check = run_command(f"ollama show {model}", f"{model} modeli kontrolü")
            if not model_check:
                print(f"📥 {model} modeli indiriliyor...")
                run_command(f"ollama pull {model}", f"{model} modeli indirme")
    
    # Klasör yapısını kontrol et
    print("\n📁 Proje klasörleri kontrol ediliyor...")
    required_dirs = [
        "data/uploads", 
        "data/vector_db", 
        "data/cache", 
        "logs", 
        "docs"
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path} klasörü hazır")
    
    # .env dosyasını kontrol et
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("\n⚙️  .env dosyası oluşturuluyor...")
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ .env dosyası .env.example'dan kopyalandı")
    
    print("\n🎉 Kurulum tamamlandı!")
    print("\n🚀 Uygulamayı çalıştırmak için:")
    print("   streamlit run app.py")
    print("\n📖 Daha fazla bilgi için README.md dosyasını inceleyin.")

if __name__ == "__main__":
    main()