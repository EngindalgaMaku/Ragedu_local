"""
RAG Altyapısı Görselleştirme Scripti
Makale için yüksek kaliteli diagramlar oluşturur
"""

import os
import subprocess
from pathlib import Path

def check_mermaid_cli():
    """Mermaid CLI'nin yüklü olup olmadığını kontrol et"""
    try:
        result = subprocess.run(['mmdc', '--version'], 
                              capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False

def install_mermaid_cli_instructions():
    """Mermaid CLI kurulum talimatları"""
    print("""
    ⚠️  Mermaid CLI yüklü değil!
    
    Kurulum için:
    
    1. Node.js yüklü olmalı (https://nodejs.org/)
    
    2. Mermaid CLI'yi global olarak yükleyin:
       npm install -g @mermaid-js/mermaid-cli
    
    3. Veya Docker kullanın:
       docker pull minlag/mermaid-cli
    """)

def generate_svg_from_mermaid(mermaid_file, output_dir="diagrams"):
    """Mermaid dosyasından SVG oluştur"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not check_mermaid_cli():
        install_mermaid_cli_instructions()
        return False
    
    # Mermaid dosyasını oku
    with open(mermaid_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Mermaid diagramlarını bul
    import re
    diagram_pattern = r'```mermaid\n(.*?)\n```'
    diagrams = re.findall(diagram_pattern, content, re.DOTALL)
    
    print(f"📊 {len(diagrams)} diagram bulundu")
    
    for i, diagram in enumerate(diagrams, 1):
        # Geçici mermaid dosyası oluştur
        temp_file = f"temp_diagram_{i}.mmd"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(diagram)
        
        # SVG oluştur
        output_file = output_path / f"diagram_{i}.svg"
        try:
            subprocess.run(['mmdc', '-i', temp_file, '-o', str(output_file), 
                          '--width', '1920', '--height', '1080',
                          '--backgroundColor', 'white'],
                         check=True)
            print(f"✅ {output_file} oluşturuldu")
        except subprocess.CalledProcessError as e:
            print(f"❌ Hata: {e}")
        finally:
            # Geçici dosyayı sil
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    return True

def generate_png_from_svg(svg_dir="diagrams", png_dir="diagrams/png"):
    """SVG'leri PNG'ye dönüştür (yüksek çözünürlük)"""
    try:
        import cairosvg
    except ImportError:
        print("""
        ⚠️  cairosvg yüklü değil!
        
        PNG export için:
        pip install cairosvg
        """)
        return False
    
    svg_path = Path(svg_dir)
    png_path = Path(png_dir)
    png_path.mkdir(exist_ok=True, parents=True)
    
    for svg_file in svg_path.glob("*.svg"):
        png_file = png_path / f"{svg_file.stem}.png"
        cairosvg.svg2png(url=str(svg_file), 
                        write_to=str(png_file),
                        output_width=1920,
                        output_height=1080)
        print(f"✅ {png_file} oluşturuldu")
    
    return True

if __name__ == "__main__":
    mermaid_file = "RAG_ALTYAPI_GORSEL_ANALIZ.md"
    
    if not os.path.exists(mermaid_file):
        print(f"❌ {mermaid_file} bulunamadı!")
        exit(1)
    
    print("🎨 RAG Altyapısı Diagramları Oluşturuluyor...\n")
    
    # SVG oluştur
    if generate_svg_from_mermaid(mermaid_file):
        print("\n✅ SVG diagramlar oluşturuldu!")
        
        # PNG oluştur (opsiyonel)
        user_input = input("\nPNG formatında da oluşturmak ister misiniz? (e/h): ")
        if user_input.lower() == 'e':
            generate_png_from_svg()
    
    print("\n📁 Diagramlar 'diagrams' klasöründe!")



