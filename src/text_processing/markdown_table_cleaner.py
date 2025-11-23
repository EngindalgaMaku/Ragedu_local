"""
Markdown Table Cleaner for Better LLM Understanding
----------------------------------------------------
Converts Markdown tables to plain text lists so LLMs can easily parse them.
"""

import re

def clean_markdown_tables(text: str) -> str:
    """
    Markdown tablolarını düz metne çevirir.
    
    Örnek:
    | • Özel giysiler kullan |
    |---|
    
    →
    
    • Özel giysiler kullan
    """
    # Tablo ayırıcı satırlarını kaldır (|---|, |:---|, etc.)
    text = re.sub(r'\n\s*\|[\s\-:]+\|\s*\n', '\n', text)
    
    # Tablo hücrelerindeki pipe'ları kaldır
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Eğer satır tablo formatındaysa (| ile başlıyor/bitiyor)
        if line.strip().startswith('|') and line.strip().endswith('|'):
            # Pipe'ları temizle
            cleaned = line.strip('|').strip()
            # Birden fazla pipe varsa (multi-column) virgülle ayır
            if '|' in cleaned:
                cells = [c.strip() for c in cleaned.split('|') if c.strip()]
                cleaned = ', '.join(cells)
            if cleaned:
                cleaned_lines.append(cleaned)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


if __name__ == "__main__":
    # Test
    sample = """
### Tablo 3
| Radyasyonun zararlı etkilerinden korunmak için, alınabilecek başlıca önlemler |
|---|
| şunlardır: |

### Tablo 4
| • Özel giysiler (kurşun önlük, özel maske) kullanılmalıdır |
|---|
| • Radyasyon kaynağından uzak durulmalı |
    """
    
    print("ÖNCE:")
    print(sample)
    print("\n" + "="*50 + "\n")
    print("SONRA:")
    print(clean_markdown_tables(sample))










