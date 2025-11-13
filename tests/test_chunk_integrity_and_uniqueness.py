import pytest
import re
from src.text_processing.semantic_chunker import create_semantic_chunks

# This text simulates the issues found in the user's screenshot.
# It contains topic headers, short sentences, and complex sentences.
# The old chunker would likely create duplicates and broken sentences from this.
PROBLEM_TEXT = """
# Biyolojinin Temelleri

Biyoloji, canlıları ve yaşam süreçlerini inceleyen bilim dalıdır. Bu alan, moleküler düzeyden ekosistemlere kadar geniş bir yelpazeyi kapsar. Canlıların yapısını, işlevini, büyümesini, kökenini, evrimini ve dağılımını inceler.

## Hücre Teorisi

Hücre teorisi, modern biyolojinin temel taşlarından biridir. Teoriye göre tüm canlılar bir veya daha fazla hücreden oluşur. Hücreler, yaşamın temel yapısal ve işlevsel birimidir. Yeni hücreler, yalnızca önceden var olan hücrelerin bölünmesiyle meydana gelir. Bu prensip, yaşamın sürekliliğini açıklar.

### Ökaryotik Hücreler

Ökaryotik hücreler, zarla çevrili bir çekirdeğe ve diğer organellere sahiptir. Bu hücreler, bitkiler, hayvanlar, mantarlar ve protistler gibi çok çeşitli canlılarda bulunur. Çekirdek, hücrenin genetik materyalini (DNA) barındırır ve hücre aktivitelerini kontrol eder. Mitokondri, enerji üretiminden sorumludur. Ribozomlar protein sentezler. Endoplazmik retikulum ve Golgi aygıtı, proteinlerin ve lipitlerin işlenmesi ve taşınmasında rol oynar. Bu karmaşık yapı, hücrenin hayatta kalmasını sağlar.

# Genetik ve Evrim

Genetik, kalıtım ve genlerin incelenmesidir. Gregor Mendel'in çalışmaları, modern genetiğin temelini atmıştır. DNA'nın yapısının keşfi, genetik anlayışımızı devrim niteliğinde değiştirmiştir. Evrim ise, canlı popülasyonlarının nesiller boyunca geçirdiği kalıtsal değişiklikler sürecidir. Charles Darwin'in doğal seçilim teorisi, evrimin ana mekanizmasını açıklar. Bu teoriye göre, çevreye daha iyi uyum sağlayan bireylerin hayatta kalma ve üreme olasılığı daha yüksektir. Bu süreç, türlerin zamanla değişmesine ve yeni türlerin ortaya çıkmasına neden olur. Genetik ve evrim, biyolojinin iki temel direğidir ve birbirini tamamlar.
"""

def test_chunk_integrity_and_uniqueness():
    """
    Tests the new sentence-based chunker for integrity, uniqueness, and completeness.
    """
    chunks = create_semantic_chunks(PROBLEM_TEXT, target_size=300, overlap_ratio=0.1)
    
    assert chunks is not None
    assert len(chunks) > 1, "Chunking should produce more than one chunk for this text."

    all_sentences_in_chunks = []
    # Use the same sentence pattern as the chunker for consistent testing.
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    # 1. Test for Sentence Integrity: No chunk should start or end with a broken sentence.
    for i, chunk in enumerate(chunks):
        assert len(chunk) > 0, f"Chunk {i} is empty."
        
        # Each chunk must be composed of full sentences.
        # A simple but effective check is that the first character should be uppercase
        # (or a non-alphabetic character like '#').
        first_char = chunk.lstrip()
        assert first_char.isupper() or not first_char.isalpha(), \
            f"Chunk {i} appears to start mid-sentence: '{chunk[:50]}...'"
        
        # And the chunk should end with valid sentence-ending punctuation.
        assert chunk.rstrip().endswith(('.', '!', '?')), \
            f"Chunk {i} does not end with proper punctuation: '...{chunk[-50:]}'"
        
        # Collect sentences for the uniqueness test.
        chunk_sentences = sentence_pattern.split(chunk)
        all_sentences_in_chunks.extend([s.strip() for s in chunk_sentences if s.strip()])

    # 2. Test for Content Uniqueness: No sentence should be duplicated excessively.
    # With the new overlap strategy, some sentences will appear exactly twice, but never more.
    sentence_counts = {}
    for sentence in all_sentences_in_chunks:
        sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
    
    for sentence, count in sentence_counts.items():
        assert count <= 2, \
            f"A sentence appears more than twice (count: {count}), indicating a duplication bug: '{sentence[:100]}...'"

    # 3. Test for Content Completeness: The combined chunks should represent the original text.
    # We check if all original sentences are present in the final collection of chunked sentences.
    original_sentences = set([s.strip() for s in sentence_pattern.split(PROBLEM_TEXT) if s.strip() and not s.strip().startswith('#')])
    chunked_sentences = set(all_sentences_in_chunks)
    
    lost_sentences = original_sentences - chunked_sentences
    
    # Headers might be processed differently, so we focus on the core text.
    # No sentences from the original text should be lost.
    assert not lost_sentences, \
        f"Content loss detected. {len(lost_sentences)} sentences are missing from the chunks. Example: '{list(lost_sentences)}'"
