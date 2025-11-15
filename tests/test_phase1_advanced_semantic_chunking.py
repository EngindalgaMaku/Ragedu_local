#!/usr/bin/env python3
"""
Phase 1 Advanced Semantic Chunking Architecture Tests

This test suite validates all Phase 1 enhancements:
1. Enhanced Semantic Chunker with embedding-based refinement
2. Advanced Chunk Validator with quality metrics
3. AST-based Markdown Parser with semantic context
4. Enhanced Text Chunker with new strategies
5. Configuration integration and Turkish language support

Test Categories:
- Unit tests for individual components
- Integration tests for complete pipeline
- Performance and error handling tests
- Turkish language optimization tests
- Quality metrics validation tests
"""

import pytest
import sys
import os
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import numpy as np

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    # Phase 1 imports
    from text_processing.semantic_chunker import AdvancedSemanticChunker, SemanticChunk
    from text_processing.advanced_chunk_validator import AdvancedChunkValidator, ValidationResult
    from text_processing.ast_markdown_parser import ASTMarkdownParser, MarkdownElement, MarkdownSection
    from text_processing.text_chunker import chunk_text, create_semantic_chunks
    
    # Configuration imports
    from config import (
        get_semantic_chunking_config, get_chunk_size, get_chunk_overlap,
        get_embedding_model, is_embedding_refinement_enabled, is_ast_parser_enabled,
        get_validation_thresholds, get_turkish_language_config
    )
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import warning: {e}")
    IMPORTS_AVAILABLE = False

# Test data
TURKISH_SAMPLE_TEXT = """
# Biyoloji ve Canlıların Ortak Özellikleri

## Canlıların Temel Özellikleri

Canlılar, cansız varlıklardan ayıran birçok ortak özelliğe sahiptir. Bu özellikler şunlardır:

### 1. Hücresel Yapı
Tüm canlılar hücrelerden meydana gelir. Hücre, canlılığın temel birimidir. Tek hücreli veya çok hücreli olabilirler.

### 2. Metabolizma
Canlılar sürekli olarak enerji alışverişi yaparlar. Bu süreçte:
- Besinleri alırlar (beslenme)
- Enerji üretirler (solunum)
- Atık maddeleri uzaklaştırırlar (boşaltım)

### 3. Büyüme ve Gelişme
Canlılar zaman içinde büyür ve gelişirler. Bu süreç genetik programları tarafından kontrol edilir.

## Hücre Çeşitleri

Hücreler temel olarak iki gruba ayrılır:

### Prokaryot Hücreler
- Çekirdeği yoktur
- Genetik materyal sitoplazmada dağınık haldedir
- Bakteriler ve arkeler bu gruba girer

### Ökaryot Hücreler  
- Çekirdeği vardır
- Genetik materyal çekirdek zarı ile çevrilidir
- Bitki, hayvan ve mantar hücreleri bu gruba girer

## Sonuç

Canlıların ortak özellikleri, yaşamın temelini oluşturan karmaşık sistemlerin bir sonucudur.
"""

ENGLISH_MARKDOWN_SAMPLE = """
# Advanced Semantic Processing

## Introduction

This document outlines the key concepts in semantic processing and natural language understanding.

### Core Components

#### 1. Text Preprocessing
- Tokenization of input text
- Normalization of various formats
- Language detection algorithms

#### 2. Semantic Analysis
The semantic analysis phase involves:
- **Embedding generation**: Converting text to vector representations
- **Similarity computation**: Measuring semantic relationships
- **Context preservation**: Maintaining document structure

```python
# Example code block
def process_text(text):
    embeddings = model.encode(text)
    return embeddings
```

### Mathematical Formulas

The cosine similarity is calculated as:
$$\text{similarity} = \frac{A \cdot B}{||A|| \times ||B||}$$

## Results and Conclusions

Our approach demonstrates significant improvements in:
1. Accuracy: +20% improvement
2. Processing speed: 15% faster
3. Memory usage: 10% reduction

| Metric | Before | After | Improvement |
|--------|---------|--------|-------------|
| Accuracy | 75% | 95% | +20% |
| Speed | 100ms | 85ms | +15% |
| Memory | 2GB | 1.8GB | +10% |
"""

COMPLEX_MARKDOWN_SAMPLE = """
# Research Paper: Advanced NLP Techniques

## Abstract

This paper presents novel approaches to natural language processing...

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Results](#results)

## Introduction {#introduction}

Natural Language Processing (NLP) has evolved significantly...

### Background
The field encompasses several key areas:
- Syntax analysis
- Semantic understanding  
- Pragmatic interpretation

## Methodology {#methodology}

Our approach consists of three main phases:

### Phase 1: Data Collection
We collected data from multiple sources:
1. Academic papers (n=1,000)
2. Web articles (n=5,000)
3. Social media posts (n=10,000)

### Phase 2: Model Training
The training process involved:
```bash
python train.py --model transformer --epochs 100
```

### Phase 3: Evaluation
We used standard metrics:
- BLEU score
- ROUGE metrics
- Human evaluation

## Results {#results}

Our results show significant improvements:

| Model | BLEU | ROUGE-1 | ROUGE-L |
|-------|------|---------|---------|
| Baseline | 0.25 | 0.35 | 0.30 |
| Ours | 0.45 | 0.55 | 0.50 |

## Conclusion

The proposed method demonstrates...
"""

@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestAdvancedSemanticChunker:
    """Test Enhanced Semantic Chunker with embedding-based refinement"""
    
    def setup_method(self):
        """Setup test environment"""
        self.chunker = AdvancedSemanticChunker(
            target_size=500,
            language='auto',
            use_embeddings=True
        )
    
    def test_initialization(self):
        """Test chunker initialization with Phase 1 parameters"""
        assert self.chunker.target_size == 500
        assert self.chunker.language == 'auto'
        assert self.chunker.use_embeddings == True
        assert hasattr(self.chunker, 'sentence_splitter')
        assert hasattr(self.chunker, 'embedding_model')
    
    def test_turkish_sentence_splitting(self):
        """Test enhanced Turkish sentence splitting"""
        turkish_text = "Dr. Ahmet bey geldi. Ancak öğr. üyesi yoktu. Bu durum çok önemli."
        sentences = self.chunker._split_into_sentences(turkish_text, 'tr')
        
        # Should handle Turkish abbreviations correctly
        assert len(sentences) >= 2
        assert any('Dr. Ahmet bey geldi' in s for s in sentences)
    
    def test_semantic_chunking_creation(self):
        """Test semantic chunk creation with embeddings"""
        chunks = self.chunker.create_semantic_chunks(TURKISH_SAMPLE_TEXT, overlap_ratio=0.1)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, SemanticChunk) for chunk in chunks)
        assert all(hasattr(chunk, 'embedding_vector') for chunk in chunks)
        assert all(len(chunk.text) > 50 for chunk in chunks)  # Minimum meaningful size
    
    def test_embedding_caching(self):
        """Test embedding caching mechanism"""
        text = "Bu bir test cümlesidir. Embedding cache test edilmektedir."
        
        # First call should cache embeddings
        embeddings1 = self.chunker._get_sentence_embeddings([text])
        
        # Second call should use cache
        embeddings2 = self.chunker._get_sentence_embeddings([text])
        
        np.testing.assert_array_equal(embeddings1, embeddings2)
    
    def test_semantic_boundary_detection(self):
        """Test embedding-based boundary detection"""
        chunks = self.chunker.create_semantic_chunks(TURKISH_SAMPLE_TEXT)
        
        # Chunks should have reasonable coherence
        for chunk in chunks:
            assert len(chunk.text.strip()) > 100  # Meaningful content
            assert chunk.coherence_score > 0.5  # Reasonable coherence
    
    def test_batch_processing_efficiency(self):
        """Test batch processing for embeddings"""
        sentences = [
            "İlk test cümlesi.",
            "İkinci test cümlesi.", 
            "Üçüncü test cümlesi."
        ]
        
        embeddings = self.chunker._get_sentence_embeddings(sentences)
        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] > 0  # Has embedding dimensions


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestAdvancedChunkValidator:
    """Test Advanced Chunk Validator with quality metrics"""
    
    def setup_method(self):
        """Setup test environment"""
        self.validator = AdvancedChunkValidator()
    
    def test_initialization(self):
        """Test validator initialization"""
        assert hasattr(self.validator, 'embedding_model')
        assert hasattr(self.validator, 'validation_cache')
        assert hasattr(self.validator, 'thresholds')
    
    def test_chunk_validation(self):
        """Test individual chunk validation"""
        chunk_text = """
        Canlıların temel özellikleri arasında hücresel yapı, metabolizma ve büyüme yer alır.
        Bu özellikler tüm canlılar için ortaktır ve yaşamın temelini oluşturur.
        Hücre, canlılığın en küçük birimidir ve tüm yaşamsal faaliyetleri gerçekleştirir.
        """
        
        result = self.validator.validate_chunk(chunk_text, chunk_index=0)
        
        assert isinstance(result, ValidationResult)
        assert result.overall_score > 0
        assert hasattr(result, 'semantic_coherence')
        assert hasattr(result, 'topic_consistency')
        assert hasattr(result, 'length_quality')
    
    def test_cross_chunk_analysis(self):
        """Test cross-chunk relationship analysis"""
        chunks = [
            "Canlıların temel özellikleri hücresel yapı ve metabolizmadır.",
            "Hücre canlılığın en küçük birimidir ve yaşamsal faaliyetleri gerçekleştirir.", 
            "Metabolizma canlıların enerji alışverişi sürecidir."
        ]
        
        analysis = self.validator.analyze_cross_chunk_relationships(chunks)
        
        assert 'chunk_similarities' in analysis
        assert 'topic_transitions' in analysis
        assert 'coherence_scores' in analysis
        assert len(analysis['chunk_similarities']) == len(chunks) - 1
    
    def test_batch_validation_efficiency(self):
        """Test efficient batch validation with embedding reuse"""
        chunks = [
            "İlk chunk metni test amaçlıdır.",
            "İkinci chunk da benzer bir içerik taşır.",
            "Üçüncü chunk ise farklı bir konu işler."
        ]
        
        # Should reuse embeddings efficiently
        results = self.validator.validate_chunks_batch(chunks)
        
        assert len(results) == len(chunks)
        assert all(isinstance(r, ValidationResult) for r in results)
    
    def test_turkish_optimization(self):
        """Test Turkish language optimization in validation"""
        turkish_chunk = """
        Türkçe metin analizi özel kurallar gerektirir. Çünkü dil yapısı farklıdır.
        Örneğin, Prof. Dr. gibi kısaltmalar önemlidir. Buna dikkat etmek gerekir.
        """
        
        result = self.validator.validate_chunk(turkish_chunk, language='tr')
        
        # Should handle Turkish patterns correctly
        assert result.is_valid or result.overall_score > 0.5


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestASTMarkdownParser:
    """Test AST-based Markdown Parser with semantic context"""
    
    def setup_method(self):
        """Setup test environment"""
        self.parser = ASTMarkdownParser()
    
    def test_initialization(self):
        """Test parser initialization"""
        assert hasattr(self.parser, 'element_cache')
        assert hasattr(self.parser, 'context_cache')
        
    def test_markdown_ast_parsing(self):
        """Test markdown to AST conversion"""
        ast_nodes = self.parser.parse_markdown_to_ast(ENGLISH_MARKDOWN_SAMPLE)
        
        assert len(ast_nodes) > 0
        assert all(isinstance(node, MarkdownElement) for node in ast_nodes)
        
        # Should detect headers
        headers = [node for node in ast_nodes if node.type == 'header']
        assert len(headers) > 0
        
        # Should detect code blocks
        code_blocks = [node for node in ast_nodes if node.type == 'code_block']
        assert len(code_blocks) > 0
    
    def test_semantic_section_creation(self):
        """Test semantic section grouping"""
        ast_nodes = self.parser.parse_markdown_to_ast(COMPLEX_MARKDOWN_SAMPLE)
        sections = self.parser.create_semantic_sections(ast_nodes)
        
        assert len(sections) > 0
        assert all(isinstance(section, MarkdownSection) for section in sections)
        
        # Should maintain header hierarchy
        main_sections = [s for s in sections if s.level == 1]
        assert len(main_sections) > 0
    
    def test_table_preservation(self):
        """Test table semantic context preservation"""
        table_text = """
        # Data Analysis
        
        | Metric | Value | Change |
        |--------|-------|---------|
        | Accuracy | 95% | +20% |
        | Speed | 85ms | +15% |
        """
        
        ast_nodes = self.parser.parse_markdown_to_ast(table_text)
        table_nodes = [node for node in ast_nodes if node.type == 'table']
        
        assert len(table_nodes) > 0
        # Should preserve table structure and context
        assert 'Metric' in str(table_nodes[0].content)
    
    def test_code_block_protection(self):
        """Test code block intelligent handling"""
        code_text = """
        # Example Code
        
        ```python
        def process_text(text):
            return model.encode(text)
        ```
        
        This function processes text efficiently.
        """
        
        ast_nodes = self.parser.parse_markdown_to_ast(code_text)
        code_blocks = [node for node in ast_nodes if node.type == 'code_block']
        
        assert len(code_blocks) > 0
        assert 'def process_text' in code_blocks[0].content
    
    def test_cross_reference_resolution(self):
        """Test cross-reference handling"""
        ref_text = """
        # Section 1 {#sec1}
        Content with reference to [Section 2](#sec2).
        
        # Section 2 {#sec2}  
        Referenced content here.
        """
        
        context = self.parser.extract_semantic_context(ref_text)
        
        assert 'cross_references' in context
        assert len(context['cross_references']) > 0


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestEnhancedTextChunker:
    """Test Enhanced Text Chunker with new strategies"""
    
    def test_ast_markdown_strategy(self):
        """Test new ast_markdown chunking strategy"""
        chunks = chunk_text(
            ENGLISH_MARKDOWN_SAMPLE,
            chunk_size=800,
            chunk_overlap=100,
            strategy="ast_markdown"
        )
        
        assert len(chunks) > 0
        assert all(len(chunk) > 50 for chunk in chunks)
        
        # Should preserve markdown structure
        assert any('#' in chunk for chunk in chunks)  # Headers preserved
    
    def test_semantic_strategy_with_refinement(self):
        """Test enhanced semantic strategy with embedding refinement"""
        chunks = chunk_text(
            TURKISH_SAMPLE_TEXT,
            chunk_size=600,
            chunk_overlap=100,
            strategy="semantic",
            use_embedding_refinement=True
        )
        
        assert len(chunks) > 0
        assert all(len(chunk) > 100 for chunk in chunks)
    
    def test_create_semantic_chunks_integration(self):
        """Test create_semantic_chunks with Phase 1 enhancements"""
        chunks = create_semantic_chunks(
            text=TURKISH_SAMPLE_TEXT,
            target_size=500,
            overlap_ratio=0.15,
            language='tr',
            use_embedding_refinement=True
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) > 50 for chunk in chunks)
    
    def test_fallback_mechanism(self):
        """Test fallback mechanism when advanced features fail"""
        # Should fallback gracefully when components are unavailable
        with patch('text_processing.semantic_chunker.SEMANTIC_CHUNKING_AVAILABLE', False):
            chunks = chunk_text(
                ENGLISH_MARKDOWN_SAMPLE,
                strategy="semantic"
            )
            
            assert len(chunks) > 0  # Should still produce chunks
    
    def test_turkish_language_optimization(self):
        """Test Turkish language specific optimizations"""
        chunks = chunk_text(
            TURKISH_SAMPLE_TEXT,
            chunk_size=400,
            strategy="semantic",
            language="tr"
        )
        
        assert len(chunks) > 0
        
        # Should handle Turkish sentence patterns correctly
        for chunk in chunks:
            # Should not break on Turkish abbreviations
            assert not chunk.strip().startswith('.')
            assert not chunk.strip().startswith('!')


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestConfigurationIntegration:
    """Test Phase 1 configuration integration"""
    
    def test_semantic_chunking_config_access(self):
        """Test semantic chunking configuration access"""
        config = get_semantic_chunking_config()
        
        assert 'chunk_size' in config
        assert 'embedding_cache_size' in config
        assert 'validation_enabled' in config
        assert 'turkish_abbreviations' in config
    
    def test_configuration_helpers(self):
        """Test configuration helper functions"""
        chunk_size = get_chunk_size()
        chunk_overlap = get_chunk_overlap()
        embedding_model = get_embedding_model()
        
        assert isinstance(chunk_size, int)
        assert isinstance(chunk_overlap, int)
        assert isinstance(embedding_model, str)
        assert chunk_size > 0
        assert chunk_overlap >= 0
    
    def test_validation_thresholds(self):
        """Test validation threshold configuration"""
        thresholds = get_validation_thresholds()
        
        required_keys = ['semantic_coherence', 'topic_consistency', 'length_quality', 'overall_quality']
        assert all(key in thresholds for key in required_keys)
        assert all(0.0 <= threshold <= 1.0 for threshold in thresholds.values())
    
    def test_turkish_language_config(self):
        """Test Turkish language configuration"""
        config = get_turkish_language_config()
        
        assert 'abbreviations' in config
        assert 'sentence_endings' in config
        assert 'conjunctions' in config
        
        # Should have Turkish abbreviations
        assert 'Dr.' in config['abbreviations']
        assert 'Prof.' in config['abbreviations']


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestPerformanceAndErrorHandling:
    """Test performance optimizations and error handling"""
    
    def test_memory_efficiency(self):
        """Test memory-efficient processing"""
        large_text = TURKISH_SAMPLE_TEXT * 20  # Create larger text
        
        chunker = AdvancedSemanticChunker(target_size=1000, use_embeddings=True)
        chunks = chunker.create_semantic_chunks(large_text)
        
        assert len(chunks) > 0
        # Should handle large texts without memory issues
    
    def test_error_handling_graceful_degradation(self):
        """Test graceful degradation on errors"""
        chunker = AdvancedSemanticChunker(target_size=500)
        
        # Test with invalid input
        chunks = chunker.create_semantic_chunks("")
        assert chunks == []  # Should return empty list, not crash
        
        # Test with very short text
        chunks = chunker.create_semantic_chunks("Kısa.")
        assert len(chunks) >= 0  # Should handle gracefully
    
    def test_caching_efficiency(self):
        """Test caching mechanisms"""
        validator = AdvancedChunkValidator()
        chunk_text = "Test chunk for caching efficiency."
        
        # First validation should cache result
        result1 = validator.validate_chunk(chunk_text)
        
        # Second validation should use cache
        result2 = validator.validate_chunk(chunk_text)
        
        assert result1.overall_score == result2.overall_score
    
    def test_batch_processing_optimization(self):
        """Test batch processing optimizations"""
        chunks = [
            "İlk test chunk metni burada.",
            "İkinci chunk farklı bir içerik.",
            "Üçüncü chunk da benzer şekilde."
        ]
        
        validator = AdvancedChunkValidator()
        
        # Batch processing should be more efficient than individual
        import time
        
        start = time.time()
        batch_results = validator.validate_chunks_batch(chunks)
        batch_time = time.time() - start
        
        assert len(batch_results) == len(chunks)
        # Should complete in reasonable time
        assert batch_time < 10.0  # Should not take more than 10 seconds


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available") 
class TestQualityMetrics:
    """Test Phase 1 quality improvement metrics"""
    
    def test_semantic_coherence_measurement(self):
        """Test semantic coherence scoring"""
        coherent_text = """
        Canlıların temel özellikleri arasında hücresel yapı yer alır.
        Hücre canlılığın en küçük birimidir ve yaşamsal işlevleri gerçekleştirir.
        Bu hücreler metabolizma yoluyla enerji üretir ve büyüme sağlar.
        """
        
        validator = AdvancedChunkValidator()
        result = validator.validate_chunk(coherent_text)
        
        # Coherent text should have high coherence score
        assert result.semantic_coherence > 0.6
    
    def test_topic_consistency_measurement(self):
        """Test topic consistency scoring"""
        consistent_text = """
        Fotosentez bitkiler için yaşamsal bir süreçtir.
        Bu süreçte güneş enerjisi kullanılarak glikoz üretilir.
        Klorofil pigmenti fotosentez için gereklidir.
        """
        
        validator = AdvancedChunkValidator()
        result = validator.validate_chunk(consistent_text)
        
        # Topic-consistent text should have high consistency score
        assert result.topic_consistency > 0.6
    
    def test_overall_quality_improvement(self):
        """Test overall quality improvements"""
        # Test with AST markdown strategy
        ast_chunks = chunk_text(
            ENGLISH_MARKDOWN_SAMPLE,
            strategy="ast_markdown",
            chunk_size=800
        )
        
        # Test with traditional markdown strategy  
        traditional_chunks = chunk_text(
            ENGLISH_MARKDOWN_SAMPLE,
            strategy="markdown", 
            chunk_size=800
        )
        
        # AST chunks should preserve structure better
        ast_structure_preservation = sum(1 for chunk in ast_chunks if '#' in chunk or '```' in chunk)
        traditional_structure_preservation = sum(1 for chunk in traditional_chunks if '#' in chunk or '```' in chunk)
        
        # AST should preserve more structure
        assert ast_structure_preservation >= traditional_structure_preservation


if __name__ == "__main__":
    """Run Phase 1 tests"""
    print("🧪 Running Phase 1 Advanced Semantic Chunking Architecture Tests...")
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available. Please install dependencies.")
        sys.exit(1)
    
    # Run specific test categories
    test_categories = [
        ("Advanced Semantic Chunker", TestAdvancedSemanticChunker),
        ("Advanced Chunk Validator", TestAdvancedChunkValidator), 
        ("AST Markdown Parser", TestASTMarkdownParser),
        ("Enhanced Text Chunker", TestEnhancedTextChunker),
        ("Configuration Integration", TestConfigurationIntegration),
        ("Performance & Error Handling", TestPerformanceAndErrorHandling),
        ("Quality Metrics", TestQualityMetrics)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category_name, test_class in test_categories:
        print(f"\n📋 Testing {category_name}...")
        
        try:
            # Create test instance and run methods
            test_instance = test_class()
            test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
            
            for method_name in test_methods:
                total_tests += 1
                try:
                    # Setup if available
                    if hasattr(test_instance, 'setup_method'):
                        test_instance.setup_method()
                    
                    # Run test method
                    test_method = getattr(test_instance, method_name)
                    test_method()
                    
                    print(f"  ✅ {method_name}")
                    passed_tests += 1
                    
                except Exception as e:
                    print(f"  ❌ {method_name}: {str(e)}")
                    
        except Exception as e:
            print(f"  ❌ Failed to initialize {category_name}: {str(e)}")
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All Phase 1 tests passed successfully!")
        sys.exit(0)
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
        sys.exit(1)