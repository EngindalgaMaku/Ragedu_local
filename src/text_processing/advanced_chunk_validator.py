"""
Advanced Chunk Quality Validation System

Bu modül semantic chunking kalitesini gerçekçi metriklerle değerlendiren gelişmiş
bir sistem sağlar. Basit karakter/cümle sayma yerine gerçek anlamsal analiz yapar.

Yeni Metrikler:
- Semantic Coherence Score (40%): Konu tutarlılığı, semantic similarity
- Context Preservation Score (25%): Bağlam korunması, referans çözümü
- Information Completeness (20%): Bilgi bütünlüğü, ana fikir tamamlanması
- Readability & Flow (15%): Doğal okuma akışı, cümle geçişleri
"""

import re
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import math


@dataclass
class ChunkQualityScore:
    """Gelişmiş chunk kalite skoru."""
    semantic_coherence: float
    context_preservation: float
    information_completeness: float
    readability_flow: float
    overall_score: float
    is_valid: bool
    detailed_analysis: Dict[str, Union[str, float, List[str]]]


class AdvancedChunkValidator:
    """Gelişmiş semantic chunk kalite değerlendirici."""
    
    def __init__(self):
        # Türkçe dil için özel pattern'lar
        self.sentence_endings = re.compile(r'[.!?]+(?:\s+|$)')
        self.reference_words = re.compile(r'\b(bu|şu|o|bunlar|şunlar|onlar|bunu|şunu|onu|bunları|şunları|onları|burada|şurada|orada|böyle|şöyle|öyle)\b', re.IGNORECASE)
        self.topic_transition_words = re.compile(r'\b(ancak|fakat|lakin|ama|halbuki|oysa|bununla birlikte|öte yandan|diğer taraftan|ayrıca|dahası|üstelik|sonuç olarak|bu nedenle|bu yüzden)\b', re.IGNORECASE)
        self.incomplete_indicators = re.compile(r'\b(örneğin|yani|şöyle ki|gibi|vs\.|vb\.|v\.s\.|devam|sürdür|ilerle)\s*$', re.IGNORECASE)
        
        # Stopwords for Turkish
        self.turkish_stopwords = {
            'bir', 'bu', 'da', 'de', 'den', 'dır', 've', 'veya', 'için', 'ile', 'ise',
            'olan', 'olarak', 'her', 'daha', 'en', 'çok', 'az', 'var', 'yok', 'gibi',
            'kadar', 'sonra', 'önce', 'şimdi', 'burada', 'orada', 'nerede', 'nasıl'
        }
        
        # Quality thresholds
        self.min_coherence_score = 0.6
        self.min_context_score = 0.5
        self.min_completeness_score = 0.6
        self.min_readability_score = 0.5
        self.overall_threshold = 0.65

    def validate_chunk_quality(self, chunk: str, previous_chunk: str = None, next_chunk: str = None) -> ChunkQualityScore:
        """Chunk kalitesini gelişmiş metriklerle değerlendirin."""
        
        if not chunk.strip():
            return self._create_invalid_score("Boş chunk")
        
        # Her bir metriği hesapla
        coherence_score = self._calculate_semantic_coherence(chunk)
        context_score = self._calculate_context_preservation(chunk, previous_chunk, next_chunk)
        completeness_score = self._calculate_information_completeness(chunk)
        readability_score = self._calculate_readability_flow(chunk)
        
        # Ağırlıklı genel skor
        overall_score = (
            coherence_score * 0.40 +
            context_score * 0.25 +
            completeness_score * 0.20 +
            readability_score * 0.15
        )
        
        # Geçerlilik kontrolü
        is_valid = (
            overall_score >= self.overall_threshold and
            coherence_score >= self.min_coherence_score and
            context_score >= self.min_context_score and
            completeness_score >= self.min_completeness_score and
            readability_score >= self.min_readability_score
        )
        
        # Detaylı analiz
        detailed_analysis = self._create_detailed_analysis(
            chunk, coherence_score, context_score, completeness_score, readability_score
        )
        
        return ChunkQualityScore(
            semantic_coherence=coherence_score,
            context_preservation=context_score,
            information_completeness=completeness_score,
            readability_flow=readability_score,
            overall_score=overall_score,
            is_valid=is_valid,
            detailed_analysis=detailed_analysis
        )

    def _calculate_semantic_coherence(self, chunk: str) -> float:
        """
        Semantic Coherence Score (40%): Konuların tutarlılığını ölçer.
        
        Faktörler:
        - Topic consistency (konu birliği)
        - Sentence-to-sentence semantic flow
        - Keyword density and distribution
        - Conceptual relationships
        """
        sentences = self._split_into_sentences(chunk)
        if len(sentences) < 2:
            return 0.3  # Tek cümle chunk'lar düşük skor
        
        coherence_factors = []
        
        # 1. Topic Consistency (Konu Birliği)
        topic_score = self._measure_topic_consistency(sentences)
        coherence_factors.append(('topic_consistency', topic_score, 0.4))
        
        # 2. Semantic Similarity between sentences
        similarity_score = self._measure_sentence_similarity(sentences)
        coherence_factors.append(('sentence_similarity', similarity_score, 0.3))
        
        # 3. Keyword density and repetition
        keyword_score = self._measure_keyword_coherence(chunk)
        coherence_factors.append(('keyword_coherence', keyword_score, 0.2))
        
        # 4. Conceptual flow
        flow_score = self._measure_conceptual_flow(sentences)
        coherence_factors.append(('conceptual_flow', flow_score, 0.1))
        
        # Ağırlıklı ortalama
        weighted_score = sum(score * weight for _, score, weight in coherence_factors)
        return min(1.0, max(0.0, weighted_score))

    def _measure_topic_consistency(self, sentences: List[str]) -> float:
        """Cümleler arası konu tutarlılığını ölçer."""
        if len(sentences) < 2:
            return 0.5
        
        # Her cümleden anahtar kelimeleri çıkar
        sentence_keywords = []
        for sentence in sentences:
            keywords = self._extract_keywords(sentence)
            sentence_keywords.append(keywords)
        
        if not sentence_keywords or all(not keywords for keywords in sentence_keywords):
            return 0.3
        
        # Ortak kelime oranını hesapla
        all_keywords = set()
        for keywords in sentence_keywords:
            all_keywords.update(keywords)
        
        if not all_keywords:
            return 0.3
        
        # Her cümle çiftinin ortak kelime oranını hesapla
        consistency_scores = []
        for i in range(len(sentence_keywords)):
            for j in range(i + 1, len(sentence_keywords)):
                keywords1 = set(sentence_keywords[i])
                keywords2 = set(sentence_keywords[j])
                
                if not keywords1 or not keywords2:
                    consistency_scores.append(0.0)
                    continue
                
                intersection = len(keywords1 & keywords2)
                union = len(keywords1 | keywords2)
                jaccard = intersection / union if union > 0 else 0.0
                consistency_scores.append(jaccard)
        
        return np.mean(consistency_scores) if consistency_scores else 0.3

    def _measure_sentence_similarity(self, sentences: List[str]) -> float:
        """Cümleler arası anlamsal benzerliği ölçer (basit TF-IDF yaklaşımı)."""
        if len(sentences) < 2:
            return 0.4
        
        # Basit word frequency vectors oluştur
        sentence_vectors = []
        all_words = set()
        
        for sentence in sentences:
            words = self._tokenize_words(sentence.lower())
            words = [w for w in words if w not in self.turkish_stopwords and len(w) > 2]
            all_words.update(words)
            sentence_vectors.append(Counter(words))
        
        if not all_words:
            return 0.3
        
        # Cosine similarity hesapla
        similarities = []
        vocab = list(all_words)
        
        for i in range(len(sentence_vectors)):
            for j in range(i + 1, len(sentence_vectors)):
                vec1 = np.array([sentence_vectors[i].get(word, 0) for word in vocab])
                vec2 = np.array([sentence_vectors[j].get(word, 0) for word in vocab])
                
                # Cosine similarity
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.3

    def _measure_keyword_coherence(self, chunk: str) -> float:
        """Anahtar kelime tutarlılığını ve dağılımını ölçer."""
        keywords = self._extract_keywords(chunk)
        
        if len(keywords) < 3:
            return 0.4  # Çok az anahtar kelime
        
        # Keyword frequency distribution
        keyword_counts = Counter(keywords)
        total_keywords = len(keywords)
        
        # Çok dominant kelimeler olmamalı (max %40)
        max_freq = max(keyword_counts.values())
        dominance_ratio = max_freq / total_keywords
        
        if dominance_ratio > 0.6:
            dominance_score = 0.3  # Çok dominant kelime var
        elif dominance_ratio > 0.4:
            dominance_score = 0.7
        else:
            dominance_score = 1.0
        
        # Keyword diversity
        unique_ratio = len(keyword_counts) / total_keywords
        diversity_score = min(1.0, unique_ratio * 2)  # İyi dağılım için 0.5+ ratio ideal
        
        return (dominance_score * 0.6 + diversity_score * 0.4)

    def _measure_conceptual_flow(self, sentences: List[str]) -> float:
        """Cümleler arası kavramsal akışı ölçer."""
        if len(sentences) < 2:
            return 0.5
        
        flow_score = 0.0
        flow_indicators = 0
        
        for i in range(len(sentences) - 1):
            current_sentence = sentences[i]
            next_sentence = sentences[i + 1]
            
            # Transition words kontrol et
            if self.topic_transition_words.search(next_sentence):
                flow_score += 0.8
                flow_indicators += 1
            
            # Reference words kontrol et
            if self.reference_words.search(next_sentence):
                flow_score += 0.6
                flow_indicators += 1
            
            # Shared concepts
            current_keywords = set(self._extract_keywords(current_sentence))
            next_keywords = set(self._extract_keywords(next_sentence))
            
            if current_keywords & next_keywords:
                flow_score += 0.5
                flow_indicators += 1
            
            flow_indicators += 1  # Her geçiş için sayaç
        
        return flow_score / max(1, flow_indicators)

    def _calculate_context_preservation(self, chunk: str, previous_chunk: str = None, next_chunk: str = None) -> float:
        """
        Context Preservation Score (25%): Bağlamsal bilginin korunmasını ölçer.
        
        Faktörler:
        - Reference resolution capability
        - Connection with adjacent chunks
        - Contextual completeness
        - Pronoun and demonstrative handling
        """
        context_factors = []
        
        # 1. Internal reference resolution
        internal_ref_score = self._measure_internal_references(chunk)
        context_factors.append(('internal_references', internal_ref_score, 0.4))
        
        # 2. External context dependency
        external_dep_score = self._measure_external_dependencies(chunk)
        context_factors.append(('external_dependencies', external_dep_score, 0.3))
        
        # 3. Chunk boundary context (if adjacent chunks provided)
        boundary_score = self._measure_boundary_context(chunk, previous_chunk, next_chunk)
        context_factors.append(('boundary_context', boundary_score, 0.3))
        
        # Ağırlıklı ortalama
        weighted_score = sum(score * weight for _, score, weight in context_factors)
        return min(1.0, max(0.0, weighted_score))

    def _measure_internal_references(self, chunk: str) -> float:
        """Chunk içindeki referansların çözümlenebilirliğini ölçer."""
        sentences = self._split_into_sentences(chunk)
        
        if len(sentences) < 2:
            return 0.8  # Tek cümle chunk'ta referans problemi yok
        
        resolvable_refs = 0
        total_refs = 0
        
        # Her cümledeki referansları kontrol et
        for i, sentence in enumerate(sentences):
            references = self.reference_words.findall(sentence.lower())
            
            for ref in references:
                total_refs += 1
                
                # Önceki cümlelerde çözümlenebilir antecedent var mı?
                if i > 0:  # İlk cümle değilse
                    for prev_sentence in sentences[:i]:
                        # Basit heuristic: önceki cümlelerde isim/kavram var mı?
                        if self._has_potential_antecedent(prev_sentence, ref):
                            resolvable_refs += 1
                            break
        
        if total_refs == 0:
            return 0.9  # Referans yok, sorun yok
        
        resolution_ratio = resolvable_refs / total_refs
        return min(1.0, resolution_ratio + 0.3)  # %30 bonus: çok katı olmamak için

    def _measure_external_dependencies(self, chunk: str) -> float:
        """Chunk'ın dış bağlam bağımlılığını ölçer (düşük bağımlılık = yüksek skor)."""
        sentences = self._split_into_sentences(chunk)
        
        dependency_indicators = 0
        total_sentences = len(sentences)
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            
            # Unresolved references at beginning
            if sentence == sentences[0] and self.reference_words.search(sentence_lower):
                dependency_indicators += 1
            
            # Continuation indicators
            if sentence_lower.startswith(('ve ', 'da ', 'de ', 'ama ', 'fakat ', 'ancak ')):
                dependency_indicators += 1
            
            # Incomplete sentence starters
            if sentence_lower.startswith(('bu nedenle', 'sonuç olarak', 'böylece', 'dolayısıyla')):
                if not self._has_supporting_context_in_chunk(sentence, chunk):
                    dependency_indicators += 1
        
        if total_sentences == 0:
            return 0.3
        
        dependency_ratio = dependency_indicators / total_sentences
        return max(0.0, 1.0 - dependency_ratio)

    def _measure_boundary_context(self, chunk: str, previous_chunk: str = None, next_chunk: str = None) -> float:
        """Chunk sınırlarındaki bağlam kalitesini ölçer."""
        boundary_score = 0.8  # Base score
        
        sentences = self._split_into_sentences(chunk)
        if not sentences:
            return 0.3
        
        first_sentence = sentences[0].strip()
        last_sentence = sentences[-1].strip()
        
        # First sentence boundary quality
        if previous_chunk:
            if self.reference_words.search(first_sentence.lower()):
                # Check if reference can be resolved in previous chunk
                if self._can_resolve_in_previous_chunk(first_sentence, previous_chunk):
                    boundary_score += 0.1
                else:
                    boundary_score -= 0.2
        
        # Last sentence boundary quality
        if next_chunk:
            if self.incomplete_indicators.search(last_sentence.lower()):
                # Check if continuation exists in next chunk
                if self._has_continuation_in_next_chunk(last_sentence, next_chunk):
                    boundary_score += 0.1
                else:
                    boundary_score -= 0.2
        
        return min(1.0, max(0.0, boundary_score))

    def _calculate_information_completeness(self, chunk: str) -> float:
        """
        Information Completeness Score (20%): Bilgi bütünlüğünü ölçer.
        
        Faktörler:
        - Complete ideas and concepts
        - Main idea + supporting details balance
        - No abrupt cuts or incomplete information
        - Structural completeness
        """
        completeness_factors = []
        
        # 1. Idea completeness
        idea_score = self._measure_idea_completeness(chunk)
        completeness_factors.append(('idea_completeness', idea_score, 0.4))
        
        # 2. Information balance (main + supporting details)
        balance_score = self._measure_information_balance(chunk)
        completeness_factors.append(('information_balance', balance_score, 0.3))
        
        # 3. Structural completeness
        structure_score = self._measure_structural_completeness(chunk)
        completeness_factors.append(('structural_completeness', structure_score, 0.3))
        
        # Ağırlıklı ortalama
        weighted_score = sum(score * weight for _, score, weight in completeness_factors)
        return min(1.0, max(0.0, weighted_score))

    def _measure_idea_completeness(self, chunk: str) -> float:
        """Ana fikirlerin tamamlanmış olup olmadığını ölçer."""
        sentences = self._split_into_sentences(chunk)
        
        if not sentences:
            return 0.0
        
        completeness_score = 0.8  # Base score
        
        # Check for incomplete endings
        last_sentence = sentences[-1].strip()
        if self.incomplete_indicators.search(last_sentence):
            completeness_score -= 0.3
        
        # Check for abrupt endings
        if not last_sentence.endswith(('.', '!', '?')):
            completeness_score -= 0.2
        
        # Check for introduction without development
        first_sentence = sentences[0].strip()
        if len(sentences) < 3 and any(word in first_sentence.lower() for word in ['örneğin', 'şunlar', 'bunlar', 'aşağıdaki']):
            completeness_score -= 0.3
        
        return max(0.0, completeness_score)

    def _measure_information_balance(self, chunk: str) -> float:
        """Ana fikir ve destekleyici detaylar arasındaki dengeyi ölçer."""
        sentences = self._split_into_sentences(chunk)
        
        if len(sentences) < 2:
            return 0.6  # Tek cümle için orta skor
        
        # Identify main ideas vs supporting details
        main_idea_indicators = 0
        detail_indicators = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Main idea indicators
            if any(indicator in sentence_lower for indicator in [
                'temel', 'ana', 'önemli', 'başlıca', 'esas', 'asıl', 'merkezi'
            ]):
                main_idea_indicators += 1
            
            # Detail indicators
            if any(indicator in sentence_lower for indicator in [
                'örneğin', 'detay', 'ayrıntı', 'özellikle', 'dahası', 'ayrıca', 'bunun yanında'
            ]):
                detail_indicators += 1
        
        total_sentences = len(sentences)
        
        # İdeal oran: %30-60 main idea, %40-70 details
        main_ratio = main_idea_indicators / total_sentences
        detail_ratio = detail_indicators / total_sentences
        
        # Balance calculation
        balance_score = 1.0
        
        if main_ratio > 0.8:  # Çok fazla ana fikir, az detay
            balance_score -= 0.3
        elif main_ratio < 0.1 and detail_ratio > 0.5:  # Çok fazla detay, az ana fikir
            balance_score -= 0.2
        
        return max(0.3, balance_score)

    def _measure_structural_completeness(self, chunk: str) -> float:
        """Yapısal bütünlüğü ölçer (listeler, bölümler, vb.)."""
        completeness_score = 0.9  # Base score
        
        lines = chunk.split('\n')
        
        # Check for incomplete lists
        numbered_items = [line for line in lines if re.match(r'^\s*\d+\.', line)]
        bullet_items = [line for line in lines if re.match(r'^\s*[-*+]', line)]
        
        # Numbered list completeness
        if numbered_items:
            numbers = []
            for item in numbered_items:
                match = re.match(r'^\s*(\d+)\.', item)
                if match:
                    numbers.append(int(match.group(1)))
            
            if numbers:
                numbers.sort()
                # Check for gaps in numbering
                expected = list(range(numbers[0], numbers[-1] + 1))
                if numbers != expected:
                    completeness_score -= 0.2
        
        # Check for incomplete headers
        headers = [line for line in lines if re.match(r'^#+\s+', line)]
        if headers:
            for i, header in enumerate(headers):
                # Check if header has content following it
                header_index = lines.index(header)
                has_content = False
                
                for j in range(header_index + 1, len(lines)):
                    if lines[j].strip():
                        if not re.match(r'^#+\s+', lines[j]):  # Not another header
                            has_content = True
                            break
                        else:  # Found another header
                            break
                
                if not has_content:
                    completeness_score -= 0.1
        
        return max(0.2, completeness_score)

    def _calculate_readability_flow(self, chunk: str) -> float:
        """
        Readability & Flow Score (15%): Doğal okuma akışını ölçer.
        
        Faktörler:
        - Natural sentence flow and transitions
        - Appropriate sentence length variation
        - Coherent paragraph structure
        - Smooth conceptual progression
        """
        readability_factors = []
        
        # 1. Sentence flow quality
        flow_score = self._measure_sentence_flow(chunk)
        readability_factors.append(('sentence_flow', flow_score, 0.4))
        
        # 2. Sentence length variation
        variation_score = self._measure_sentence_length_variation(chunk)
        readability_factors.append(('length_variation', variation_score, 0.3))
        
        # 3. Paragraph structure
        paragraph_score = self._measure_paragraph_structure(chunk)
        readability_factors.append(('paragraph_structure', paragraph_score, 0.3))
        
        # Ağırlıklı ortalama
        weighted_score = sum(score * weight for _, score, weight in readability_factors)
        return min(1.0, max(0.0, weighted_score))

    def _measure_sentence_flow(self, chunk: str) -> float:
        """Cümleler arası akışın kalitesini ölçer."""
        sentences = self._split_into_sentences(chunk)
        
        if len(sentences) < 2:
            return 0.7
        
        flow_score = 0.0
        transitions = 0
        
        for i in range(len(sentences) - 1):
            current_sentence = sentences[i].strip()
            next_sentence = sentences[i + 1].strip()
            
            transitions += 1
            
            # Good transition words
            if self.topic_transition_words.search(next_sentence):
                flow_score += 1.0
            
            # Reference continuity
            elif self.reference_words.search(next_sentence):
                flow_score += 0.8
            
            # Repeated key concepts
            elif self._has_conceptual_bridge(current_sentence, next_sentence):
                flow_score += 0.6
            
            # No clear transition (still acceptable)
            else:
                flow_score += 0.4
        
        return flow_score / max(1, transitions)

    def _measure_sentence_length_variation(self, chunk: str) -> float:
        """Cümle uzunluğu çeşitliliğini ölçer."""
        sentences = self._split_into_sentences(chunk)
        
        if not sentences:
            return 0.0
        
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        
        if len(sentence_lengths) < 2:
            # Single sentence: check if reasonable length (8-30 words)
            length = sentence_lengths[0]
            if 8 <= length <= 30:
                return 0.8
            elif 5 <= length <= 40:
                return 0.6
            else:
                return 0.3
        
        # Calculate variation coefficient
        mean_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)
        
        if mean_length == 0:
            return 0.0
        
        variation_coefficient = std_length / mean_length
        
        # İdeal variation: 0.3-0.7 arası
        if 0.3 <= variation_coefficient <= 0.7:
            return 1.0
        elif 0.2 <= variation_coefficient <= 0.8:
            return 0.8
        elif 0.1 <= variation_coefficient <= 1.0:
            return 0.6
        else:
            return 0.4

    def _measure_paragraph_structure(self, chunk: str) -> float:
        """Paragraf yapısının kalitesini ölçer."""
        paragraphs = [p.strip() for p in chunk.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return 0.3
        
        structure_score = 0.8  # Base score
        
        # Single very long paragraph is problematic
        if len(paragraphs) == 1 and len(paragraphs[0].split()) > 100:
            structure_score -= 0.3
        
        # Multiple very short paragraphs
        short_paragraphs = sum(1 for p in paragraphs if len(p.split()) < 3)
        if short_paragraphs > len(paragraphs) * 0.5:
            structure_score -= 0.2
        
        # Good paragraph length distribution (20-80 words per paragraph)
        good_length_paragraphs = sum(1 for p in paragraphs if 20 <= len(p.split()) <= 80)
        if good_length_paragraphs > 0:
            structure_score += 0.1
        
        return max(0.2, structure_score)

    # Helper methods
    def _split_into_sentences(self, text: str) -> List[str]:
        """Metni cümlelere böl."""
        sentences = self.sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Metinden anahtar kelimeleri çıkar."""
        words = self._tokenize_words(text.lower())
        keywords = []
        
        for word in words:
            if (len(word) >= min_length and 
                word not in self.turkish_stopwords and 
                word.isalpha()):
                keywords.append(word)
        
        return keywords

    def _tokenize_words(self, text: str) -> List[str]:
        """Metni kelimelere böl."""
        return re.findall(r'\b\w+\b', text)

    def _has_potential_antecedent(self, sentence: str, reference: str) -> bool:
        """Cümlede referans için potansiyel antecedent var mı?"""
        # Basit heuristic: isim ve kavram kelimelerine bak
        words = self._tokenize_words(sentence.lower())
        content_words = [w for w in words if len(w) > 3 and w not in self.turkish_stopwords]
        return len(content_words) > 0

    def _has_supporting_context_in_chunk(self, sentence: str, chunk: str) -> bool:
        """Cümlenin chunk içinde destekleyici bağlamı var mı?"""
        sentence_keywords = set(self._extract_keywords(sentence))
        chunk_keywords = set(self._extract_keywords(chunk))
        
        shared_keywords = sentence_keywords & chunk_keywords
        return len(shared_keywords) > len(sentence_keywords) * 0.3

    def _can_resolve_in_previous_chunk(self, sentence: str, previous_chunk: str) -> bool:
        """Referans önceki chunk'ta çözümlenebilir mi?"""
        if not previous_chunk:
            return False
        
        sentence_keywords = set(self._extract_keywords(sentence))
        prev_keywords = set(self._extract_keywords(previous_chunk))
        
        # At least 30% keyword overlap
        if not sentence_keywords:
            return False
        
        overlap = len(sentence_keywords & prev_keywords) / len(sentence_keywords)
        return overlap >= 0.3

    def _has_continuation_in_next_chunk(self, sentence: str, next_chunk: str) -> bool:
        """Cümlenin devamı bir sonraki chunk'ta var mı?"""
        if not next_chunk:
            return False
        
        # Simple check for thematic continuity
        sentence_keywords = set(self._extract_keywords(sentence))
        next_keywords = set(self._extract_keywords(next_chunk))
        
        if not sentence_keywords:
            return False
        
        overlap = len(sentence_keywords & next_keywords) / len(sentence_keywords)
        return overlap >= 0.2

    def _has_conceptual_bridge(self, sentence1: str, sentence2: str) -> bool:
        """İki cümle arasında kavramsal köprü var mı?"""
        keywords1 = set(self._extract_keywords(sentence1))
        keywords2 = set(self._extract_keywords(sentence2))
        
        if not keywords1 or not keywords2:
            return False
        
        intersection = keywords1 & keywords2
        return len(intersection) > 0

    def _create_detailed_analysis(self, chunk: str, coherence: float, context: float, 
                                completeness: float, readability: float) -> Dict:
        """Detaylı analiz raporu oluştur."""
        
        analysis = {
            'chunk_length': len(chunk),
            'sentence_count': len(self._split_into_sentences(chunk)),
            'word_count': len(chunk.split()),
            'paragraph_count': len([p for p in chunk.split('\n\n') if p.strip()]),
            'avg_sentence_length': 0,
            'keyword_count': len(self._extract_keywords(chunk)),
            'reference_count': len(self.reference_words.findall(chunk.lower())),
            'transition_count': len(self.topic_transition_words.findall(chunk.lower())),
            'quality_issues': [],
            'strengths': []
        }
        
        sentences = self._split_into_sentences(chunk)
        if sentences:
            word_counts = [len(s.split()) for s in sentences]
            analysis['avg_sentence_length'] = np.mean(word_counts)
        
        # Quality issues
        if coherence < 0.6:
            analysis['quality_issues'].append("Düşük anlamsal tutarlılık")
        if context < 0.5:
            analysis['quality_issues'].append("Bağlam korunması zayıf")
        if completeness < 0.6:
            analysis['quality_issues'].append("Bilgi tamamlanması yetersiz")
        if readability < 0.5:
            analysis['quality_issues'].append("Okunabilirlik problemi")
        
        # Strengths
        if coherence >= 0.8:
            analysis['strengths'].append("Güçlü anlamsal tutarlılık")
        if context >= 0.8:
            analysis['strengths'].append("İyi bağlam korunması")
        if completeness >= 0.8:
            analysis['strengths'].append("Tamamlanmış bilgi")
        if readability >= 0.8:
            analysis['strengths'].append("Akıcı okuma deneyimi")
        
        return analysis

    def _create_invalid_score(self, reason: str) -> ChunkQualityScore:
        """Geçersiz chunk için skor oluştur."""
        return ChunkQualityScore(
            semantic_coherence=0.0,
            context_preservation=0.0,
            information_completeness=0.0,
            readability_flow=0.0,
            overall_score=0.0,
            is_valid=False,
            detailed_analysis={'reason': reason, 'quality_issues': [reason]}
        )