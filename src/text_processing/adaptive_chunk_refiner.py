"""
Adaptive Semantic Chunk Refinement System

Bu modül düşük kaliteli chunk'ları LLM ile analiz ederek anlamsal olarak iyileştirir.

SORUN TESPİTİ:
❌ Semantic Coherence < 0.6 → Konu dağınıklığı
❌ Context Preservation < 0.5 → Bağlam kopukluğu  
❌ Information Completeness < 0.6 → Eksik bilgi
❌ Readability & Flow < 0.5 → Akış problemi

ÇÖZÜMLEr:
✅ LLM ile anlamsal sınır düzeltmesi
✅ Geçiş kelimesi ekleme
✅ Bağlam köprüsü oluşturma
✅ İçerik yeniden organizasyonu
"""

from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass
from ..utils.cloud_llm_client import get_cloud_llm_client
from ..utils.language_detector import detect_query_language
from ..utils.logger import get_logger
from ..config import get_config
from .advanced_chunk_validator import AdvancedChunkValidator, ChunkQualityScore


@dataclass
class ChunkRefinementSuggestion:
    """Chunk iyileştirme önerisi."""
    issue_type: str
    severity: float
    suggestion: str
    refined_content: str
    confidence: float


@dataclass
class RefinementResult:
    """Chunk iyileştirme sonucu."""
    original_chunk: str
    refined_chunks: List[str]
    improvement_score: float
    applied_fixes: List[str]
    success: bool
    reasoning: str


class AdaptiveChunkRefiner:
    """LLM destekli chunk kalite iyileştirici."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__, self.config)
        self.llm_client = get_cloud_llm_client()
        self.validator = AdvancedChunkValidator()
        
        # LLM model configuration
        self.refinement_model = "llama-3.1-8b-instant"
        
        # Quality thresholds for intervention
        self.coherence_threshold = 0.6
        self.context_threshold = 0.5
        self.completeness_threshold = 0.6
        self.readability_threshold = 0.5
        
        # Refinement strategies
        self.refinement_strategies = {
            'coherence': self._refine_semantic_coherence,
            'context': self._refine_context_preservation,
            'completeness': self._refine_information_completeness,
            'readability': self._refine_readability_flow
        }

    def refine_low_quality_chunk(
        self, 
        chunk: str, 
        quality_score: ChunkQualityScore,
        previous_chunk: str = None,
        next_chunk: str = None,
        language: str = "auto"
    ) -> RefinementResult:
        """
        Düşük kaliteli chunk'ı LLM ile analiz ederek iyileştir.
        
        Args:
            chunk: İyileştirilecek chunk
            quality_score: Mevcut kalite skoru
            previous_chunk: Önceki chunk (bağlam için)
            next_chunk: Sonraki chunk (bağlam için)
            language: Metin dili
            
        Returns:
            İyileştirme sonucu
        """
        if language == "auto":
            language = detect_query_language(chunk)
        
        self.logger.info(f"Chunk iyileştirme başlatıldı (skor: {quality_score.overall_score:.3f})")
        
        # Hangi alanlar iyileştirme gerektirir?
        issues = self._identify_refinement_needs(quality_score)
        
        if not issues:
            return RefinementResult(
                original_chunk=chunk,
                refined_chunks=[chunk],
                improvement_score=0.0,
                applied_fixes=[],
                success=True,
                reasoning="Chunk kalitesi iyileştirme eşiğinin üzerinde"
            )
        
        # Her sorun için iyileştirme önerileri al
        refinement_suggestions = []
        for issue in issues:
            suggestion = self._get_refinement_suggestion(
                chunk, issue, quality_score, previous_chunk, next_chunk, language
            )
            if suggestion:
                refinement_suggestions.append(suggestion)
        
        # En iyi iyileştirme stratejisini uygula
        if refinement_suggestions:
            return self._apply_best_refinement(chunk, refinement_suggestions, quality_score)
        else:
            # LLM iyileştirme başarısız, mekanik fallback
            return self._fallback_mechanical_split(chunk)

    def _identify_refinement_needs(self, quality_score: ChunkQualityScore) -> List[str]:
        """Hangi alanların iyileştirme gerektiğini belirle."""
        issues = []
        
        if quality_score.semantic_coherence < self.coherence_threshold:
            issues.append('coherence')
        if quality_score.context_preservation < self.context_threshold:
            issues.append('context')
        if quality_score.information_completeness < self.completeness_threshold:
            issues.append('completeness')
        if quality_score.readability_flow < self.readability_threshold:
            issues.append('readability')
        
        return issues

    def _get_refinement_suggestion(
        self,
        chunk: str,
        issue_type: str,
        quality_score: ChunkQualityScore,
        previous_chunk: str,
        next_chunk: str,
        language: str
    ) -> Optional[ChunkRefinementSuggestion]:
        """Belirli bir sorun türü için LLM'den iyileştirme önerisi al."""
        
        try:
            strategy_func = self.refinement_strategies.get(issue_type)
            if not strategy_func:
                return None
            
            return strategy_func(chunk, quality_score, previous_chunk, next_chunk, language)
            
        except Exception as e:
            self.logger.warning(f"Refinement suggestion hatası ({issue_type}): {e}")
            return None

    def _refine_semantic_coherence(
        self,
        chunk: str,
        quality_score: ChunkQualityScore,
        previous_chunk: str,
        next_chunk: str,
        language: str
    ) -> Optional[ChunkRefinementSuggestion]:
        """Anlamsal tutarlılığı iyileştir."""
        
        if language == "tr" or language == "turkish":
            system_prompt = """Sen bir metin düzenleme uzmanısın. Verilen chunk'ın anlamsal tutarlılığını artırmak için yeniden düzenliyorsun.

GÖREVIN:
1. Chunk'taki konu dağınıklığını tespit et
2. İlgisiz cümleleri ayıkla veya gruplara böl  
3. Benzer konuları bir araya getir
4. Geçiş cümlelerini güçlendir
5. Ana fikri netleştir

Yanıtını JSON formatında ver."""
            
            user_prompt = f"""Bu chunk'ın anlamsal tutarlılığını artır:

CHUNK:
{chunk}

MEVCUT COHERENCE SKORU: {quality_score.semantic_coherence:.3f}
HEDEF: 0.7+

Çıktı formatı:
{{
    "issue_analysis": "Tespit edilen sorunlar",
    "refined_content": "İyileştirilmiş metin",
    "improvement_reasoning": "Neden bu değişiklikleri yaptın",
    "confidence": 0.8
}}"""
        else:
            system_prompt = """You are a text editing expert. Your task is to improve semantic coherence of the given chunk by reorganizing content for better topic consistency.

YOUR TASKS:
1. Identify topic scatter in the chunk
2. Remove or group unrelated sentences
3. Group similar topics together  
4. Strengthen transitions
5. Clarify main ideas

Respond in JSON format."""
            
            user_prompt = f"""Improve semantic coherence of this chunk:

CHUNK:
{chunk}

CURRENT COHERENCE SCORE: {quality_score.semantic_coherence:.3f}
TARGET: 0.7+

Output format:
{{
    "issue_analysis": "Identified issues",
    "refined_content": "Improved text",
    "improvement_reasoning": "Why you made these changes",
    "confidence": 0.8
}}"""
        
        try:
            response = self.llm_client.generate(
                model=self.refinement_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            suggestion_data = json.loads(response)
            
            return ChunkRefinementSuggestion(
                issue_type='coherence',
                severity=1.0 - quality_score.semantic_coherence,
                suggestion=suggestion_data.get('issue_analysis', ''),
                refined_content=suggestion_data.get('refined_content', chunk),
                confidence=suggestion_data.get('confidence', 0.7)
            )
            
        except Exception as e:
            self.logger.error(f"Coherence refinement hatası: {e}")
            return None

    def _refine_context_preservation(
        self,
        chunk: str,
        quality_score: ChunkQualityScore,
        previous_chunk: str,
        next_chunk: str,
        language: str
    ) -> Optional[ChunkRefinementSuggestion]:
        """Bağlam korunmasını iyileştir."""
        
        context_info = ""
        if previous_chunk:
            context_info += f"ÖNCEKİ CHUNK: {previous_chunk[-100:]}...\n\n"
        if next_chunk:
            context_info += f"SONRAKİ CHUNK: {next_chunk[:100]}...\n\n"
        
        if language == "tr" or language == "turkish":
            system_prompt = """Sen bir bağlam analizi uzmanısın. Chunk'ların arasındaki bağlamsal bütünlüğü koruyacak şekilde düzenleme yapıyorsun.

GÖREVIN:
1. Çözülemeyen referansları tespit et (bu, o, bunlar, vb.)
2. Eksik bağlam bilgilerini ekle  
3. Önceki/sonraki chunk ile uyumunu artır
4. Geçiş köprülerini güçlendir
5. Bağımsız anlaşılabilirliği artır

JSON formatında yanıt ver."""
            
            user_prompt = f"""Bu chunk'ın bağlam korunmasını iyileştir:

{context_info}DÜZENLENECEK CHUNK:
{chunk}

MEVCUT CONTEXT SKORU: {quality_score.context_preservation:.3f}
HEDEF: 0.7+

Çıktı formatı:
{{
    "context_issues": "Bağlam sorunları",
    "refined_content": "İyileştirilmiş metin",  
    "improvement_reasoning": "Yapılan iyileştirmeler",
    "confidence": 0.8
}}"""
        else:
            system_prompt = """You are a context analysis expert. You improve contextual coherence between chunks while maintaining standalone comprehensibility.

YOUR TASKS:
1. Identify unresolvable references (this, that, these, etc.)
2. Add missing context information
3. Improve compatibility with previous/next chunks
4. Strengthen transitional bridges
5. Enhance standalone understandability

Respond in JSON format."""
            
            user_prompt = f"""Improve context preservation of this chunk:

{context_info}CHUNK TO REFINE:
{chunk}

CURRENT CONTEXT SCORE: {quality_score.context_preservation:.3f}
TARGET: 0.7+

Output format:
{{
    "context_issues": "Context problems",
    "refined_content": "Improved text",
    "improvement_reasoning": "Applied improvements",
    "confidence": 0.8
}}"""
        
        try:
            response = self.llm_client.generate(
                model=self.refinement_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            suggestion_data = json.loads(response)
            
            return ChunkRefinementSuggestion(
                issue_type='context',
                severity=1.0 - quality_score.context_preservation,
                suggestion=suggestion_data.get('context_issues', ''),
                refined_content=suggestion_data.get('refined_content', chunk),
                confidence=suggestion_data.get('confidence', 0.7)
            )
            
        except Exception as e:
            self.logger.error(f"Context refinement hatası: {e}")
            return None

    def _refine_information_completeness(
        self,
        chunk: str,
        quality_score: ChunkQualityScore,
        previous_chunk: str,
        next_chunk: str,
        language: str
    ) -> Optional[ChunkRefinementSuggestion]:
        """Bilgi tamamlanmasını iyileştir."""
        
        if language == "tr" or language == "turkish":
            system_prompt = """Sen bir içerik tamamlama uzmanısın. Chunk'ların eksik bilgilerini tespir edip tamamlıyorsun.

GÖREVIN:
1. Yarım kalan fikirleri tespit et
2. Destekleyici detay eksikliklerini bul
3. Ana fikir-destekleyici bilgi dengesini kur
4. Eksik sonuç/özet cümlelerini ekle
5. Yapısal bütünlüğü sağla

JSON formatında yanıt ver."""
            
            user_prompt = f"""Bu chunk'ın bilgi tamamlanmasını iyileştir:

CHUNK:
{chunk}

MEVCUT COMPLETENESS SKORU: {quality_score.information_completeness:.3f}
HEDEF: 0.7+

Çıktı formatı:
{{
    "completeness_issues": "Eksik bilgi alanları",
    "refined_content": "Tamamlanmış metin",
    "improvement_reasoning": "Eklenen bilgiler",
    "confidence": 0.8
}}"""
        else:
            system_prompt = """You are a content completion expert. You identify and complete missing information in chunks.

YOUR TASKS:
1. Identify incomplete ideas
2. Find missing supporting details
3. Balance main ideas with supporting information
4. Add missing conclusion/summary sentences
5. Ensure structural completeness

Respond in JSON format."""
            
            user_prompt = f"""Improve information completeness of this chunk:

CHUNK:
{chunk}

CURRENT COMPLETENESS SCORE: {quality_score.information_completeness:.3f}
TARGET: 0.7+

Output format:
{{
    "completeness_issues": "Missing information areas",
    "refined_content": "Completed text",
    "improvement_reasoning": "Added information",
    "confidence": 0.8
}}"""
        
        try:
            response = self.llm_client.generate(
                model=self.refinement_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            suggestion_data = json.loads(response)
            
            return ChunkRefinementSuggestion(
                issue_type='completeness',
                severity=1.0 - quality_score.information_completeness,
                suggestion=suggestion_data.get('completeness_issues', ''),
                refined_content=suggestion_data.get('refined_content', chunk),
                confidence=suggestion_data.get('confidence', 0.7)
            )
            
        except Exception as e:
            self.logger.error(f"Completeness refinement hatası: {e}")
            return None

    def _refine_readability_flow(
        self,
        chunk: str,
        quality_score: ChunkQualityScore,
        previous_chunk: str,
        next_chunk: str,
        language: str
    ) -> Optional[ChunkRefinementSuggestion]:
        """Okunabilirlik ve akışı iyileştir."""
        
        if language == "tr" or language == "turkish":
            system_prompt = """Sen bir metin akışı uzmanısın. Chunk'ların okunabilirliğini ve doğal akışını iyileştiriyorsun.

GÖREVIN:
1. Cümle geçişlerini güçlendir
2. Uygun bağlaçları ekle
3. Cümle uzunluk çeşitliliğini artır
4. Paragraf yapısını düzenle
5. Doğal okuma ritmi oluştur

JSON formatında yanıt ver."""
            
            user_prompt = f"""Bu chunk'ın okunabilirlik ve akışını iyileştir:

CHUNK:
{chunk}

MEVCUT READABILITY SKORU: {quality_score.readability_flow:.3f}
HEDEF: 0.7+

Çıktı formatı:
{{
    "flow_issues": "Akış sorunları",
    "refined_content": "Akışı iyileştirilmiş metin",
    "improvement_reasoning": "Yapılan düzeltmeler",
    "confidence": 0.8
}}"""
        else:
            system_prompt = """You are a text flow expert. You improve readability and natural flow of chunks.

YOUR TASKS:
1. Strengthen sentence transitions
2. Add appropriate conjunctions
3. Increase sentence length variety
4. Organize paragraph structure
5. Create natural reading rhythm

Respond in JSON format."""
            
            user_prompt = f"""Improve readability and flow of this chunk:

CHUNK:
{chunk}

CURRENT READABILITY SCORE: {quality_score.readability_flow:.3f}
TARGET: 0.7+

Output format:
{{
    "flow_issues": "Flow problems",
    "refined_content": "Flow-improved text",
    "improvement_reasoning": "Applied corrections",
    "confidence": 0.8
}}"""
        
        try:
            response = self.llm_client.generate(
                model=self.refinement_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            suggestion_data = json.loads(response)
            
            return ChunkRefinementSuggestion(
                issue_type='readability',
                severity=1.0 - quality_score.readability_flow,
                suggestion=suggestion_data.get('flow_issues', ''),
                refined_content=suggestion_data.get('refined_content', chunk),
                confidence=suggestion_data.get('confidence', 0.7)
            )
            
        except Exception as e:
            self.logger.error(f"Readability refinement hatası: {e}")
            return None

    def _apply_best_refinement(
        self,
        original_chunk: str,
        suggestions: List[ChunkRefinementSuggestion],
        original_quality: ChunkQualityScore
    ) -> RefinementResult:
        """En iyi iyileştirme önerisini uygula."""
        
        # En yüksek confidence'a sahip öneriyi seç
        best_suggestion = max(suggestions, key=lambda s: s.confidence * s.severity)
        
        refined_chunk = best_suggestion.refined_content
        
        # İyileştirilmiş chunk'ı değerlendir
        new_quality = self.validator.validate_chunk_quality(refined_chunk)
        improvement_score = new_quality.overall_score - original_quality.overall_score
        
        applied_fixes = [s.issue_type for s in suggestions]
        
        success = improvement_score > 0.05  # En az %5 iyileşme
        
        reasoning = f"En kritik sorun ({best_suggestion.issue_type}) için LLM iyileştirmesi uygulandı. "
        reasoning += f"Confidence: {best_suggestion.confidence:.2f}, "
        reasoning += f"Skor değişimi: {improvement_score:+.3f}"
        
        if success:
            self.logger.info(f"Chunk başarıyla iyileştirildi: {improvement_score:+.3f}")
            return RefinementResult(
                original_chunk=original_chunk,
                refined_chunks=[refined_chunk],
                improvement_score=improvement_score,
                applied_fixes=applied_fixes,
                success=True,
                reasoning=reasoning
            )
        else:
            # İyileştirme yeterli değil, mekanik bölme dene
            self.logger.warning(f"LLM iyileştirmesi yetersiz ({improvement_score:+.3f}), mekanik bölmeye geçiliyor")
            return self._fallback_mechanical_split(original_chunk)

    def _fallback_mechanical_split(self, chunk: str) -> RefinementResult:
        """LLM iyileştirme başarısızsa mekanik bölme uygula."""
        
        sentences = chunk.split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        if len(sentences) > 2:
            # Ortadan böl
            mid_point = len(sentences) // 2
            first_half = ' '.join(sentences[:mid_point])
            second_half = ' '.join(sentences[mid_point:])
            
            return RefinementResult(
                original_chunk=chunk,
                refined_chunks=[first_half, second_half],
                improvement_score=0.1,  # Minimal mechanical improvement
                applied_fixes=['mechanical_split'],
                success=True,
                reasoning="LLM iyileştirme başarısız, mekanik cümle bölmesi uygulandı"
            )
        else:
            # Çok kısa, bölemez
            return RefinementResult(
                original_chunk=chunk,
                refined_chunks=[chunk],
                improvement_score=0.0,
                applied_fixes=[],
                success=False,
                reasoning="Chunk çok kısa, iyileştirme uygulanamadı"
            )

    def should_refine_chunk(self, quality_score: ChunkQualityScore) -> bool:
        """Bu chunk'ın iyileştirme gerekip gerekmediğini belirle."""
        return (
            quality_score.semantic_coherence < self.coherence_threshold or
            quality_score.context_preservation < self.context_threshold or
            quality_score.information_completeness < self.completeness_threshold or
            quality_score.readability_flow < self.readability_threshold
        )