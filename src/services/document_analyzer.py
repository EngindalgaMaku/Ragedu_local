"""
Document Analyzer and Automatic Prompt Suggestion System
Döküman Analizi ve Otomatik Prompt Önerisi Sistemi

Bu modül yüklenen dökümanları LLM ile analiz eder ve içeriğe uygun 
prompt önerileri oluşturur.
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import ollama
from datetime import datetime

from src import config as app_config
from src.utils.language_detector import detect_query_language
from src.utils.cloud_llm_client import CloudLLMClient
from src.services.prompt_manager import (
    TeacherPromptManager, CustomPrompt, PromptCommand,
    PromptCategory, PromptComplexity, teacher_prompt_manager
)

# Enhanced document processing
try:
    from src.document_processing.enhanced_pdf_processor import (
        process_pdf_with_analysis, MARKER_AVAILABLE
    )
    ENHANCED_PDF_AVAILABLE = True
except ImportError:
    ENHANCED_PDF_AVAILABLE = False
    process_pdf_with_analysis = None


class ContentType(Enum):
    """Döküman içerik türleri"""
    TEXTBOOK = "textbook"           # Ders kitabı
    LECTURE_NOTES = "lecture_notes" # Ders notları  
    EXERCISE = "exercise"           # Alıştırma/Ödev
    EXAM = "exam"                   # Sınav/Test
    PRESENTATION = "presentation"   # Sunum
    RESEARCH = "research"           # Araştırma makalesi
    REFERENCE = "reference"         # Referans/El kitabı
    STORY = "story"                 # Hikaye/Edebiyat
    MIXED = "mixed"                 # Karışık içerik
    UNKNOWN = "unknown"             # Belirsiz


class EducationalLevel(Enum):
    """Eğitim seviyeleri"""
    PRESCHOOL = "preschool"         # Anaokulu
    ELEMENTARY = "elementary"       # İlkokul (1-4)
    MIDDLE_SCHOOL = "middle_school" # Ortaokul (5-8)
    HIGH_SCHOOL = "high_school"     # Lise (9-12)
    UNIVERSITY = "university"       # Üniversite
    GRADUATE = "graduate"           # Lisansüstü
    ADULT = "adult"                 # Yetişkin eğitimi
    UNKNOWN = "unknown"             # Belirsiz


@dataclass
class DocumentAnalysis:
    """Döküman analizi sonucu"""
    # Temel bilgiler
    filename: str
    content_type: ContentType
    educational_level: EducationalLevel
    subject_area: str
    language: str
    
    # İçerik analizi
    main_topics: List[str]
    difficulty_score: float  # 0-1 arası, 1 en zor
    complexity_indicators: List[str]
    
    # Yapısal özellikler
    has_examples: bool
    has_exercises: bool
    has_questions: bool
    has_images_tables: bool
    
    # Eğitsel özellikler
    teaching_methods: List[str]  # "görsel", "adım-adım", "analoji", vb.
    target_skills: List[str]     # "problem-çözme", "analiz", "sentez", vb.
    
    # Metrikler
    word_count: int
    readability_score: float
    concept_density: float
    
    # Analiz metadatası
    analyzed_at: str
    confidence_score: float  # Analizin güvenilirlik skoru
    
    # Ek notlar ve öneriler
    analysis_notes: str
    improvement_suggestions: List[str]


@dataclass
class PromptSuggestion:
    """Otomatik prompt önerisi"""
    # Temel bilgiler
    suggested_name: str
    suggested_description: str
    prompt_template: str
    category: PromptCategory
    complexity: PromptComplexity
    
    # Özelleştirme
    variables: List[str]
    suggested_tags: List[str]
    
    # Önerilme nedeni
    reasoning: str
    confidence: float
    
    # Kullanım önerileri
    use_cases: List[str]
    example_inputs: Dict[str, str]
    
    # Komut önerisi (opsiyonel)
    suggested_command: Optional[str] = None
    command_parameters: Optional[List[str]] = None


class DocumentAnalyzer:
    """Döküman analizi ve prompt önerisi sistemi"""
    
    def __init__(self):
        self.prompt_manager = teacher_prompt_manager
    
    def analyze_document(
        self,
        content: str,
        filename: str,
        generation_model: Optional[str] = None
    ) -> DocumentAnalysis:
        """
        Döküman içeriğini LLM ile analiz eder.
        """
        try:
            # Dil tespiti
            detected_language = detect_query_language(content[:1000])
            
            # Analiz prompt'u oluştur
            analysis_prompt = self._create_analysis_prompt(content, filename, detected_language)
            
            # LLM ile analiz yap
            analysis_result = self._execute_analysis(analysis_prompt, generation_model)
            
            # Sonucu parse et
            document_analysis = self._parse_analysis_result(
                analysis_result, filename, detected_language, content
            )
            
            return document_analysis
            
        except Exception as e:
            # Hata durumunda basit analiz döndür
            return self._create_fallback_analysis(filename, content, str(e))
    
    def analyze_pdf_with_enhanced_processing(
        self,
        pdf_path: str,
        generation_model: Optional[str] = None
    ) -> Tuple[DocumentAnalysis, Dict[str, Any]]:
        """
        PDF'i enhanced processing ile analiz eder (Marker kullanarak)
        
        Returns:
            Tuple of (document_analysis, processing_metadata)
        """
        try:
            if ENHANCED_PDF_AVAILABLE and MARKER_AVAILABLE:
                # Marker ile gelişmiş PDF işleme
                content, processing_metadata = process_pdf_with_analysis(pdf_path)
                
                # İçerik varsa analiz et
                if content:
                    document_analysis = self.analyze_document(
                        content,
                        os.path.basename(pdf_path),
                        generation_model
                    )
                    
                    # Processing metadata'yı document analysis'e ekle
                    document_analysis.analysis_notes += f"\n\n[PDF Processing: {processing_metadata.get('processing_method', 'unknown')}]"
                    if processing_metadata.get('extraction_quality'):
                        document_analysis.analysis_notes += f"\n[Quality: {processing_metadata['extraction_quality']}]"
                    
                    return document_analysis, processing_metadata
                else:
                    raise ValueError("PDF içeriği çıkarılamadı")
            else:
                # Fallback: Normal PDF processing
                from src.document_processing.document_processor import process_document
                content = process_document(pdf_path)
                
                if content:
                    document_analysis = self.analyze_document(
                        content,
                        os.path.basename(pdf_path),
                        generation_model
                    )
                    
                    fallback_metadata = {
                        "processing_method": "basic_pdf",
                        "extraction_quality": "basic",
                        "marker_available": False
                    }
                    
                    return document_analysis, fallback_metadata
                else:
                    raise ValueError("PDF içeriği çıkarılamadı (fallback)")
                    
        except Exception as e:
            # Hata durumunda minimal analiz döndür
            error_analysis = self._create_fallback_analysis(
                os.path.basename(pdf_path),
                "",
                f"PDF analizi başarısız: {str(e)}"
            )
            
            error_metadata = {
                "processing_method": "failed",
                "error": str(e),
                "marker_available": ENHANCED_PDF_AVAILABLE and MARKER_AVAILABLE
            }
            
            return error_analysis, error_metadata
    
    def _create_analysis_prompt(self, content: str, filename: str, language: str) -> str:
        """Döküman analizi için LLM prompt'u oluştur"""
        
        # İçeriği kısalt (LLM token limitine uygun)
        max_chars = 4000
        sample_content = content[:max_chars]
        if len(content) > max_chars:
            sample_content += "\n... (devamı var)"
        
        if language == 'tr':
            return f"""Sen bir eğitim uzmanısın. Aşağıdaki dökümanı detaylı şekilde analiz et ve JSON formatında rapor ver.

DOSYA ADI: {filename}

İÇERİK ÖRNEĞİ:
{sample_content}

Lütfen aşağıdaki bilgileri JSON formatında ver:

{{
  "content_type": "textbook|lecture_notes|exercise|exam|presentation|research|reference|story|mixed|unknown",
  "educational_level": "preschool|elementary|middle_school|high_school|university|graduate|adult|unknown",
  "subject_area": "matematik|fen|türkçe|sosyal|tarih|coğrafya|biyoloji|kimya|fizik|ingilizce|bilgisayar|sanat|müzik|beden|other",
  "main_topics": ["konu1", "konu2", "konu3"],
  "difficulty_score": 0.0-1.0,
  "complexity_indicators": ["karmaşık kavramlar", "formüller", "teknik terimler"],
  "has_examples": true/false,
  "has_exercises": true/false,
  "has_questions": true/false,
  "has_images_tables": true/false,
  "teaching_methods": ["görsel", "adım-adım", "analoji", "problem-çözme", "tartışma"],
  "target_skills": ["anlama", "analiz", "sentez", "değerlendirme", "uygulama"],
  "readability_score": 0.0-1.0,
  "concept_density": 0.0-1.0,
  "confidence_score": 0.0-1.0,
  "analysis_notes": "Bu dökümana dair notlar",
  "improvement_suggestions": ["öneri1", "öneri2"]
}}

SADECE JSON döndür, başka açıklama yapma:"""

        else:  # English
            return f"""You are an education expert. Analyze the following document in detail and provide a report in JSON format.

FILENAME: {filename}

CONTENT SAMPLE:
{sample_content}

Please provide the following information in JSON format:

{{
  "content_type": "textbook|lecture_notes|exercise|exam|presentation|research|reference|story|mixed|unknown",
  "educational_level": "preschool|elementary|middle_school|high_school|university|graduate|adult|unknown", 
  "subject_area": "mathematics|science|language|social|history|geography|biology|chemistry|physics|english|computer|art|music|physical|other",
  "main_topics": ["topic1", "topic2", "topic3"],
  "difficulty_score": 0.0-1.0,
  "complexity_indicators": ["complex concepts", "formulas", "technical terms"],
  "has_examples": true/false,
  "has_exercises": true/false,
  "has_questions": true/false,
  "has_images_tables": true/false,
  "teaching_methods": ["visual", "step-by-step", "analogy", "problem-solving", "discussion"],
  "target_skills": ["comprehension", "analysis", "synthesis", "evaluation", "application"],
  "readability_score": 0.0-1.0,
  "concept_density": 0.0-1.0,
  "confidence_score": 0.0-1.0,
  "analysis_notes": "Notes about this document",
  "improvement_suggestions": ["suggestion1", "suggestion2"]
}}

Return ONLY JSON, no other explanation:"""
    
    def _execute_analysis(self, prompt: str, generation_model: Optional[str]) -> str:
        """LLM ile analizi çalıştır"""
        
        model_to_use = generation_model or app_config.OLLAMA_GENERATION_MODEL
        
        try:
            if app_config.LLM_PROVIDER == "cloud":
                cloud_client = CloudLLMClient()
                result = cloud_client.generate(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "Sen eğitim uzmanı bir analistsin. Dökümanları objektif şekilde analiz edersin."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Düşük temperature - tutarlı sonuçlar için
                    max_tokens=1024
                )
                return result
            else:
                # Ollama kullan
                client = ollama.Client(host=app_config.OLLAMA_BASE_URL)
                response = client.chat(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "Sen eğitim uzmanı bir analistsin. Dökümanları objektif şekilde analiz edersin."},
                        {"role": "user", "content": prompt}
                    ],
                    options={
                        "temperature": 0.1,
                        "num_predict": 1024
                    }
                )
                return response.get("message", {}).get("content", "")
        
        except Exception as e:
            raise Exception(f"LLM analizi başarısız: {e}")
    
    def _parse_analysis_result(
        self, 
        llm_response: str, 
        filename: str, 
        language: str, 
        original_content: str
    ) -> DocumentAnalysis:
        """LLM yanıtını parse ederek DocumentAnalysis oluştur"""
        
        try:
            # JSON'u temizle ve parse et
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis_data = json.loads(json_str)
            else:
                raise ValueError("JSON bulunamadı")
            
            # Enum değerlerini dönüştür
            content_type = ContentType(analysis_data.get("content_type", "unknown"))
            educational_level = EducationalLevel(analysis_data.get("educational_level", "unknown"))
            
            # Temel metrikler hesapla
            word_count = len(original_content.split())
            
            return DocumentAnalysis(
                filename=filename,
                content_type=content_type,
                educational_level=educational_level,
                subject_area=analysis_data.get("subject_area", "other"),
                language=language,
                main_topics=analysis_data.get("main_topics", []),
                difficulty_score=float(analysis_data.get("difficulty_score", 0.5)),
                complexity_indicators=analysis_data.get("complexity_indicators", []),
                has_examples=analysis_data.get("has_examples", False),
                has_exercises=analysis_data.get("has_exercises", False),
                has_questions=analysis_data.get("has_questions", False),
                has_images_tables=analysis_data.get("has_images_tables", False),
                teaching_methods=analysis_data.get("teaching_methods", []),
                target_skills=analysis_data.get("target_skills", []),
                word_count=word_count,
                readability_score=float(analysis_data.get("readability_score", 0.5)),
                concept_density=float(analysis_data.get("concept_density", 0.5)),
                analyzed_at=datetime.now().isoformat(),
                confidence_score=float(analysis_data.get("confidence_score", 0.7)),
                analysis_notes=analysis_data.get("analysis_notes", ""),
                improvement_suggestions=analysis_data.get("improvement_suggestions", [])
            )
            
        except Exception as e:
            # Parse hatası durumunda fallback
            return self._create_fallback_analysis(filename, original_content, f"Parse hatası: {e}")
    
    def _create_fallback_analysis(self, filename: str, content: str, error: str) -> DocumentAnalysis:
        """Hata durumunda basit analiz oluştur"""
        
        language = detect_query_language(content[:500])
        word_count = len(content.split())
        
        # Basit kural tabanlı analiz
        subject_area = "other"
        if any(word in filename.lower() for word in ["matematik", "math"]):
            subject_area = "matematik"
        elif any(word in filename.lower() for word in ["fen", "science", "fizik", "biyoloji", "kimya"]):
            subject_area = "fen"
        elif any(word in filename.lower() for word in ["türkçe", "turkish", "edebiyat"]):
            subject_area = "türkçe"
        elif any(word in filename.lower() for word in ["tarih", "history", "sosyal"]):
            subject_area = "sosyal"
        
        return DocumentAnalysis(
            filename=filename,
            content_type=ContentType.UNKNOWN,
            educational_level=EducationalLevel.UNKNOWN,
            subject_area=subject_area,
            language=language,
            main_topics=[],
            difficulty_score=0.5,
            complexity_indicators=[],
            has_examples=False,
            has_exercises=False,
            has_questions=False,
            has_images_tables=False,
            teaching_methods=[],
            target_skills=[],
            word_count=word_count,
            readability_score=0.5,
            concept_density=0.5,
            analyzed_at=datetime.now().isoformat(),
            confidence_score=0.3,
            analysis_notes=f"Otomatik analiz başarısız. Hata: {error}",
            improvement_suggestions=["Manuel analiz önerilir"]
        )
    
    def generate_prompt_suggestions(
        self, 
        analysis: DocumentAnalysis,
        max_suggestions: int = 5
    ) -> List[PromptSuggestion]:
        """Döküman analizine göre prompt önerileri oluştur"""
        
        suggestions = []
        
        # 1. İçerik tipine göre öneriler
        suggestions.extend(self._suggest_by_content_type(analysis))
        
        # 2. Eğitim seviyesine göre öneriler  
        suggestions.extend(self._suggest_by_educational_level(analysis))
        
        # 3. Konu alanına göre öneriler
        suggestions.extend(self._suggest_by_subject_area(analysis))
        
        # 4. Yapısal özelliklere göre öneriler
        suggestions.extend(self._suggest_by_structural_features(analysis))
        
        # 5. Öğretim yöntemlerine göre öneriler
        suggestions.extend(self._suggest_by_teaching_methods(analysis))
        
        # Güven skoruna göre sırala ve sınırla
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions[:max_suggestions]
    
    def _suggest_by_content_type(self, analysis: DocumentAnalysis) -> List[PromptSuggestion]:
        """İçerik tipine göre prompt önerileri"""
        
        suggestions = []
        lang_code = analysis.language
        
        if analysis.content_type == ContentType.TEXTBOOK:
            suggestions.append(PromptSuggestion(
                suggested_name="Ders Kitabı Açıklama" if lang_code == 'tr' else "Textbook Explanation",
                suggested_description="Ders kitabı içeriğini öğrenci seviyesine uygun şekilde açıklar",
                prompt_template=f"""Sen bir {analysis.subject_area} öğretmenisin. Verilen ders kitabı içeriğini {{seviye}} seviyesindeki öğrenciler için {{aciklama_stili}} şeklinde açıkla.

İÇERİK: {{icerik}}
ÖĞRENCİ SEVİYESİ: {{seviye}}
AÇIKLAMA STİLİ: {{aciklama_stili}}

KURALLAR:
- Ders kitabının dil seviyesine uygun açıkla
- Ana kavramları vurgula
- Kitaptaki örnekleri referans al
- Konu bütünlüğünü koru

AÇIKLAMA:""",
                category=PromptCategory.EDUCATIONAL,
                complexity=PromptComplexity.INTERMEDIATE,
                variables=["icerik", "seviye", "aciklama_stili"],
                suggested_tags=["ders-kitabı", analysis.subject_area, "açıklama"],
                reasoning="Ders kitabı içeriği tespit edildi, sistematik açıklama gerekiyor",
                confidence=0.9,
                use_cases=["Konu anlatımı", "Ders tekrarı", "Sınıf içi açıklama"],
                example_inputs={
                    "icerik": f"{analysis.main_topics[0] if analysis.main_topics else 'Temel kavramlar'}",
                    "seviye": analysis.educational_level.value,
                    "aciklama_stili": "adım-adım"
                },
                suggested_command="/kitap-aciklaruhi",
                command_parameters=["icerik", "seviye", "aciklama_stili"]
            ))
        
        elif analysis.content_type == ContentType.EXERCISE:
            suggestions.append(PromptSuggestion(
                suggested_name="Alıştırma Rehberi" if lang_code == 'tr' else "Exercise Guide",
                suggested_description="Alıştırma sorularını çözme sürecinde rehberlik eder",
                prompt_template=f"""Sen bir {analysis.subject_area} öğretmenisin. Öğrencinin alıştırma sorusunu çözmesine adım adım rehberlik et.

SORU: {{soru}}
ÖĞRENCİ SEVİYESİ: {{seviye}}
REHBERLIK TİPİ: {{tip}}

KURALLAR:
- Doğrudan cevap verme, rehberlik et
- Her adımı açıkla
- Öğrenciyi düşünmeye teşvik et
- Hatalarını fark etmesini sağla

REHBERLİK:""",
                category=PromptCategory.EDUCATIONAL,
                complexity=PromptComplexity.ADVANCED,
                variables=["soru", "seviye", "tip"],
                suggested_tags=["alıştırma", "rehberlik", analysis.subject_area],
                reasoning="Alıştırma içeriği tespit edildi, adım adım rehberlik gerekiyor",
                confidence=0.85,
                use_cases=["Ödev yardımı", "Sınıf içi problem çözme", "Bireysel rehberlik"],
                example_inputs={
                    "soru": "Örnek alıştırma sorusu",
                    "seviye": analysis.educational_level.value,
                    "tip": "adım-adım"
                },
                suggested_command="/alistirma-rehberi",
                command_parameters=["soru", "seviye", "tip"]
            ))
        
        return suggestions
    
    def _suggest_by_educational_level(self, analysis: DocumentAnalysis) -> List[PromptSuggestion]:
        """Eğitim seviyesine göre öneriler"""
        
        suggestions = []
        
        if analysis.educational_level == EducationalLevel.ELEMENTARY:
            suggestions.append(PromptSuggestion(
                suggested_name="İlkokul Seviye Açıklama",
                suggested_description="Konuları ilkokul öğrencilerine uygun basit dilde açıklar",
                prompt_template="""Sen çok sabırlı ve sevecen bir öğretmensin. {konu} konusunu ilkokul {sinif} öğrencileri için en basit şekilde açıkla.

KONU: {konu}
SINIF: {sinif}
ANLATIM STİLİ: {stil}

KURALLAR:
- Çok basit kelimeler kullan
- Günlük hayattan örnekler ver
- Renkli ve eğlenceli anlatım yap
- Kısa cümleler kur
- Merak uyandır

İLKOKUL AÇIKLAMASI:""",
                category=PromptCategory.EDUCATIONAL,
                complexity=PromptComplexity.BEGINNER,
                variables=["konu", "sinif", "stil"],
                suggested_tags=["ilkokul", "basit", "çocuklar"],
                reasoning="İlkokul seviyesi içerik tespit edildi",
                confidence=0.8,
                use_cases=["İlkokul ders anlatımı", "Çocuk dostu açıklama"],
                example_inputs={
                    "konu": analysis.main_topics[0] if analysis.main_topics else "temel kavram",
                    "sinif": "2. sınıf",
                    "stil": "hikaye gibi"
                }
            ))
            
        elif analysis.educational_level == EducationalLevel.HIGH_SCHOOL:
            suggestions.append(PromptSuggestion(
                suggested_name="Lise Seviye Detaylı Analiz",
                suggested_description="Konuları lise öğrencileri için analitik yaklaşımla açıklar",
                prompt_template=f"""Sen deneyimli bir {analysis.subject_area} öğretmenisin. {{konu}} konusunu lise {{sinif}} seviyesinde detaylı şekilde analiz et.

KONU: {{konu}}
SINIF SEVİYESİ: {{sinif}}
ANALİZ TİPİ: {{tip}}

KURALLAR:
- Bilimsel yaklaşım kullan
- Neden-sonuç ilişkilerini açıkla
- Kritik düşünmeyi teşvik et
- Güncel örnekler ver
- Sınav hazırlığına uygun detay

LİSE SEVİYESİ ANALİZ:""",
                category=PromptCategory.EDUCATIONAL,
                complexity=PromptComplexity.ADVANCED,
                variables=["konu", "sinif", "tip"],
                suggested_tags=["lise", "analitik", "detaylı"],
                reasoning="Lise seviyesi içerik tespit edildi, analitik yaklaşım gerekiyor",
                confidence=0.85,
                use_cases=["Lise ders anlatımı", "Sınav hazırlığı", "Proje rehberliği"],
                example_inputs={
                    "konu": analysis.main_topics[0] if analysis.main_topics else "temel konu",
                    "sinif": "11. sınıf",
                    "tip": "detaylı analiz"
                }
            ))
        
        return suggestions
    
    def _suggest_by_subject_area(self, analysis: DocumentAnalysis) -> List[PromptSuggestion]:
        """Konu alanına göre öneriler"""
        
        suggestions = []
        
        if analysis.subject_area == "matematik":
            if analysis.has_exercises:
                suggestions.append(PromptSuggestion(
                    suggested_name="Matematik Problem Çözme",
                    suggested_description="Matematik problemlerini adım adım çözme rehberi",
                    prompt_template="""Sen matematik öğretmenisin. Öğrencinin matematik problemini adım adım çözmesine yardım et.

PROBLEM: {problem}
SINIF SEVİYESİ: {seviye}
KONU ALANI: {alan}

ÇÖZÜM STRATEJİSİ:
1. Problemde ne istendiğini anla
2. Verilen bilgileri listele
3. Hangi işlemleri yapacağını planla
4. Adım adım çöz
5. Sonucu kontrol et

PROBLEM ÇÖZME REHBERİ:""",
                    category=PromptCategory.SUBJECT_SPECIFIC,
                    complexity=PromptComplexity.INTERMEDIATE,
                    variables=["problem", "seviye", "alan"],
                    suggested_tags=["matematik", "problem-çözme", "adım-adım"],
                    reasoning="Matematik içeriği ve alıştırma sorularının varlığı tespit edildi",
                    confidence=0.9,
                    use_cases=["Matematik ödev yardımı", "Problem çözme dersleri"],
                    example_inputs={
                        "problem": "Örnek matematik problemi",
                        "seviye": analysis.educational_level.value,
                        "alan": analysis.main_topics[0] if analysis.main_topics else "genel matematik"
                    },
                    suggested_command="/matematik-coz",
                    command_parameters=["problem", "seviye", "alan"]
                ))
        
        elif analysis.subject_area == "fen":
            suggestions.append(PromptSuggestion(
                suggested_name="Fen Kavram Açıklama",
                suggested_description="Fen bilimleri kavramlarını örneklerle açıklar",
                prompt_template="""Sen fen bilimleri öğretmenisin. {kavram} kavramını {seviye} seviyesindeki öğrenciler için günlük hayattan örneklerle açıkla.

KAVRAM: {kavram}
ÖĞRENCİ SEVİYESİ: {seviye}
AÇIKLAMA STİLİ: {stil}

KURALLAR:
- Bilimsel doğruluğu koru
- Günlük hayattan örnekler ver
- Görsel imgeler kullan
- Deneyi varsa açıkla
- Merak uyandır

FEN AÇIKLAMASI:""",
                category=PromptCategory.SUBJECT_SPECIFIC,
                complexity=PromptComplexity.INTERMEDIATE,
                variables=["kavram", "seviye", "stil"],
                suggested_tags=["fen", "kavram", "günlük-hayat"],
                reasoning="Fen bilimleri içeriği tespit edildi",
                confidence=0.85,
                use_cases=["Fen dersi anlatımı", "Kavram öğretimi"],
                example_inputs={
                    "kavram": analysis.main_topics[0] if analysis.main_topics else "temel fen kavramı",
                    "seviye": analysis.educational_level.value,
                    "stil": "örneklerle"
                }
            ))
        
        return suggestions
    
    def _suggest_by_structural_features(self, analysis: DocumentAnalysis) -> List[PromptSuggestion]:
        """Yapısal özelliklere göre öneriler"""
        
        suggestions = []
        
        if analysis.has_questions:
            suggestions.append(PromptSuggestion(
                suggested_name="Soru-Cevap Rehberi",
                suggested_description="Dokümandaki soruları cevaplanmaya yardım eder",
                prompt_template="""Sen öğrencilere soru cevaplama konusunda yardım eden bir öğretmensin. 

SORU: {soru}
KONU ALANI: {alan}
CEVAP TİPİ: {tip}

KURALLAR:
- Önce soruyu analiz et
- Hangi bilgilere ihtiyaç var belirle
- Sistematik cevap ver
- Örnek ve açıklama ekle

SORU-CEVAP REHBERİ:""",
                category=PromptCategory.EDUCATIONAL,
                complexity=PromptComplexity.INTERMEDIATE,
                variables=["soru", "alan", "tip"],
                suggested_tags=["soru-cevap", "rehberlik"],
                reasoning="Dokümanda soru yapısı tespit edildi",
                confidence=0.8,
                use_cases=["Sınav hazırlığı", "Ödev kontrolü"],
                example_inputs={
                    "soru": "Örnek soru",
                    "alan": analysis.subject_area,
                    "tip": "detaylı açıklama"
                }
            ))
        
        if analysis.has_examples:
            suggestions.append(PromptSuggestion(
                suggested_name="Örnek Genişletme",
                suggested_description="Dokümandaki örnekleri genişleterek açıklar",
                prompt_template="""Sen örnekleri detaylandırarak öğreten bir eğitmensin.

MEVCUT ÖRNEK: {ornek}
KONU: {konu}
HEDEF SEVİYE: {seviye}

KURALLAR:
- Örneği detaylandır
- Benzer örnekler ekle
- Neden bu örneği verdiğini açıkla
- Öğrenci anlayışını destekle

GENİŞLETİLMİŞ ÖRNEK AÇIKLAMASI:""",
                category=PromptCategory.EDUCATIONAL,
                complexity=PromptComplexity.BEGINNER,
                variables=["ornek", "konu", "seviye"],
                suggested_tags=["örnekler", "detaylandırma"],
                reasoning="Dokümanda örnek içerik tespit edildi",
                confidence=0.75,
                use_cases=["Konu pekiştirme", "Öğrenci anlayışını artırma"],
                example_inputs={
                    "ornek": "Dokümandaki örnek",
                    "konu": analysis.main_topics[0] if analysis.main_topics else "temel konu",
                    "seviye": analysis.educational_level.value
                }
            ))
        
        return suggestions
    
    def _suggest_by_teaching_methods(self, analysis: DocumentAnalysis) -> List[PromptSuggestion]:
        """Öğretim yöntemlerine göre öneriler"""
        
        suggestions = []
        
        if "görsel" in analysis.teaching_methods:
            suggestions.append(PromptSuggestion(
                suggested_name="Görsel Açıklama",
                suggested_description="Konuları görsel imgelerle açıklar",
                prompt_template="""Sen görsel öğrenmeyi destekleyen bir eğitmensin. {konu} konusunu görsel imgeler ve betimlemenler kullanarak açıkla.

KONU: {konu}
GÖRSEL TİP: {gorsel_tip}
SEVİYE: {seviye}

KURALLAR:
- Zihinsel görüntüler oluştur
- Renk ve şekil betimlemeleri yap
- Görsel analojiler kullan
- Şema ve diyagram tarif et

GÖRSEL AÇIKLAMA:""",
                category=PromptCategory.EDUCATIONAL,
                complexity=PromptComplexity.INTERMEDIATE,
                variables=["konu", "gorsel_tip", "seviye"],
                suggested_tags=["görsel", "imgeleme", "betimleme"],
                reasoning="Dokümanda görsel öğretim yöntemi tespit edildi",
                confidence=0.8,
                use_cases=["Görsel öğrenci desteği", "Hayal kurma"],
                example_inputs={
                    "konu": analysis.main_topics[0] if analysis.main_topics else "temel konu",
                    "gorsel_tip": "şema",
                    "seviye": analysis.educational_level.value
                }
            ))
        
        if "problem-çözme" in analysis.teaching_methods:
            suggestions.append(PromptSuggestion(
                suggested_name="Problem Çözme Stratejisi",
                suggested_description="Problem çözme becerilerini geliştirici yaklaşım",
                prompt_template="""Sen problem çözme uzmanı bir öğretmensin. Öğrenciye {problem_tipi} tipinde problemleri çözme stratejisini öğret.

PROBLEM TİPİ: {problem_tipi}
KONU ALANI: {alan}
STRATEJİ SEVİYESİ: {seviye}

PROBLEM ÇÖZME STRATEJİLERİ:
1. Problemi anlama
2. Plan yapma
3. Planı uygulama
4. Gözden geçirme

STRATEJİ AÇIKLAMASI:""",
                category=PromptCategory.EDUCATIONAL,
                complexity=PromptComplexity.ADVANCED,
                variables=["problem_tipi", "alan", "seviye"],
                suggested_tags=["problem-çözme", "strateji", "analitik"],
                reasoning="Problem çözme yöntemi tespit edildi",
                confidence=0.85,
                use_cases=["Problem çözme dersleri", "Analitik düşünce"],
                example_inputs={
                    "problem_tipi": "matematik",
                    "alan": analysis.subject_area,
                    "seviye": "orta"
                }
            ))
        
        return suggestions


# Global instance
document_analyzer = DocumentAnalyzer()


def analyze_document_and_suggest_prompts(
    content: str,
    filename: str,
    generation_model: Optional[str] = None
) -> Tuple[DocumentAnalysis, List[PromptSuggestion]]:
    """
    Dokumento analiz eder ve prompt önerileri üretir.
    
    Returns:
        Tuple of (document_analysis, prompt_suggestions)
    """
    try:
        # Dökümanı analiz et
        analysis = document_analyzer.analyze_document(content, filename, generation_model)
        
        # Prompt önerileri oluştur
        suggestions = document_analyzer.generate_prompt_suggestions(analysis, max_suggestions=5)
        
        return analysis, suggestions
        
    except Exception as e:
        # Hata durumunda boş sonuç döndür
        fallback_analysis = document_analyzer._create_fallback_analysis(filename, content, str(e))
        return fallback_analysis, []