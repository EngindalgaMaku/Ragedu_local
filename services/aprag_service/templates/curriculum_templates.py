"""
Curriculum-Aware Prompt Templates for Module Extraction
Supports Turkish MEB curriculum standards and other educational systems
"""

import logging
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)


class CurriculumTemplateManager:
    """Manages curriculum-specific templates for module extraction"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
        logger.info("Curriculum Template Manager initialized")
    
    def _initialize_templates(self):
        """Initialize all curriculum templates"""
        # Use generic template for all curricula
        # Specific templates can be added here if needed in the future
        # For now, we use a dynamic generic template that adapts to any curriculum
        self.templates = {}
        
        # Note: We removed hardcoded MEB_2018 templates as they are outdated
        # All templates now use the generic, dynamic template that adapts to
        # the curriculum_standard, subject_area, and grade_level from course_info
    
    def get_template(
        self,
        curriculum: str,
        subject: str,
        grade_level: str,
        topics: List[Dict[str, Any]]
    ) -> str:
        """
        Get appropriate template and populate with topics
        
        Args:
            curriculum: Curriculum system (e.g., 'MEB_2018')
            subject: Subject area (e.g., 'biology')
            grade_level: Grade level (e.g., '9')
            topics: List of topics to organize
            
        Returns:
            Formatted prompt template
        """
        # Always use generic template - it's dynamic and adapts to any curriculum
        # This avoids hardcoding outdated curriculum versions (like MEB_2018)
        template = self._get_generic_template(subject, grade_level)
        logger.info(f"Using generic template for {curriculum}/{subject}/{grade_level}")
        
        # Format template with topics
        topics_context = self._format_topics_for_template(topics)
        
        # Prepare format kwargs with all possible placeholders
        format_kwargs = {"topics_list": topics_context}
        
        # Subject name mapping (English to Turkish)
        subject_names = {
            "biology": "biyoloji",
            "mathematics": "matematik",
            "physics": "fizik",
            "chemistry": "kimya",
            "history": "tarih",
            "geography": "coğrafya",
            "literature": "edebiyat",
            "general": "genel"
        }
        subject_name = subject_names.get(subject.lower(), subject)
        subject_name_upper = subject_name.upper()
        
        # Curriculum standard formatting
        curriculum_display = curriculum.replace("_", " ").upper() if curriculum else "MEB"
        
        # Add all possible placeholders
        format_kwargs.update({
            "subject": subject,
            "subject_name": subject_name,
            "subject_name_upper": subject_name_upper,
            "grade_level": grade_level,
            "curriculum_standard": curriculum_display
        })
        
        return template.format(**format_kwargs)
    
    def _format_topics_for_template(self, topics: List[Dict[str, Any]]) -> str:
        """Format topics for inclusion in prompts"""
        formatted_topics = []
        
        for i, topic in enumerate(topics, 1):
            topic_info = f"""
{i}. Konu ID: {topic['topic_id']}
   Başlık: {topic['topic_title']}
   Açıklama: {topic.get('description', topic.get('topic_description', 'Açıklama yok'))}
   Zorluk: {topic.get('estimated_difficulty', 'orta')}
   Anahtar kelimeler: {', '.join(topic.get('keywords', []))}
   İlgili chunk'lar: {len(topic.get('related_chunk_ids', []))} adet"""
            
            formatted_topics.append(topic_info)
        
        return '\n'.join(formatted_topics)
    
    # =======================================================================
    # Turkish MEB 2018 Biology Templates
    # =======================================================================
    
    def _get_meb_biology_templates(self) -> Dict[str, str]:
        """Get MEB Biology templates for all grades"""
        return {
            '9': self._get_meb_biology_9_template(),
            '10': self._get_meb_biology_10_template(),
            '11': self._get_meb_biology_11_template(),
            '12': self._get_meb_biology_12_template(),
        }
    
    def _get_meb_biology_9_template(self) -> str:
        """MEB 9th Grade Biology template"""
        return """Sen bir Türk Milli Eğitim Bakanlığı {subject_name} müfredatı uzmanısın.
{grade_level}. sınıf {subject_name} konularını resmi {curriculum_standard} müfredatına uygun modüllere organize et.

ÖNEMLİ: Aşağıdaki konuları modüllere organize etmelisin. Sadece konu listesi döndürme!
Her modül bir başlık ve açıklama içermeli, ve içinde ilgili konuları gruplamalı.

{curriculum_standard} {grade_level}. SINIF {subject_name_upper} MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Yaşam Bilimi Biyoloji (8 ders saati)
- B.9.1.1: Biyolojinin çalışma alanlarını ve diğer bilimlerle ilişkisini açıklar
- B.9.1.2: Bilimsel yöntemi basamaklarını belirterek uygular
- B.9.1.3: Mikroskobun çalışma prensibini açıklar ve kullanır

ÜNİTE 2: Hücre ve Organelleri (40 ders saati)
- B.9.2.1: Canlıların temel birimi olan hücreyi tanıyabilir
- B.9.2.2: Hücre zarının yapı ve görevlerini açıklayabilir
- B.9.2.3: Hücre organellerinin yapı ve görevlerini karşılaştırır
- B.9.2.4: Prokaryot ve ökaryot hücreleri karşılaştırır
- B.9.2.5: Hücre zarından madde geçişlerini analiz eder

ÜNİTE 3: Canlılarda Çoğalma, Büyüme ve Gelişim (32 ders saati)
- B.9.3.1: Mitoz ve mayoz bölünmeleri karşılaştırır
- B.9.3.2: Üreme türlerini örneklerle açıklar
- B.9.3.3: Büyüme ve gelişmeyi etkileyen faktörleri analiz eder
- B.9.3.4: Kalıtım kavramını örneklerle açıklar

ÜNİTE 4: Canlılar ve Enerji İlişkileri (32 ders saati)
- B.9.4.1: Fotosentez ve solunum olaylarını karşılaştırır
- B.9.4.2: ATP'nin enerji metabolizmasındaki rolünü açıklar
- B.9.4.3: Enzimlerin çalışma prensiplerini açıklar
- B.9.4.4: Beslenme türlerini örneklerle sınıflandırır

ÜNİTE 5: Ekosistemdeki Döngüler ve Çevre Sorunları (32 ders saati)
- B.9.5.1: Ekosistemdeki enerji akışını ve madde döngülerini analiz eder
- B.9.5.2: Çevre sorunlarının nedenlerini ve çözüm yollarını tartışır
- B.9.5.3: Biyoçeşitliliğin önemini kavrayarak koruma yollarını önerir
- B.9.5.4: İnsan faaliyetlerinin çevre üzerindeki etkilerini değerlendirir

MODÜL ORGANİZASYON PRENSİPLERİ:
1. Her modül resmi müfredat ünitelerine uygun olmalı
2. Konular mantıklı öğrenme sırası takip etmeli (basit → karmaşık)
3. Öğrenci seviyesine uygun zorluk dağılımı olmalı
4. Praktik uygulamalar ve laboratuvar çalışmaları içermeli
5. Ölçme değerlendirme yöntemleri MEB standartlarına uygun olmalı
6. Her modül 3-12 konu içermeli
7. Önkoşul ilişkileri açık şekilde belirtilmeli

MEVCUT KONULAR:
{topics_list}

ÇIKTI FORMATI (JSON):
{{
  "extraction_metadata": {{
    "total_modules": 0,
    "curriculum_alignment_score": 0.0,
    "extraction_confidence": 0.0,
    "curriculum_system": "MEB_2018",
    "subject_area": "biology",
    "grade_level": "9"
  }},
  "modules": [
    {{
      "module_title": "Hücre ve Yaşamın Temelleri",
      "module_description": "Canlıların temel birimi olan hücreyi anlama ve yaşam süreçlerini kavrama",
      "module_order": 1,
      "estimated_duration_hours": 20,
      "difficulty_level": "intermediate",
      "curriculum_standards": ["B.9.2.1", "B.9.2.2"],
      "curriculum_unit": "Hücre ve Organelleri",
      "learning_outcomes": [
        "Hücre teorisini açıklayabilir",
        "Prokaryot ve ökaryot hücreleri karşılaştırabilir",
        "Hücre organellerinin görevlerini tanır"
      ],
      "topics": [
        {{
          "topic_id": 123,
          "topic_title": "Hücre Nedir",
          "topic_order_in_module": 1,
          "importance_level": "critical",
          "content_coverage_percentage": 100.0,
          "prerequisites": [],
          "relationship_type": "contains"
        }}
      ],
      "prerequisites": [],
      "assessment_methods": ["quiz", "laboratory", "project"],
      "time_allocation": {{
        "theory_hours": 12,
        "laboratory_hours": 6,
        "assessment_hours": 2
      }}
    }}
  ]
}}

ÖNEMLİ KURALLAR:
- Sadece geçerli JSON çıktısı ver, başka açıklama yapma
- MUTLAKA modül organizasyonu yap: Konuları mantıklı gruplara ayır ve her gruba bir modül başlığı ver
- Sadece konu listesi döndürme! Konuları modüllere organize etmelisin
- JSON anahtar kelimeleri İNGİLİZCE olmalı: "modules" (Türkçe "moduller" değil), "module_title" (Türkçe "baslik" değil)
- Çıktı formatı: {{"modules": [{{"module_title": "...", "topics": [...]}}]}} şeklinde olmalı
- Konu ID'lerini doğru kullan (mevcut konular listesinden)
- Her modül mutlaka MEB kazanımlarına uygun olmalı
- Her modül en az 3-12 konu içermeli
- Zorluk seviyesi: başlangıç, orta, ileri
- Önkoşul ilişkilerini mantıklı şekilde belirt
- JSON formatında çıktı verirken İNGİLİZCE anahtar kelimeler kullan: "modules", "module_title", "module_description", "topics", "topic_id" vb."""

    def _get_meb_biology_10_template(self) -> str:
        """MEB 10th Grade Biology template"""
        return """Sen bir MEB biyoloji müfredatı uzmanısın. 10. sınıf biyoloji konularını organize et.

MEB 2018 10. SINIF BİYOLOJİ MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Hücre Bölünmeleri ve Üreme (32 ders saati)
ÜNİTE 2: Kalıtım (40 ders saati) 
ÜNİTE 3: Canlıların Çeşitliliği ve Sınıflandırılması (40 ders saati)
ÜNİTE 4: Bitkilerde Yaşam (32 ders saati)

MEVCUT KONULAR:
{topics_list}

10. sınıf seviyesine uygun, daha detaylı ve karmaşık konuları içeren modüller oluştur.
Sadece JSON formatında çıktı ver."""

    def _get_meb_biology_11_template(self) -> str:
        """MEB 11th Grade Biology template"""
        return """Sen bir MEB biyoloji müfredatı uzmanısın. 11. sınıf biyoloji konularını organize et.

MEB 2018 11. SINIF BİYOLOJİ MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: İnsan Fizyolojisi (48 ders saati)
ÜNİTE 2: Sinir Sistemi (32 ders saati)
ÜNİTE 3: Endokrin Sistem (24 ders saati)
ÜNİTE 4: Üreme Sistemi ve Embriyonik Gelişim (24 ders saati)
ÜNİTE 5: Duyu Organları (16 ders saati)

MEVCUT KONULAR:
{topics_list}

11. sınıf seviyesine uygun, insan biyolojisi ağırlıklı modüller oluştur.
Sadece JSON formatında çıktı ver."""

    def _get_meb_biology_12_template(self) -> str:
        """MEB 12th Grade Biology template"""
        return """Sen bir MEB biyoloji müfredatı uzmanısın. 12. sınıf biyoloji konularını organize et.

MEB 2018 12. SINIF BİYOLOJİ MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Genden Protiine (32 ders saati)
ÜNİTE 2: Canlılarda Enerji Dönüşümleri (32 ders saati)
ÜNİTE 3: Canlılarda Homeostasis (24 ders saati)
ÜNİTE 4: Popülasyonlar (16 ders saati)
ÜNİTE 5: Ekosistem Ekolojisi ve Güncel Çevre Sorunları (40 ders saati)

MEVCUT KONULAR:
{topics_list}

12. sınıf seviyesine uygun, moleküler biyoloji ve ekoloji ağırlıklı modüller oluştur.
Sadece JSON formatında çıktı ver."""

    # =======================================================================
    # Turkish MEB 2018 Mathematics Templates
    # =======================================================================
    
    def _get_meb_mathematics_templates(self) -> Dict[str, str]:
        """Get MEB Mathematics templates for all grades"""
        return {
            '9': self._get_meb_mathematics_9_template(),
            '10': self._get_meb_mathematics_10_template(),
            '11': self._get_meb_mathematics_11_template(),
            '12': self._get_meb_mathematics_12_template(),
        }
    
    def _get_meb_mathematics_9_template(self) -> str:
        """MEB 9th Grade Mathematics template"""
        return """Sen bir MEB matematik müfredatı uzmanısın. 9. sınıf matematik konularını organize et.

MEB 2018 9. SINIF MATEMATİK MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Sayılar ve İşlemler (24 ders saati)
- Reel sayılar
- Kök alma işlemleri
- Üslü sayılar

ÜNİTE 2: Cebirsel İfadeler (32 ders saati)
- Çarpanlara ayırma
- İkinci dereceden denklemler
- Eşitsizlikler

ÜNİTE 3: Fonksiyonlar (28 ders saati)
- Fonksiyon kavramı
- Fonksiyon türleri
- Fonksiyon işlemleri

ÜNİTE 4: Geometri (36 ders saati)
- Üçgenlerde benzerlik
- Üçgenlerde trigonometri
- Çember ve daire

ÜNİTE 5: Veri Analizi (24 ders saati)
- İstatistik
- Olasılık

MEVCUT KONULAR:
{topics_list}

MATEMATİK ÖĞRETİM PRENSİPLERİ:
1. Kavramsal anlama öncelikli
2. Problem çözme becerileri geliştirme
3. Matematiksel modelleme
4. Teknoloji entegrasyonu
5. Günlük yaşamla ilişkilendirme
6. Matematiksel iletişim

Sadece JSON formatında çıktı ver."""

    def _get_meb_mathematics_10_template(self) -> str:
        """MEB 10th Grade Mathematics template"""
        return """Sen bir MEB matematik müfredatı uzmanısın. 10. sınıf matematik konularını organize et.

MEB 2018 10. SINIF MATEMATİK MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Fonksiyonlar (32 ders saati)
- Fonksiyon kavramı ve gösterimi
- Fonksiyon türleri ve özellikleri
- Fonksiyon işlemleri

ÜNİTE 2: Polinom Fonksiyonlar (24 ders saati)
- Polinom kavramı
- Polinomlarla işlemler
- Polinom fonksiyonların grafikleri

ÜNİTE 3: İkinci Dereceden Fonksiyonlar (28 ders saati)
- İkinci dereceden fonksiyon
- Parabol grafikleri
- Uygulamalar

ÜNİTE 4: Trigonometri ve Trigonometrik Fonksiyonlar (36 ders saati)
- Trigonometrik oranlar
- Trigonometrik fonksiyonlar
- Trigonometrik özdeşlikler

ÜNİTE 5: Diziler ve Seriler (24 ders saati)
- Aritmetik diziler
- Geometrik diziler
- Seriler

MEVCUT KONULAR:
{topics_list}

10. sınıf seviyesinde fonksiyon kavramı merkeze alınarak modüller oluştur.
Sadece JSON formatında çıktı ver."""

    def _get_meb_mathematics_11_template(self) -> str:
        """MEB 11th Grade Mathematics template"""
        return """Sen bir MEB matematik müfredatı uzmanısın. 11. sınıf matematik konularını organize et.

MEB 2018 11. SINIF MATEMATİK MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Limit ve Süreklilik (32 ders saati)
ÜNİTE 2: Türev (40 ders saati)
ÜNİTE 3: İntegral (32 ders saati)

MEVCUT KONULAR:
{topics_list}

11. sınıf seviyesinde analiz matematik konularını içeren modüller oluştur.
Sadece JSON formatında çıktı ver."""

    def _get_meb_mathematics_12_template(self) -> str:
        """MEB 12th Grade Mathematics template"""
        return """Sen bir MEB matematik müfredatı uzmanısın. 12. sınıf matematik konularını organize et.

MEB 2018 12. SINIF MATEMATİK MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Üstel ve Logaritmik Fonksiyonlar (24 ders saati)
ÜNİTE 2: Analitik Geometri (32 ders saati)
ÜNİTE 3: İstatistik ve Olasılık (32 ders saati)

MEVCUT KONULAR:
{topics_list}

12. sınıf seviyesinde ileri matematik konularını içeren modüller oluştur.
Sadece JSON formatında çıktı ver."""

    # =======================================================================
    # Turkish MEB 2018 Physics Templates
    # =======================================================================
    
    def _get_meb_physics_templates(self) -> Dict[str, str]:
        """Get MEB Physics templates for all grades"""
        return {
            '9': self._get_meb_physics_9_template(),
            '10': self._get_meb_physics_10_template(),
            '11': self._get_meb_physics_11_template(),
            '12': self._get_meb_physics_12_template(),
        }
    
    def _get_meb_physics_9_template(self) -> str:
        """MEB 9th Grade Physics template"""
        return """Sen bir MEB fizik müfredatı uzmanısın. 9. sınıf fizik konularını organize et.

MEB 2018 9. SINIF FİZİK MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Fizik Bilimine Giriş (10 ders saati)
- F.9.1.1: Fizik biliminin doğası ve çalışma alanları
- F.9.1.2: Fiziksel büyüklükler ve ölçme
- F.9.1.3: Bilimsel yöntem

ÜNİTE 2: Madde ve Özellikleri (16 ders saati)
- F.9.2.1: Maddenin yapısı ve halleri
- F.9.2.2: Maddenin ısıl özellikleri
- F.9.2.3: Gazlarda basınç

ÜNİTE 3: Hareket ve Kuvvet (24 ders saati)
- F.9.3.1: Kinematik
- F.9.3.2: Dinamik
- F.9.3.3: Newton kanunları

ÜNİTE 4: Enerji (20 ders saati)
- F.9.4.1: İş, güç ve enerji
- F.9.4.2: Enerjinin korunumu
- F.9.4.3: Basit makineler

ÜNİTE 5: Isı ve Sıcaklık (16 ders saati)
- F.9.5.1: Sıcaklık ve termometreler
- F.9.5.2: Isı ve ısı transferi
- F.9.5.3: Hal değişimi

ÜNİTE 6: Elektrostatik (22 ders saati)
- F.9.6.1: Elektriksel kuvvet ve alan
- F.9.6.2: Elektriksel potansiyel
- F.9.6.3: Kapasitörler

MEVCUT KONULAR:
{topics_list}

FİZİK ÖĞRETİM PRENSİPLERİ:
1. Kavramsal anlama ve matematiksel ifade
2. Deney ve gözlem
3. Problem çözme stratejileri
4. Günlük yaşamla ilişkilendirme
5. Laboratuvar uygulamaları

Sadece JSON formatında çıktı ver."""

    def _get_meb_physics_10_template(self) -> str:
        """MEB 10th Grade Physics template"""
        return """Sen bir MEB fizik müfredatı uzmanısın. 10. sınıf fizik konularını organize et.

MEB 2018 10. SINIF FİZİK MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Elektrik ve Manyetizma (40 ders saati)
ÜNİTE 2: Basınç ve Kaldırma Kuvveti (20 ders saati)
ÜNİTE 3: Dalgalar (24 ders saati)
ÜNİTE 4: Optik (24 ders saati)

MEVCUT KONULAR:
{topics_list}

10. sınıf seviyesinde elektrik, manyetizma ve dalga konularını içeren modüller oluştur.
Sadece JSON formatında çıktı ver."""

    def _get_meb_physics_11_template(self) -> str:
        """MEB 11th Grade Physics template"""
        return """Sen bir MEB fizik müfredatı uzmanısın. 11. sınıf fizik konularını organize et.

MEB 2018 11. SINIF FİZİK MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Çembersel Hareket (16 ders saati)
ÜNİTE 2: Basit Harmonik Hareket (16 ders saati)
ÜNİTE 3: Dalga Mekaniği (20 ders saati)
ÜNİTE 4: Elektrik ve Manyetizma (36 ders saati)
ÜNİTE 5: Modern Fizik (20 ders saati)

MEVCUT KONULAR:
{topics_list}

11. sınıf seviyesinde ileri fizik konularını içeren modüller oluştur.
Sadece JSON formatında çıktı ver."""

    def _get_meb_physics_12_template(self) -> str:
        """MEB 12th Grade Physics template"""
        return """Sen bir MEB fizik müfredatı uzmanısın. 12. sınıf fizik konularını organize et.

MEB 2018 12. SINIF FİZİK MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Çembersel Hareket ve Gravitasyon (24 ders saati)
ÜNİTE 2: Basit Harmonik Hareket (20 ders saati)
ÜNİTE 3: Elektrik ve Manyetizma (32 ders saati)
ÜNİTE 4: Modern Fizik (32 ders saati)

MEVCUT KONULAR:
{topics_list}

12. sınıf seviyesinde modern fizik dahil ileri konuları içeren modüller oluştur.
Sadece JSON formatında çıktı ver."""

    # =======================================================================
    # Turkish MEB 2018 Chemistry Templates
    # =======================================================================
    
    def _get_meb_chemistry_templates(self) -> Dict[str, str]:
        """Get MEB Chemistry templates for all grades"""
        return {
            '9': self._get_meb_chemistry_9_template(),
            '10': self._get_meb_chemistry_10_template(),
            '11': self._get_meb_chemistry_11_template(),
            '12': self._get_meb_chemistry_12_template(),
        }
    
    def _get_meb_chemistry_9_template(self) -> str:
        """MEB 9th Grade Chemistry template"""
        return """Sen bir MEB kimya müfredatı uzmanısın. 9. sınıf kimya konularını organize et.

MEB 2018 9. SINIF KİMYA MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Kimya Bilimi (8 ders saati)
ÜNİTE 2: Atom ve Periyodik Sistem (32 ders saati)
ÜNİTE 3: Kimyasal Türler Arası Etkileşimler (32 ders saati)
ÜNİTE 4: Maddenin Halleri (16 ders saati)
ÜNİTE 5: Doğa ve Kimya (20 ders saati)

MEVCUT KONULAR:
{topics_list}

KİMYA ÖĞRETİM PRENSİPLERİ:
1. Makroskopik gözlemden mikroskopik açıklamaya
2. Laboratuvar çalışmaları
3. Günlük yaşamla ilişkilendirme
4. Çevre bilinci

Sadece JSON formatında çıktı ver."""

    def _get_meb_chemistry_10_template(self) -> str:
        """MEB 10th Grade Chemistry template"""
        return """Sen bir MEB kimya müfredatı uzmanısın. 10. sınıf kimya konularını organize et.

MEB 2018 10. SINIF KİMYA MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Kimyasal Türler ve Tepkimeler (32 ders saati)
ÜNİTE 2: Asit, Baz ve Tuz (24 ders saati)
ÜNİTE 3: Endüstriyel Hammaddeler (20 ders saati)

MEVCUT KONULAR:
{topics_list}

10. sınıf seviyesinde kimyasal tepkimeler ve endüstriyel kimya konularını içeren modüller oluştur.
Sadece JSON formatında çıktı ver."""

    def _get_meb_chemistry_11_template(self) -> str:
        """MEB 11th Grade Chemistry template"""
        return """Sen bir MEB kimya müfredatı uzmanısın. 11. sınıf kimya konularını organize et.

MEB 2018 11. SINIF KİMYA MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Modern Atom Teorisi (16 ders saati)
ÜNİTE 2: Gazlar (20 ders saati)
ÜNİTE 3: Sıvılar (12 ders saati)
ÜNİTE 4: Çözeltiler ve Çözünürlük (20 ders saati)
ÜNİTE 5: Termodinamik ve Kimyasal Tepkimeler (20 ders saati)

MEVCUT KONULAR:
{topics_list}

11. sınıf seviyesinde fiziksel kimya ağırlıklı modüller oluştur.
Sadece JSON formatında çıktı ver."""

    def _get_meb_chemistry_12_template(self) -> str:
        """MEB 12th Grade Chemistry template"""
        return """Sen bir MEB kimya müfredatı uzmanısın. 12. sınıf kimya konularını organize et.

MEB 2018 12. SINIF KİMYA MÜFREDAT ÜNİTELERİ:

ÜNİTE 1: Kimyasal Tepkimelerde Hız (20 ders saati)
ÜNİTE 2: Kimyasal Tepkimelerde Denge (20 ders saati)
ÜNİTE 3: Çözünürlük Dengesi (16 ders saati)
ÜNİTE 4: Elektrokimya (16 ders saati)
ÜNİTE 5: Organik Kimya (36 ders saati)

MEVCUT KONULAR:
{topics_list}

12. sınıf seviyesinde kimyasal denge ve organik kimya ağırlıklı modüller oluştur.
Sadece JSON formatında çıktı ver."""

    # =======================================================================
    # Generic Templates
    # =======================================================================
    
    def _get_generic_template(self, subject: str, grade_level: str) -> str:
        """Get generic template for any subject/grade combination"""
        # Use regular string with placeholders, not f-string
        # All placeholders will be filled in get_template() method
        return """Sen bir eğitim uzmanısın. {grade_level}. sınıf {subject_name} konularını {curriculum_standard} müfredatına uygun modüllere organize et.

ÖNEMLİ: Aşağıdaki konuları modüllere organize etmelisin. Sadece konu listesi döndürme!
Her modül bir başlık ve açıklama içermeli, ve içinde ilgili konuları gruplamalı.

MEVCUT KONULAR:
{{topics_list}}

MODÜL ORGANİZASYON PRENSİPLERİ:
1. Konular mantıklı öğrenme sırası takip etmeli (basit → karmaşık)
2. Her modül 3-12 konu içermeli
3. Zorluk derecesi kademeli artmalı
4. Önkoşul ilişkileri belirtilmeli
5. {grade_level}. sınıf seviyesine uygun olmalı
6. {curriculum_standard} müfredat standartlarına uygun olmalı
7. ÇOK ÖNEMLİ: Her konu SADECE BİR modülde olmalı - aynı konuyu birden fazla modüle koyma!
8. ÇOK ÖNEMLİ: TÜM konular modüllere dağıtılmalı - hiçbir konu boşta kalmamalı!
9. Konuları modül başlıklarına ve açıklamalarına göre ilgili modüllere yerleştir

MODÜL BAŞLIK FORMATI (ÇOK ÖNEMLİ):
- Modül başlıkları Türkiye eğitim müfredatındaki ünite başlıklarına benzer formatta olmalı
- Her kelimenin SADECE İLK HARFİ BÜYÜK, diğer harfler KÜÇÜK olmalı (Title Case)
- Örnek format: "Biyoloji ve Canlıların Ortak Özellikleri - I" (YANLIŞ: "BİYOLOJİ VE CANLILARIN ORTAK ÖZELLİKLERİ")
- Örnek format: "Organik Bileşikler - Karbonhidratlar" (YANLIŞ: "ORGANİK BİLEŞİKLER - KARBONHİDRATLAR")
- Örnek format: "Nükleik Asitler (DNA)" (YANLIŞ: "NÜKLEİK ASİTLER (DNA)")
- Örnek format: "Enzimlerin Çalışmasına Etki Eden Faktörler" (YANLIŞ: "ENZİMLERİN ÇALIŞMASINA ETKİ EDEN FAKTÖRLER")
- Başlıklar Türkçe karakterleri doğru kullanmalı (ı, İ, ş, Ş, ğ, Ğ, ü, Ü, ö, Ö, ç, Ç)
- Başlıklar anlamlı ve müfredat standartlarına uygun olmalı

ÇIKTI FORMATI (JSON):
{{{{
  "extraction_metadata": {{{{
    "total_modules": 0,
    "curriculum_alignment_score": 0.5,
    "extraction_confidence": 0.7,
    "curriculum_system": "{curriculum_standard}",
    "subject_area": "{subject}",
    "grade_level": "{grade_level}"
  }}}},
  "modules": [
    {{{{
      "module_title": "Öğrenme Modülü 1",
      "module_description": "Temel konuları içeren modül",
      "module_order": 1,
      "estimated_duration_hours": 20,
      "difficulty_level": "intermediate",
      "curriculum_standards": [],
      "learning_outcomes": ["Temel kavramları anlar"],
      "topics": [
        {{{{
          "topic_id": 123,
          "topic_order_in_module": 1,
          "importance_level": "medium",
          "relationship_type": "contains"
        }}}}
      ],
      "prerequisites": [],
      "assessment_methods": ["quiz", "assignment"]
    }}}}
  ]
}}}}

ÖNEMLİ KURALLAR:
- MUTLAKA modül organizasyonu yap: Konuları mantıklı gruplara ayır ve her gruba bir modül başlığı ver
- Sadece konu listesi döndürme! Konuları modüllere organize etmelisin
- Çıktı formatı: {{"modules": [{{"module_title": "...", "topics": [...]}}]}} şeklinde olmalı
- JSON çıktısında İNGİLİZCE anahtar kelimeler kullan:
  * "modules" (Türkçe "moduller" değil)
  * "module_title" (Türkçe "baslik" değil)
  * "module_description" (Türkçe "aciklama" değil)
  * "topics" (Türkçe "konular" değil)
  * "topic_id" (Türkçe "konu_id" değil)
- Her modül en az 3-12 konu içermeli
- KRİTİK: Her konu SADECE BİR modülde olmalı - aynı topic_id'yi birden fazla modülde kullanma!
- KRİTİK: MEVCUT KONULAR listesindeki TÜM konuları modüllere dağıt - hiçbir konu eksik kalmamalı!
- Konuları modül başlıklarına ve açıklamalarına göre en uygun modüle yerleştir
- İlgili konuları aynı modülde grupla, ancak her konu sadece bir modülde olmalı

MODÜL BAŞLIK FORMATI (MUTLAKA UYULMALI):
- Modül başlıkları (module_title) Türkiye eğitim müfredatındaki ünite başlıklarına benzer formatta olmalı
- Her kelimenin SADECE İLK HARFİ BÜYÜK, diğer harfler KÜÇÜK olmalı (Title Case)
- Örnek DOĞRU format: "Biyoloji ve Canlıların Ortak Özellikleri - I"
- Örnek DOĞRU format: "Biyoloji ve Canlıların Ortak Özellikleri - II"
- Örnek DOĞRU format: "Canlıların Yapısında Bulunan Temel Bileşikler"
- Örnek DOĞRU format: "Organik Bileşikler - Karbonhidratlar"
- Örnek DOĞRU format: "Lipitler"
- Örnek DOĞRU format: "Proteinler"
- Örnek DOĞRU format: "Enzimler"
- Örnek DOĞRU format: "Enzimlerin Çalışmasına Etki Eden Faktörler"
- Örnek DOĞRU format: "Hormonlar ve Vitaminler"
- Örnek DOĞRU format: "Nükleik Asitler (DNA)"
- Örnek DOĞRU format: "Nükleik Asitler (RNA) ve ATP"
- Örnek DOĞRU format: "Sağlıklı Beslenme"
- Örnek DOĞRU format: "Hücre Teorisi ve Prokaryot Hücre"
- Örnek YANLIŞ format: "BİYOLOJİ VE CANLILARIN ORTAK ÖZELLİKLERİ" (TÜMÜ BÜYÜK HARF - YANLIŞ!)
- Örnek YANLIŞ format: "biyoloji ve canlıların ortak özellikleri" (TÜMÜ KÜÇÜK HARF - YANLIŞ!)
- Başlıklar Türkçe karakterleri doğru kullanmalı (ı, İ, ş, Ş, ğ, Ğ, ü, Ü, ö, Ö, ç, Ç)
- Başlıklar anlamlı, müfredat standartlarına uygun ve eğitimsel olmalı
- Modül başlıkları konuların içeriğini yansıtmalı ve öğrenciler için anlaşılır olmalı
- Müfredat yönergesindeki örnek başlık formatlarını TAM OLARAK takip et

Sadece JSON çıktısı ver, açıklama yapma."""

    # =======================================================================
    # Utility Methods
    # =======================================================================
    
    def get_supported_curricula(self) -> List[str]:
        """Get list of supported curriculum systems"""
        return list(self.templates.keys())
    
    def get_supported_subjects(self, curriculum: str) -> List[str]:
        """Get list of supported subjects for a curriculum"""
        return list(self.templates.get(curriculum, {}).keys())
    
    def get_supported_grades(self, curriculum: str, subject: str) -> List[str]:
        """Get list of supported grade levels for a curriculum/subject"""
        return list(self.templates.get(curriculum, {}).get(subject, {}).keys())
    
    def is_template_available(self, curriculum: str, subject: str, grade_level: str) -> bool:
        """Check if a specific template is available"""
        return (
            curriculum in self.templates and
            subject in self.templates[curriculum] and
            grade_level in self.templates[curriculum][subject]
        )
    
    def get_template_info(self) -> Dict[str, Any]:
        """Get comprehensive template availability information"""
        template_info = {}
        
        for curriculum in self.templates:
            template_info[curriculum] = {}
            for subject in self.templates[curriculum]:
                template_info[curriculum][subject] = {
                    "available_grades": list(self.templates[curriculum][subject].keys()),
                    "template_count": len(self.templates[curriculum][subject])
                }
        
        return template_info


# Global instance
curriculum_template_manager = CurriculumTemplateManager()


def get_curriculum_template_manager() -> CurriculumTemplateManager:
    """Get the global curriculum template manager instance"""
    return curriculum_template_manager