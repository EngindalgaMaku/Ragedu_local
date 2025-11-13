"""
Advanced Prompt Management System for Teachers
Öğretmenler için Gelişmiş Prompt Yönetim Sistemi

Bu modül öğretmenlerin:
- Özel prompt şablonları oluşturmasına
- Prompt performansını izlemesine  
- Çeşitli öğretim senaryoları için özel komutlar tanımlamasına
- Prompt A/B testleri yapmasına olanak sağlar.
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib

from src.utils.prompt_templates import BilingualPromptManager, LanguageCode


class PromptCategory(Enum):
    """Prompt kategorileri"""
    SYSTEM = "system"
    USER = "user" 
    EDUCATIONAL = "educational"
    SUBJECT_SPECIFIC = "subject_specific"
    ASSESSMENT = "assessment"
    FEEDBACK = "feedback"
    CREATIVE = "creative"


class PromptComplexity(Enum):
    """Prompt karmaşıklık seviyeleri"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"  
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class CustomPrompt:
    """Özel prompt şablonu"""
    id: str
    name: str
    description: str
    template: str
    category: PromptCategory
    complexity: PromptComplexity
    language: LanguageCode
    variables: List[str]  # Prompt içindeki değişkenler {variable} formatında
    tags: List[str]
    created_by: str
    created_at: str
    updated_at: str
    usage_count: int = 0
    avg_rating: float = 0.0
    is_active: bool = True


@dataclass 
class PromptCommand:
    """Öğretmen komutları - önceden tanımlı prompt makroları"""
    id: str
    command: str  # örn: "/explain-simple"
    name: str
    description: str
    prompt_template: str
    parameters: List[str]
    examples: List[str]
    subject_area: str
    grade_level: str
    created_by: str
    created_at: str
    usage_count: int = 0
    is_active: bool = True


@dataclass
class PromptPerformance:
    """Prompt performans metrikleri"""
    prompt_id: str
    execution_time: float
    user_rating: Optional[float]
    response_quality: Optional[float]
    educational_effectiveness: Optional[float]
    engagement_score: Optional[float]
    timestamp: str
    session_id: str
    user_feedback: Optional[str]


class TeacherPromptManager:
    """Öğretmenler için gelişmiş prompt yönetim sistemi"""
    
    def __init__(self, db_path: str = "data/analytics/teacher_prompts.db"):
        self.db_path = db_path
        self.base_manager = BilingualPromptManager()
        self._ensure_db_exists()
        
        # Varsayılan eğitim komutlarını yükle
        self._load_default_educational_commands()
    
    def _ensure_db_exists(self):
        """Veritabanını ve tabloları oluştur"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Özel prompt'lar tablosu
            conn.execute("""
                CREATE TABLE IF NOT EXISTS custom_prompts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    template TEXT NOT NULL,
                    category TEXT NOT NULL,
                    complexity TEXT NOT NULL,
                    language TEXT NOT NULL,
                    variables TEXT,  -- JSON array
                    tags TEXT,      -- JSON array
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    avg_rating REAL DEFAULT 0.0,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Prompt komutları tablosu
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_commands (
                    id TEXT PRIMARY KEY,
                    command TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    prompt_template TEXT NOT NULL,
                    parameters TEXT,    -- JSON array
                    examples TEXT,      -- JSON array
                    subject_area TEXT,
                    grade_level TEXT,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Performans metrikleri tablosu
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id TEXT NOT NULL,
                    execution_time REAL,
                    user_rating REAL,
                    response_quality REAL,
                    educational_effectiveness REAL,
                    engagement_score REAL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    user_feedback TEXT
                )
            """)
    
    def _load_default_educational_commands(self):
        """Varsayılan eğitim komutlarını yükle"""
        default_commands = [
            {
                "command": "/basit-anlat",
                "name": "Basit Anlatım",
                "description": "Konuyu basit ve anlaşılır şekilde açıkla",
                "prompt_template": """Sen bir eğitim asistanısın. Verilen konuyu öğrenci seviyesine uygun, basit ve anlaşılır şekilde açıkla.

KONU: {topic}
ÖĞRENCİ SEVİYESİ: {grade_level}

KURALLAR:
- Basit kelimeler kullan
- Kısa cümleler kur  
- Örnekler ver
- Adım adım açıkla
- Öğrenci merakını uyandır

ÖĞRENCİ DOSTU AÇIKLAMA:""",
                "parameters": ["topic", "grade_level"],
                "examples": ["/basit-anlat topic='Fotosentez' grade_level='5. sınıf'"],
                "subject_area": "Genel",
                "grade_level": "İlkokul-Ortaokul"
            },
            {
                "command": "/analoji-yap",
                "name": "Analoji ile Açıklama",
                "description": "Konuyu günlük hayattan örneklerle açıkla",
                "prompt_template": """Sen bir eğitim asistanısın. Verilen konuyu günlük hayattan tanıdık örnekler ve analojiler kullanarak açıkla.

KONU: {topic}
HEDEF KİTLE: {audience}

KURALLAR:
- Günlük yaşamdan örnekler kullan
- Tanıdık durumlarla karşılaştır
- Görsel imgeler oluştur
- Kolay hatırlanabilir analojiler bul
- Öğrencinin deneyim dünyasından referans al

ANALOJİ İLE AÇIKLAMA:""",
                "parameters": ["topic", "audience"],
                "examples": ["/analoji-yap topic='Atom yapısı' audience='9. sınıf'"],
                "subject_area": "Genel",
                "grade_level": "Tüm Seviyeler"
            },
            {
                "command": "/soru-sor",
                "name": "Öğrenci Katılımı",
                "description": "Konuyla ilgili düşündürücü sorular sor",
                "prompt_template": """Sen bir eğitmen asistanısın. Verilen konuyla ilgili öğrenci katılımını artıracak, düşündürücü sorular oluştur.

KONU: {topic}
ÖĞRENCİ SEVİYESİ: {level}
SORU TİPİ: {question_type}

KURALLAR:
- Açık uçlu sorular sor
- Eleştirel düşünmeyi teşvik et  
- Öğrenci deneyimlerine bağla
- Tartışma ortamı yarat
- Merak uyandır

KATILIM SORULARI:""",
                "parameters": ["topic", "level", "question_type"],
                "examples": ["/soru-sor topic='Çevre kirliliği' level='7. sınıf' question_type='tartışma'"],
                "subject_area": "Genel",
                "grade_level": "Tüm Seviyeler"
            },
            {
                "command": "/ozet-cikar",
                "name": "Özet Çıkarma",
                "description": "Uzun metni önemli noktalarıyla özetle",
                "prompt_template": """Sen bir eğitim asistanısın. Verilen metni öğrenci seviyesine uygun şekilde özetle.

METİN: {text}
ÖĞRENCİ SEVİYESİ: {grade}
ÖZET UZUNLUĞU: {length}

KURALLAR:
- Ana fikirleri vurgula
- Önemli detayları koru
- Mantıklı sıra takip et
- Basit dil kullan
- Madde madde düzenle

ÖZET:""",
                "parameters": ["text", "grade", "length"],
                "examples": ["/ozet-cikar text='Uzun tarih metni' grade='8. sınıf' length='kısa'"],
                "subject_area": "Genel",
                "grade_level": "Tüm Seviyeler"
            },
            {
                "command": "/test-hazirla",
                "name": "Test Soruları",
                "description": "Konuyla ilgili test soruları hazırla",
                "prompt_template": """Sen bir eğitim asistanısın. Verilen konu için seviyeye uygun test soruları hazırla.

KONU: {topic}
ÖĞRENCİ SEVİYESİ: {grade}
SORU SAYISI: {count}
SORU TİPİ: {type}

KURALLAR:
- Konu kapsamından çıkma
- Seviyeye uygun zorluktuta tut
- Net ve anlaşılır sor
- Dikkat çekici seçenekler oluştur
- Doğru cevapları işaretle

TEST SORULARI:""",
                "parameters": ["topic", "grade", "count", "type"],
                "examples": ["/test-hazirla topic='Osmanlı Devleti' grade='10. sınıf' count='5' type='çoktan seçmeli'"],
                "subject_area": "Genel",
                "grade_level": "Tüm Seviyeler"
            }
        ]
        
        # Varsayılan komutları veritabanına ekle
        for cmd_data in default_commands:
            cmd_id = hashlib.md5(cmd_data["command"].encode()).hexdigest()
            cmd = PromptCommand(
                id=cmd_id,
                command=cmd_data["command"],
                name=cmd_data["name"], 
                description=cmd_data["description"],
                prompt_template=cmd_data["prompt_template"],
                parameters=cmd_data["parameters"],
                examples=cmd_data["examples"],
                subject_area=cmd_data["subject_area"],
                grade_level=cmd_data["grade_level"],
                created_by="system",
                created_at=datetime.now().isoformat()
            )
            self._save_command_if_not_exists(cmd)
    
    def _save_command_if_not_exists(self, command: PromptCommand):
        """Komut zaten yoksa kaydet"""
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute("SELECT id FROM prompt_commands WHERE command = ?", (command.command,)).fetchone()
            if not existing:
                self.save_prompt_command(command)
    
    def create_custom_prompt(self, name: str, description: str, template: str, 
                           category: PromptCategory, complexity: PromptComplexity,
                           language: LanguageCode, created_by: str,
                           tags: Optional[List[str]] = None) -> CustomPrompt:
        """Yeni özel prompt oluştur"""
        
        # Template içindeki değişkenleri tespit et
        import re
        variables = list(set(re.findall(r'\{(\w+)\}', template)))
        
        prompt_id = hashlib.md5(f"{name}_{created_by}_{datetime.now()}".encode()).hexdigest()
        
        prompt = CustomPrompt(
            id=prompt_id,
            name=name,
            description=description,
            template=template,
            category=category,
            complexity=complexity,
            language=language,
            variables=variables,
            tags=tags or [],
            created_by=created_by,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.save_custom_prompt(prompt)
        return prompt
    
    def save_custom_prompt(self, prompt: CustomPrompt):
        """Özel prompt'u veritabanına kaydet"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO custom_prompts 
                (id, name, description, template, category, complexity, language, 
                 variables, tags, created_by, created_at, updated_at, usage_count, avg_rating, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prompt.id, prompt.name, prompt.description, prompt.template,
                prompt.category.value, prompt.complexity.value, prompt.language,
                json.dumps(prompt.variables), json.dumps(prompt.tags),
                prompt.created_by, prompt.created_at, prompt.updated_at,
                prompt.usage_count, prompt.avg_rating, prompt.is_active
            ))
    
    def save_prompt_command(self, command: PromptCommand):
        """Prompt komutunu veritabanına kaydet"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO prompt_commands
                (id, command, name, description, prompt_template, parameters, examples,
                 subject_area, grade_level, created_by, created_at, usage_count, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                command.id, command.command, command.name, command.description,
                command.prompt_template, json.dumps(command.parameters),
                json.dumps(command.examples), command.subject_area,
                command.grade_level, command.created_by, command.created_at,
                command.usage_count, command.is_active
            ))
    
    def get_custom_prompts(self, created_by: Optional[str] = None, 
                          category: Optional[PromptCategory] = None) -> List[CustomPrompt]:
        """Özel prompt'ları getir"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM custom_prompts WHERE is_active = 1"
            params = []
            
            if created_by:
                query += " AND created_by = ?"
                params.append(created_by)
            
            if category:
                query += " AND category = ?"  
                params.append(category.value)
            
            query += " ORDER BY created_at DESC"
            
            rows = conn.execute(query, params).fetchall()
            
            prompts = []
            for row in rows:
                prompts.append(CustomPrompt(
                    id=row[0], name=row[1], description=row[2], template=row[3],
                    category=PromptCategory(row[4]), complexity=PromptComplexity(row[5]),
                    language=row[6], variables=json.loads(row[7]), tags=json.loads(row[8]),
                    created_by=row[9], created_at=row[10], updated_at=row[11],
                    usage_count=row[12], avg_rating=row[13], is_active=bool(row[14])
                ))
            
            return prompts
    
    def get_prompt_commands(self, subject_area: Optional[str] = None) -> List[PromptCommand]:
        """Prompt komutlarını getir"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM prompt_commands WHERE is_active = 1"
            params = []
            
            if subject_area:
                query += " AND (subject_area = ? OR subject_area = 'Genel')"
                params.append(subject_area)
            
            query += " ORDER BY usage_count DESC, created_at DESC"
            
            rows = conn.execute(query, params).fetchall()
            
            commands = []
            for row in rows:
                commands.append(PromptCommand(
                    id=row[0], command=row[1], name=row[2], description=row[3],
                    prompt_template=row[4], parameters=json.loads(row[5]),
                    examples=json.loads(row[6]), subject_area=row[7],
                    grade_level=row[8], created_by=row[9], created_at=row[10],
                    usage_count=row[11], is_active=bool(row[12])
                ))
            
            return commands
    
    def execute_prompt_command(self, command: str, **kwargs) -> Tuple[str, Optional[str]]:
        """Prompt komutunu çalıştır"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT prompt_template, parameters FROM prompt_commands WHERE command = ? AND is_active = 1",
                (command,)
            ).fetchone()
            
            if not row:
                return "", f"Komut bulunamadı: {command}"
            
            template, params_json = row
            required_params = json.loads(params_json)
            
            # Gerekli parametrelerin kontrolü
            missing_params = [p for p in required_params if p not in kwargs]
            if missing_params:
                return "", f"Eksik parametreler: {', '.join(missing_params)}"
            
            # Template'i doldur
            try:
                filled_prompt = template.format(**kwargs)
                
                # Kullanım sayısını artır
                conn.execute(
                    "UPDATE prompt_commands SET usage_count = usage_count + 1 WHERE command = ?",
                    (command,)
                )
                
                return filled_prompt, None
            except KeyError as e:
                return "", f"Template hatası: {str(e)}"
    
    def record_prompt_performance(self, performance: PromptPerformance):
        """Prompt performansını kaydet"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO prompt_performance
                (prompt_id, execution_time, user_rating, response_quality, 
                 educational_effectiveness, engagement_score, timestamp, session_id, user_feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                performance.prompt_id, performance.execution_time, performance.user_rating,
                performance.response_quality, performance.educational_effectiveness,
                performance.engagement_score, performance.timestamp,
                performance.session_id, performance.user_feedback
            ))
            
            # Ortalama rating'i güncelle
            if performance.user_rating:
                avg_rating = conn.execute(
                    "SELECT AVG(user_rating) FROM prompt_performance WHERE prompt_id = ? AND user_rating IS NOT NULL",
                    (performance.prompt_id,)
                ).fetchone()[0]
                
                conn.execute(
                    "UPDATE custom_prompts SET avg_rating = ? WHERE id = ?",
                    (avg_rating, performance.prompt_id)
                )
    
    def get_prompt_analytics(self, prompt_id: Optional[str] = None, 
                           days: int = 30) -> Dict[str, Any]:
        """Prompt analitiklerini getir"""
        with sqlite3.connect(self.db_path) as conn:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            if prompt_id:
                # Belirli bir prompt için
                stats = conn.execute("""
                    SELECT 
                        COUNT(*) as usage_count,
                        AVG(execution_time) as avg_execution_time,
                        AVG(user_rating) as avg_user_rating,
                        AVG(response_quality) as avg_response_quality,
                        AVG(educational_effectiveness) as avg_educational_effectiveness,
                        AVG(engagement_score) as avg_engagement_score
                    FROM prompt_performance 
                    WHERE prompt_id = ? AND timestamp > ?
                """, (prompt_id, since_date)).fetchone()
                
                return {
                    "prompt_id": prompt_id,
                    "usage_count": stats[0],
                    "avg_execution_time": stats[1],
                    "avg_user_rating": stats[2],
                    "avg_response_quality": stats[3],
                    "avg_educational_effectiveness": stats[4],
                    "avg_engagement_score": stats[5]
                }
            else:
                # Genel istatistikler
                overall_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_executions,
                        COUNT(DISTINCT prompt_id) as unique_prompts_used,
                        AVG(execution_time) as avg_execution_time,
                        AVG(user_rating) as avg_user_rating,
                        AVG(response_quality) as avg_response_quality,
                        AVG(educational_effectiveness) as avg_educational_effectiveness
                    FROM prompt_performance 
                    WHERE timestamp > ?
                """, (since_date,)).fetchone()
                
                # En popüler komutlar
                popular_commands = conn.execute("""
                    SELECT command, name, usage_count 
                    FROM prompt_commands 
                    WHERE is_active = 1 
                    ORDER BY usage_count DESC 
                    LIMIT 10
                """).fetchall()
                
                return {
                    "total_executions": overall_stats[0],
                    "unique_prompts_used": overall_stats[1],
                    "avg_execution_time": overall_stats[2], 
                    "avg_user_rating": overall_stats[3],
                    "avg_response_quality": overall_stats[4],
                    "avg_educational_effectiveness": overall_stats[5],
                    "popular_commands": [
                        {"command": row[0], "name": row[1], "usage_count": row[2]}
                        for row in popular_commands
                    ]
                }
    
    def search_prompts_and_commands(self, query: str, language: Optional[LanguageCode] = None) -> Dict[str, List]:
        """Prompt ve komutlarda arama yap"""
        with sqlite3.connect(self.db_path) as conn:
            # Prompt'larda ara
            prompt_query = """
                SELECT * FROM custom_prompts 
                WHERE is_active = 1 AND (
                    name LIKE ? OR description LIKE ? OR template LIKE ?
                )
            """
            params = [f"%{query}%", f"%{query}%", f"%{query}%"]
            
            if language:
                prompt_query += " AND language = ?"
                params.append(language)
            
            prompt_rows = conn.execute(prompt_query, params).fetchall()
            
            # Komutlarda ara
            command_query = """
                SELECT * FROM prompt_commands
                WHERE is_active = 1 AND (
                    command LIKE ? OR name LIKE ? OR description LIKE ? OR prompt_template LIKE ?
                )
            """
            command_params = [f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%"]
            command_rows = conn.execute(command_query, command_params).fetchall()
            
            # Sonuçları dönüştür
            prompts = []
            for row in prompt_rows:
                prompts.append({
                    "id": row[0], "name": row[1], "description": row[2],
                    "category": row[4], "complexity": row[5],
                    "usage_count": row[12], "avg_rating": row[13]
                })
            
            commands = []
            for row in command_rows:
                commands.append({
                    "id": row[0], "command": row[1], "name": row[2],
                    "description": row[3], "subject_area": row[7],
                    "usage_count": row[11]
                })
            
            return {"prompts": prompts, "commands": commands}

# Global instance
teacher_prompt_manager = TeacherPromptManager()