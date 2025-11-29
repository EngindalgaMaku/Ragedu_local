"""
LLM-based Module Organization Logic
Uses curriculum-aware prompts to organize topics into educational modules
Integrates with existing model inference service
"""

import logging
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import requests
import os

from templates.curriculum_templates import get_curriculum_template_manager

logger = logging.getLogger(__name__)

# Environment variables - compatible with existing APRAG service
MODEL_INFERENCER_URL = os.getenv("MODEL_INFERENCER_URL", os.getenv("MODEL_INFERENCE_URL", "http://model-inference-service:8002"))


class LLMModuleOrganizer:
    """Handles LLM-based topic organization into modules"""

    def __init__(self):
        self.template_manager = get_curriculum_template_manager()
        self.logger = logging.getLogger(__name__ + '.LLMModuleOrganizer')
        self.logger.info("LLM Module Organizer initialized")

    async def organize_topics_into_modules(
        self,
        topics: List[Dict[str, Any]],
        course_info: Dict[str, Any],
        template: Optional[Dict[str, Any]] = None,
        strategy: str = "curriculum_aligned",
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Main organization method using LLM analysis
        
        Args:
            topics: List of topics to organize
            course_info: Course information including curriculum details
            template: Curriculum template (optional, will be auto-selected if None)
            strategy: Organization strategy
            options: Additional options
            
        Returns:
            List of organized module candidates
        """
        self.logger.info(f"Starting LLM-based organization for {len(topics)} topics using {strategy} strategy")

        if not topics:
            self.logger.warning("No topics provided for organization")
            return []

        llm_response = None
        try:
            # 1. Prepare curriculum-aware prompt
            prompt = await self._prepare_organization_prompt(
                topics, course_info, template, options or {}
            )

            # 2. Get LLM response with proper model selection
            model_to_use = self._select_model_for_organization(course_info, options)
            llm_response = await self._call_llm_service(prompt, model_to_use, options)

            # 3. Parse and validate response
            module_candidates = await self._parse_llm_response(llm_response, topics)

            # 4. Apply post-processing refinements
            refined_modules = await self._refine_module_candidates(
                module_candidates, options or {}, course_info
            )

            self.logger.info(f"LLM organization complete: {len(refined_modules)} modules created")
            return refined_modules

        except ValueError as e:
            # If it's already a ValueError with LLM response, re-raise as-is
            if "LLM Yanıtı" in str(e) or llm_response is None:
                raise
            
            # Otherwise, add LLM response to the error
            response_preview = llm_response[:1000] if len(llm_response) > 1000 else llm_response
            if len(llm_response) > 1000:
                response_preview += f"\n\n... (toplam {len(llm_response)} karakter, ilk 1000 karakter gösteriliyor)"
            
            error_msg = (
                f"{str(e)}\n\n"
                f"LLM Yanıtı (ilk 1000 karakter):\n{response_preview}"
            )
            self.logger.error(f"LLM organization failed: {e}\nLLM Response: {llm_response[:500]}")
            raise ValueError(error_msg) from e
            
        except Exception as e:
            # For other exceptions, add LLM response if available
            if llm_response:
                response_preview = llm_response[:1000] if len(llm_response) > 1000 else llm_response
                if len(llm_response) > 1000:
                    response_preview += f"\n\n... (toplam {len(llm_response)} karakter, ilk 1000 karakter gösteriliyor)"
                
                error_msg = (
                    f"LLM organizasyonu başarısız oldu: {str(e)}\n\n"
                    f"LLM Yanıtı (ilk 1000 karakter):\n{response_preview}\n\n"
                    "Lütfen LLM servisini kontrol edin ve tekrar deneyin."
                )
            else:
                error_msg = (
                    f"LLM organizasyonu başarısız oldu: {str(e)}\n\n"
                    "LLM servisine çağrı yapılamadı veya yanıt alınamadı.\n"
                    "Lütfen LLM servisini kontrol edin ve tekrar deneyin."
                )
            
            self.logger.error(f"LLM organization failed: {e}\nLLM Response: {llm_response[:500] if llm_response else 'No response'}")
            raise ValueError(error_msg) from e

    async def _prepare_organization_prompt(
        self,
        topics: List[Dict[str, Any]],
        course_info: Dict[str, Any],
        template: Optional[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> str:
        """Prepare curriculum-specific prompt for LLM"""
        
        # Check if user provided custom curriculum_prompt
        user_curriculum_prompt = options.get('curriculum_prompt')
        if user_curriculum_prompt:
            # Use user's custom curriculum prompt directly
            topics_context = self._format_topics_for_prompt(topics)
            
            # Build prompt with user's instruction
            full_prompt = f"""{user_curriculum_prompt}

MEVCUT KONULAR:
{topics_context}

MODÜL ORGANİZASYON PRENSİPLERİ:
1. Konular mantıklı öğrenme sırası takip etmeli (basit → karmaşık)
2. Her modül 3-12 konu içermeli
3. Zorluk derecesi kademeli artmalı
4. Önkoşul ilişkileri belirtilmeli
5. ÇOK ÖNEMLİ: Her konu SADECE BİR modülde olmalı - aynı konuyu birden fazla modüle koyma!
6. ÇOK ÖNEMLİ: TÜM konular modüllere dağıtılmalı - hiçbir konu boşta kalmamalı!
7. Konuları modül başlıklarına ve açıklamalarına göre ilgili modüllere yerleştir

MODÜL BAŞLIK FORMATI (MUTLAKA UYULMALI):
- Modül başlıkları (module_title) Türkiye eğitim müfredatındaki ünite başlıklarına benzer formatta olmalı
- Her kelimenin SADECE İLK HARFİ BÜYÜK, diğer harfler KÜÇÜK olmalı (Title Case)
- Örnek DOĞRU format: "Biyoloji ve Canlıların Ortak Özellikleri - I"
- Örnek DOĞRU format: "Organik Bileşikler - Karbonhidratlar"
- Örnek DOĞRU format: "Nükleik Asitler (DNA)"
- Örnek DOĞRU format: "Enzimlerin Çalışmasına Etki Eden Faktörler"
- Örnek YANLIŞ format: "BİYOLOJİ VE CANLILARIN ORTAK ÖZELLİKLERİ" (TÜMÜ BÜYÜK HARF - YANLIŞ!)
- Örnek YANLIŞ format: "biyoloji ve canlıların ortak özellikleri" (TÜMÜ KÜÇÜK HARF - YANLIŞ!)
- Başlıklar Türkçe karakterleri doğru kullanmalı (ı, İ, ş, Ş, ğ, Ğ, ü, Ü, ö, Ö, ç, Ç)
- Başlıklar anlamlı, müfredat standartlarına uygun ve eğitimsel olmalı

ÇIKTI FORMATI (JSON):
{{
  "modules": [
    {{
      "module_title": "Modül Başlığı (Title Case Formatında)",
      "module_description": "Modül açıklaması",
      "module_order": 1,
      "estimated_duration_hours": 20,
      "difficulty_level": "intermediate",
      "learning_outcomes": ["Öğrenme hedefi 1", "Öğrenme hedefi 2"],
      "topics": [
        {{
          "topic_id": 123,
          "topic_order_in_module": 1,
          "importance_level": "medium"
        }}
      ]
    }}
  ]
}}

ÖNEMLİ KURALLAR:
- MUTLAKA modül organizasyonu yap: Konuları mantıklı gruplara ayır ve her gruba bir modül başlığı ver
- Sadece konu listesi döndürme! Konuları modüllere organize etmelisin
- JSON çıktısında İNGİLİZCE anahtar kelimeler kullan: "modules", "module_title", "module_description", "topics", "topic_id"
- Her modül en az 3-12 konu içermeli
- KRİTİK: Her konu SADECE BİR modülde olmalı - aynı topic_id'yi birden fazla modülde kullanma!
- KRİTİK: MEVCUT KONULAR listesindeki TÜM konuları modüllere dağıt - hiçbir konu eksik kalmamalı!
- Modül başlıkları Title Case formatında olmalı (her kelimenin sadece ilk harfi büyük)

Sadece JSON çıktısı ver, açıklama yapma."""
            
            self.logger.info(f"Using user-provided curriculum prompt (length: {len(user_curriculum_prompt)} chars)")
            return full_prompt
        
        # Get curriculum template if not provided
        if not template:
            curriculum_type = course_info.get('curriculum_standard', 'generic')
            subject_area = course_info.get('subject_area', 'general')
            grade_level = course_info.get('grade_level', '9')
            
            template_text = self.template_manager.get_template(
                curriculum=curriculum_type,
                subject=subject_area,
                grade_level=grade_level,
                topics=topics
            )
        else:
            # Use provided template and format with topics
            template_text = template.get('prompt_template', '')
            topics_context = self._format_topics_for_prompt(topics)
            
            # Prepare all format kwargs - default template needs grade and subject
            format_kwargs = {"topics_list": topics_context}
            format_kwargs["grade"] = course_info.get('grade_level', '9')
            format_kwargs["subject"] = course_info.get('subject_area', 'genel')
            
            try:
                template_text = template_text.format(**format_kwargs)
            except KeyError as e:
                # Log the problematic template for debugging
                self.logger.error(f"Template format error: {e}")
                self.logger.error(f"Template preview (first 500 chars): {template_text[:500]}")
                self.logger.error(f"Format kwargs keys: {list(format_kwargs.keys())}")
                # Try to find the problematic placeholder in template
                import re
                placeholders = re.findall(r'\{([^}]+)\}', template_text)
                self.logger.error(f"Found placeholders in template: {placeholders}")
                raise ValueError(
                    f"Template format hatası: {str(e)}\n\n"
                    f"Template içinde beklenmeyen placeholder bulundu. "
                    f"Lütfen template'i kontrol edin veya sistem yöneticisine başvurun.\n\n"
                    f"Template (ilk 500 karakter):\n{template_text[:500]}"
                ) from e

        # Add organization constraints based on options
        constraints = self._build_organization_constraints(options)
        
        # Add strategy-specific instructions
        strategy_instructions = self._get_strategy_instructions(options.get('strategy', 'curriculum_aligned'))

        # Combine all elements
        full_prompt = f"""
{template_text}

EK ORGANİZASYON KISITLARI:
{constraints}

STRATEJİ TALİMATLARI:
{strategy_instructions}

ÖNEMLI: Sadece geçerli JSON formatında çıktı ver. Hiçbir açıklama ekleme.
"""

        self.logger.debug(f"Generated prompt length: {len(full_prompt)} characters")
        return full_prompt

    def _format_topics_for_prompt(self, topics: List[Dict[str, Any]]) -> str:
        """Format topics for inclusion in prompts"""
        formatted_topics = []
        
        for i, topic in enumerate(topics, 1):
            # Extract keywords safely
            keywords = topic.get('keywords', [])
            if isinstance(keywords, str):
                try:
                    keywords = json.loads(keywords)
                except:
                    keywords = [keywords] if keywords else []
            
            # Extract related chunks count
            related_chunks = topic.get('related_chunk_ids', [])
            if isinstance(related_chunks, str):
                try:
                    related_chunks = json.loads(related_chunks)
                except:
                    related_chunks = []
            
            topic_info = f"""
{i}. Konu ID: {topic['topic_id']}
   Başlık: {topic['topic_title']}
   Açıklama: {topic.get('description', 'Açıklama yok')}
   Zorluk: {topic.get('estimated_difficulty', 'orta')}
   Anahtar kelimeler: {', '.join(keywords) if keywords else 'Yok'}
   İlgili içerik: {len(related_chunks)} parça
   Sıra: {topic.get('topic_order', i)}"""
            
            formatted_topics.append(topic_info)

        return '\n'.join(formatted_topics)

    def _normalize_modules_format(self, raw_modules: List[Any]) -> List[Dict[str, Any]]:
        """Normalize any module format to standard format"""
        normalized = []
        
        for i, modul_item in enumerate(raw_modules):
            if not isinstance(modul_item, dict):
                self.logger.warning(f"Skipping non-dict module item: {type(modul_item)}")
                continue
            
            # Extract module title from any possible key
            module_title = (
                modul_item.get('başlık') or modul_item.get('baslik') or 
                modul_item.get('module_title') or modul_item.get('title') or
                modul_item.get('name') or f"Modül {i + 1}"
            )
            
            # Extract topics from any possible structure
            topics = []
            
            # Try different ways topics might be stored
            # 1. Direct "konular" or "topics" array
            konular = (
                modul_item.get('konular') or modul_item.get('konular') or
                modul_item.get('topics') or modul_item.get('topic_list') or []
            )
            
            # 2. Single "konuID" or "topic_id"
            konu_id = (
                modul_item.get('konuID') or modul_item.get('konu_id') or
                modul_item.get('topic_id') or modul_item.get('topicId')
            )
            
            # 3. "modül" key (nested structure)
            modul_nested = modul_item.get('modül') or modul_item.get('modul')
            
            # Process konular array
            if konular:
                for topic_item in konular:
                    topic_data = self._extract_topic_data(topic_item, module_title)
                    if topic_data:
                        topics.append(topic_data)
            
            # Process single konu_id
            if konu_id:
                try:
                    topic_id = int(konu_id)
                    topics.append({
                        'topic_id': topic_id,
                        'topic_title': module_title,
                        'topic_order_in_module': len(topics) + 1,
                        'importance_level': 'medium'
                    })
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid konu_id format: {konu_id}")
            
            # Process nested modül
            if modul_nested:
                if isinstance(modul_nested, list):
                    for topic_item in modul_nested:
                        topic_data = self._extract_topic_data(topic_item, module_title)
                        if topic_data:
                            topics.append(topic_data)
                elif isinstance(modul_nested, dict):
                    topic_data = self._extract_topic_data(modul_nested, module_title)
                    if topic_data:
                        topics.append(topic_data)
            
            # If no topics found but module has an "id" field, use it as topic_id
            if not topics:
                modul_id = modul_item.get('id') or modul_item.get('module_id')
                if modul_id:
                    try:
                        topic_id = int(modul_id)
                        topics.append({
                            'topic_id': topic_id,
                            'topic_title': module_title,
                            'topic_order_in_module': 1,
                            'importance_level': 'medium'
                        })
                    except (ValueError, TypeError):
                        pass
            
            normalized.append({
                'module_title': module_title,
                'module_description': (
                    modul_item.get('aciklama') or modul_item.get('açıklama') or
                    modul_item.get('module_description') or modul_item.get('description') or
                    f"Eğitim modülü - {len(topics)} konu"
                ),
                'module_order': (
                    modul_item.get('sira') or modul_item.get('sıra') or
                    modul_item.get('module_order') or modul_item.get('order') or
                    i + 1
                ),
                'difficulty_level': (
                    modul_item.get('zorluk') or modul_item.get('difficulty_level') or
                    modul_item.get('difficulty') or 'intermediate'
                ),
                'topics': topics
            })
        
        return normalized
    
    def _extract_topic_data(self, topic_item: Any, default_title: str = '') -> Optional[Dict[str, Any]]:
        """Extract topic data from any format"""
        if topic_item is None:
            return None
        
        # Handle integer/string topic ID
        if isinstance(topic_item, (int, str)):
            try:
                topic_id = int(topic_item)
                return {
                    'topic_id': topic_id,
                    'topic_title': default_title,
                    'topic_order_in_module': 1,
                    'importance_level': 'medium'
                }
            except (ValueError, TypeError):
                return None
        
        # Handle dict topic
        if not isinstance(topic_item, dict):
            return None
        
        # Extract topic_id from any possible key
        topic_id = (
            topic_item.get('id') or topic_item.get('konuID') or topic_item.get('konu_id') or
            topic_item.get('topic_id') or topic_item.get('topicId') or
            topic_item.get('konuID') or topic_item.get('konuId')
        )
        
        if topic_id is None:
            return None
        
        try:
            topic_id = int(topic_id)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid topic_id format: {topic_id}")
            return None
        
        return {
            'topic_id': topic_id,
            'topic_title': (
                topic_item.get('başlık') or topic_item.get('baslik') or
                topic_item.get('title') or topic_item.get('name') or
                default_title
            ),
            'topic_order_in_module': (
                topic_item.get('sira') or topic_item.get('sıra') or
                topic_item.get('topic_order_in_module') or topic_item.get('order') or
                1
            ),
            'importance_level': (
                topic_item.get('onem') or topic_item.get('önem') or
                topic_item.get('importance_level') or topic_item.get('importance') or
                'medium'
            )
        }

    def _build_organization_constraints(self, options: Dict[str, Any]) -> str:
        """Build organization constraints from options"""
        constraints = []

        max_modules = options.get('max_modules_per_course', 12)
        min_topics = options.get('min_topics_per_module', 3)
        max_topics = options.get('max_topics_per_module', 15)

        constraints.append(f"- Maksimum {max_modules} modül oluştur")
        constraints.append(f"- Her modülde en az {min_topics} konu olmalı")
        constraints.append(f"- Her modülde en fazla {max_topics} konu olmalı")

        if options.get('include_prerequisites', True):
            constraints.append("- Önkoşul ilişkilerini mantıklı şekilde belirt")

        alignment_threshold = options.get('alignment_threshold', 0.7)
        constraints.append(f"- Müfredat uyumu en az %{alignment_threshold*100:.0f} olmalı")

        if options.get('progressive_difficulty', True):
            constraints.append("- Zorluk seviyesi kademeli artmalı (kolay → zor)")

        return '\n'.join(constraints) if constraints else "Genel organizasyon kurallarını takip et"

    def _get_strategy_instructions(self, strategy: str) -> str:
        """Get strategy-specific instructions"""
        instructions = {
            'curriculum_aligned': """
- Resmi müfredat ünitelerine kesinlikle uygun organize et
- Her modül belirli müfredat kazanımlarını karşılamalı
- Müfredat sırasını takip et
- Öğretim programındaki zaman dağılımını dikkate al""",
            
            'semantic_clustering': """
- Konuları içerik benzerliğine göre grupla
- Semantik olarak ilişkili konuları bir araya getir
- Kavramsal bağlantıları öncelikle
- Anahtar kelimeleri analiz ederek grupla""",
            
            'hybrid': """
- Hem müfredat uyumunu hem semantik benzerliği dikkate al
- Önce müfredat ünitelerine uygun grupla, sonra semantik optimizasyon yap
- En iyi öğrenme deneyimi için iki yaklaşımı birleştir
- Esnek ama tutarlı organizasyon sağla""",
            
            'difficulty_progressive': """
- Zorluk seviyesine göre organize et
- Basit konulardan karmaşık konulara doğru ilerle
- Her modülde zorluk kademeli artsın
- Öğrenci hazır bulunuşluğunu dikkate al"""
        }
        
        return instructions.get(strategy, instructions['curriculum_aligned'])

    def _select_model_for_organization(self, course_info: Dict[str, Any], options: Optional[Dict[str, Any]]) -> str:
        """Select appropriate model for organization task"""
        # Use options-specified model if available
        if options and options.get('model'):
            return options['model']
        
        # For Turkish curriculum, prefer models good with Turkish
        subject = course_info.get('subject_area', '').lower()
        
        # Complex subjects might need more powerful models
        if subject in ['mathematics', 'physics', 'chemistry']:
            return "llama-3.1-70b-instant"  # More capable model for complex subjects
        else:
            return "llama-3.1-8b-instant"  # Fast model for general organization

    async def _call_llm_service(self, prompt: str, model: str, options: Dict[str, Any]) -> str:
        """Call the model inference service"""
        # Determine timeout based on model and complexity
        timeout_seconds = 300 if "70b" in model else 120  # Longer timeout for larger models
        
        # Smart truncation for token limits
        final_prompt = prompt
        if "instant" in model.lower() and len(prompt) > 15000:
            # Truncate for Groq models to avoid token limits
            final_prompt = prompt[:15000] + "\n\nSadece JSON çıktısı ver, başka açıklama yapma."
            self.logger.warning(f"Prompt truncated for model {model}: {len(prompt)} -> 15000 chars")

        try:
            self.logger.info(f"Calling LLM service with model: {model}")
            
            response = requests.post(
                f"{MODEL_INFERENCER_URL}/models/generate",
                json={
                    "prompt": final_prompt,
                    "model": model,
                    "max_tokens": 4096,
                    "temperature": options.get('temperature', 0.3)
                },
                timeout=timeout_seconds
            )

            if response.status_code != 200:
                error_detail = self._extract_error_from_response(response)
                self.logger.error(f"LLM service error: {response.status_code} - {error_detail}")
                raise Exception(f"LLM service error: {error_detail}")

            result = response.json()
            llm_output = result.get("response", "")
            
            self.logger.info(f"LLM response received: {len(llm_output)} characters")
            return llm_output

        except requests.exceptions.Timeout:
            self.logger.error(f"LLM request timeout after {timeout_seconds}s with model {model}")
            raise Exception(f"LLM request timeout with model {model}")
        
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Failed to connect to LLM service at {MODEL_INFERENCER_URL}")
            raise Exception("Could not connect to LLM service")
        
        except Exception as e:
            self.logger.error(f"Unexpected error in LLM call: {e}")
            raise

    def _extract_error_from_response(self, response) -> str:
        """Safely extract error message from response"""
        try:
            if hasattr(response, 'text') and response.text:
                return response.text[:500]
            elif hasattr(response, 'content') and response.content:
                return response.content.decode('utf-8', errors='ignore')[:500]
            else:
                return f"HTTP {response.status_code} - No response body"
        except Exception:
            return f"HTTP {response.status_code} - Error reading response"

    async def _parse_llm_response(
        self,
        llm_response: str,
        original_topics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse and validate LLM response with comprehensive error handling"""
        
        self.logger.debug(f"Parsing LLM response length: {len(llm_response)}")
        
        try:
            # Step 1: Extract JSON from response
            json_str = self._extract_json_from_response(llm_response)
            if not json_str:
                raise ValueError("No JSON found in LLM response")
            
            # Step 2: Parse JSON (with repair attempt on error)
            try:
                parsed_response = json.loads(json_str)
            except json.JSONDecodeError as parse_error:
                # Try to repair JSON
                self.logger.warning(f"Initial JSON parse failed: {parse_error}. Attempting repair...")
                repaired_json = self._repair_json(llm_response)
                if repaired_json:
                    parsed_response = json.loads(repaired_json)
                    self.logger.info("JSON successfully repaired and parsed")
                else:
                    raise  # Re-raise original error if repair failed
            
            # Step 3: Validate structure
            # Check if response is just a list of topics (wrong format)
            if isinstance(parsed_response, list):
                # LLM returned just a list of topics instead of organized modules
                error_msg = (
                    "LLM yanıtı yanlış formatta: Konular modüllere organize edilmemiş.\n\n"
                    "LLM sadece konu listesi döndürmüş, modül organizasyonu yapmamış.\n"
                    "Beklenen format: {{'modules': [...]}} şeklinde modül organizasyonu.\n"
                    "Dönen format: Sadece konu listesi.\n\n"
                    "Lütfen prompt'u kontrol edin veya farklı bir model deneyin."
                )
                raise ValueError(error_msg)
            
            if not isinstance(parsed_response, dict):
                raise ValueError("Response is not a JSON object")
            
            # Support multiple formats: English, Turkish, and alternative Turkish formats
            # Use a flexible approach that handles any format
            modules = None
            
            # Try to find modules list in any possible key
            possible_keys = [
                'modules', 'moduller', 'modüller', 'ders_modulleri', 'dersModulleri',
                'ders_modülleri', 'moduls', 'module_list', 'moduleList'
            ]
            
            for key in possible_keys:
                if key in parsed_response:
                    raw_modules = parsed_response[key]
                    if isinstance(raw_modules, list):
                        modules = self._normalize_modules_format(raw_modules)
                        self.logger.info(f"Found modules under key: {key}, count: {len(modules)}")
                        break
            
            # If still not found, try to find any list that looks like modules
            if modules is None:
                # Try to find any list-like structure
                for key, value in parsed_response.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Check if it looks like modules
                        first_item = value[0]
                        if isinstance(first_item, dict):
                            # Check for any module-like keys
                            module_keys = ['baslik', 'başlık', 'module_title', 'title', 'name', 
                                         'konular', 'konular', 'topics', 'konuID', 'konu_id', 'topic_id']
                            if any(k in first_item for k in module_keys):
                                modules = self._normalize_modules_format(value)
                                self.logger.warning(f"Found modules under unexpected key: {key}, normalized count: {len(modules)}")
                                break
                
                if modules is None:
                    raise ValueError("Response missing 'modules', 'moduller', or 'ders_modülleri' key")
                    
                    # Handle case where each module has a single konuID (not an array)
                    konu_id = modul_item.get('konuID') or modul_item.get('konu_id') or modul_item.get('topic_id')
                    konular = modul_item.get('konular') or modul_item.get('topics', [])
                    
                    topics = []
                    if konu_id:
                        # Single topic ID
                        try:
                            topic_id = int(konu_id)
                            topics.append({
                                'topic_id': topic_id,
                                'topic_title': module_title,  # Use module title as topic title
                                'topic_order_in_module': 1,
                                'importance_level': 'medium'
                            })
                        except (ValueError, TypeError):
                            self.logger.warning(f"Invalid konuID format: {konu_id}")
                    
                    # Also handle konular array if present
                    if konular:
                        for topic_item in konular:
                            if isinstance(topic_item, (int, str)):
                                try:
                                    topic_id = int(topic_item)
                                    topics.append({
                                        'topic_id': topic_id,
                                        'topic_title': '',
                                        'topic_order_in_module': len(topics) + 1,
                                        'importance_level': 'medium'
                                    })
                                except (ValueError, TypeError):
                                    self.logger.warning(f"Invalid topic ID format: {topic_item}")
                                    continue
                            elif isinstance(topic_item, dict):
                                topic_id = topic_item.get('id') or topic_item.get('konuID') or topic_item.get('konu_id') or topic_item.get('topic_id')
                                if topic_id:
                                    try:
                                        topic_id = int(topic_id)
                                        topics.append({
                                            'topic_id': topic_id,
                                            'topic_title': topic_item.get('baslik') or topic_item.get('başlık') or topic_item.get('title', ''),
                                            'topic_order_in_module': len(topics) + 1,
                                            'importance_level': topic_item.get('onem') or topic_item.get('importance_level', 'medium')
                                        })
                                    except (ValueError, TypeError):
                                        self.logger.warning(f"Invalid topic ID format: {topic_id}")
                    
                    modules.append({
                        'module_title': module_title,
                        'module_description': modul_item.get('aciklama') or modul_item.get('module_description', ''),
                        'module_order': modul_item.get('sira') or modul_item.get('module_order', len(modules) + 1),
                        'difficulty_level': modul_item.get('zorluk') or modul_item.get('difficulty_level', 'intermediate'),
                        'topics': topics
                    })
            elif 'ders_modülleri' in parsed_response:
                # Handle "ders_modülleri" format
                ders_modulleri = parsed_response['ders_modülleri']
                modules = []
                for modul_item in ders_modulleri:
                    # Handle structure: {"başlık": "...", "konular": [...]} or {"başlık": "...", "modül": [...]}
                    module_title = modul_item.get('başlık') or modul_item.get('baslik') or modul_item.get('module_title', '')
                    # Try "konular" first (most common), then "modül", then "topics"
                    modul_list = modul_item.get('konular') or modul_item.get('modül') or modul_item.get('modul') or modul_item.get('topics', [])
                    
                    # Convert topics from this format to expected format
                    topics = []
                    for topic_item in modul_list:
                        # Handle case where "konular" is an array of integers (topic IDs)
                        if isinstance(topic_item, (int, str)):
                            # Direct topic ID (integer or string)
                            try:
                                topic_id = int(topic_item)
                                topics.append({
                                    'topic_id': topic_id,
                                    'topic_title': '',  # Will be matched by ID
                                    'topic_order_in_module': len(topics) + 1,
                                    'importance_level': 'medium'
                                })
                                continue
                            except (ValueError, TypeError):
                                self.logger.warning(f"Invalid topic ID format: {topic_item}")
                                continue
                        
                        # Handle case where "konular" is an array of objects
                        if not isinstance(topic_item, dict):
                            self.logger.warning(f"Unexpected topic item type: {type(topic_item)}, value: {topic_item}")
                            continue
                        
                        # Try to match topic by title or other identifiers
                        topic_title = topic_item.get('başlık') or topic_item.get('baslik') or topic_item.get('title', '')
                        # Try "id" first (as LLM returns), then "konu_id", then "topic_id"
                        topic_id = topic_item.get('id') or topic_item.get('konu_id') or topic_item.get('topic_id')
                        
                        if topic_id is None:
                            self.logger.warning(f"Topic item missing ID: {topic_item}")
                            continue
                        
                        topics.append({
                            'topic_id': topic_id,
                            'topic_title': topic_title,  # Store title for matching
                            'topic_order_in_module': len(topics) + 1,
                            'importance_level': topic_item.get('onem') or topic_item.get('importance_level', 'medium')
                        })
                    
                    modules.append({
                        'module_title': module_title,
                        'module_description': modul_item.get('aciklama') or modul_item.get('module_description', ''),
                        'module_order': modul_item.get('sira') or modul_item.get('module_order', len(modules) + 1),
                        'difficulty_level': modul_item.get('zorluk') or modul_item.get('difficulty_level', 'intermediate'),
                        'topics': topics
                    })
            elif 'modüller' in parsed_response or 'moduller' in parsed_response:
                # Convert Turkish format to English format (handle both with and without Turkish characters)
                turkish_modules = parsed_response.get('modüller') or parsed_response.get('moduller')
                modules = []
                for modul in turkish_modules:
                    # Convert Turkish keys to English (handle both "baslik" and "başlık")
                    module = {
                        'module_title': modul.get('başlık') or modul.get('baslik') or modul.get('module_title', ''),
                        'module_description': modul.get('aciklama') or modul.get('açıklama') or modul.get('module_description', ''),
                        'module_order': modul.get('sira') or modul.get('sıra') or modul.get('module_order', len(modules) + 1),
                        'difficulty_level': modul.get('zorluk') or modul.get('difficulty_level', 'intermediate'),
                        'topics': []
                    }
                    # Handle topics if present (handle both "konular" and "konular")
                    konular_list = modul.get('konular') or modul.get('konular') or []
                    if konular_list:
                        for konu in konular_list:
                            # Handle case where "konular" is an array of integers (topic IDs)
                            if isinstance(konu, (int, str)):
                                # Direct topic ID (integer or string)
                                try:
                                    topic_id = int(konu)
                                    module['topics'].append({
                                        'topic_id': topic_id,
                                        'topic_title': '',  # Will be matched by ID
                                        'topic_order_in_module': len(module['topics']) + 1,
                                        'importance_level': 'medium'
                                    })
                                    continue
                                except (ValueError, TypeError):
                                    self.logger.warning(f"Invalid topic ID format: {konu}")
                                    continue
                            
                            # Handle case where "konular" is an array of objects
                            if not isinstance(konu, dict):
                                self.logger.warning(f"Unexpected topic item type: {type(konu)}, value: {konu}")
                                continue
                            
                            # Try "id" first (as LLM sometimes returns), then "konu_id", then "topic_id"
                            topic_id = konu.get('id') or konu.get('konu_id') or konu.get('topic_id')
                            if topic_id is None:
                                self.logger.warning(f"Topic item missing ID: {konu}")
                                continue
                            
                            module['topics'].append({
                                'topic_id': topic_id,
                                'topic_title': konu.get('baslik') or konu.get('başlık') or konu.get('topic_title', ''),  # Store title for matching
                                'topic_order_in_module': konu.get('sira') or konu.get('topic_order_in_module', len(module['topics']) + 1),
                                'importance_level': konu.get('onem') or konu.get('importance_level', 'medium')
                            })
                    elif 'topics' in modul:
                        module['topics'] = modul['topics']
                    # If single topic_id at module level, add it
                    elif 'konu_id' in modul:
                        module['topics'].append({
                            'topic_id': modul['konu_id'],
                            'topic_order_in_module': 1,
                            'importance_level': 'medium'
                        })
                    modules.append(module)
            
            if modules is None:
                # Try to find any list-like structure
                for key, value in parsed_response.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Check if it looks like modules
                        first_item = value[0]
                        if isinstance(first_item, dict) and ('baslik' in first_item or 'başlık' in first_item or 'module_title' in first_item):
                            modules = value
                            self.logger.warning(f"Found modules under unexpected key: {key}")
                            break
                
                if modules is None:
                    raise ValueError("Response missing 'modules', 'moduller', or 'ders_modülleri' key")
            if not isinstance(modules, list):
                raise ValueError("'modules' is not a list")
            
            # Step 4: Validate each module with shared used_topic_ids tracking
            validated_modules = []
            topic_id_map = {t['topic_id']: t for t in original_topics}
            used_topic_ids = set()  # Track used topics across ALL modules
            
            for i, module in enumerate(modules):
                validated_module = await self._validate_module_structure(
                    module, topic_id_map, i + 1, used_topic_ids
                )
                if validated_module:
                    validated_modules.append(validated_module)
            
            if not validated_modules:
                raise ValueError("No valid modules found in response")
            
            # Step 5: Ensure all topics are assigned to modules
            all_topic_ids = set(topic_id_map.keys())
            unused_topic_ids = all_topic_ids - used_topic_ids
            
            if unused_topic_ids:
                self.logger.warning(f"{len(unused_topic_ids)} topics were not assigned to any module. Distributing them...")
                # Distribute unused topics to modules
                unused_topics = [topic_id_map[tid] for tid in unused_topic_ids]
                self._distribute_unused_topics(validated_modules, unused_topics, used_topic_ids)
            
            self.logger.info(f"Successfully parsed {len(validated_modules)} valid modules with {len(used_topic_ids)}/{len(all_topic_ids)} topics assigned")
            return validated_modules

        except json.JSONDecodeError as e:
            # Try to repair JSON before giving up
            self.logger.warning(f"JSON parse error: {e}. Attempting to repair JSON...")
            try:
                repaired_json = self._repair_json(llm_response)
                if repaired_json:
                    # Retry parsing with repaired JSON
                    parsed_response = json.loads(repaired_json)
                    # Continue with normal flow
                    # (We'll need to handle this differently - let's extract JSON first)
                    json_str = repaired_json
                else:
                    raise  # Re-raise original error if repair failed
            except:
                # If repair fails, show original error
                response_preview = llm_response[:1000] if len(llm_response) > 1000 else llm_response
                if len(llm_response) > 1000:
                    response_preview += f"\n\n... (toplam {len(llm_response)} karakter, ilk 1000 karakter gösteriliyor)"
                
                error_msg = (
                    f"LLM yanıtı JSON formatında parse edilemedi: {str(e)}\n\n"
                    f"LLM Yanıtı (ilk 1000 karakter):\n{response_preview}\n\n"
                    "LLM servisi geçersiz JSON döndürdü. Lütfen:\n"
                    "1. LLM servisinin çalıştığından emin olun\n"
                    "2. Model'in JSON formatında yanıt verdiğinden emin olun\n"
                    "3. Daha sonra tekrar deneyin"
                )
                self.logger.error(f"JSON parsing failed: {e}\nLLM Response: {llm_response[:500]}")
                raise ValueError(error_msg) from e
            
            # If we got here, we successfully repaired and parsed
            # Re-run the parsing logic with repaired JSON
            # (This is a bit hacky, but we need to re-enter the parsing flow)
            try:
                # Validate structure
                if isinstance(parsed_response, list):
                    error_msg = (
                        "LLM yanıtı yanlış formatta: Konular modüllere organize edilmemiş.\n\n"
                        "LLM sadece konu listesi döndürmüş, modül organizasyonu yapmamış.\n"
                        "Beklenen format: {{'modules': [...]}} şeklinde modül organizasyonu.\n"
                        "Dönen format: Sadece konu listesi.\n\n"
                        "Lütfen prompt'u kontrol edin veya farklı bir model deneyin."
                    )
                    raise ValueError(error_msg)
                
                if not isinstance(parsed_response, dict):
                    raise ValueError("Response is not a JSON object")
                
                # Continue with module extraction logic (same as above)
                # ... (we'll handle this in the main flow)
                # For now, let's just extract JSON and re-parse
                json_str = repaired_json
                parsed_response = json.loads(json_str)
            except Exception as repair_error:
                # If repair parsing also fails, show original error
                response_preview = llm_response[:1000] if len(llm_response) > 1000 else llm_response
                if len(llm_response) > 1000:
                    response_preview += f"\n\n... (toplam {len(llm_response)} karakter, ilk 1000 karakter gösteriliyor)"
                
                error_msg = (
                    f"LLM yanıtı JSON formatında parse edilemedi: {str(e)}\n\n"
                    f"JSON onarım denemesi de başarısız: {str(repair_error)}\n\n"
                    f"LLM Yanıtı (ilk 1000 karakter):\n{response_preview}\n\n"
                    "LLM servisi geçersiz JSON döndürdü. Lütfen:\n"
                    "1. LLM servisinin çalıştığından emin olun\n"
                    "2. Model'in JSON formatında yanıt verdiğinden emin olun\n"
                    "3. Daha sonra tekrar deneyin"
                )
                self.logger.error(f"JSON parsing and repair failed: {e}, {repair_error}\nLLM Response: {llm_response[:500]}")
                raise ValueError(error_msg) from e
        
        except (ValueError, KeyError) as e:
            # Truncate response if too long for error message
            response_preview = llm_response[:1000] if len(llm_response) > 1000 else llm_response
            if len(llm_response) > 1000:
                response_preview += f"\n\n... (toplam {len(llm_response)} karakter, ilk 1000 karakter gösteriliyor)"
            
            error_msg = (
                f"LLM yanıtı doğrulanamadı: {str(e)}\n\n"
                f"LLM Yanıtı (ilk 1000 karakter):\n{response_preview}\n\n"
                "LLM servisi beklenen formatta yanıt döndürmedi. Lütfen:\n"
                "1. LLM servisinin doğru çalıştığından emin olun\n"
                "2. Model'in doğru formatta yanıt verdiğinden emin olun\n"
                "3. Daha sonra tekrar deneyin"
            )
            self.logger.error(f"Response validation failed: {e}\nLLM Response: {llm_response[:500]}")
            raise ValueError(error_msg) from e
        
        except Exception as e:
            # Truncate response if too long for error message
            response_preview = llm_response[:1000] if len(llm_response) > 1000 else llm_response
            if len(llm_response) > 1000:
                response_preview += f"\n\n... (toplam {len(llm_response)} karakter, ilk 1000 karakter gösteriliyor)"
            
            error_msg = (
                f"LLM yanıtı işlenirken beklenmeyen hata: {str(e)}\n\n"
                f"LLM Yanıtı (ilk 1000 karakter):\n{response_preview}\n\n"
                "Lütfen LLM servisini kontrol edin ve tekrar deneyin."
            )
            self.logger.error(f"Unexpected parsing error: {e}\nLLM Response: {llm_response[:500]}")
            raise ValueError(error_msg) from e

    def _repair_json(self, json_str: str) -> Optional[str]:
        """Attempt to repair common JSON errors"""
        try:
            # Try to extract JSON first
            json_content = self._extract_json_from_response(json_str)
            if not json_content:
                return None
            
            # Common repairs
            repaired = json_content
            
            # Fix trailing commas before closing braces/brackets
            repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
            
            # Fix missing commas between objects in arrays
            repaired = re.sub(r'}\s*{', r'}, {', repaired)
            
            # Try to parse to see if it's valid now
            json.loads(repaired)
            return repaired
        except:
            return None
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON content from LLM response"""
        
        # Method 1: Look for JSON code blocks (use greedy match to get full JSON)
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_block_pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        
        # Method 2: Find complete JSON object by counting braces
        # Find the first opening brace
        start_pos = response.find('{')
        if start_pos == -1:
            self.logger.warning("No opening brace found in LLM response")
            return None
        
        # Count braces to find the complete JSON object
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(response[start_pos:], start_pos):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[start_pos:i+1]
                        self.logger.debug(f"Extracted JSON: {len(json_str)} characters")
                        return json_str
        
        # If we get here, JSON might be incomplete or malformed
        # Try to extract what we have anyway (might be valid partial JSON)
        if brace_count > 0:
            self.logger.warning(f"JSON appears incomplete: {brace_count} unclosed braces")
            # Try to find the last closing brace
            last_brace = response.rfind('}')
            if last_brace > start_pos:
                json_str = response[start_pos:last_brace+1]
                self.logger.debug(f"Extracted partial JSON: {len(json_str)} characters")
                return json_str
        
        self.logger.warning("Could not extract complete JSON from response")
        return None

    async def _validate_module_structure(
        self,
        module: Dict[str, Any],
        topic_id_map: Dict[int, Dict[str, Any]],
        module_number: int,
        used_topic_ids: set  # Shared across all modules
    ) -> Optional[Dict[str, Any]]:
        """Validate individual module structure"""
        
        # Check required fields
        required_fields = ['module_title']
        for field in required_fields:
            if field not in module or not module[field]:
                self.logger.warning(f"Module {module_number} missing required field: {field}. Module keys: {list(module.keys())}")
                return None
        
        # Log module info for debugging
        self.logger.debug(f"Validating module {module_number}: title='{module.get('module_title')}', topics_count={len(module.get('topics', []))}")
        
        # Validate and fix topics
        valid_topics = []
        module_topics = module.get('topics', [])
        module_used_topic_ids = set()  # Track topics used in THIS module to prevent duplicates
        
        if not module_topics:
            self.logger.warning(f"Module {module_number} has no topics")
            # Don't reject - this might be intentional
        
        # Create a list of all available topics for fallback matching
        all_topics_list = list(topic_id_map.values())
        
        for topic_ref in module_topics:
            if not isinstance(topic_ref, dict):
                self.logger.warning(f"Module {module_number}: topic_ref is not a dict: {type(topic_ref)}, value: {topic_ref}")
                continue
            
            topic_id = topic_ref.get('topic_id')
            matched_topic = None
            
            # Log for debugging
            self.logger.debug(f"Module {module_number}: Processing topic_ref with topic_id={topic_id}, topic_id type={type(topic_id)}")
            
            # Try to match topic_id (handle both int and string)
            if topic_id is not None:
                # Try direct match (int or string)
                if topic_id in topic_id_map:
                    # Check if already used before matching
                    if topic_id not in used_topic_ids and topic_id not in module_used_topic_ids:
                        matched_topic = topic_id_map[topic_id]
                else:
                    # Try converting string to int
                    try:
                        topic_id_int = int(topic_id)
                        if topic_id_int in topic_id_map:
                            # Check if already used before matching
                            if topic_id_int not in used_topic_ids and topic_id_int not in module_used_topic_ids:
                                matched_topic = topic_id_map[topic_id_int]
                    except (ValueError, TypeError):
                        # Not a numeric string, try to find by topic_title if available
                        topic_title = topic_ref.get('topic_title', '')
                        if topic_title:
                            # Convert to string if it's not already
                            topic_title_str = str(topic_title).lower() if topic_title else ''
                            if topic_title_str:
                                for orig_topic_id, orig_topic in topic_id_map.items():
                                    orig_topic_title = orig_topic.get('topic_title', '')
                                    orig_topic_title_str = str(orig_topic_title).lower() if orig_topic_title else ''
                                    # Skip if this topic is already used (to prevent duplicates)
                                    if orig_topic_id in used_topic_ids or orig_topic_id in module_used_topic_ids:
                                        continue
                                    
                                    if orig_topic_title_str == topic_title_str:
                                        matched_topic = orig_topic
                                        self.logger.info(f"Matched topic '{topic_title}' to topic_id {orig_topic_id}")
                                        break
            
            # If still no match, try to match by order (fallback: use topics in order)
            if not matched_topic:
                # Use topic_order_in_module to select from available topics
                topic_order = topic_ref.get('topic_order_in_module', len(valid_topics) + 1)
                # Try to find an unused topic by order (exclude both global and module-level used topics)
                available_topics = [
                    t for t in all_topics_list 
                    if t.get('topic_id') not in used_topic_ids 
                    and t.get('topic_id') not in module_used_topic_ids
                ]
                if available_topics and topic_order <= len(available_topics):
                    # Use topic by order from available topics
                    matched_topic = available_topics[topic_order - 1]
                    self.logger.info(f"Matched topic by order {topic_order} to topic_id {matched_topic.get('topic_id')}")
                elif available_topics:
                    # If order is out of range, use next available topic
                    matched_topic = available_topics[0]
                    self.logger.info(f"Matched topic by fallback to topic_id {matched_topic.get('topic_id')}")
            
            if matched_topic:
                matched_topic_id = matched_topic.get('topic_id')
                
                # Check if topic is already used in THIS module (prevent duplicates within same module)
                if matched_topic_id in module_used_topic_ids:
                    self.logger.warning(f"Topic {matched_topic_id} ({matched_topic.get('topic_title', 'N/A')}) already added to module {module_number}, skipping duplicate")
                    continue
                
                # Check if topic is already used in ANOTHER module (prevent duplicates across modules)
                if matched_topic_id in used_topic_ids:
                    self.logger.warning(f"Topic {matched_topic_id} ({matched_topic.get('topic_title', 'N/A')}) already assigned to another module, skipping duplicate in module {module_number}")
                    continue
                
                # Topic is valid - add it
                used_topic_ids.add(matched_topic_id)
                module_used_topic_ids.add(matched_topic_id)
                
                # Merge with original topic data and add module-specific info
                valid_topic = {
                    **matched_topic,
                    'topic_order_in_module': topic_ref.get('topic_order_in_module', len(valid_topics) + 1),
                    'importance_level': topic_ref.get('importance_level', 'medium'),
                    'relationship_type': topic_ref.get('relationship_type', 'contains'),
                    'content_coverage_percentage': topic_ref.get('content_coverage_percentage', 100.0)
                }
                valid_topics.append(valid_topic)
            else:
                self.logger.warning(f"Topic {topic_id} not found in original topics. Available topic_ids: {list(topic_id_map.keys())[:10]}")
        
        # Set defaults for missing fields
        validated_module = {
            'module_title': module['module_title'],
            'module_description': module.get('module_description', f"Eğitim modülü - {len(valid_topics)} konu"),
            'module_order': module.get('module_order', module_number),
            'estimated_duration_hours': module.get('estimated_duration_hours', max(len(valid_topics) * 2, 10)),
            'difficulty_level': module.get('difficulty_level', 'intermediate'),
            'curriculum_standards': module.get('curriculum_standards', []),
            'learning_outcomes': module.get('learning_outcomes', []),
            'prerequisites': module.get('prerequisites', []),
            'assessment_methods': module.get('assessment_methods', ['quiz']),
            'topics': valid_topics
        }
        
        return validated_module

    def _distribute_unused_topics(
        self,
        modules: List[Dict[str, Any]],
        unused_topics: List[Dict[str, Any]],
        used_topic_ids: set
    ) -> None:
        """Distribute unused topics to most relevant modules based on semantic/keyword matching"""
        if not unused_topics or not modules:
            return
        
        self.logger.info(f"Distributing {len(unused_topics)} unused topics to {len(modules)} modules based on relevance")
        
        # For each unused topic, find the best matching module
        for topic in unused_topics:
            topic_id = topic.get('topic_id')
            if topic_id in used_topic_ids:
                continue
            
            # Find best matching module
            best_module = self._find_best_module_for_topic(topic, modules)
            
            if best_module:
                # Get current max order in module
                current_topics = best_module.get('topics', [])
                max_order = max([t.get('topic_order_in_module', 0) for t in current_topics], default=0)
                
                # Add topic to best matching module
                used_topic_ids.add(topic_id)
                topic_entry = {
                    **topic,
                    'topic_order_in_module': max_order + 1,
                    'importance_level': 'medium',
                    'relationship_type': 'contains',
                    'content_coverage_percentage': 100.0
                }
                current_topics.append(topic_entry)
                best_module['topics'] = current_topics
                
                self.logger.info(f"Added unused topic {topic_id} ({topic.get('topic_title', 'N/A')}) to module '{best_module.get('module_title')}' based on relevance")
            else:
                # Fallback: add to module with fewest topics
                modules_sorted = sorted(modules, key=lambda m: len(m.get('topics', [])))
                if modules_sorted:
                    best_module = modules_sorted[0]
                    current_topics = best_module.get('topics', [])
                    max_order = max([t.get('topic_order_in_module', 0) for t in current_topics], default=0)
                    
                    used_topic_ids.add(topic_id)
                    topic_entry = {
                        **topic,
                        'topic_order_in_module': max_order + 1,
                        'importance_level': 'medium',
                        'relationship_type': 'contains',
                        'content_coverage_percentage': 100.0
                    }
                    current_topics.append(topic_entry)
                    best_module['topics'] = current_topics
                    
                    self.logger.info(f"Added unused topic {topic_id} ({topic.get('topic_title', 'N/A')}) to module '{best_module.get('module_title')}' (fallback: smallest module)")
        
        self.logger.info(f"Distributed {len(unused_topics)} unused topics to modules based on relevance")

    def _find_best_module_for_topic(
        self,
        topic: Dict[str, Any],
        modules: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find the best matching module for a topic based on keyword and title similarity"""
        if not modules:
            return None
        
        topic_title = str(topic.get('topic_title', '')).lower()
        topic_keywords = topic.get('keywords', [])
        if isinstance(topic_keywords, str):
            topic_keywords = [k.strip() for k in topic_keywords.split(',') if k.strip()]
        topic_keywords_lower = [str(k).lower() for k in topic_keywords]
        
        best_module = None
        best_score = 0
        
        for module in modules:
            score = 0
            
            # Check module title similarity
            module_title = str(module.get('module_title', '')).lower()
            module_description = str(module.get('module_description', '')).lower()
            module_text = f"{module_title} {module_description}"
            
            # Check if topic title words appear in module title/description
            topic_words = set(topic_title.split())
            module_words = set(module_text.split())
            common_words = topic_words.intersection(module_words)
            if common_words:
                score += len(common_words) * 2  # Title matches are more important
            
            # Check keyword matches
            for keyword in topic_keywords_lower:
                if keyword in module_text:
                    score += 3  # Keyword matches are very important
                # Also check if keyword appears in module's existing topics
                existing_topics = module.get('topics', [])
                for existing_topic in existing_topics:
                    existing_title = str(existing_topic.get('topic_title', '')).lower()
                    existing_keywords = existing_topic.get('keywords', [])
                    if isinstance(existing_keywords, str):
                        existing_keywords = [k.strip() for k in existing_keywords.split(',') if k.strip()]
                    existing_keywords_lower = [str(k).lower() for k in existing_keywords]
                    
                    if keyword in existing_title or keyword in existing_keywords_lower:
                        score += 2  # Topic is related to existing topics in module
            
            # Prefer modules with fewer topics (to balance distribution)
            topic_count = len(module.get('topics', []))
            if topic_count < 5:
                score += 1  # Small bonus for smaller modules
            
            if score > best_score:
                best_score = score
                best_module = module
        
        return best_module

    async def _attempt_json_repair(
        self,
        response: str,
        original_topics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Attempt to repair malformed JSON"""
        
        self.logger.info("Attempting JSON repair")
        
        try:
            # Find JSON-like content
            start = response.find('{')
            end = response.rfind('}')
            
            if start >= 0 and end > start:
                json_candidate = response[start:end + 1]
                
                # Common repairs
                repairs = [
                    (r',\s*}', '}'),  # Remove trailing commas before }
                    (r',\s*]', ']'),  # Remove trailing commas before ]
                    (r'}\s*{', '},{'), # Add missing commas between objects
                    (r']\s*{', '],{'), # Add missing commas after arrays
                    (r'([0-9])\s*"', r'\1,"'), # Add missing commas after numbers
                ]
                
                for pattern, replacement in repairs:
                    json_candidate = re.sub(pattern, replacement, json_candidate)
                
                # Try to parse repaired JSON
                repaired_data = json.loads(json_candidate)
                
                if 'modules' in repaired_data and isinstance(repaired_data['modules'], list):
                    self.logger.info("JSON repair successful")
                    return await self._parse_llm_response(json_candidate, original_topics)
            
        except Exception as e:
            self.logger.warning(f"JSON repair failed: {e}")
        
        # If repair fails, raise error - no fallback
        # Truncate response if too long for error message
        response_preview = response[:1000] if len(response) > 1000 else response
        if len(response) > 1000:
            response_preview += f"\n\n... (toplam {len(response)} karakter, ilk 1000 karakter gösteriliyor)"
        
        error_msg = (
            f"LLM yanıtı düzeltilemedi ve parse edilemedi: {str(e)}\n\n"
            f"LLM Yanıtı (ilk 1000 karakter):\n{response_preview}\n\n"
            "LLM servisi geçersiz veya bozuk JSON döndürdü. Lütfen:\n"
            "1. LLM servisinin çalıştığından emin olun\n"
            "2. Model'in JSON formatında yanıt verdiğinden emin olun\n"
            "3. Daha sonra tekrar deneyin"
        )
        raise ValueError(error_msg) from e

    async def _refine_module_candidates(
        self,
        module_candidates: List[Dict[str, Any]],
        options: Dict[str, Any],
        course_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply post-processing refinements to module candidates"""
        
        refined_modules = []
        
        for i, module in enumerate(module_candidates):
            # Ensure module order
            if 'module_order' not in module:
                module['module_order'] = i + 1
            
            # Sort topics within module by order or importance
            topics = module.get('topics', [])
            topics.sort(key=lambda t: (
                t.get('topic_order_in_module', 999),
                self._get_importance_sort_key(t.get('importance_level', 'medium'))
            ))
            
            # Assign topic orders if missing
            for j, topic in enumerate(topics):
                if 'topic_order_in_module' not in topic:
                    topic['topic_order_in_module'] = j + 1
            
            module['topics'] = topics
            
            # Calculate estimated duration if not provided or unrealistic
            if ('estimated_duration_hours' not in module or 
                module['estimated_duration_hours'] <= 0 or
                module['estimated_duration_hours'] > 100):
                
                topic_count = len(topics)
                avg_difficulty = self._calculate_average_difficulty(topics)
                base_hours = max(topic_count * 2, 8)  # At least 8 hours per module
                
                difficulty_multipliers = {
                    'beginner': 0.8, 
                    'başlangıç': 0.8,
                    'intermediate': 1.0, 
                    'orta': 1.0,
                    'advanced': 1.3,
                    'ileri': 1.3
                }
                
                multiplier = difficulty_multipliers.get(avg_difficulty, 1.0)
                module['estimated_duration_hours'] = int(base_hours * multiplier)
            
            # Ensure reasonable limits
            module['estimated_duration_hours'] = min(
                max(module['estimated_duration_hours'], 8), 
                80  # Max 80 hours per module
            )
            
            refined_modules.append(module)
        
        self.logger.info(f"Refined {len(refined_modules)} modules")
        return refined_modules

    def _get_importance_sort_key(self, importance_level) -> int:
        """Get sort key for importance level"""
        # Handle both string and integer importance levels
        if isinstance(importance_level, int):
            # Map integer to sort key: 1=high, 2=medium, 3=low
            if importance_level == 1:
                return 0  # high/critical
            elif importance_level == 2:
                return 1  # medium
            else:
                return 2  # low
        elif isinstance(importance_level, str):
            importance_map = {
                'critical': 0,
                'kritik': 0,
                'high': 1,
                'yüksek': 1,
                'medium': 2,
                'orta': 2,
                'low': 3,
                'düşük': 3
            }
            return importance_map.get(importance_level.lower(), 2)
        else:
            # Default to medium
            return 1

    def _calculate_average_difficulty(self, topics: List[Dict[str, Any]]) -> str:
        """Calculate average difficulty level for topics"""
        
        if not topics:
            return 'intermediate'
        
        difficulty_scores = {
            'beginner': 1, 'başlangıç': 1, 'baslangic': 1,
            'intermediate': 2, 'orta': 2,
            'advanced': 3, 'ileri': 3, 'gelişmiş': 3
        }
        
        total_score = 0
        valid_count = 0
        
        for topic in topics:
            difficulty = topic.get('estimated_difficulty', 'intermediate').lower()
            if difficulty in difficulty_scores:
                total_score += difficulty_scores[difficulty]
                valid_count += 1
        
        if valid_count == 0:
            return 'intermediate'
        
        avg_score = total_score / valid_count
        
        if avg_score <= 1.5:
            return 'beginner'
        elif avg_score <= 2.5:
            return 'intermediate'
        else:
            return 'advanced'

    async def _fallback_organization(
        self,
        topics: List[Dict[str, Any]],
        course_info: Dict[str, Any],
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback organization method when LLM fails"""
        
        self.logger.warning("Using fallback organization method")
        
        max_topics_per_module = options.get('max_topics_per_module', 8)
        modules = []
        
        # Simple organization by splitting topics into equal-sized modules
        for i in range(0, len(topics), max_topics_per_module):
            module_topics = topics[i:i + max_topics_per_module]
            
            # Determine module title based on course info
            subject = course_info.get('subject_area', 'genel')
            module_title = f"{subject.title()} Modülü {len(modules) + 1}"
            
            module = {
                'module_title': module_title,
                'module_description': f"Temel {subject} konuları - {len(module_topics)} konu içerir",
                'module_order': len(modules) + 1,
                'estimated_duration_hours': max(len(module_topics) * 3, 12),
                'difficulty_level': self._calculate_average_difficulty(module_topics),
                'curriculum_standards': [],
                'learning_outcomes': [f"{module_title} temel hedefleri"],
                'topics': [
                    {
                        **topic,
                        'topic_order_in_module': j + 1,
                        'importance_level': 'medium',
                        'relationship_type': 'contains',
                        'content_coverage_percentage': 100.0
                    }
                    for j, topic in enumerate(module_topics)
                ],
                'prerequisites': [],
                'assessment_methods': ['quiz', 'assignment']
            }
            
            modules.append(module)
        
        self.logger.info(f"Fallback organization created {len(modules)} modules")
        return modules

    # Utility methods for integration
    def get_supported_strategies(self) -> List[str]:
        """Get list of supported organization strategies"""
        return ['curriculum_aligned', 'semantic_clustering', 'hybrid', 'difficulty_progressive']
    
    def get_recommended_model(self, course_info: Dict[str, Any]) -> str:
        """Get recommended model for a course"""
        return self._select_model_for_organization(course_info, None)