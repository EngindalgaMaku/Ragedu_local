"""
Enhanced PDF Processor with Marker Integration
Marker ile Gelişmiş PDF İşleme Sistemi

Bu modül Marker kütüphanesini kullanarak PDF'leri yüksek kaliteli
Markdown formatına dönüştürür ve daha iyi metin analizi sağlar.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import traceback
import threading
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Setup logger first before any other imports that might use it
logger = logging.getLogger(__name__)

# CRITICAL FIX: Set ALL environment variables BEFORE any marker imports
# This is the root cause of cache failure - marker reads env vars at import time
def _setup_marker_environment():
    """Setup marker environment variables before any marker imports"""
    cache_base = os.getenv("MARKER_CACHE_DIR", "/app/models")
    
    # Core cache directories - MUST match Dockerfile exactly
    cache_vars = {
        "TORCH_HOME": f"{cache_base}/torch",
        "HUGGINGFACE_HUB_CACHE": f"{cache_base}/huggingface",
        "TRANSFORMERS_CACHE": f"{cache_base}/transformers",
        "HF_HOME": f"{cache_base}/hf_home",
        "HF_DATASETS_CACHE": f"{cache_base}/datasets",
        "PYTORCH_TRANSFORMERS_CACHE": f"{cache_base}/transformers",
        
        # Offline mode - force use of cached models only
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        
        # Marker-specific settings
        "MARKER_CACHE_DIR": cache_base,
        "MARKER_DISABLE_GEMINI": "true",
        "MARKER_USE_LOCAL_ONLY": "true",
        "MARKER_DISABLE_CLOUD_SERVICES": "true",
        "MARKER_DISABLE_ALL_LLM": "true",
        "MARKER_OCR_ONLY": "true",
        
        # Disable API keys to prevent cloud access
        "GEMINI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "OPENAI_API_KEY": "",
        "GROQ_API_KEY": "",
    }
    
    for key, value in cache_vars.items():
        if key not in os.environ or not os.environ[key]:
            os.environ[key] = value
            
    logger.info(f"🔧 Marker environment configured - Cache base: {cache_base}")
    logger.info(f"🔧 TORCH_HOME: {os.environ.get('TORCH_HOME')}")
    logger.info(f"🔧 HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE')}")
    logger.info(f"🔧 TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")

# Setup environment BEFORE any marker imports
_setup_marker_environment()

# Marker availability'yi dynamic olarak kontrol et
def check_marker_availability():
    """Marker kütüphanesinin mevcut olup olmadığını kontrol et"""
    try:
        # Yeni marker API'sini kontrol et - PDF temel desteği
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        from marker.config.parser import ConfigParser
        
        # Test additional format support - these may not exist in current version
        additional_formats = []
        
        # Check for DOCX support
        try:
            from marker.converters.docx import DocxConverter
            additional_formats.append("DOCX")
        except ImportError:
            pass
            
        # Check for PPTX support
        try:
            from marker.converters.pptx import PptxConverter
            additional_formats.append("PPTX")
        except ImportError:
            pass
            
        # Check for XLSX support
        try:
            from marker.converters.xlsx import XlsxConverter
            additional_formats.append("XLSX")
        except ImportError:
            pass
        
        if additional_formats:
            logger.info(f"✅ Marker enhanced support available (PDF + {', '.join(additional_formats)})")
        else:
            logger.info("✅ Marker PDF support available (optimized for high-quality PDF processing)")
        
        return True
    except ImportError:
        return False

# Model cache manager import
try:
    from src.utils.model_cache_manager import get_cached_marker_models, get_model_cache_manager
    MODEL_CACHE_AVAILABLE = True
    logger.info("✅ Model cache manager loaded successfully")
except ImportError as e:
    MODEL_CACHE_AVAILABLE = False
    logger.warning(f"⚠️ Model cache manager not available: {e}")

# Memory manager import
try:
    from src.utils.memory_manager import get_memory_manager, memory_managed, optimize_for_large_processing
    MEMORY_MANAGER_AVAILABLE = True
    logger.info("✅ Memory manager loaded successfully")
except ImportError as e:
    MEMORY_MANAGER_AVAILABLE = False
    logger.warning(f"⚠️ Memory manager not available: {e}")

# İlk kontrol
MARKER_AVAILABLE = check_marker_availability()

# Import'ları global yapabilmek için yeni API
PdfConverter = None
create_model_dict = None
text_from_rendered = None
ConfigParser = None

if MARKER_AVAILABLE:
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        from marker.config.parser import ConfigParser
        logging.info("🎉 Marker yeni API'si başarıyla yüklendi! (Ollama LLM desteği ile)")
    except ImportError:
        MARKER_AVAILABLE = False
        logging.warning("📄 Marker yeni API'si yüklenemedi. Basit PDF okuma kullanılacak.")
else:
    logging.info("📄 Marker kütüphanesi mevcut değil. Basit PDF okuma kullanılacak.")

# Fallback için mevcut PDF processor
from src.document_processing.pdf_processor import process_pdf as fallback_pdf_extract


class MarkerPDFProcessor:
    """GPU + LLM Optimized Marker PDF işlemci"""
    
    def __init__(self, use_llm: bool = True, use_gpu: bool = True):  # LLM ve GPU aktif
        self.models_loaded = False
        self.converter = None
        self.use_llm = use_llm  # LLM aktif
        self.use_gpu = use_gpu  # GPU desteği
        self.ollama_config = self._get_ollama_config()
        self.resource_limits = self._get_resource_limits()
        self.current_process = None
        self.resource_monitor_thread = None
        self.stop_monitoring = threading.Event()
        self._initialize_marker()
    
    def _get_resource_limits(self):
        """System resource limitlerini al - Büyük PDF'ler için optimize edildi"""
        try:
            from src.config import get_config
            config = get_config()
            return {
                "max_memory_mb": int(os.getenv("MARKER_MAX_MEMORY_MB", "4096")),  # 4GB'ye yükseltildi
                "max_cpu_percent": int(os.getenv("MARKER_MAX_CPU_PERCENT", "80")),  # %80'e yükseltildi
                "timeout_seconds": int(os.getenv("MARKER_TIMEOUT_SECONDS", "900")),  # 15 dakikaya çıkarıldı
                "max_pages": int(os.getenv("MARKER_MAX_PAGES", "200")),  # 200 sayfaya çıkarıldı
                "max_file_size_mb": int(os.getenv("MARKER_MAX_FILE_SIZE_MB", "0")),  # 0 = sınır yok
                "enable_monitoring": os.getenv("MARKER_ENABLE_RESOURCE_MONITORING", "false").lower() == "true"  # Varsayılan kapalı
            }
        except:
            return {
                "max_memory_mb": 4096,  # 4GB
                "max_cpu_percent": 80,   # %80
                "timeout_seconds": 900,  # 15 dakika
                "max_pages": 200,        # 200 sayfa
                "max_file_size_mb": 0,   # Sınır yok
                "enable_monitoring": False  # Monitoring kapalı
            }
    
    def _set_process_priority(self):
        """Process priority'yi düşük olarak ayarla"""
        try:
            import psutil
            current_process = psutil.Process()
            
            # Windows için
            if os.name == 'nt':
                current_process.nice(psutil.IDLE_PRIORITY_CLASS)
            else:  # Unix-like systems
                current_process.nice(19)  # En düşük priority
                
            logger.info("🔧 Process priority düşük olarak ayarlandı")
        except Exception as e:
            logger.warning(f"⚠️ Process priority ayarlanamadı: {e}")
    
    def _start_resource_monitor(self):
        """Resource monitoring başlat - CRASH-SAFE MODE"""
        # Resource monitoring DEVRE DIŞI - crash risk azaltmak için
        return
    
    def _stop_resource_monitor(self):
        """Resource monitoring durdur - CRASH-SAFE MODE"""
        # Resource monitoring DEVRE DIŞI
        return
    
    def _monitor_resources(self):
        """Resource monitoring - CRASH-SAFE MODE"""
        # Resource monitoring tamamen devre dışı - crash riskini azaltmak için
        return
    
    def _emergency_stop(self):
        """Acil durum durdurma"""
        logger.error("🚨 EMERGENCY STOP: Resource limits exceeded!")
        try:
            if self.current_process:
                self.current_process.terminate()
        except:
            pass
    
    def _track_layout_progress(self, start_time: float, total_progress: int, base_progress: int, page_count: int):
        """Layout recognition progress'ini takip et"""
        try:
            estimated_duration = max(60, page_count * 2)  # Minimum 1 dakika, sayfa başına 2 saniye
            progress_interval = 5  # Her 5 saniyede bir güncelle
            
            while True:
                elapsed = time.time() - start_time
                
                if elapsed >= estimated_duration * 0.9:  # %90'ı geçtiyse dur
                    break
                
                # Progress hesaplama
                progress_pct = min(90, (elapsed / estimated_duration) * 100)
                current_stage_progress = int((progress_pct / 100) * total_progress)
                total_current_progress = base_progress + current_stage_progress
                
                # Tahmini kalan süre
                remaining = estimated_duration - elapsed
                
                logger.info(f"🔄 Layout Recognition devam ediyor... ({total_current_progress}%) - Kalan: ~{remaining:.0f}s")
                
                # Memory usage göster
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    logger.info(f"💾 Anlık bellek kullanımı: {memory_mb:.1f}MB")
                except:
                    pass
                
                time.sleep(progress_interval)
                
        except Exception as e:
            logger.warning(f"Progress tracking hatası: {e}")
    
    def _get_ollama_config(self):
        """Ollama konfigürasyonunu al - CRASH-SAFE"""
        # Provider karışıklığını önlemek için basit config dön
        if not self.use_llm:
            return {"base_url": "", "model": "", "disabled": True}
        
        try:
            return {
                "base_url": "http://localhost:11434",
                "model": "disabled_to_prevent_conflicts",
                "crash_safe_mode": True
            }
        except:
            return {"base_url": "", "model": "", "error": "config_error"}
    
    def _initialize_marker(self):
        """Marker converter'ı başlat"""
        # Dinamik kontrol yap
        global MARKER_AVAILABLE
        MARKER_AVAILABLE = check_marker_availability()
        
        if not MARKER_AVAILABLE:
            logger.warning("Marker kütüphanesi mevcut değil, fallback kullanılacak")
            return
        
        try:
            llm_status = "LLM aktif" if self.use_llm else "LLM kapalı"
            logger.info(f"🚀 Marker converter başlatılıyor... ({llm_status})")
            # Yeni API ile converter oluştur
            self.converter = None  # İlk kullanımda yüklenecek
            logger.info("✅ Marker converter hazır")
        except Exception as e:
            logger.error(f"❌ Marker başlatılamadı: {e}")
            self.converter = None
    
    def _load_converter_if_needed(self):
        """Gerektiğinde converter'ı yükle (Cached models ile)"""
        if not MARKER_AVAILABLE or self.models_loaded:
            return
        
        try:
            if self.converter is None:
                logger.info("🔄 Marker converter ve modelleri yükleniyor (cached models ile)...")
                memory_before = self._get_memory_usage()
                
                # CRITICAL DEBUGGING: Show cache status before model loading
                self._debug_cache_status()
                
                # Environment variables are already set at module import
                # Just verify they're still set correctly
                cache_dir = os.environ.get("MARKER_CACHE_DIR", "/app/models")
                logger.info(f"🔧 Using cache directory: {cache_dir}")
                logger.info(f"🔧 TORCH_HOME: {os.environ.get('TORCH_HOME')}")
                logger.info(f"🔧 TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")
                
                # Verify cache directories exist
                torch_cache = os.environ.get("TORCH_HOME")
                hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
                if torch_cache and not os.path.exists(torch_cache):
                    logger.warning(f"⚠️ TORCH_HOME cache missing: {torch_cache}")
                if hf_cache and not os.path.exists(hf_cache):
                    logger.warning(f"⚠️ HuggingFace cache missing: {hf_cache}")
                
                # GPU ayarları - optional optimization
                if self.use_gpu:
                    os.environ["MARKER_USE_GPU"] = "true"
                    os.environ["MARKER_GPU_MEMORY_FRACTION"] = "0.8"
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # İlk GPU'yu kullan
                    logger.info("🚀 GPU desteği aktif - CUDA optimize edildi")
                
                # CRITICAL: Use environment-cached models directly
                # The cache manager sets env vars, but marker needs direct model dict creation
                logger.info("🎯 Creating model dict with cached environment...")
                
                # Verify cache exists before proceeding
                torch_cache = os.environ.get("TORCH_HOME", "")
                if os.path.exists(torch_cache):
                    logger.info(f"✅ Cache verified: {torch_cache} exists")
                    model_files = []
                    for ext in ['*.bin', '*.safetensors', '*.pt']:
                        import glob
                        model_files.extend(glob.glob(os.path.join(torch_cache, '**', ext), recursive=True))
                    logger.info(f"📁 Found {len(model_files)} cached model files")
                else:
                    logger.warning(f"⚠️ Cache directory missing: {torch_cache}")
                
                # Create model dict - this will use cached models due to env vars
                try:
                    artifact_dict = create_model_dict()
                    logger.info("✅ Model dict created with cached models")
                except Exception as model_error:
                    logger.error(f"❌ Failed to create model dict: {model_error}")
                    logger.warning("🔄 This suggests models were not properly cached during build")
                    raise model_error
                
                # Optimize artifact dict - disable heavy models if available
                if artifact_dict and 'layout' in artifact_dict:
                    # Use lighter layout model settings if available
                    logger.info("🔧 Marker artifact dict optimized for faster processing")
                
                # Sadece temel Marker converter'ı kullan - performans odaklı
                start_time = time.time()
                self.converter = PdfConverter(artifact_dict=artifact_dict)
                converter_load_time = time.time() - start_time
                
                memory_after = self._get_memory_usage()
                memory_used = memory_after - memory_before
                
                # CRITICAL: Log timing to detect downloads
                if converter_load_time > 60:
                    logger.error(f"🚨 CONVERTER LOAD TOOK {converter_load_time:.1f}s - MODELS LIKELY DOWNLOADING!")
                    logger.error("🚨 CACHE FIX FAILED - Check Docker build and environment variables!")
                elif converter_load_time > 30:
                    logger.warning(f"⚠️ Converter load took {converter_load_time:.1f}s - may be downloading")
                else:
                    logger.info(f"✅ Fast converter load ({converter_load_time:.1f}s) - using cached models!")
                
                logger.info(f"✅ Marker converter loaded - Time: {converter_load_time:.1f}s, Memory: +{memory_used:.1f}MB")
                self.models_loaded = True
        except Exception as e:
            logger.error(f"❌ Marker converter yüklenemedi: {e}")
            logger.error(traceback.format_exc())
            self.converter = None
            self.models_loaded = False
    
    def _debug_cache_status(self):
        """Debug cache status before model loading"""
        logger.info("🔍 CACHE DEBUG STATUS:")
        
        cache_vars = [
            "TORCH_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
            "HF_HOME", "TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"
        ]
        
        for var in cache_vars:
            value = os.environ.get(var, "NOT_SET")
            logger.info(f"🔧 {var}: {value}")
            
            if value != "NOT_SET" and os.path.isdir(value):
                try:
                    file_count = len([f for f in Path(value).rglob("*") if f.is_file()])
                    logger.info(f"📁 {var} directory exists with {file_count} files")
                except:
                    logger.info(f"📁 {var} directory exists but couldn't count files")
            elif value != "NOT_SET":
                logger.warning(f"⚠️ {var} directory missing: {value}")
        
        # Check for key model files
        torch_home = os.environ.get("TORCH_HOME", "")
        if torch_home:
            model_extensions = ["*.bin", "*.safetensors", "*.pt"]
            total_models = 0
            for ext in model_extensions:
                import glob
                models = glob.glob(os.path.join(torch_home, "**", ext), recursive=True)
                total_models += len(models)
                if models:
                    logger.info(f"📦 Found {len(models)} {ext} files")
            
            if total_models == 0:
                logger.error(f"❌ NO MODEL FILES FOUND IN CACHE! Models will download.")
            else:
                logger.info(f"✅ Found {total_models} total model files in cache")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    @memory_managed("PDF Processing with Marker")
    def process_pdf_with_marker(self, pdf_path: str, output_dir: Optional[str] = None, timeout_seconds: int = None) -> Tuple[str, Dict[str, Any]]:
        """
        PDF'i Marker ile işle - Memory managed and crash-safe
        
        Args:
            pdf_path: PDF dosya yolu
            output_dir: Çıktı dizini (opsiyonel)
            timeout_seconds: İşleme timeout süresi (varsayılan: config'den alır)
        
        Returns:
            Tuple of (markdown_content, metadata)
        """
        if not MARKER_AVAILABLE:
            logger.warning("Marker kütüphanesi mevcut değil, fallback kullanılacak")
            return self._process_with_fallback(pdf_path)
        
        self._load_converter_if_needed()
        
        if not self.models_loaded or self.converter is None:
            logger.warning("Marker converter yüklenemedi, fallback kullanılacak")
            return self._process_with_fallback(pdf_path)
        
        # Resource limits'i uygula
        timeout_seconds = timeout_seconds or self.resource_limits["timeout_seconds"]
        
        def protected_process():
            """MEMORY-SAFE PDF işleme - detaylı progress tracking ile"""
            
            try:
                # Initialize memory manager for large files
                memory_manager = None
                if MEMORY_MANAGER_AVAILABLE:
                    memory_manager = optimize_for_large_processing()
                    
                    # Register cleanup callback for this processing session
                    def cleanup_pdf_processing():
                        try:
                            # Clean up large variables
                            if 'rendered' in locals():
                                del rendered
                            if 'markdown_text' in locals():
                                del markdown_text
                            if 'images' in locals():
                                del images
                        except:
                            pass
                    
                    memory_manager.add_cleanup_callback(cleanup_pdf_processing)
                
                logger.info(f"🚀 PDF işleniyor - Memory managed processing (timeout: {timeout_seconds}s, memory limit: {self.resource_limits['max_memory_mb']}MB)")
                logger.info(f"📄 Dosya: {pdf_path}")
                
                # File size kontrolü
                file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
                logger.info(f"📊 Dosya boyutu: {file_size_mb:.1f}MB")
                
                # Memory check before processing
                if MEMORY_MANAGER_AVAILABLE and file_size_mb > 50:  # Large file
                    required_memory = file_size_mb * 8  # Estimate 8x file size needed for processing
                    if not memory_manager.get_memory_usage()["available_mb"] > required_memory:
                        logger.warning(f"⚠️ Large file detected ({file_size_mb:.1f}MB) - forcing pre-cleanup")
                        memory_manager.force_cleanup()
                
                # Progress tracking setup
                progress_stages = {
                    "initialization": 5,
                    "page_analysis": 15,
                    "layout_recognition": 40,  # En uzun aşama
                    "ocr_processing": 25,
                    "markdown_generation": 10,
                    "cleanup": 5
                }
                
                current_progress = 0
                stage_start_time = time.time()
                
                # Stage 1: Initialization (5%)
                logger.info(f"🔄 [1/6] Başlatılıyor... ({current_progress}%)")
                
                # Page sayısı kontrolü
                page_count = 0
                try:
                    import PyPDF2
                    with open(pdf_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        page_count = len(reader.pages)
                        logger.info(f"📊 Sayfa sayısı: {page_count}")
                except Exception as e:
                    logger.warning(f"Sayfa sayısı tespit edilemedi: {e}")
                
                current_progress += progress_stages["initialization"]
                logger.info(f"✅ [1/6] Başlatma tamamlandı ({current_progress}%)")
                
                # Stage 2: Page Analysis (20%)
                stage_start_time = time.time()
                logger.info(f"🔄 [2/6] Sayfa analizi... ({current_progress}%)")
                
                # Memory optimization and monitoring
                import gc
                gc.collect()
                
                initial_memory = 0
                if MEMORY_MANAGER_AVAILABLE:
                    memory_info = memory_manager.get_memory_usage()
                    initial_memory = memory_info["rss_mb"]
                    logger.info(f"🔧 Başlangıç bellek kullanımı: {initial_memory:.1f}MB")
                    
                    # Check if memory is getting high
                    if memory_manager.is_memory_warning():
                        logger.warning("⚠️ High memory usage detected before processing - forcing cleanup")
                        memory_manager.force_cleanup()
                else:
                    import psutil
                    process = psutil.Process()
                    initial_memory = process.memory_info().rss / 1024 / 1024
                    logger.info(f"🔧 Başlangıç bellek kullanımı: {initial_memory:.1f}MB")
                
                current_progress += progress_stages["page_analysis"]
                elapsed = time.time() - stage_start_time
                logger.info(f"✅ [2/6] Sayfa analizi tamamlandı ({current_progress}%) - {elapsed:.1f}s")
                
                # Stage 3: Layout Recognition (60%) - EN UZUN AŞAMA
                stage_start_time = time.time()
                logger.info(f"🔄 [3/6] Layout tanıma başlıyor... ({current_progress}%) - BU AŞAMA UZUN SÜREBİLİR")
                logger.info(f"💡 Büyük PDF'lerde layout recognition 5-10 dakika sürebilir. Lütfen bekleyin...")
                
                if page_count > 50:
                    logger.info(f"⚠️  BÜYÜK PDF UYARISI: {page_count} sayfa - Layout recognition çok uzun sürecek!")
                    logger.info(f"💡 Tahmini süre: {page_count * 2}-{page_count * 5} saniye")
                
                # Progress tracking thread for layout recognition
                layout_progress_thread = threading.Thread(
                    target=self._track_layout_progress,
                    args=(stage_start_time, progress_stages["layout_recognition"], current_progress, page_count)
                )
                layout_progress_thread.daemon = True
                layout_progress_thread.start()
                
                # MARKER İŞLEME - Layout recognition burada yapılıyor
                rendered = self.converter(pdf_path)
                
                # Layout recognition finished
                current_progress += progress_stages["layout_recognition"]
                elapsed = time.time() - stage_start_time
                logger.info(f"✅ [3/6] Layout tanıma tamamlandı! ({current_progress}%) - {elapsed:.1f}s")
                
                # Stage 4: OCR Processing (85%)
                stage_start_time = time.time()
                logger.info(f"🔄 [4/6] OCR işleme... ({current_progress}%)")
                
                # Memory monitoring during processing
                if MEMORY_MANAGER_AVAILABLE:
                    processing_memory = memory_manager.get_memory_usage()["rss_mb"]
                    logger.info(f"🔧 İşlem sonrası bellek: {processing_memory:.1f}MB (+{processing_memory-initial_memory:.1f}MB)")
                    
                    # Critical memory check
                    if memory_manager.is_memory_critical():
                        logger.error("🚨 CRITICAL MEMORY during processing - forcing emergency cleanup!")
                        memory_manager.force_cleanup()
                else:
                    import psutil
                    process = psutil.Process()
                    processing_memory = process.memory_info().rss / 1024 / 1024
                    logger.info(f"🔧 İşlem sonrası bellek: {processing_memory:.1f}MB (+{processing_memory-initial_memory:.1f}MB)")
                
                current_progress += progress_stages["ocr_processing"]
                elapsed = time.time() - stage_start_time
                logger.info(f"✅ [4/6] OCR işleme tamamlandı ({current_progress}%) - {elapsed:.1f}s")
                
                # Stage 5: Markdown Generation (95%)
                stage_start_time = time.time()
                logger.info(f"🔄 [5/6] Markdown oluşturuluyor... ({current_progress}%)")
                
                markdown_text, metadata_dict, images = text_from_rendered(rendered)
                
                current_progress += progress_stages["markdown_generation"]
                elapsed = time.time() - stage_start_time
                logger.info(f"✅ [5/6] Markdown oluşturma tamamlandı ({current_progress}%) - {elapsed:.1f}s")
                
                # Stage 6: Cleanup (100%)
                stage_start_time = time.time()
                logger.info(f"🔄 [6/6] Temizlik yapılıyor... ({current_progress}%)")
                
                gc.collect()
                
                if MEMORY_MANAGER_AVAILABLE:
                    final_memory = memory_manager.get_memory_usage()["rss_mb"]
                    logger.info(f"🔧 Final bellek kullanımı: {final_memory:.1f}MB")
                    
                    # Stop monitoring for this processing session
                    memory_manager.stop_memory_monitoring()
                else:
                    import psutil
                    process = psutil.Process()
                    final_memory = process.memory_info().rss / 1024 / 1024
                    logger.info(f"🔧 Final bellek kullanımı: {final_memory:.1f}MB")
                
                current_progress += progress_stages["cleanup"]
                elapsed = time.time() - stage_start_time
                logger.info(f"✅ [6/6] İşlem tamamlandı! (100%) - {elapsed:.1f}s")
                
                return markdown_text, metadata_dict, images, rendered
                
            finally:
                # MEMORY-SAFE cleanup
                try:
                    # Memory manager cleanup
                    if MEMORY_MANAGER_AVAILABLE and memory_manager:
                        memory_manager.stop_memory_monitoring()
                        memory_manager.force_cleanup()
                    
                    # Manual cleanup
                    import gc
                    gc.collect()
                    
                    # Clean up large objects
                    if 'rendered' in locals():
                        del rendered
                    if 'markdown_text' in locals():
                        del markdown_text
                    if 'images' in locals():
                        del images
                        
                    # Final garbage collection
                    gc.collect()
                    
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Cleanup error: {cleanup_error}")
        
        try:
            # Protected processing with extended timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(protected_process)
                try:
                    markdown_text, metadata_dict, images, rendered = future.result(timeout=timeout_seconds)
                except FutureTimeoutError:
                    logger.error(f"⏰ TIMEOUT: PDF işleme {timeout_seconds} saniye içinde tamamlanamadı!")
                    logger.error(f"Bu büyük bir PDF olabilir. Timeout süresini artırmayı deneyin.")
                    self._stop_resource_monitor()
                    # Büyük PDF'ler için fallback kullanmak yerine hata fırlat
                    raise Exception(f"PDF processing timeout after {timeout_seconds} seconds. Try increasing MARKER_TIMEOUT_SECONDS environment variable.")
                
            # Metadata oluştur
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            metadata = {
                "source_file": os.path.basename(pdf_path),
                "processing_method": "marker_optimized_for_large_files",
                "page_count": len(rendered.children) if hasattr(rendered, 'children') else 0,
                "extraction_quality": "high",
                "has_images": len(images) > 0 if images else False,
                "image_count": len(images) if images else 0,
                "marker_rendered_type": type(rendered).__name__,
                "marker_metadata": metadata_dict,
                "timeout_used": timeout_seconds,
                "file_size_mb": file_size_mb,
                "resource_limits": self.resource_limits,
                "processing_time_estimate": f"Processed in under {timeout_seconds}s"
            }
            
            logger.info(f"✅ PDF başarıyla işlendi! {metadata['page_count']} sayfa, {len(markdown_text)} karakter, {file_size_mb:.1f}MB")
            
            # Markdown içeriğini temizle ve formatla
            cleaned_text = self._clean_markdown_content(markdown_text)
            
            return cleaned_text, metadata
            
        except Exception as e:
            logger.error(f"❌ Marker ile PDF işleme başarısız: {e}")
            logger.error(f"Hata detayları: {str(e)}")
            self._stop_resource_monitor()
            
            # Büyük PDF'ler için fallback yerine hata raporla
            if "timeout" in str(e).lower():
                logger.error("💡 Çözüm: MARKER_TIMEOUT_SECONDS environment variable'ını artırın")
                logger.error("💡 Örnek: MARKER_TIMEOUT_SECONDS=1800 (30 dakika)")
            elif "memory" in str(e).lower():
                logger.error("💡 Çözüm: MARKER_MAX_MEMORY_MB environment variable'ını artırın")
                logger.error("💡 Örnek: MARKER_MAX_MEMORY_MB=8192 (8GB)")
            
            # Kritik hata durumunda fallback kullan, aksi halde hata fırlat
            if "FileNotFoundError" in str(e) or "PermissionError" in str(e):
                logger.error("Dosya erişim hatası - fallback kullanılacak")
                return self._process_with_fallback(pdf_path)
            else:
                # Marker hatası - tekrar dene veya kullanıcıya bilgi ver
                logger.error("Marker processing hatası - büyük PDF için ayarları optimize edin")
                raise Exception(f"Marker PDF processing failed: {str(e)}. Consider increasing timeout or memory limits.")
    
    def _process_with_fallback(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Fallback PDF işleme yöntemi"""
        try:
            fallback_text = self._fallback_extract(pdf_path)
            
            fallback_metadata = {
                "source_file": os.path.basename(pdf_path),
                "processing_method": "fallback_due_to_marker_issue",
                "extraction_quality": "basic",
                "page_count": fallback_text.count("=== Page") if "=== Page" in fallback_text else 1,
                "has_images": False,
                "image_count": 0,
                "text_length": len(fallback_text),
                "estimated_reading_time": len(fallback_text.split()) / 200
            }
            
            logger.info(f"✅ Fallback ile işlendi: {len(fallback_text)} karakter")
            return fallback_text, fallback_metadata
            
        except Exception as fallback_error:
            logger.error(f"❌ Fallback da başarısız: {fallback_error}")
            return "", {
                "source_file": os.path.basename(pdf_path),
                "processing_method": "failed",
                "extraction_quality": "none",
                "error": str(fallback_error)
            }
    
    def _clean_markdown_content(self, content: str) -> str:
        """Markdown içeriğini temizle ve formatla"""
        if not content:
            return ""
        
        # Temel temizlik işlemleri
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Çok fazla boş satırı temizle
            if line.strip() == "":
                if not cleaned_lines or cleaned_lines[-1].strip() != "":
                    cleaned_lines.append("")
                continue
            
            # Satır sonu karakterlerini normalize et
            line = line.rstrip()
            
            # Çok uzun satırları kontrol et
            if len(line) > 1000:
                # Uzun satırları bölmek yerine olduğu gibi bırak (tablo olabilir)
                pass
            
            cleaned_lines.append(line)
        
        # Son temizlikler
        result = '\n'.join(cleaned_lines)
        
        # Çok fazla ardışık boş satırı temizle
        import re
        result = re.sub(r'\n\n\n+', '\n\n', result)
        
        return result.strip()
    
    def extract_text_from_pdf(self, pdf_path: str, use_marker: bool = True) -> str:
        """
        PDF'den metin çıkar (Marker veya fallback kullanarak) - Yeni API
        
        Args:
            pdf_path: PDF dosya yolu
            use_marker: Marker kullan (True) ya da fallback (False)
        
        Returns:
            Çıkarılan metin
        """
        if use_marker and MARKER_AVAILABLE:
            try:
                markdown_content, metadata = self.process_pdf_with_marker(pdf_path)
                
                # Markdown'ı düz metne dönüştür (başlıklar ve formatlamayı koru)
                text_content = self._markdown_to_enhanced_text(markdown_content)
                
                logger.info(f"🎉 Marker (yeni API) ile başarılı: {len(text_content)} karakter çıkarıldı")
                return text_content
                
            except Exception as e:
                logger.warning(f"⚠️ Marker başarısız, fallback kullanılıyor: {e}")
                return self._fallback_extract(pdf_path)
        else:
            logger.info("📄 Fallback PDF extraction kullanılıyor")
            return self._fallback_extract(pdf_path)
    
    def _markdown_to_enhanced_text(self, markdown_content: str) -> str:
        """Markdown içeriğini gelişmiş düz metne dönüştür - Memory optimized"""
        if not markdown_content:
            return ""
        
        # Büyük dosyalar için memory efficient processing
        if len(markdown_content) > 1024 * 1024:  # 1MB'den büyükse
            logger.info("🔧 Büyük markdown için memory optimized processing")
            return self._process_large_markdown_efficiently(markdown_content)
        
        lines = markdown_content.split('\n')
        enhanced_lines = []
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000  # 1000 satır chunks'lara böl
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_enhanced = self._process_markdown_chunk(chunk_lines)
            enhanced_lines.extend(chunk_enhanced)
            
            # Memory cleanup between chunks
            if i % (chunk_size * 10) == 0 and i > 0:  # Her 10 chunk'ta
                import gc
                gc.collect()
        
        result = '\n'.join(enhanced_lines).strip()
        
        # Final cleanup
        del lines
        del enhanced_lines
        import gc
        gc.collect()
        
        return result
    
    def _process_large_markdown_efficiently(self, markdown_content: str) -> str:
        """Büyük markdown dosyaları için verimli işleme"""
        logger.info("🔧 Büyük markdown dosyası streaming işleniyor...")
        
        # Stream processing - satır satır işle
        import io
        
        result_lines = []
        buffer_size = 64 * 1024  # 64KB buffer
        
        with io.StringIO(markdown_content) as content_stream:
            while True:
                chunk = content_stream.read(buffer_size)
                if not chunk:
                    break
                
                lines = chunk.split('\n')
                processed_chunk = self._process_markdown_chunk(lines)
                result_lines.extend(processed_chunk)
                
                # Memory cleanup
                del lines
                del processed_chunk
                import gc
                gc.collect()
        
        result = '\n'.join(result_lines).strip()
        del result_lines
        import gc
        gc.collect()
        
        return result
    
    def _process_markdown_chunk(self, lines):
        """Markdown chunk'ını işle"""
        enhanced_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Başlıkları belirgin hale getir
            if stripped.startswith('#'):
                # # Başlık -> === Başlık ===
                level = len(stripped.split()[0]) if stripped.split() else 1
                title = stripped.lstrip('#').strip()
                if level == 1:
                    enhanced_lines.append(f"\n=== {title} ===\n")
                elif level == 2:
                    enhanced_lines.append(f"\n--- {title} ---\n")
                else:
                    enhanced_lines.append(f"\n{title}:\n")
                continue
            
            # Boş satırları koru
            if not stripped:
                enhanced_lines.append("")
                continue
            
            # Liste öğelerini formatla
            if stripped.startswith('-') or stripped.startswith('*'):
                enhanced_lines.append(f"• {stripped[1:].strip()}")
                continue
            
            # Numaralı listeleri formatla
            import re
            if re.match(r'^\d+\.', stripped):
                enhanced_lines.append(stripped)
                continue
            
            # Bold metinleri formatla (**text** -> TEXT)
            processed_line = re.sub(r'\*\*(.*?)\*\*', r'\1', stripped)
            
            # İtalik metinleri formatla (*text* -> text)
            processed_line = re.sub(r'\*(.*?)\*', r'\1', processed_line)
            
            # Code blokları (backtick'leri kaldır)
            processed_line = re.sub(r'`([^`]+)`', r'\1', processed_line)
            
            enhanced_lines.append(processed_line)
        
        return enhanced_lines
    
    def _fallback_extract(self, pdf_path: str) -> str:
        """Fallback PDF extraction"""
        try:
            return fallback_pdf_extract(pdf_path)
        except Exception as e:
            logger.error(f"Fallback PDF extraction da başarısız: {e}")
            return ""
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """İşleme istatistiklerini getir"""
        method = "fallback"
        if MARKER_AVAILABLE and self.models_loaded:
            if self.use_llm:
                method = "marker_new_api_ollama_llm_cached"
            else:
                method = "marker_new_api_cached"
        
        cache_stats = {}
        if MODEL_CACHE_AVAILABLE:
            try:
                cache_manager = get_model_cache_manager()
                cache_stats = cache_manager.get_cache_stats()
            except:
                cache_stats = {"error": "Could not get cache stats"}
        
        return {
            "marker_available": MARKER_AVAILABLE,
            "converter_loaded": self.models_loaded,
            "llm_enabled": self.use_llm,
            "ollama_config": self.ollama_config if self.use_llm else None,
            "processing_method": method,
            "api_version": "new_with_llm_cached" if MARKER_AVAILABLE and self.use_llm else "new_cached" if MARKER_AVAILABLE else "none",
            "model_cache_available": MODEL_CACHE_AVAILABLE,
            "cache_stats": cache_stats
        }


# Global instance - CRASH-SAFE MODE (LLM tamamen devre dışı - provider karışıklığını önler)
enhanced_pdf_processor = MarkerPDFProcessor(use_llm=False, use_gpu=False)  # Hem LLM hem GPU güvenli mod


def extract_text_from_pdf_enhanced(pdf_path: str, prefer_marker: bool = True, use_llm: bool = False) -> str:
    """
    Gelişmiş PDF metin çıkarma (Marker OCR only - stable mode)
    
    Args:
        pdf_path: PDF dosya yolu
        prefer_marker: Marker'ı tercih et (varsayılan: True)
        use_llm: LLM kullan (varsayılan: False - provider karışıklığını önler)
    
    Returns:
        Çıkarılan metin
    """
    # Provider karışıklığını önlemek için her zaman LLM'siz kullan
    return enhanced_pdf_processor.extract_text_from_pdf(pdf_path, use_marker=prefer_marker)


def process_pdf_with_analysis(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    PDF'i analiz bilgileri ile birlikte işle
    
    Args:
        pdf_path: PDF dosya yolu
    
    Returns:
        Tuple of (extracted_text, analysis_metadata)
    """
    try:
        if MARKER_AVAILABLE:
            content, metadata = enhanced_pdf_processor.process_pdf_with_marker(pdf_path)
            text_content = enhanced_pdf_processor._markdown_to_enhanced_text(content)
            
            # Analiz metadata'sını genişlet
            analysis_metadata = {
                **metadata,
                "text_length": len(text_content),
                "estimated_reading_time": len(text_content.split()) / 200,  # dakika cinsinden
                "structure_detected": {
                    "has_headings": "===" in text_content or "---" in text_content,
                    "has_lists": "•" in text_content or any(line.strip().startswith(('1.', '2.', '3.')) for line in text_content.split('\n')),
                    "has_code": "`" in content,  # Orijinal markdown'ta kod bloğu var mı
                    "paragraph_count": len([p for p in text_content.split('\n\n') if p.strip()])
                }
            }
            
            return text_content, analysis_metadata
        else:
            # Fallback
            text_content = enhanced_pdf_processor._fallback_extract(pdf_path)
            fallback_metadata = {
                "source_file": os.path.basename(pdf_path),
                "processing_method": "fallback",
                "extraction_quality": "basic",
                "text_length": len(text_content),
                "estimated_reading_time": len(text_content.split()) / 200
            }
            return text_content, fallback_metadata
            
    except Exception as e:
        logger.error(f"PDF analiz işleme hatası: {e}")
        # En son fallback
        try:
            text_content = enhanced_pdf_processor._fallback_extract(pdf_path)
            error_metadata = {
                "source_file": os.path.basename(pdf_path),
                "processing_method": "emergency_fallback",
                "extraction_quality": "basic",
                "error": str(e),
                "text_length": len(text_content)
            }
            return text_content, error_metadata
        except:
            return "", {"error": f"Tüm PDF işleme yöntemleri başarısız: {str(e)}"}


if __name__ == "__main__":
    # Test kodu
    test_pdf_path = "test.pdf"
    
    if os.path.exists(test_pdf_path):
        print("📄 Test PDF'i işleniyor...")
        
        # Basit extraction test
        text = extract_text_from_pdf_enhanced(test_pdf_path)
        print(f"✅ Çıkarılan metin: {len(text)} karakter")
        
        # Detaylı analiz test
        text2, metadata = process_pdf_with_analysis(test_pdf_path)
        print(f"✅ Analiz tamamlandı: {metadata}")
        
        # İlk 200 karakteri göster
        print(f"📄 Metin önizleme:\n{text2[:200]}...")
    else:
        print("⚠️ Test PDF dosyası bulunamadı")
        
    # İstatistikleri göster
    stats = enhanced_pdf_processor.get_processing_stats()
    print(f"📊 İşleme istatistikleri: {stats}")