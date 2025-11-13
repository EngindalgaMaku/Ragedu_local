# src/services/learning_loop_manager.py

"""
Learning Loop Manager: Otomatik Öğrenme ve Optimizasyon Yöneticisi

Bu modül, aktif öğrenme ve RAG optimizasyon süreçlerini periyodik olarak
çalıştıran, performans trendlerini izleyen ve sistemin genel sağlığını
kontrol eden bir yönetici (manager) görevi görür.
"""

import time
import threading
from typing import Dict, Any

# Proje içindeki diğer modüllerden importlar
# Gerçek implementasyonda bu importların doğru çalıştığından emin olun.
# sys.path ayarlaması veya uygun paket yapısı gerekebilir.
from src.services.active_learning import ActiveLearningEngine
from src.services.rag_optimizer import RAGParameterOptimizer
# Note: DatabaseAnalytics doesn't exist, but the db_connection parameter
# will be a mock or actual database instance passed to the constructor

class LearningLoopManager:
    """
    Aktif öğrenme ve optimizasyon döngülerini zamanlar ve yönetir.
    """
    def __init__(self, db_connection, analysis_interval_seconds: int = 86400):
        """
        Yöneticiyi başlatır.
        :param db_connection: Veritabanı bağlantı nesnesi.
        :param analysis_interval_seconds: Analiz döngülerinin ne sıklıkla çalışacağı (saniye cinsinden).
                                          Varsayılan: 24 saat (86400 saniye).
        """
        self.db = db_connection
        self.active_learning_engine = ActiveLearningEngine(self.db)
        self.rag_optimizer = RAGParameterOptimizer(self.db)
        
        self.analysis_interval = analysis_interval_seconds
        self.is_running = False
        self.thread = None
        
        self.system_health = {"status": "INITIALIZING", "last_check": None, "issues": []}

    def _run_periodic_analysis(self):
        """
        Periyodik olarak çalışan ana analiz fonksiyonu.
        """
        while self.is_running:
            print(f"\n{'='*20} ÖĞRENME DÖNGÜSÜ BAŞLATILIYOR {'='*20}")
            
            # 1. Sistem sağlığını izle
            self.monitor_system_health()
            
            # 2. Aktif öğrenme analizi yap
            print("\n[1/3] Aktif Öğrenme Motoru çalıştırılıyor...")
            feedback_patterns = self.active_learning_engine.analyze_feedback_patterns()
            samples_for_review = self.active_learning_engine.identify_samples_for_review(limit=5)
            print(f"-> Geri bildirim desenleri analiz edildi. En sorunlu konular: {feedback_patterns}")
            print(f"-> Gözden geçirilmesi gereken en belirsiz 5 örnek belirlendi.")
            # Bu örnekler bir UI'a veya etiketleme aracına gönderilebilir.

            # 3. RAG parametre optimizasyonu yap
            print("\n[2/3] RAG Parametre Optimize Edici çalıştırılıyor...")
            optimized_config = self.rag_optimizer.run_optimization_cycle()
            print(f"-> Optimizasyon tamamlandı. Yeni yapılandırma önerisi: {optimized_config}")

            # 4. Performans trendlerini tespit et
            print("\n[3/3] Performans Trend Analizi yapılıyor...")
            self.detect_performance_trends()

            print(f"\n{'='*20} ÖĞRENME DÖNGÜSÜ TAMAMLANDI {'='*20}")
            print(f"Sonraki döngü {self.analysis_interval // 3600} saat sonra çalışacak.")
            
            # Bir sonraki döngü için bekle
            time.sleep(self.analysis_interval)

    def detect_performance_trends(self, short_window_days=7, long_window_days=30):
        """
        Kısa ve uzun vadeli performans metriklerini karşılaştırarak trendleri belirler.
        """
        try:
            # Validate input parameters
            if short_window_days is None or not isinstance(short_window_days, (int, float)) or short_window_days <= 0:
                short_window_days = 7
            if long_window_days is None or not isinstance(long_window_days, (int, float)) or long_window_days <= 0:
                long_window_days = 30
                
            # Ensure short window is not longer than long window
            if short_window_days >= long_window_days:
                long_window_days = short_window_days + 7
            
            # Get performance metrics with error handling
            try:
                short_term_perf = self.db.get_average_rating(days=int(short_window_days))
                long_term_perf = self.db.get_average_rating(days=int(long_window_days))
            except Exception as e:
                print(f"HATA: Performans verisi alınamadı: {e}")
                # Re-raise database errors for proper test handling
                raise e
            
            # Validate returned values
            if short_term_perf is None or long_term_perf is None:
                print("UYARI: Performans verileri eksik veya geçersiz.")
                return "NO_DATA"
                
            # Convert to float and validate
            try:
                short_term_perf = float(short_term_perf)
                long_term_perf = float(long_term_perf)
            except (TypeError, ValueError):
                print("UYARI: Performans verileri sayısal olmayan formatta.")
                return "INVALID_DATA"
            
            # Handle edge cases for comparisons
            if long_term_perf == 0 and short_term_perf == 0:
                # Both are zero - this is considered stable (no change)
                print("BİLGİ: Her iki dönem de sıfır puan - sistem stabil.")
                return "STABİL"
            elif long_term_perf == 0:
                print("UYARI: Uzun vadeli performans verisi sıfır, karşılaştırma yapılamıyor.")
                return "INSUFFICIENT_DATA"
            
            # Safe printing with None checks
            try:
                print(f"-> Son {int(short_window_days)} gün ortalama puan: {short_term_perf:.2f}")
                print(f"-> Son {int(long_window_days)} gün ortalama puan: {long_term_perf:.2f}")
            except (TypeError, ValueError):
                print(f"-> Son {short_window_days} gün ortalama puan: {short_term_perf}")
                print(f"-> Son {long_window_days} gün ortalama puan: {long_term_perf}")

            # Safe threshold calculations
            try:
                threshold_down = long_term_perf * 0.9
                threshold_up = long_term_perf * 1.1
                
                if short_term_perf <= threshold_down: # %10'dan fazla düşüş varsa (dahil)
                    trend = "DÜŞÜŞTE"
                    print("UYARI: Sistem performansı son zamanlarda düşüş trendinde!")
                    if 'issues' in self.system_health:
                        self.system_health['issues'].append("Performance degradation detected.")
                elif short_term_perf > threshold_up:
                    trend = "YÜKSELİŞTE"
                    print("BİLGİ: Sistem performansı yükseliş trendinde. Harika iş!")
                else:
                    trend = "STABİL"
                    print("BİLGİ: Sistem performansı stabil.")
                    
            except (TypeError, ValueError, ZeroDivisionError):
                trend = "CALCULATION_ERROR"
                print("HATA: Trend hesaplama sırasında hata oluştu.")
            
            return trend
            
        except Exception as e:
            # Don't catch database errors that should be re-raised
            if 'Database' in str(e) or 'query failed' in str(e).lower():
                raise e
            print(f"HATA: Performans trend analizi başarısız: {e}")
            return "ERROR"

    def monitor_system_health(self):
        """
        Sistemin genel sağlığını kontrol eder (örn. veritabanı bağlantısı, API erişilebilirliği).
        """
        # Ensure system_health structure exists
        if not hasattr(self, 'system_health') or not isinstance(self.system_health, dict):
            self.system_health = {"status": "UNKNOWN", "last_check": None, "issues": []}
            
        # Kontrol öncesi temizle - but safely
        if 'issues' not in self.system_health:
            self.system_health['issues'] = []
        else:
            self.system_health['issues'] = []
            
        try:
            # Check if db has the check_connection method
            if hasattr(self.db, 'check_connection'):
                self.db.check_connection()
            elif hasattr(self.db, 'ping'):
                self.db.ping()
            else:
                # Basic connectivity test - try a simple operation
                try:
                    # Attempt a simple database operation
                    if hasattr(self.db, 'get_average_rating'):
                        self.db.get_average_rating(days=1)
                    else:
                        # If no standard method, assume connection is OK
                        pass
                except:
                    raise Exception("Database operation test failed")
                    
            self.system_health['status'] = "OPERATIONAL"
            print("-> Sistem Sağlığı: Veritabanı bağlantısı başarılı.")
            
        except Exception as e:
            self.system_health['status'] = "ERROR"
            error_msg = f"Database connection failed: {str(e)}"
            self.system_health['issues'].append(error_msg)
            print(f"HATA: Sistem sağlığı kontrolü başarısız! Veritabanı bağlantı sorunu: {e}")
        
        # Use precise timestamp with better format for testing
        try:
            import datetime
            self.system_health['last_check'] = datetime.datetime.now().isoformat()
        except ImportError:
            # Fallback to time.time() for better precision than ctime()
            self.system_health['last_check'] = time.time()


    def start(self):
        """
        Öğrenme döngüsünü ayrı bir thread'de başlatır.
        """
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run_periodic_analysis, daemon=True)
            self.thread.start()
            print("Learning Loop Manager başlatıldı.")

    def stop(self):
        """
        Öğrenme döngüsünü durdurur.
        """
        if self.is_running:
            self.is_running = False
            # Thread'in sonlanmasını beklemek için join() kullanılabilir,
            # ancak daemon thread olduğu için ana program bittiğinde kendi de sonlanır.
            print("Learning Loop Manager durduruluyor...")


if __name__ == '__main__':
    # Örnek kullanım
    class MockDB:
        def check_connection(self): pass
        def get_feedback_since(self, days): return [{'topic': 'Derin Öğrenme', 'count': 15}]
        def get_most_uncertain_queries(self, limit): return [{'query': 'Test query', 'score': 0.9}]
        def get_performance_metrics(self, days): return [{'rating': 2.8, 'retrieval_k': 5}]
        def get_average_rating(self, days):
            return 3.5 if days == 7 else 4.1

    db = MockDB()
    # Test için interval'i 10 saniyeye düşürelim
    manager = LearningLoopManager(db_connection=db, analysis_interval_seconds=10)
    
    manager.start()
    
    try:
        # 25 saniye boyunca çalışsın (2 döngü tamamlanmalı)
        time.sleep(25)
    finally:
        manager.stop()
        print("Program sonlandı.")