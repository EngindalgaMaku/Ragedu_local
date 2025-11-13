# src/services/rag_optimizer.py

"""
RAG Parameter Optimizer: Performansa Dayalı Optimizasyon Motoru

Bu modül, sistemin genel performansını analiz ederek RAG ardışık düzeninin
(pipeline) parametrelerini (örn. chunk boyutu, geri getirme sayısı, yönlendirme kuralları)
dinamik olarak optimize eder.
"""

from typing import Dict, Any, List
import json

class RAGParameterOptimizer:
    """
    Performans verilerini analiz eder ve RAG yapılandırmasını iyileştirmek için
    önerilerde bulunur veya otomatik güncellemeler yapar.
    """
    def __init__(self, db_connection, config_path='src/config.py'):
        self.db = db_connection
        self.config_path = config_path
        self.current_config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Mevcut yapılandırma dosyasını yükler.
        Not: Bu basit bir örnek. Gerçekte, config.py'yi doğrudan parse etmek yerine
        daha yönetilebilir bir format (JSON, YAML) kullanmak daha iyidir.
        """
        # Örnek olarak, config'in bir JSON dosyasında olduğunu varsayalım.
        # config.py'yi dinamik olarak yönetmek karmaşıktır.
        # Bu yüzden, yönetilecek parametreler için ayrı bir JSON dosyası kullanalım.
        self.json_config_path = 'rag_config.json'
        try:
            with open(self.json_config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Varsayılan yapılandırma
            default_config = {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "retrieval_k": 5,
                "query_routing_rules": {
                    "keyword_based": ["tanım", "nedir"],
                    "semantic_based": ["nasıl", "neden", "karşılaştır"]
                }
            }
            self.update_config(default_config)
            return default_config


    def update_config(self, new_config: Dict[str, Any]):
        """
        Yapılandırma dosyasını yeni parametrelerle günceller.
        """
        with open(self.json_config_path, 'w') as f:
            json.dump(new_config, f, indent=4)
        self.current_config = new_config
        print(f"Yapılandırma güncellendi: {self.json_config_path}")

    def analyze_performance_data(self, time_window_days: int = 7) -> Dict[str, Any]:
        """
        Belirli bir zaman aralığındaki performans metriklerini analiz eder.
        (örn. geri getirme isabet oranı, yanıt kalitesi puanları)
        """
        # Veritabanından performans verilerini çek
        performance_data = self.db.get_performance_metrics(time_window_days)
        
        # Örnek analiz: Düşük puanlı sorguların chunk boyutlarını incele
        analysis_summary = {
            "average_rating": 0,
            "low_rated_retrieval_k": [],
            "high_rated_retrieval_k": []
        }
        if not performance_data:
            return analysis_summary

        total_rating = 0
        for record in performance_data:
            total_rating += record['rating']
            if record['rating'] < 3:
                analysis_summary['low_rated_retrieval_k'].append(record['retrieval_k'])
            else:
                analysis_summary['high_rated_retrieval_k'].append(record['retrieval_k'])
        
        analysis_summary['average_rating'] = total_rating / len(performance_data)
        return analysis_summary

    def tune_retrieval_parameters(self, performance_analysis: Dict[str, Any]) -> bool:
        """
        Geri getirme (retrieval) parametrelerini (örn. k sayısı) performansa göre ayarlar.
        """
        avg_rating = performance_analysis.get('average_rating', 0)
        
        if avg_rating < 3.5 and self.current_config['retrieval_k'] < 10:
            new_k = self.current_config['retrieval_k'] + 1
            print(f"Ortalama puan ({avg_rating:.2f}) düşük. Retrieval 'k' değeri {new_k}'ye yükseltiliyor.")
            self.current_config['retrieval_k'] = new_k
            return True
        elif avg_rating > 4.5 and self.current_config['retrieval_k'] > 3:
            new_k = self.current_config['retrieval_k'] - 1
            print(f"Ortalama puan ({avg_rating:.2f}) yüksek. Kaynakları optimize etmek için 'k' değeri {new_k}'ye düşürülüyor.")
            self.current_config['retrieval_k'] = new_k
            return True
        
        return False

    def optimize_routing_rules(self, feedback_data: List[Dict[str, Any]]):
        """
        Geri bildirimlere dayanarak sorgu yönlendirme kurallarını iyileştirir.
        Örn: Belirli anahtar kelimeler sürekli yanlış zincire yönlendiriliyorsa.
        """
        # Bu fonksiyon, hangi sorgu türlerinin hangi zincirlerde daha iyi/kötü
        # performans gösterdiğini analiz eder ve kuralları günceller.
        # Şimdilik basit bir sayaç mekanizması kuralım.
        print("Sorgu yönlendirme kuralları optimizasyonu henüz implemente edilmedi.")
        pass

    def run_optimization_cycle(self):
        """
        Tam bir optimizasyon döngüsünü çalıştırır.
        """
        print("RAG Optimizasyon Döngüsü Başlatılıyor...")
        
        # 1. Performans verilerini analiz et
        performance_analysis = self.analyze_performance_data(time_window_days=7)
        print(f"Performans Analizi Tamamlandı. Ortalama Puan: {performance_analysis.get('average_rating', 'N/A'):.2f}")

        # 2. Parametreleri ayarla
        config_changed = self.tune_retrieval_parameters(performance_analysis)
        
        # 3. Geri bildirim verilerini al (routing için)
        feedback = self.db.get_feedback_since(days=7)
        self.optimize_routing_rules(feedback)

        # 4. Değişiklik varsa yapılandırmayı güncelle
        if config_changed:
            self.update_config(self.current_config)
        
        print("Optimizasyon Döngüsü Tamamlandı.")
        return self.current_config


if __name__ == '__main__':
    # Örnek kullanım
    class MockDB:
        def get_performance_metrics(self, days):
            # Düşük performans simülasyonu
            return [
                {'query': '...', 'rating': 2, 'retrieval_k': 5},
                {'query': '...', 'rating': 3, 'retrieval_k': 5},
                {'query': '...', 'rating': 2.5, 'retrieval_k': 5},
            ]
        def get_feedback_since(self, days):
            return []

    db = MockDB()
    optimizer = RAGParameterOptimizer(db_connection=db)
    
    print("Mevcut Yapılandırma:", optimizer.current_config)
    
    # Optimizasyon döngüsünü çalıştır
    new_config = optimizer.run_optimization_cycle()
    
    print("Optimize Edilmiş Yeni Yapılandırma:", new_config)