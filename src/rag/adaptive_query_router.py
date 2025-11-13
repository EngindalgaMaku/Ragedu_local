# src/rag/adaptive_query_router.py

"""
Adaptive Query Router: Akıllı Sorgu Yönlendirme Mekanizması

Bu modül, gelen sorguları analiz ederek en uygun RAG stratejisine veya
zincirine yönlendirir. Kararlarını öğrenci profili, geçmiş performans
verileri ve sorgu belirsizliği gibi dinamik faktörlere dayandırır.
"""

from typing import Dict, Any, List

class AdaptiveQueryRouter:
    """
    Sorguları, özelliklerine ve sistemin mevcut durumuna göre en uygun
    işlem ardışık düzenine (pipeline) yönlendirir.
    """
    def __init__(self, db_connection, active_learning_engine):
        self.db = db_connection
        self.ale = active_learning_engine
        # Mevcut RAG stratejileri (örneğin, farklı chunk boyutları, farklı modeller)
        self.rag_strategies = {
            "precise_retrieval": "Küçük chunk'lar ve yüksek benzerlik eşiği ile çalışan zincir.",
            "broad_context_retrieval": "Geniş chunk'lar ve daha fazla doküman getiren zincir.",
            "hybrid_search": "Anahtar kelime ve anlamsal aramayı birleştiren zincir."
        }

    def get_student_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Öğrencinin geçmiş etkileşimlerine dayalı bir profil getirir.
        """
        # Veritabanından öğrenci verilerini çek
        # Örnek: Öğrencinin daha önce zorlandığı konular
        profile = self.db.get_user_profile(user_id)
        if not profile:
            return {"expertise_level": "beginner", "past_topics": []}
        return profile

    def get_historical_performance(self, query_type: str) -> Dict[str, Any]:
        """
        Benzer sorgu türleri için hangi stratejinin geçmişte daha iyi
        performans gösterdiğini analiz eder.
        """
        performance = self.db.get_strategy_performance_by_query_type(query_type)
        return performance

    def select_strategy_dynamically(self, query: str, user_id: str) -> str:
        """
        Bir sorgu için en uygun RAG stratejisini dinamik olarak seçer.
        """
        # 1. Öğrenci profilini al
        student_profile = self.get_student_profile(user_id)
        
        # 2. Sorgu türünü belirle (basit bir örnek)
        query_type = "definition" if "nedir" in query.lower() else "explanation"

        # 3. Geçmiş performansı kontrol et
        historical_perf = self.get_historical_performance(query_type)
        
        # Öncelikli stratejiyi belirle
        if historical_perf and historical_perf['best_strategy']:
            print(f"Geçmiş performansa göre en iyi strateji: {historical_perf['best_strategy']}")
            return historical_perf['best_strategy']

        # 4. Profil bazlı kural
        if student_profile['expertise_level'] == 'beginner' and query_type == "definition":
            # Yeni başlayanlar için net, kısa cevaplar daha iyi olabilir
            return "precise_retrieval"
        
        # Varsayılan strateji
        return "hybrid_search"

    def route_query(self, query: str, user_id: str, query_metadata: Dict[str, Any]) -> str:
        """
        Sorguyu yönlendirmek için ana fonksiyon. Belirsizlik tabanlı geri çekilme
        (fallback) mekanizmasını içerir.
        """
        # Modelden gelen yanıttaki olasılıkları kullanarak belirsizlik skoru hesapla
        # Bu, genellikle bir ön-işlem veya ilk RAG geçişinden sonra yapılır.
        probabilities = query_metadata.get('probabilities', [])
        uncertainty_score = self.ale.sampler.entropy(probabilities)

        # 1. Belirsizlik çok yüksekse, daha sağlam bir stratejiye yönlendir
        if uncertainty_score > 0.8:
            print(f"Yüksek belirsizlik ({uncertainty_score:.2f}) tespit edildi. Geniş bağlam stratejisine yönlendiriliyor.")
            return "broad_context_retrieval"

        # 2. Belirsizlik düşükse, dinamik strateji seçimi yap
        selected_strategy = self.select_strategy_dynamically(query, user_id)
        print(f"Dinamik olarak seçilen strateji: {selected_strategy}")
        
        return selected_strategy


if __name__ == '__main__':
    # Örnek kullanım
    from active_learning import ActiveLearningEngine

    class MockDB:
        def get_user_profile(self, user_id):
            if user_id == "user123":
                return {"expertise_level": "intermediate", "past_topics": ["RAG", "Python"]}
            return None
        
        def get_strategy_performance_by_query_type(self, query_type):
            if query_type == "definition":
                return {"best_strategy": "precise_retrieval", "avg_rating": 4.5}
            return None
        
        def get_feedback_since(self, days): return []
        def get_most_uncertain_queries(self, limit): return []

    db = MockDB()
    ale = ActiveLearningEngine(db_connection=db)
    router = AdaptiveQueryRouter(db_connection=db, active_learning_engine=ale)

    # --- Senaryo 1: Geçmiş performansa dayalı yönlendirme ---
    query1 = "RAG nedir?"
    user1 = "user456"
    print(f"\n--- Sorgu 1: '{query1}' ---")
    strategy1 = router.route_query(query1, user1, {'probabilities': [0.7, 0.2, 0.1]})
    print(f"Sonuç: Sorgu '{strategy1}' stratejisine yönlendirildi.")
    
    # --- Senaryo 2: Yüksek belirsizlik durumunda fallback ---
    query2 = "Gelecekteki AI ajanları ve insan etkileşimi etiği üzerine bir analiz"
    user2 = "user123"
    print(f"\n--- Sorgu 2: '{query2}' ---")
    # Yüksek entropili olasılıklar -> yüksek belirsizlik
    strategy2 = router.route_query(query2, user2, {'probabilities': [0.3, 0.3, 0.2, 0.2]})
    print(f"Sonuç: Sorgu '{strategy2}' stratejisine yönlendirildi.")

    # --- Senaryo 3: Profil bazlı yönlendirme (geçmiş data yoksa) ---
    query3 = "Python'da 'yield' nasıl çalışır?"
    user3 = "user789" # Yeni kullanıcı -> beginner profili
    print(f"\n--- Sorgu 3: '{query3}' ---")
    strategy3 = router.route_query(query3, user3, {'probabilities': [0.8, 0.1, 0.1]})
    print(f"Sonuç: Sorgu '{strategy3}' stratejisine yönlendirildi.")