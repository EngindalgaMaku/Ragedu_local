import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
import logging

from src.services.active_learning import UncertaintySampler, ActiveLearningEngine


@pytest.fixture
def mock_db():
    """Veritabanı sınıfı için bir MagicMock nesnesi oluşturur."""
    return MagicMock()


@pytest.fixture
def active_learning_engine(mock_db):
    """ActiveLearningEngine'i mock veritabanı ile başlatır."""
    return ActiveLearningEngine(db_connection=mock_db)


@pytest.fixture
def sample_probabilities():
    """Test için örnek olasılık dağılımı."""
    return [0.6, 0.3, 0.1]


@pytest.fixture
def sample_feedback_data():
    """Test için örnek feedback verisi."""
    return [
        {'query': 'RAG nedir?', 'rating': 1, 'topic': 'Genel AI'},
        {'query': 'Python lambda fonksiyonu nasıl yazılır?', 'rating': 5, 'topic': 'Python'},
        {'query': 'Transformer mimarisi detayları', 'rating': 2, 'topic': 'Derin Öğrenme'},
        {'query': 'Transformer nedir?', 'rating': 2, 'topic': 'Derin Öğrenme'},
    ]


@pytest.fixture
def sample_query_responses():
    """Test için örnek query-response verileri."""
    return [
        {'query': 'RAG nedir?', 'response': 'RAG açıklaması...', 'probabilities': [0.6, 0.3, 0.1]},
        {'query': 'En iyi LLM hangisi?', 'response': 'LLM karşılaştırması...', 'probabilities': [0.4, 0.35, 0.25]},
    ]


class TestUncertaintySampler:
    """UncertaintySampler sınıfı için testler."""
    
    @pytest.mark.unit
    def test_entropy_normal_distribution(self, sample_probabilities):
        """Normal olasılık dağılımı ile entropi hesaplama testi."""
        result = UncertaintySampler.entropy(sample_probabilities)
        
        # Entropi manual hesaplama: -0.6*log2(0.6) - 0.3*log2(0.3) - 0.1*log2(0.1)
        expected = -0.6 * np.log2(0.6) - 0.3 * np.log2(0.3) - 0.1 * np.log2(0.1)
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.unit
    def test_entropy_uniform_distribution(self):
        """Uniform dağılım ile entropi hesaplama testi."""
        uniform_probs = [0.25, 0.25, 0.25, 0.25]
        result = UncertaintySampler.entropy(uniform_probs)
        
        # Uniform dağılım için entropi = log2(n) = log2(4) = 2
        expected = 2.0
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.unit
    def test_entropy_empty_list(self):
        """Boş liste ile entropi hesaplama testi."""
        result = UncertaintySampler.entropy([])
        assert result == 0.0
    
    @pytest.mark.unit
    def test_entropy_with_zero_probabilities(self):
        """Sıfır olasılıklar ile entropi hesaplama testi."""
        probs_with_zero = [0.5, 0.0, 0.3, 0.2]
        result = UncertaintySampler.entropy(probs_with_zero)
        
        # Sadece sıfır olmayan değerler için hesaplama yapılmalı
        expected = -0.5 * np.log2(0.5) - 0.3 * np.log2(0.3) - 0.2 * np.log2(0.2)
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.unit
    def test_entropy_single_probability(self):
        """Tek olasılık ile entropi hesaplama testi."""
        result = UncertaintySampler.entropy([1.0])
        assert result == 0.0
    
    @pytest.mark.unit
    def test_least_confidence_normal_distribution(self, sample_probabilities):
        """Normal olasılık dağılımı ile least confidence hesaplama testi."""
        result = UncertaintySampler.least_confidence(sample_probabilities)
        
        # 1 - max(probabilities) = 1 - 0.6 = 0.4
        expected = 1 - 0.6
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.unit
    def test_least_confidence_high_confidence(self):
        """Yüksek güven ile least confidence hesaplama testi."""
        high_confidence_probs = [0.9, 0.08, 0.02]
        result = UncertaintySampler.least_confidence(high_confidence_probs)
        
        expected = 1 - 0.9
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.unit
    def test_least_confidence_empty_list(self):
        """Boş liste ile least confidence hesaplama testi."""
        result = UncertaintySampler.least_confidence([])
        assert result == 1.0
    
    @pytest.mark.unit
    def test_least_confidence_single_probability(self):
        """Tek olasılık ile least confidence hesaplama testi."""
        result = UncertaintySampler.least_confidence([0.7])
        expected = 1 - 0.7
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.unit
    def test_margin_sampling_normal_distribution(self, sample_probabilities):
        """Normal olasılık dağılımı ile margin sampling hesaplama testi."""
        result = UncertaintySampler.margin_sampling(sample_probabilities)
        
        # sorted: [0.6, 0.3, 0.1], margin = 0.6 - 0.3 = 0.3, result = 1 - 0.3 = 0.7
        expected = 1 - (0.6 - 0.3)
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.unit
    def test_margin_sampling_small_margin(self):
        """Küçük margin ile margin sampling hesaplama testi."""
        close_probs = [0.51, 0.49]
        result = UncertaintySampler.margin_sampling(close_probs)
        
        # margin = 0.51 - 0.49 = 0.02, result = 1 - 0.02 = 0.98
        expected = 1 - (0.51 - 0.49)
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.unit
    def test_margin_sampling_single_probability(self):
        """Tek olasılık ile margin sampling hesaplama testi."""
        result = UncertaintySampler.margin_sampling([0.7])
        assert result == 0.0
    
    @pytest.mark.unit
    def test_margin_sampling_empty_list(self):
        """Boş liste ile margin sampling hesaplama testi."""
        result = UncertaintySampler.margin_sampling([])
        assert result == 0.0
    
    @pytest.mark.unit
    def test_margin_sampling_identical_probabilities(self):
        """Aynı olasılıklar ile margin sampling hesaplama testi."""
        identical_probs = [0.5, 0.5]
        result = UncertaintySampler.margin_sampling(identical_probs)
        
        # margin = 0, result = float('inf')
        assert result == float('inf')


class TestActiveLearningEngineInit:
    """ActiveLearningEngine constructor testleri."""
    
    @pytest.mark.unit
    def test_init_success(self, mock_db):
        """ActiveLearningEngine'in başarılı başlatılması testi."""
        engine = ActiveLearningEngine(db_connection=mock_db)
        
        assert engine.db == mock_db
        assert isinstance(engine.sampler, UncertaintySampler)
    
    @pytest.mark.unit
    def test_init_with_none_db(self):
        """None veritabanı ile başlatma testi."""
        engine = ActiveLearningEngine(db_connection=None)
        
        assert engine.db is None
        assert isinstance(engine.sampler, UncertaintySampler)


class TestAnalyzeFeedbackPatterns:
    """analyze_feedback_patterns metodu için testler."""
    
    @pytest.mark.unit
    def test_analyze_feedback_patterns_success(self, active_learning_engine, mock_db, sample_feedback_data):
        """Feedback pattern analizinin başarılı yapılması testi."""
        mock_db.get_feedback_since.return_value = sample_feedback_data
        
        result = active_learning_engine.analyze_feedback_patterns(time_window_days=7)
        
        mock_db.get_feedback_since.assert_called_once_with(7)
        
        # Negatif feedback'ler (rating < 3): 'Genel AI' ve 'Derin Öğrenme'
        assert len(result) == 2
        
        # Sonuçları topic'e göre kontrol et
        topic_counts = {item['topic']: item['negative_feedback_count'] for item in result}
        assert topic_counts.get('Genel AI') == 1
        assert topic_counts.get('Derin Öğrenme') == 2
    
    @pytest.mark.unit
    def test_analyze_feedback_patterns_no_negative_feedback(self, active_learning_engine, mock_db):
        """Negatif feedback bulunmayan durum testi."""
        positive_feedback_data = [
            {'query': 'Test query', 'rating': 4, 'topic': 'Test Topic'},
            {'query': 'Another query', 'rating': 5, 'topic': 'Another Topic'}
        ]
        mock_db.get_feedback_since.return_value = positive_feedback_data
        
        result = active_learning_engine.analyze_feedback_patterns()
        
        assert result == []
    
    @pytest.mark.unit
    def test_analyze_feedback_patterns_empty_feedback(self, active_learning_engine, mock_db):
        """Hiç feedback bulunmayan durum testi."""
        mock_db.get_feedback_since.return_value = []
        
        result = active_learning_engine.analyze_feedback_patterns()
        
        assert result == []
    
    @pytest.mark.unit
    def test_analyze_feedback_patterns_custom_time_window(self, active_learning_engine, mock_db, sample_feedback_data):
        """Özel zaman penceresi ile pattern analizi testi."""
        mock_db.get_feedback_since.return_value = sample_feedback_data
        
        active_learning_engine.analyze_feedback_patterns(time_window_days=14)
        
        mock_db.get_feedback_since.assert_called_once_with(14)
    
    @pytest.mark.unit
    def test_analyze_feedback_patterns_missing_topic(self, active_learning_engine, mock_db):
        """Topic bilgisi eksik olan feedback'ler için test."""
        feedback_without_topic = [
            {'query': 'Test query', 'rating': 1},
            {'query': 'Another query', 'rating': 2, 'topic': 'Known Topic'}
        ]
        mock_db.get_feedback_since.return_value = feedback_without_topic
        
        result = active_learning_engine.analyze_feedback_patterns()
        
        topic_counts = {item['topic']: item['negative_feedback_count'] for item in result}
        assert topic_counts.get('unknown') == 1
        assert topic_counts.get('Known Topic') == 1
    
    @pytest.mark.unit
    def test_analyze_feedback_patterns_database_error(self, active_learning_engine, mock_db):
        """Veritabanı hatası durumu testi."""
        error_message = "Database connection failed"
        mock_db.get_feedback_since.side_effect = Exception(error_message)
        
        with pytest.raises(Exception, match=error_message):
            active_learning_engine.analyze_feedback_patterns()


class TestScoreModelConfidence:
    """score_model_confidence metodu için testler."""
    
    @pytest.mark.unit
    def test_score_model_confidence_success(self, active_learning_engine, sample_query_responses):
        """Model güven skorlarının başarılı hesaplanması testi."""
        result = active_learning_engine.score_model_confidence(sample_query_responses)
        
        assert len(result) == 2
        
        # İlk response kontrolü
        first_response = result[0]
        assert 'confidence_scores' in first_response
        assert 'uncertainty_score' in first_response
        
        confidence_scores = first_response['confidence_scores']
        assert 'entropy' in confidence_scores
        assert 'least_confidence' in confidence_scores
        assert 'margin' in confidence_scores
        
        # Uncertainty score ortalama olarak hesaplanmalı
        expected_uncertainty = np.mean(list(confidence_scores.values()))
        assert abs(first_response['uncertainty_score'] - expected_uncertainty) < 1e-10
    
    @pytest.mark.unit
    def test_score_model_confidence_empty_probabilities(self, active_learning_engine):
        """Olasılık bilgisi olmayan response'lar için test."""
        responses_without_probs = [
            {'query': 'Test query', 'response': 'Test response'}
        ]
        
        result = active_learning_engine.score_model_confidence(responses_without_probs)
        
        assert len(result) == 1
        confidence_scores = result[0]['confidence_scores']
        
        # Boş olasılık için beklenen değerler
        assert confidence_scores['entropy'] == 0.0
        assert confidence_scores['least_confidence'] == 1.0
        assert confidence_scores['margin'] == 0.0
    
    @pytest.mark.unit
    def test_score_model_confidence_empty_list(self, active_learning_engine):
        """Boş response listesi için test."""
        result = active_learning_engine.score_model_confidence([])
        
        assert result == []
    
    @pytest.mark.unit
    def test_score_model_confidence_various_probability_distributions(self, active_learning_engine):
        """Farklı olasılık dağılımları ile confidence scoring testi."""
        diverse_responses = [
            {'query': 'High confidence', 'probabilities': [0.9, 0.08, 0.02]},
            {'query': 'Low confidence', 'probabilities': [0.4, 0.35, 0.25]},
            {'query': 'Uniform confidence', 'probabilities': [0.33, 0.33, 0.34]}
        ]
        
        result = active_learning_engine.score_model_confidence(diverse_responses)
        
        assert len(result) == 3
        
        # Yüksek güven düşük uncertainty'ye sahip olmalı
        high_conf_uncertainty = result[0]['uncertainty_score']
        low_conf_uncertainty = result[1]['uncertainty_score']
        uniform_conf_uncertainty = result[2]['uncertainty_score']
        
        # Uniform dağılım en yüksek uncertainty'ye sahip olmalı
        assert uniform_conf_uncertainty > low_conf_uncertainty > high_conf_uncertainty


class TestRecommendLearningStrategy:
    """recommend_learning_strategy metodu için testler."""
    
    @pytest.mark.unit
    def test_recommend_high_uncertainty_strategy(self, active_learning_engine):
        """Yüksek uncertainty durumu için strateji önerisi testi."""
        uncertainty_score = 0.8
        patterns = []
        
        result = active_learning_engine.recommend_learning_strategy(uncertainty_score, patterns)
        
        assert "HIGH_UNCERTAINTY" in result
        assert "Modeli daha fazla çeşitli veri ile yeniden eğitin" in result
    
    @pytest.mark.unit
    def test_recommend_focused_negative_feedback_strategy(self, active_learning_engine):
        """Odaklanmış negatif feedback durumu için strateji önerisi testi."""
        uncertainty_score = 0.3
        patterns = [
            {'topic': 'Problem Topic', 'negative_feedback_count': 15}
        ]
        
        result = active_learning_engine.recommend_learning_strategy(uncertainty_score, patterns)
        
        assert "FOCUSED_NEGATIVE_FEEDBACK" in result
        assert "Belirli konulardaki belgeleri ve chunking stratejisini iyileştirin" in result
    
    @pytest.mark.unit
    def test_recommend_moderate_uncertainty_strategy(self, active_learning_engine):
        """Orta seviye uncertainty durumu için strateji önerisi testi."""
        uncertainty_score = 0.6
        patterns = [
            {'topic': 'Normal Topic', 'negative_feedback_count': 5}
        ]
        
        result = active_learning_engine.recommend_learning_strategy(uncertainty_score, patterns)
        
        assert "MODERATE_UNCERTAINTY" in result
        assert "İnsan etiketlemesi için en belirsiz sorguları önceliklendirin" in result
    
    @pytest.mark.unit
    def test_recommend_low_uncertainty_strategy(self, active_learning_engine):
        """Düşük uncertainty durumu için strateji önerisi testi."""
        uncertainty_score = 0.2
        patterns = []
        
        result = active_learning_engine.recommend_learning_strategy(uncertainty_score, patterns)
        
        assert "LOW_UNCERTAINTY" in result
        assert "Sistem stabil. Performansı izlemeye devam edin" in result
    
    @pytest.mark.unit
    def test_recommend_strategy_priority_high_uncertainty(self, active_learning_engine):
        """Yüksek uncertainty'nin diğer koşullara göre öncelik testi."""
        uncertainty_score = 0.8
        patterns = [
            {'topic': 'Problem Topic', 'negative_feedback_count': 15}
        ]
        
        result = active_learning_engine.recommend_learning_strategy(uncertainty_score, patterns)
        
        # Yüksek uncertainty öncelikli olmalı
        assert "HIGH_UNCERTAINTY" in result
    
    @pytest.mark.unit
    def test_recommend_strategy_priority_focused_feedback(self, active_learning_engine):
        """Focused negative feedback'in moderate uncertainty'ye göre öncelik testi."""
        uncertainty_score = 0.6
        patterns = [
            {'topic': 'Problem Topic', 'negative_feedback_count': 12}
        ]
        
        result = active_learning_engine.recommend_learning_strategy(uncertainty_score, patterns)
        
        # Focused negative feedback moderate uncertainty'den öncelikli olmalı
        assert "FOCUSED_NEGATIVE_FEEDBACK" in result


class TestIdentifySamplesForReview:
    """identify_samples_for_review metodu için testler."""
    
    @pytest.mark.unit
    def test_identify_samples_for_review_success(self, active_learning_engine, mock_db):
        """Gözden geçirilecek örneklerin başarılı tanımlanması testi."""
        mock_samples = [
            {'query': 'Belirsiz query 1', 'uncertainty_score': 0.85},
            {'query': 'Belirsiz query 2', 'uncertainty_score': 0.81},
        ]
        mock_db.get_most_uncertain_queries.return_value = mock_samples
        
        result = active_learning_engine.identify_samples_for_review()
        
        mock_db.get_most_uncertain_queries.assert_called_once_with(10)
        assert result == mock_samples
    
    @pytest.mark.unit
    def test_identify_samples_for_review_custom_limit(self, active_learning_engine, mock_db):
        """Özel limit ile gözden geçirilecek örneklerin tanımlanması testi."""
        mock_samples = [
            {'query': 'Query 1', 'uncertainty_score': 0.9},
            {'query': 'Query 2', 'uncertainty_score': 0.8},
            {'query': 'Query 3', 'uncertainty_score': 0.7},
            {'query': 'Query 4', 'uncertainty_score': 0.6},
            {'query': 'Query 5', 'uncertainty_score': 0.5},
        ]
        mock_db.get_most_uncertain_queries.return_value = mock_samples
        
        result = active_learning_engine.identify_samples_for_review(limit=5)
        
        mock_db.get_most_uncertain_queries.assert_called_once_with(5)
        assert len(result) == 5
    
    @pytest.mark.unit
    def test_identify_samples_for_review_empty_result(self, active_learning_engine, mock_db):
        """Belirsiz örnek bulunmayan durum testi."""
        mock_db.get_most_uncertain_queries.return_value = []
        
        result = active_learning_engine.identify_samples_for_review()
        
        assert result == []
    
    @pytest.mark.unit
    def test_identify_samples_for_review_database_error(self, active_learning_engine, mock_db):
        """Veritabanı hatası durumu testi."""
        error_message = "Query execution failed"
        mock_db.get_most_uncertain_queries.side_effect = Exception(error_message)
        
        with pytest.raises(Exception, match=error_message):
            active_learning_engine.identify_samples_for_review()


class TestActiveLearningEngineIntegration:
    """ActiveLearningEngine entegrasyon testleri."""
    
    @pytest.mark.unit
    def test_full_active_learning_workflow(self, active_learning_engine, mock_db):
        """Tam aktif öğrenme iş akışı testi."""
        # Mock data setup
        feedback_data = [
            {'query': 'RAG nasıl çalışır?', 'rating': 1, 'topic': 'RAG'},
            {'query': 'Python nedir?', 'rating': 5, 'topic': 'Python'},
        ]
        query_responses = [
            {'query': 'RAG nasıl çalışır?', 'probabilities': [0.4, 0.4, 0.2]},
            {'query': 'Python nedir?', 'probabilities': [0.8, 0.15, 0.05]},
        ]
        uncertain_samples = [
            {'query': 'Belirsiz soru', 'uncertainty_score': 0.9}
        ]
        
        mock_db.get_feedback_since.return_value = feedback_data
        mock_db.get_most_uncertain_queries.return_value = uncertain_samples
        
        # 1. Feedback pattern analizi
        patterns = active_learning_engine.analyze_feedback_patterns()
        
        # 2. Confidence scoring
        scored_responses = active_learning_engine.score_model_confidence(query_responses)
        
        # 3. Ortalama uncertainty hesaplama
        avg_uncertainty = np.mean([r['uncertainty_score'] for r in scored_responses])
        
        # 4. Strateji önerisi
        strategy = active_learning_engine.recommend_learning_strategy(avg_uncertainty, patterns)
        
        # 5. Gözden geçirilecek örnekleri belirleme
        samples_for_review = active_learning_engine.identify_samples_for_review()
        
        # Assertions
        assert len(patterns) == 1
        assert patterns[0]['topic'] == 'RAG'
        assert len(scored_responses) == 2
        assert isinstance(strategy, str)
        assert len(samples_for_review) == 1
    
    @pytest.mark.unit
    def test_uncertainty_calculations_consistency(self, active_learning_engine):
        """Uncertainty hesaplamalarının tutarlılığı testi."""
        test_cases = [
            [0.9, 0.08, 0.02],  # Yüksek güven
            [0.4, 0.35, 0.25],  # Orta güven
            [0.33, 0.33, 0.34]  # Düşük güven (uniform)
        ]
        
        responses = [
            {'query': f'Query {i}', 'probabilities': probs}
            for i, probs in enumerate(test_cases)
        ]
        
        scored = active_learning_engine.score_model_confidence(responses)
        
        # Uncertainty skorları beklenen sırada olmalı: uniform > orta > yüksek güven
        uncertainties = [r['uncertainty_score'] for r in scored]
        
        assert uncertainties[2] > uncertainties[1] > uncertainties[0]


class TestActiveLearningEdgeCases:
    """ActiveLearning kenar durum testleri."""
    
    @pytest.mark.unit
    def test_extremely_small_probabilities(self, active_learning_engine):
        """Çok küçük olasılıklar ile çalışma testi."""
        responses = [
            {'query': 'Test', 'probabilities': [1e-10, 1e-15, 1 - 1e-10 - 1e-15]}
        ]
        
        result = active_learning_engine.score_model_confidence(responses)
        
        assert len(result) == 1
        assert 'uncertainty_score' in result[0]
        assert not np.isnan(result[0]['uncertainty_score'])
        assert not np.isinf(result[0]['uncertainty_score'])
    
    @pytest.mark.unit
    def test_large_probability_arrays(self, active_learning_engine):
        """Büyük olasılık dizileri ile çalışma testi."""
        # 100 elemanlı uniform dağılım
        large_probs = [0.01] * 100
        responses = [
            {'query': 'Large array test', 'probabilities': large_probs}
        ]
        
        result = active_learning_engine.score_model_confidence(responses)
        
        assert len(result) == 1
        assert 'uncertainty_score' in result[0]
    
    @pytest.mark.unit
    def test_feedback_patterns_with_extreme_ratings(self, active_learning_engine, mock_db):
        """Aşırı rating değerleri ile pattern analizi testi."""
        extreme_feedback = [
            {'query': 'Test 1', 'rating': -5, 'topic': 'Negative'},
            {'query': 'Test 2', 'rating': 100, 'topic': 'Positive'},
            {'query': 'Test 3', 'rating': 0, 'topic': 'Zero'}
        ]
        mock_db.get_feedback_since.return_value = extreme_feedback
        
        result = active_learning_engine.analyze_feedback_patterns()
        
        # Rating < 3 olan tüm değerler negative olarak sayılmalı
        negative_topics = [item['topic'] for item in result]
        assert 'Negative' in negative_topics
        assert 'Zero' in negative_topics
        assert 'Positive' not in negative_topics
    
    @pytest.mark.unit
    def test_strategy_recommendation_boundary_values(self, active_learning_engine):
        """Sınır değerler ile strateji önerisi testi."""
        # Tam sınır değerleri
        boundary_tests = [
            (0.7, [], "HIGH_UNCERTAINTY"),
            (0.5, [], "MODERATE_UNCERTAINTY"),
            (0.3, [{'negative_feedback_count': 10}], "FOCUSED_NEGATIVE_FEEDBACK")
        ]
        
        for uncertainty, patterns, expected_type in boundary_tests:
            result = active_learning_engine.recommend_learning_strategy(uncertainty, patterns)
            assert expected_type in result


class TestActiveLearningErrorHandling:
    """ActiveLearning hata yönetimi testleri."""
    
    @pytest.mark.unit
    def test_score_model_confidence_with_invalid_probabilities(self, active_learning_engine):
        """Geçersiz olasılık değerleri ile confidence scoring testi."""
        invalid_responses = [
            {'query': 'Test', 'probabilities': [-0.1, 0.5, 0.6]},  # Negatif değer
            {'query': 'Test2', 'probabilities': [0.5, 0.6, 0.7]},  # Toplam > 1
        ]
        
        # Bu durumda bile sistem çalışmaya devam etmeli
        result = active_learning_engine.score_model_confidence(invalid_responses)
        
        assert len(result) == 2
        for r in result:
            assert 'uncertainty_score' in r
    
    @pytest.mark.unit
    def test_analyze_feedback_patterns_with_malformed_data(self, active_learning_engine, mock_db):
        """Bozuk veri formatı ile pattern analizi testi."""
        malformed_data = [
            {'query': 'Test 1'},  # rating eksik
            {'rating': 1},  # query eksik
            {'query': 'Test 3', 'rating': 'invalid'},  # geçersiz rating
        ]
        mock_db.get_feedback_since.return_value = malformed_data
        
        # Sistem hata vermeden çalışmalı, sadece geçerli verileri işlemeli
        result = active_learning_engine.analyze_feedback_patterns()
        
        # Malformed data nedeniyle boş sonuç bekleniyor
        assert isinstance(result, list)
    
    @pytest.mark.unit
    def test_recommend_strategy_with_none_values(self, active_learning_engine):
        """None değerler ile strateji önerisi testi."""
        result = active_learning_engine.recommend_learning_strategy(None, None)
        
        # None değerlerle bile sistem bir strateji döndürmeli
        assert isinstance(result, str)
        assert len(result) > 0