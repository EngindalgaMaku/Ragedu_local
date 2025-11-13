import pytest
import sys
import os
from unittest.mock import MagicMock

# Projenin kök dizinini (src) Python yoluna ekle
# Bu, testlerin 'src' içindeki modülleri bulmasını sağlar.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analytics.database import ExperimentDatabase


@pytest.fixture(scope="session")
def project_root():
    """Proje kök dizinini döndüren bir fixture."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="function")
def db_instance():
    """
    Her test fonksiyonu için bellek içi (in-memory) bir veritabanı örneği oluşturur.
    'yield' kullanılarak, test bittikten sonra veritabanı bağlantısı kapatılır.
    """
    # ':memory:' özel dizesi, veritabanını RAM'de oluşturur.
    db = ExperimentDatabase(db_path=":memory:")
    yield db
    # Test bittikten sonra herhangi bir temizleme işlemi gerekirse buraya eklenebilir,
    # ancak in-memory veritabanı için genellikle gerekli değildir.

@pytest.fixture(scope="function")
def mock_db():
    """Mock database fixture for testing."""
    mock = MagicMock()
    mock.get_feedback_since.return_value = []
    mock.get_performance_metrics.return_value = []
    mock.get_average_rating.return_value = 3.5
    mock.get_most_uncertain_queries.return_value = []
    mock.check_connection.return_value = True
    return mock

# Gelecekteki testler için ortak fixture'lar buraya eklenecek.
# Örnekler:
# - API testleri için bir TestClient fixture'ı
# - Mock data üreten fixture'lar