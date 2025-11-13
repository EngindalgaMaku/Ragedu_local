
import pytest
import hashlib
import time
from unittest.mock import MagicMock, patch, call, mock_open
from concurrent.futures import TimeoutError
from typing import Dict, Any, List

from src.rag.rag_pipeline import RAGPipeline


@pytest.fixture
def mock_config():
    """Mock configuration for RAGPipeline."""
    return {
        "ollama_base_url": "http://localhost:11434",
        "ollama_embedding_model": "mxbai-embed-large",
        "ollama_generation_model": "llama3.1:8b",
        "temperature": 0.7,
        "max_tokens": 512,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "enable_cache": True,
        "cache_ttl": 3600,
        "ollama_request_timeout": 120
    }


@pytest.fixture
def mock_faiss_store():
    """Mock FaissVectorStore for RAGPipeline."""
    mock_store = MagicMock()
    mock_store.search.return_value = [
        ("Document 1 content", 0.95, {"source": "doc1.pdf", "page": 1}),
        ("Document 2 content", 0.87, {"source": "doc2.pdf", "page": 2})
    ]
    return mock_store


@pytest.fixture
def mock_cache():
    """Mock cache instance."""
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_cache.set.return_value = None
    return mock_cache


@pytest.fixture
def mock_memory_manager():
    """Mock memory manager instance."""
    mock_mm = MagicMock()
    mock_mm.get_memory_usage.return_value = {"percent": 45.0}
    mock_mm.auto_gc_check.return_value = False
    return mock_mm


@pytest.fixture
def mock_logger():
    """Mock logger instance."""
    return MagicMock()


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client."""
    mock_client = MagicMock()
    mock_client.list.return_value = {"models": []}
    mock_client.chat.return_value = {
        "message": {
            "content": "Test response from Ollama"
        }
    }
    return mock_client


@pytest.fixture
def rag_pipeline(mock_config, mock_faiss_store):
    """RAGPipeline instance with mocked dependencies."""
    with patch('src.rag.rag_pipeline.get_logger') as mock_get_logger, \
         patch('src.rag.rag_pipeline.get_cache') as mock_get_cache, \
         patch('src.rag.rag_pipeline.get_memory_manager') as mock_get_memory_manager, \
         patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True), \
         patch('src.rag.rag_pipeline.ollama') as mock_ollama_module:
        
        mock_get_logger.return_value = MagicMock()
        mock_get_cache.return_value = MagicMock()
        mock_get_memory_manager.return_value = MagicMock()
        
        # Mock the ollama module and its Client class
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}
        mock_ollama_module.Client.return_value = mock_client
        
        pipeline = RAGPipeline(mock_config, mock_faiss_store)
        yield pipeline


@pytest.fixture
def sample_retrieved_context():
    """Sample retrieved context for testing."""
    return [
        {
            "text": "RAG (Retrieval-Augmented Generation) is a technique...",
            "score": 0.95,
            "metadata": {"source": "rag_doc.pdf", "page": 1}
        },
        {
            "text": "Vector databases store embeddings...",
            "score": 0.87,
            "metadata": {"source": "vector_doc.pdf", "page": 3}
        }
    ]


class TestRAGPipelineInit:
    """Tests for RAGPipeline initialization."""
    
    @patch('src.rag.rag_pipeline.get_logger')
    @patch('src.rag.rag_pipeline.get_cache')
    @patch('src.rag.rag_pipeline.get_memory_manager')
    @patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True)
    @patch('src.rag.rag_pipeline.ollama')
    def test_init_success(self, mock_ollama_module, mock_available, mock_get_memory_manager,
                         mock_get_cache, mock_get_logger, mock_config, mock_faiss_store):
        """Tests successful initialization of RAGPipeline."""
        mock_logger = MagicMock()
        mock_cache = MagicMock()
        mock_memory_manager = MagicMock()
        mock_client = MagicMock()
        
        mock_get_logger.return_value = mock_logger
        mock_get_cache.return_value = mock_cache
        mock_get_memory_manager.return_value = mock_memory_manager
        mock_ollama_module.Client.return_value = mock_client
        mock_client.list.return_value = {"models": []}
        
        pipeline = RAGPipeline(mock_config, mock_faiss_store)
        
        assert pipeline.config == mock_config
        assert pipeline.faiss_store == mock_faiss_store
        assert pipeline.logger == mock_logger
        assert pipeline.cache == mock_cache
        assert pipeline.memory_manager == mock_memory_manager
        assert pipeline.ollama_client == mock_client
        
        mock_get_logger.assert_called_once_with('src.rag.rag_pipeline', mock_config)
        mock_get_cache.assert_called_once_with(ttl=3600)
        mock_get_memory_manager.assert_called_once_with(mock_config)
        mock_ollama_module.Client.assert_called_once_with(host="http://localhost:11434")
    
    @patch('src.rag.rag_pipeline.get_logger')
    @patch('src.rag.rag_pipeline.get_cache')
    @patch('src.rag.rag_pipeline.get_memory_manager')
    @patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True)
    @patch('src.rag.rag_pipeline.ollama')
    def test_init_cache_disabled(self, mock_ollama_module, mock_available, mock_get_memory_manager,
                                mock_get_cache, mock_get_logger, mock_faiss_store):
        """Tests initialization with cache disabled."""
        config_no_cache = {
            "ollama_base_url": "http://localhost:11434",
            "enable_cache": False
        }
        
        mock_get_logger.return_value = MagicMock()
        mock_get_memory_manager.return_value = MagicMock()
        mock_client = MagicMock()
        mock_ollama_module.Client.return_value = mock_client
        mock_client.list.return_value = {"models": []}
        
        pipeline = RAGPipeline(config_no_cache, mock_faiss_store)
        
        assert pipeline.cache is None
        mock_get_cache.assert_not_called()
    
    @patch('src.rag.rag_pipeline.get_logger')
    @patch('src.rag.rag_pipeline.get_cache')
    @patch('src.rag.rag_pipeline.get_memory_manager')
    @patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True)
    @patch('src.rag.rag_pipeline.ollama')
    def test_init_custom_cache_ttl(self, mock_ollama_module, mock_available, mock_get_memory_manager,
                                  mock_get_cache, mock_get_logger, mock_faiss_store):
        """Tests initialization with custom cache TTL."""
        config_custom_ttl = {
            "ollama_base_url": "http://localhost:11434",
            "enable_cache": True,
            "cache_ttl": 7200
        }
        
        mock_get_logger.return_value = MagicMock()
        mock_get_memory_manager.return_value = MagicMock()
        mock_client = MagicMock()
        mock_ollama_module.Client.return_value = mock_client
        mock_client.list.return_value = {"models": []}
        
        RAGPipeline(config_custom_ttl, mock_faiss_store)
        
        mock_get_cache.assert_called_once_with(ttl=7200)


class TestInitOllamaClient:
    """Tests for _init_ollama_client method."""
    
    @patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True)
    @patch('src.rag.rag_pipeline.ollama')
    @patch('src.rag.rag_pipeline.time.sleep')
    def test_init_ollama_client_success_first_attempt(self, mock_sleep, mock_ollama_module, mock_available, rag_pipeline):
        """Tests successful Ollama client initialization on first attempt."""
        mock_client = MagicMock()
        mock_ollama_module.Client.return_value = mock_client
        mock_client.list.return_value = {"models": []}
        
        result = rag_pipeline._init_ollama_client()
        
        assert result == mock_client
        mock_ollama_module.Client.assert_called_once_with(host=rag_pipeline.config.get("ollama_base_url"))
        mock_client.list.assert_called_once()
        mock_sleep.assert_not_called()
        rag_pipeline.logger.info.assert_called()
    
    @patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True)
    @patch('src.rag.rag_pipeline.ollama')
    @patch('src.rag.rag_pipeline.time.sleep')
    def test_init_ollama_client_retry_then_success(self, mock_sleep, mock_ollama_module, mock_available, rag_pipeline):
        """Tests Ollama client initialization with retry logic."""
        mock_client = MagicMock()
        mock_ollama_module.Client.return_value = mock_client
        
        # First call fails, second succeeds
        mock_client.list.side_effect = [Exception("Connection failed"), {"models": []}]
        
        result = rag_pipeline._init_ollama_client()
        
        assert result == mock_client
        assert mock_client.list.call_count == 2
        mock_sleep.assert_called_once_with(2)
        rag_pipeline.logger.warning.assert_called()
        rag_pipeline.logger.info.assert_called()
    
    @patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True)
    @patch('src.rag.rag_pipeline.ollama')
    @patch('src.rag.rag_pipeline.time.sleep')
    def test_init_ollama_client_all_retries_fail(self, mock_sleep, mock_ollama_module, mock_available, rag_pipeline):
        """Tests Ollama client initialization when all retries fail."""
        mock_client = MagicMock()
        mock_ollama_module.Client.return_value = mock_client
        mock_client.list.side_effect = Exception("Connection failed")
        
        result = rag_pipeline._init_ollama_client()
        
        assert result is None
        assert mock_client.list.call_count == 3  # 3 attempts
        assert mock_sleep.call_count == 2  # 2 sleeps between 3 attempts
        rag_pipeline.logger.error.assert_called_with("Failed to connect to Ollama after all retries")
    
    @patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True)
    @patch('src.rag.rag_pipeline.ollama')
    @patch('src.rag.rag_pipeline.time.sleep')
    def test_init_ollama_client_exponential_backoff(self, mock_sleep, mock_ollama_module, mock_available, rag_pipeline):
        """Tests exponential backoff in Ollama client retry logic."""
        mock_client = MagicMock()
        mock_ollama_module.Client.return_value = mock_client
        mock_client.list.side_effect = Exception("Connection failed")
        
        rag_pipeline._init_ollama_client()
        
        # Verify exponential backoff: 2s, then 4s
        expected_calls = [call(2), call(4)]
        mock_sleep.assert_has_calls(expected_calls)
        
    def test_init_ollama_client_module_unavailable(self, rag_pipeline):
        """Tests _init_ollama_client when Ollama module is not available."""
        with patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', False):
            result = rag_pipeline._init_ollama_client()
            
            assert result is None
            rag_pipeline.logger.warning.assert_called_with("Ollama module not available - running in test/development mode")


class TestRetrieve:
    """Tests for retrieve method."""
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_retrieve_success(self, mock_generate_embeddings, rag_pipeline, mock_faiss_store):
        """Tests successful document retrieval."""
        query = "What is RAG?"
        query_embedding = [[0.1, 0.2, 0.3]]
        expected_results = [
            ("Document content 1", 0.95, {"source": "doc1.pdf"}),
            ("Document content 2", 0.87, {"source": "doc2.pdf"})
        ]
        
        mock_generate_embeddings.return_value = query_embedding
        mock_faiss_store.search.return_value = expected_results
        
        result = rag_pipeline.retrieve(query, top_k=2)
        
        mock_generate_embeddings.assert_called_once_with(
            [query],
            model=rag_pipeline.config.get("ollama_embedding_model"),
            use_cache=rag_pipeline.config.get("enable_cache", True)
        )
        mock_faiss_store.search.assert_called_once_with(query_embedding, 2)
        
        assert len(result) == 2
        assert result[0]["text"] == "Document content 1"
        assert result[0]["score"] == 0.95
        assert result[1]["text"] == "Document content 2"
        assert result[1]["score"] == 0.87
        
        rag_pipeline.logger.info.assert_any_call("Retrieving documents for query: 'What is RAG?' (top_k=2)")
        rag_pipeline.logger.info.assert_any_call("Retrieved 2 chunks from vector store.")
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_retrieve_with_cache_hit(self, mock_generate_embeddings, rag_pipeline):
        """Tests retrieve method with cache hit."""
        query = "Cached query"
        cached_result = [{"text": "Cached content", "score": 0.9, "metadata": {}}]
        
        cache_key = f"retrieve:{hashlib.md5(f'{query}:5'.encode()).hexdigest()}"
        rag_pipeline.cache.get.return_value = cached_result
        
        result = rag_pipeline.retrieve(query)
        
        rag_pipeline.cache.get.assert_called_once_with(cache_key)
        mock_generate_embeddings.assert_not_called()
        rag_pipeline.faiss_store.search.assert_not_called()
        
        assert result == cached_result
        rag_pipeline.logger.info.assert_any_call("Found cached retrieval result for query")
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_retrieve_embedding_generation_failure(self, mock_generate_embeddings, rag_pipeline):
        """Tests retrieve when embedding generation fails."""
        query = "Failed query"
        mock_generate_embeddings.return_value = None
        
        result = rag_pipeline.retrieve(query)
        
        assert result == []
        rag_pipeline.logger.error.assert_called_with("Failed to generate query embedding.")
        rag_pipeline.faiss_store.search.assert_not_called()
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    def test_retrieve_search_exception(self, mock_generate_embeddings, rag_pipeline, mock_faiss_store):
        """Tests retrieve when vector store search throws exception."""
        query = "Error query"
        mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_faiss_store.search.side_effect = Exception("Search failed")
        
        result = rag_pipeline.retrieve(query)
        
        assert result == []
        rag_pipeline.logger.error.assert_called_with("Error during document retrieval: Search failed")
    
    def test_retrieve_cache_set_on_success(self, rag_pipeline, mock_faiss_store):
        """Tests that successful retrieval results are cached."""
        query = "Cache test query"
        
        with patch('src.rag.rag_pipeline.generate_embeddings') as mock_gen:
            mock_gen.return_value = [[0.1, 0.2, 0.3]]
            mock_faiss_store.search.return_value = [
                ("Content", 0.9, {"source": "test.pdf"})
            ]
            rag_pipeline.cache.get.return_value = None  # No cache hit
            
            result = rag_pipeline.retrieve(query)
            
            cache_key = f"retrieve:{hashlib.md5(f'{query}:5'.encode()).hexdigest()}"
            rag_pipeline.cache.set.assert_called_once_with(cache_key, result)
    
    def test_retrieve_no_cache_when_disabled(self, mock_config, mock_faiss_store):
        """Tests retrieve behavior when cache is disabled."""
        mock_config["enable_cache"] = False
        
        with patch('src.rag.rag_pipeline.get_logger'), \
             patch('src.rag.rag_pipeline.get_cache') as mock_get_cache, \
             patch('src.rag.rag_pipeline.get_memory_manager'), \
             patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True), \
             patch('src.rag.rag_pipeline.ollama') as mock_ollama_module, \
             patch('src.rag.rag_pipeline.generate_embeddings') as mock_gen:
            
            mock_get_cache.return_value = None
            mock_gen.return_value = [[0.1, 0.2, 0.3]]
            mock_faiss_store.search.return_value = [("Content", 0.9, {})]
            
            mock_client = MagicMock()
            mock_client.list.return_value = {"models": []}
            mock_ollama_module.Client.return_value = mock_client
            
            pipeline = RAGPipeline(mock_config, mock_faiss_store)
            pipeline.cache = None
            
            result = pipeline.retrieve("Test query")
            
            # Should not attempt cache operations
            assert len(result) == 1


class TestGenerateCacheKey:
    """Tests for _generate_cache_key method."""
    
    def test_generate_cache_key_consistent(self, rag_pipeline):
        """Tests that cache key generation is consistent for same inputs."""
        query = "Test query"
        context = [
            {"text": "Context 1", "metadata": {}},
            {"text": "Context 2", "metadata": {}}
        ]
        
        key1 = rag_pipeline._generate_cache_key(query, context)
        key2 = rag_pipeline._generate_cache_key(query, context)
        
        assert key1 == key2
        assert key1.startswith("generate:")
    
    def test_generate_cache_key_different_inputs(self, rag_pipeline):
        """Tests that different inputs produce different cache keys."""
        query1 = "Test query 1"
        query2 = "Test query 2"
        context = [{"text": "Context", "metadata": {}}]
        
        key1 = rag_pipeline._generate_cache_key(query1, context)
        key2 = rag_pipeline._generate_cache_key(query2, context)
        
        assert key1 != key2
    
    def test_generate_cache_key_empty_context(self, rag_pipeline):
        """Tests cache key generation with empty context."""
        query = "Test query"
        context = []
        
        key = rag_pipeline._generate_cache_key(query, context)
        
        assert isinstance(key, str)
        assert key.startswith("generate:")
    
    def test_generate_cache_key_missing_text_fields(self, rag_pipeline):
        """Tests cache key generation when context items missing text fields."""
        query = "Test query"
        context = [
            {"metadata": {"source": "doc1"}},  # Missing text
            {"text": "Valid text", "metadata": {}}
        ]
        
        key = rag_pipeline._generate_cache_key(query, context)
        
        assert isinstance(key, str)
        assert key.startswith("generate:")


class TestGenerate:
    """Tests for generate method."""
    
    def test_generate_success(self, rag_pipeline, sample_retrieved_context):
        """Tests successful answer generation."""
        query = "What is RAG?"
        expected_answer = "RAG is a technique that combines retrieval and generation..."
        
        rag_pipeline.ollama_client.chat.return_value = {
            "message": {"content": expected_answer}
        }
        rag_pipeline.cache.get.return_value = None
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_future = MagicMock()
            mock_future.result.return_value = expected_answer.strip()
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            result = rag_pipeline.generate(query, sample_retrieved_context)
        
        assert result == expected_answer.strip()
        rag_pipeline.logger.info.assert_any_call("Generating answer using Ollama LLM.")
    
    def test_generate_no_ollama_client(self, rag_pipeline, sample_retrieved_context):
        """Tests generate when Ollama client is None."""
        rag_pipeline.ollama_client = None
        query = "Test query"
        
        result = rag_pipeline.generate(query, sample_retrieved_context)
        
        expected_message = "Üzgünüm, Ollama istemcisi mevcut olmadığından cevap oluşturulamıyor."
        assert result == expected_message
    
    def test_generate_with_cache_hit(self, rag_pipeline, sample_retrieved_context):
        """Tests generate with cached result."""
        query = "Cached query"
        cached_answer = "This is a cached answer"
        
        cache_key = rag_pipeline._generate_cache_key(query, sample_retrieved_context)
        rag_pipeline.cache.get.return_value = cached_answer
        
        result = rag_pipeline.generate(query, sample_retrieved_context)
        
        assert result == cached_answer
        rag_pipeline.logger.info.assert_any_call("Found cached answer for query")
        # Should not call Ollama
        rag_pipeline.ollama_client.chat.assert_not_called()
    
    def test_generate_context_truncation(self, rag_pipeline):
        """Tests context truncation when content is too long."""
        query = "Test query"
        long_context = [{"text": "A" * 5000, "metadata": {}}]  # Very long context
        
        rag_pipeline.cache.get.return_value = None
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_future = MagicMock()
            mock_future.result.return_value = "Generated answer"
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            rag_pipeline.generate(query, long_context)
        
        rag_pipeline.logger.info.assert_any_call("Context truncated due to length limit")
    
    def test_generate_timeout_error(self, rag_pipeline, sample_retrieved_context):
        """Tests generate when Ollama request times out."""
        query = "Timeout query"
        rag_pipeline.cache.get.return_value = None
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_future = MagicMock()
            mock_future.result.side_effect = TimeoutError()
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            result = rag_pipeline.generate(query, sample_retrieved_context)
        
        expected_message = "Üzgünüm, istek zaman aşımına uğradı. Lütfen daha kısa bir soru deneyin."
        assert result == expected_message
        rag_pipeline.logger.error.assert_any_call("Ollama request timed out")
    
    def test_generate_ollama_api_error(self, rag_pipeline, sample_retrieved_context):
        """Tests generate when Ollama API call fails."""
        query = "Error query"
        rag_pipeline.cache.get.return_value = None
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_future = MagicMock()
            mock_future.result.side_effect = Exception("API Error")
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            result = rag_pipeline.generate(query, sample_retrieved_context)
        
        expected_message = "Üzgünüm, cevap oluşturulurken bir hata oluştu."
        assert result == expected_message
        rag_pipeline.logger.error.assert_called()
    
    def test_generate_executor_setup_error(self, rag_pipeline, sample_retrieved_context):
        """Tests generate when ThreadPoolExecutor setup fails."""
        query = "Executor error query"
        rag_pipeline.cache.get.return_value = None
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_executor.side_effect = Exception("Executor setup failed")
            
            result = rag_pipeline.generate(query, sample_retrieved_context)
        
        expected_message = "Üzgünüm, cevap oluşturulurken bir hata oluştu."
        assert result == expected_message
        rag_pipeline.logger.error.assert_called()
    
    def test_generate_cache_set_on_success(self, rag_pipeline, sample_retrieved_context):
        """Tests that successful generation results are cached."""
        query = "Cache test query"
        expected_answer = "Generated answer"
        
        rag_pipeline.cache.get.return_value = None
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_future = MagicMock()
            mock_future.result.return_value = expected_answer
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            result = rag_pipeline.generate(query, sample_retrieved_context)
            
            cache_key = rag_pipeline._generate_cache_key(query, sample_retrieved_context)
            rag_pipeline.cache.set.assert_called_once_with(cache_key, expected_answer)


class TestCallOllama:
    """Tests for _call_ollama method."""
    
    def test_call_ollama_success(self, rag_pipeline):
        """Tests successful Ollama API call."""
        query = "Test query"
        system_prompt = "System prompt"
        user_prompt = "User prompt"
        expected_response = "Ollama response"
        
        rag_pipeline.ollama_client.chat.return_value = {
            "message": {"content": expected_response}
        }
        
        result = rag_pipeline._call_ollama(query, system_prompt, user_prompt)
        
        assert result == expected_response
        rag_pipeline.ollama_client.chat.assert_called_once()
        
        # Verify chat call arguments
        call_args = rag_pipeline.ollama_client.chat.call_args
        assert call_args[1]["model"] == rag_pipeline.config.get("ollama_generation_model")
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][1]["role"] == "user"
        
        # Verify options
        options = call_args[1]["options"]
        assert options["temperature"] == 0.7
        assert options["num_predict"] == 512
        assert options["top_k"] == 40
        assert options["top_p"] == 0.9
        assert options["repeat_penalty"] == 1.1
    
    def test_call_ollama_with_custom_config(self, mock_faiss_store):
        """Tests _call_ollama with custom configuration values."""
        custom_config = {
            "ollama_generation_model": "custom-model",
            "temperature": 0.5,
            "max_tokens": 256,
            "top_k": 20,
            "top_p": 0.8,
            "repeat_penalty": 1.2
        }
        
        with patch('src.rag.rag_pipeline.get_logger'), \
             patch('src.rag.rag_pipeline.get_cache'), \
             patch('src.rag.rag_pipeline.get_memory_manager'), \
             patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True), \
             patch('src.rag.rag_pipeline.ollama') as mock_ollama_module:
            
            mock_client = MagicMock()
            mock_ollama_module.Client.return_value = mock_client
            mock_client.list.return_value = {}
            mock_client.chat.return_value = {"message": {"content": "Response"}}
            
            pipeline = RAGPipeline(custom_config, mock_faiss_store)
            pipeline._call_ollama("query", "system", "user")
            
            call_args = mock_client.chat.call_args
            assert call_args[1]["model"] == "custom-model"
            options = call_args[1]["options"]
            assert options["temperature"] == 0.5
            assert options["num_predict"] == 256
            assert options["top_k"] == 20
            assert options["top_p"] == 0.8
            assert options["repeat_penalty"] == 1.2
    
    def test_call_ollama_strips_response(self, rag_pipeline):
        """Tests that _call_ollama strips whitespace from response."""
        response_with_whitespace = "  \n  Response with whitespace  \n  "
        
        rag_pipeline.ollama_client.chat.return_value = {
            "message": {"content": response_with_whitespace}
        }
        
        result = rag_pipeline._call_ollama("query", "system", "user")
        
        assert result == "Response with whitespace"


class TestFormatResponse:
    """Tests for format_response method."""
    
    def test_format_response_success(self, rag_pipeline, sample_retrieved_context):
        """Tests successful response formatting."""
        answer = "This is the generated answer."
        
        result = rag_pipeline.format_response(answer, sample_retrieved_context)
        
        assert result["answer"] == answer
        assert result["sources"] == sample_retrieved_context
        assert len(result["sources"]) == 2
    
    def test_format_response_empty_sources(self, rag_pipeline):
        """Tests response formatting with empty sources."""
        answer = "Answer without sources"
        sources = []
        
        result = rag_pipeline.format_response(answer, sources)
        
        assert result["answer"] == answer
        assert result["sources"] == []
    
    def test_format_response_preserves_source_structure(self, rag_pipeline):
        """Tests that format_response preserves original source structure."""
        answer = "Test answer"
        complex_sources = [
            {
                "text": "Complex text",
                "score": 0.95,
                "metadata": {
                    "source": "complex.pdf",
                    "page": 1,
                    "section": "Introduction",
                    "custom_field": "custom_value"
                }
            }
        ]
        
        result = rag_pipeline.format_response(answer, complex_sources)
        
        assert result["sources"] == complex_sources
        assert result["sources"][0]["metadata"]["custom_field"] == "custom_value"


class TestExecute:
    """Tests for execute method (full pipeline)."""
    
    @patch('src.rag.rag_pipeline.time.time')
    def test_execute_success(self, mock_time, rag_pipeline):
        """Tests successful full pipeline execution."""
        query = "What is RAG?"
        mock_time.side_effect = [1000.0, 1005.5]  # 5.5 second execution
        
        # Mock retrieve
        retrieved_context = [
            {"text": "RAG content", "score": 0.9, "metadata": {"source": "doc.pdf"}}
        ]
        
        # Mock generate
        generated_answer = "RAG is a technique..."
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = retrieved_context
            mock_generate.return_value = generated_answer
            
            result = rag_pipeline.execute(query)
        
        mock_retrieve.assert_called_once_with(query, top_k=5)
        mock_generate.assert_called_once_with(query, retrieved_context)
        
        assert result["answer"] == generated_answer
        assert result["sources"] == retrieved_context
        
        # Verify logging
        rag_pipeline.logger.info.assert_any_call("Executing RAG pipeline for query: 'What is RAG?' (top_k=5)")
        rag_pipeline.logger.info.assert_any_call(
            "RAG pipeline execution finished for query: 'What is RAG?' in 5.50s (Memory: 45.0%)"
        )
    
    def test_execute_with_strategy_params(self, rag_pipeline):
        """Tests execute with custom strategy parameters."""
        query = "Custom strategy query"
        strategy_params = {"top_k": 10}
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [{"text": "content", "score": 0.9, "metadata": {}}]
            mock_generate.return_value = "Generated answer"
            
            rag_pipeline.execute(query, strategy_params)
        
        mock_retrieve.assert_called_once_with(query, top_k=10)
    
    def test_execute_no_context_retrieved(self, rag_pipeline):
        """Tests execute when no context is retrieved."""
        query = "No context query"
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve:
            mock_retrieve.return_value = []
            
            result = rag_pipeline.execute(query)
        
        expected_message = "Üzgünüm, bu konuda bilgiye sahip değilim. Lütfen başka bir soru sorun."
        assert result["answer"] == expected_message
        assert result["sources"] == []
        
        rag_pipeline.logger.warning.assert_called_with("No context retrieved for query")
    
    def test_execute_empty_generated_answer(self, rag_pipeline):
        """Tests execute when generated answer is empty."""
        query = "Empty answer query"
        retrieved_context = [{"text": "content", "score": 0.9, "metadata": {}}]
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = retrieved_context
            mock_generate.return_value = ""
            
            result = rag_pipeline.execute(query)
        
        expected_message = "Üzgünüm, bu soruya cevap oluşturamadım."
        assert result["answer"] == expected_message
        assert result["sources"] == retrieved_context
    
    def test_execute_memory_management(self, rag_pipeline):
        """Tests memory management during pipeline execution."""
        query = "Memory test query"
        
        # Setup memory manager responses
        rag_pipeline.memory_manager.get_memory_usage.side_effect = [
            {"percent": 45.0},  # Before execution
            {"percent": 80.0}   # After execution (high usage)
        ]
        rag_pipeline.memory_manager.auto_gc_check.side_effect = [True, True]  # GC performed
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [{"text": "content", "score": 0.9, "metadata": {}}]
            mock_generate.return_value = "Generated answer"
            
            rag_pipeline.execute(query)
        
        # Verify memory checks and GC calls
        assert rag_pipeline.memory_manager.get_memory_usage.call_count == 2
        assert rag_pipeline.memory_manager.auto_gc_check.call_count == 2
        
        rag_pipeline.logger.info.assert_any_call("Garbage collection performed before RAG execution")
    
    def test_execute_pipeline_error_with_cleanup(self, rag_pipeline):
        """Tests execute error handling with cleanup."""
        query = "Error query"
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve:
            mock_retrieve.side_effect = Exception("Retrieval failed")
            
            result = rag_pipeline.execute(query)
        
        expected_message = "Üzgünüm, sorunuzu işlerken bir hata oluştu. Lütfen tekrar deneyin."
        assert result["answer"] == expected_message
        assert result["sources"] == []
        
        # Verify error logging and cleanup
        rag_pipeline.logger.error.assert_called()
        rag_pipeline.memory_manager.auto_gc_check.assert_called()
    
    def test_execute_cleanup_error_handling(self, rag_pipeline):
        """Tests execute when cleanup itself fails."""
        query = "Cleanup error query"
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve:
            mock_retrieve.side_effect = Exception("Retrieval failed")
            rag_pipeline.memory_manager.auto_gc_check.side_effect = Exception("Cleanup failed")
            
            # Should not raise exception despite cleanup failure
            result = rag_pipeline.execute(query)
        
        expected_message = "Üzgünüm, sorunuzu işlerken bir hata oluştu. Lütfen tekrar deneyin."
        assert result["answer"] == expected_message


class TestGetCurrentParameters:
    """Tests for get_current_parameters method."""
    
    def test_get_current_parameters_success(self, rag_pipeline):
        """Tests successful retrieval of current parameters."""
        result = rag_pipeline.get_current_parameters()
        
        expected_keys = [
            "embedding_model", "generation_model", "top_k", "temperature",
            "max_tokens", "top_p", "repeat_penalty", "cache_enabled", "cache_ttl"
        ]
        
        for key in expected_keys:
            assert key in result
        
        assert result["embedding_model"] == rag_pipeline.config.get("ollama_embedding_model")
        assert result["generation_model"] == rag_pipeline.config.get("ollama_generation_model")
        assert result["top_k"] == 5  # Default value
        assert result["temperature"] == 0.7
        assert result["cache_enabled"] is True
    
    def test_get_current_parameters_with_defaults(self, mock_faiss_store):
        """Tests get_current_parameters with minimal config (using defaults)."""
        minimal_config = {"ollama_base_url": "http://localhost:11434"}
        
        with patch('src.rag.rag_pipeline.get_logger'), \
             patch('src.rag.rag_pipeline.get_cache'), \
             patch('src.rag.rag_pipeline.get_memory_manager'), \
             patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True), \
             patch('src.rag.rag_pipeline.ollama') as mock_ollama_module:
            
            mock_client = MagicMock()
            mock_client.list.return_value = {"models": []}
            mock_ollama_module.Client.return_value = mock_client
            
            pipeline = RAGPipeline(minimal_config, mock_faiss_store)
            result = pipeline.get_current_parameters()
        
        # Verify default values are returned
        assert result["top_k"] == 5
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 512
        assert result["top_p"] == 0.9
        assert result["repeat_penalty"] == 1.1
        assert result["cache_enabled"] is True
        assert result["cache_ttl"] == 3600
    
    def test_get_current_parameters_custom_values(self, mock_faiss_store):
        """Tests get_current_parameters with custom configuration values."""
        custom_config = {
            "ollama_embedding_model": "custom-embed",
            "ollama_generation_model": "custom-gen",
            "top_k": 8,
            "temperature": 0.3,
            "max_tokens": 1024,
            "top_p": 0.95,
            "repeat_penalty": 1.05,
            "enable_cache": False,
            "cache_ttl": 7200
        }
        
        with patch('src.rag.rag_pipeline.get_logger'), \
             patch('src.rag.rag_pipeline.get_cache'), \
             patch('src.rag.rag_pipeline.get_memory_manager'), \
             patch('src.rag.rag_pipeline.OLLAMA_AVAILABLE', True), \
             patch('src.rag.rag_pipeline.ollama') as mock_ollama_module:
            
            mock_client = MagicMock()
            mock_client.list.return_value = {"models": []}
            mock_ollama_module.Client.return_value = mock_client
            
            pipeline = RAGPipeline(custom_config, mock_faiss_store)
            result = pipeline.get_current_parameters()
        
        assert result["embedding_model"] == "custom-embed"
        assert result["generation_model"] == "custom-gen"
        assert result["top_k"] == 8
        assert result["temperature"] == 0.3
        assert result["max_tokens"] == 1024
        assert result["top_p"] == 0.95
        assert result["repeat_penalty"] == 1.05
        assert result["cache_enabled"] is False
        assert result["cache_ttl"] == 7200


class TestRAGPipelineIntegration:
    """Integration tests for RAGPipeline workflows."""
    
    @patch('src.rag.rag_pipeline.generate_embeddings')
    @patch('src.rag.rag_pipeline.time.time')
    def test_full_pipeline_workflow(self, mock_time, mock_generate_embeddings, rag_pipeline, mock_faiss_store):
        """Tests complete pipeline workflow from query to response."""
        query = "What is machine learning?"
        mock_time.side_effect = [2000.0, 2003.2]  # 3.2 second execution
        
        # Setup mocks for complete workflow
        mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_faiss_store.search.return_value = [
            ("Machine learning is...", 0.92, {"source": "ml.pdf", "page": 1}),
            ("AI algorithms learn...", 0.85, {"source": "ai.pdf", "page": 2})
        ]
        
        expected_answer = "Machine learning is a subset of artificial intelligence..."
        rag_pipeline.cache.get.return_value = None  # No cache hits
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_future = MagicMock()
            mock_future.result.return_value = expected_answer
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            result = rag_pipeline.execute(query)
        
        # Verify complete workflow
        assert result["answer"] == expected_answer
        assert len(result["sources"]) == 2
        assert result["sources"][0]["text"] == "Machine learning is..."
        assert result["sources"][1]["text"] == "AI algorithms learn..."
        
        # Verify all components were called
        mock_generate_embeddings.assert_called_once()
        mock_faiss_store.search.assert_called_once()
        rag_pipeline.cache.set.assert_called()  # Results cached
        
        # Verify logging of complete workflow
        rag_pipeline.logger.info.assert_any_call("Executing RAG pipeline for query: 'What is machine learning?' (top_k=5)")
        rag_pipeline.logger.info.assert_any_call("Retrieved 2 chunks from vector store.")
    
    def test_pipeline_with_different_strategies(self, rag_pipeline):
        """Tests pipeline execution with different strategy parameters."""
        query = "Strategy test query"
        strategies_to_test = [
            {"top_k": 3},      # precise_retrieval style
            {"top_k": 10},     # broad_context_retrieval style
            {"top_k": 7}       # hybrid_search style
        ]
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [{"text": "content", "score": 0.9, "metadata": {}}]
            mock_generate.return_value = "Generated answer"
            
            for strategy in strategies_to_test:
                result = rag_pipeline.execute(query, strategy)
                
                assert result["answer"] == "Generated answer"
                mock_retrieve.assert_called_with(query, top_k=strategy["top_k"])
    
    def test_end_to_end_caching_behavior(self, rag_pipeline):
        """Tests end-to-end caching behavior across multiple calls."""
        query = "Caching test query"
        context = [{"text": "Cached content", "score": 0.9, "metadata": {}}]
        
        # First call - no cache
        rag_pipeline.cache.get.side_effect = [None, None]  # retrieve cache miss, generate cache miss
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = context
            mock_generate.return_value = "Generated answer"
            
            result1 = rag_pipeline.execute(query)
        
        # Second call - cache hits
        rag_pipeline.cache.get.side_effect = [context, "Cached answer"]
        
        result2 = rag_pipeline.execute(query)
        
        # Verify both results are the same
        assert result1["answer"] == "Generated answer"
        assert result2["answer"] == "Cached answer"
        
        # Verify cache was used properly
        assert rag_pipeline.cache.set.call_count == 2  # retrieve + generate results cached


class TestRAGPipelineErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_retrieve_with_corrupted_embeddings(self, rag_pipeline):
        """Tests retrieve when embeddings are corrupted/malformed."""
        query = "Corrupted embeddings query"
        
        with patch('src.rag.rag_pipeline.generate_embeddings') as mock_gen:
            # Test various corrupted embedding scenarios
            corrupted_cases = [
                [],           # Empty embeddings
                None,         # None embeddings
                [[]],         # Empty inner array
                [[None]],     # None values in embeddings
            ]
            
            for corrupted_embedding in corrupted_cases:
                mock_gen.return_value = corrupted_embedding if corrupted_embedding != [] else None
                
                result = rag_pipeline.retrieve(query)
                
                assert result == []
                if corrupted_embedding != [] and corrupted_embedding is not None:
                    rag_pipeline.faiss_store.search.assert_called_with(corrupted_embedding, 5)
    
    def test_generate_with_malformed_context(self, rag_pipeline):
        """Tests generate with malformed context data."""
        query = "Malformed context query"
        malformed_contexts = [
            [{"invalid": "no text field"}],                    # Missing text field
            [{"text": None, "metadata": {}}],                  # None text
            [{"text": "", "score": "invalid", "metadata": {}}], # Invalid score type
            [{"text": 12345, "metadata": {}}],                 # Non-string text
        ]
        
        rag_pipeline.cache.get.return_value = None
        
        for context in malformed_contexts:
            with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                mock_future = MagicMock()
                mock_future.result.return_value = "Handled gracefully"
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
                
                # Should handle malformed context gracefully
                result = rag_pipeline.generate(query, context)
                
                assert isinstance(result, str)
    
    def test_execute_with_invalid_strategy_params(self, rag_pipeline):
        """Tests execute with invalid strategy parameters."""
        query = "Invalid params query"
        invalid_params_cases = [
            {"top_k": -1},          # Negative top_k
            {"top_k": 0},           # Zero top_k
            {"top_k": "invalid"},   # String top_k
            {"top_k": None},        # None top_k
            {"invalid_param": 123}, # Unknown parameter
        ]
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [{"text": "content", "score": 0.9, "metadata": {}}]
            mock_generate.return_value = "Generated answer"
            
            for params in invalid_params_cases:
                # Should handle invalid params gracefully or use defaults
                try:
                    result = rag_pipeline.execute(query, params)
                    assert isinstance(result, dict)
                    assert "answer" in result
                except (TypeError, ValueError):
                    # Some invalid params may raise exceptions, which is acceptable
                    pass
    
    def test_memory_manager_failure_handling(self, rag_pipeline):
        """Tests pipeline behavior when memory manager fails."""
        query = "Memory manager error query"
        
        # Memory manager throws errors
        rag_pipeline.memory_manager.get_memory_usage.side_effect = Exception("Memory check failed")
        rag_pipeline.memory_manager.auto_gc_check.side_effect = Exception("GC failed")
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [{"text": "content", "score": 0.9, "metadata": {}}]
            mock_generate.return_value = "Generated answer"
            
            # Pipeline should continue working despite memory manager errors
            result = rag_pipeline.execute(query)
            
            assert result["answer"] == "Generated answer"
    
    def test_ollama_client_recovery_after_failure(self, rag_pipeline):
        """Tests Ollama client behavior after connection failure."""
        query = "Recovery test query"
        context = [{"text": "Test content", "score": 0.9, "metadata": {}}]
        
        # First call fails
        rag_pipeline.ollama_client = None
        result1 = rag_pipeline.generate(query, context)
        
        expected_error_msg = "Üzgünüm, Ollama istemcisi mevcut olmadığından cevap oluşturulamıyor."
        assert result1 == expected_error_msg
        
        # Simulate client recovery
        rag_pipeline.ollama_client = MagicMock()
        rag_pipeline.ollama_client.chat.return_value = {"message": {"content": "Recovered response"}}
        rag_pipeline.cache.get.return_value = None
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_future = MagicMock()
            mock_future.result.return_value = "Recovered response"
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            result2 = rag_pipeline.generate(query, context)
        
        assert result2 == "Recovered response"


class TestRAGPipelineEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_retrieve_with_extreme_top_k_values(self, rag_pipeline):
        """Tests retrieve with extreme top_k values."""
        query = "Extreme top_k test"
        
        with patch('src.rag.rag_pipeline.generate_embeddings') as mock_gen:
            mock_gen.return_value = [[0.1, 0.2, 0.3]]
            rag_pipeline.faiss_store.search.return_value = []
            
            extreme_values = [0, 1, 1000, 999999]
            
            for top_k in extreme_values:
                result = rag_pipeline.retrieve(query, top_k=top_k)
                
                assert isinstance(result, list)
                rag_pipeline.faiss_store.search.assert_called_with([[0.1, 0.2, 0.3]], top_k)
    
    def test_generate_with_extreme_context_sizes(self, rag_pipeline):
        """Tests generate with extremely large and small context sizes."""
        query = "Extreme context test"
        
        # Very large context
        huge_context = [{"text": "A" * 10000, "metadata": {}} for _ in range(100)]
        rag_pipeline.cache.get.return_value = None
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_future = MagicMock()
            mock_future.result.return_value = "Handled large context"
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            result = rag_pipeline.generate(query, huge_context)
            
            assert isinstance(result, str)
            # Should log context truncation
            rag_pipeline.logger.info.assert_any_call("Context truncated due to length limit")
        
        # Empty context
        empty_context = []
        result = rag_pipeline.generate(query, empty_context)
        assert isinstance(result, str)
    
    def test_execute_with_unicode_and_special_characters(self, rag_pipeline):
        """Tests execute with Unicode and special characters in query."""
        special_queries = [
            "🤖 AI nedir? Türkçe açıklayabilir misin?",
            "Что такое искусственный интеллект?",
            "机器学习是什么？",
            "¿Qué es el aprendizaje automático?",
            "query with\nnewlines\tand\ttabs",
            "query with \"quotes\" and 'apostrophes'",
            "query with <tags> and & symbols",
            "",  # Empty query
            " " * 1000,  # Whitespace-only query
        ]
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [{"text": "content", "score": 0.9, "metadata": {}}]
            mock_generate.return_value = "Generated answer"
            
            for query in special_queries:
                result = rag_pipeline.execute(query)
                
                assert isinstance(result, dict)
                assert "answer" in result
                assert "sources" in result
    
    def test_cache_key_collision_resistance(self, rag_pipeline):
        """Tests that cache keys are resistant to collisions."""
        # Test queries that might produce similar hashes
        similar_queries = [
            ("query1", [{"text": "context1", "metadata": {}}]),
            ("query2", [{"text": "context1", "metadata": {}}]),
            ("query1", [{"text": "context2", "metadata": {}}]),
            ("query1", [{"text": "context1", "metadata": {"extra": "data"}}]),
        ]
        
        cache_keys = []
        for query, context in similar_queries:
            key = rag_pipeline._generate_cache_key(query, context)
            cache_keys.append(key)
        
        # All cache keys should be unique
        assert len(cache_keys) == len(set(cache_keys))
    
    def test_concurrent_pipeline_execution_safety(self, rag_pipeline):
        """Tests pipeline thread safety with concurrent-like execution simulation."""
        queries = [f"Concurrent query {i}" for i in range(5)]
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [{"text": "content", "score": 0.9, "metadata": {}}]
            mock_generate.return_value = "Generated answer"
            
            # Simulate concurrent execution by rapid sequential calls
            results = []
            for query in queries:
                result = rag_pipeline.execute(query)
                results.append(result)
            
            # All results should be valid and independent
            assert len(results) == 5
            for result in results:
                assert result["answer"] == "Generated answer"
                assert len(result["sources"]) == 1


class TestRAGPipelineLogging:
    """Tests for logging behavior verification."""
    
    def test_comprehensive_logging_during_execution(self, rag_pipeline):
        """Tests that all major operations are properly logged."""
        query = "Comprehensive logging test"
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [{"text": "content", "score": 0.9, "metadata": {}}]
            mock_generate.return_value = "Generated answer"
            
            rag_pipeline.execute(query)
        
        # Verify key logging points
        expected_log_messages = [
            "Executing RAG pipeline for query:",
            "Retrieved 1 chunks from vector store.",
            "Generating answer using Ollama LLM.",
            "RAG pipeline execution finished for query:"
        ]
        
        info_calls = [call[0][0] for call in rag_pipeline.logger.info.call_args_list]
        
        for expected_msg in expected_log_messages:
            assert any(expected_msg in call for call in info_calls), f"Expected log message not found: {expected_msg}"
    
    def test_error_logging_during_failures(self, rag_pipeline):
        """Tests proper error logging during various failure scenarios."""
        query = "Error logging test"
        
        # Test retrieval error logging
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve:
            mock_retrieve.side_effect = Exception("Retrieval error")
            
            rag_pipeline.execute(query)
            
            rag_pipeline.logger.error.assert_called()
            error_calls = [str(call) for call in rag_pipeline.logger.error.call_args_list]
            assert any("Error in RAG pipeline execution" in call for call in error_calls)
    
    def test_debug_logging_levels(self, rag_pipeline):
        """Tests debug-level logging for memory management."""
        query = "Debug logging test"
        
        # Setup memory manager to return specific values
        rag_pipeline.memory_manager.get_memory_usage.return_value = {"percent": 67.5}
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [{"text": "content", "score": 0.9, "metadata": {}}]
            mock_generate.return_value = "Generated answer"
            
            rag_pipeline.execute(query)
        
        # Verify debug logging was called for memory usage
        rag_pipeline.logger.debug.assert_called()
        debug_calls = [call[0][0] for call in rag_pipeline.logger.debug.call_args_list]
        assert any("Memory usage before execution: 67.5%" in call for call in debug_calls)


class TestRAGPipelinePerformance:
    """Tests for performance and resource management."""
    
    @patch('src.rag.rag_pipeline.time.time')
    def test_execution_time_tracking(self, mock_time, rag_pipeline):
        """Tests accurate execution time tracking."""
        query = "Performance timing test"
        
        # Mock time progression
        start_time = 1234567890.0
        end_time = start_time + 2.75  # 2.75 seconds
        mock_time.side_effect = [start_time, end_time]
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [{"text": "content", "score": 0.9, "metadata": {}}]
            mock_generate.return_value = "Generated answer"
            
            rag_pipeline.execute(query)
        
        # Verify timing was logged correctly
        info_calls = [call[0][0] for call in rag_pipeline.logger.info.call_args_list]
        timing_logs = [call for call in info_calls if "2.75s" in call]
        assert len(timing_logs) == 1
    
    def test_memory_usage_monitoring(self, rag_pipeline):
        """Tests memory usage monitoring throughout pipeline execution."""
        query = "Memory monitoring test"
        
        # Setup progressive memory usage
        rag_pipeline.memory_manager.get_memory_usage.side_effect = [
            {"percent": 30.0},  # Before execution
            {"percent": 85.0}   # After execution (high usage)
        ]
        rag_pipeline.memory_manager.auto_gc_check.side_effect = [False, True]  # GC triggered after
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [{"text": "content", "score": 0.9, "metadata": {}}]
            mock_generate.return_value = "Generated answer"
            
            rag_pipeline.execute(query)
        
        # Verify memory checks were performed
        assert rag_pipeline.memory_manager.get_memory_usage.call_count == 2
        assert rag_pipeline.memory_manager.auto_gc_check.call_count == 2
        
        # Verify logging
        debug_calls = [call[0][0] for call in rag_pipeline.logger.debug.call_args_list]
        assert any("Memory usage before execution: 30.0%" in call for call in debug_calls)
    
    def test_cache_performance_optimization(self, rag_pipeline):
        """Tests that caching provides performance optimization."""
        query = "Cache performance test"
        context = [{"text": "cached content", "score": 0.9, "metadata": {}}]
        
        # First call - populate cache
        rag_pipeline.cache.get.return_value = None
        
        with patch.object(rag_pipeline, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline, 'generate') as mock_generate:
            
            mock_retrieve.return_value = context
            mock_generate.return_value = "Generated answer"
            
            result1 = rag_pipeline.execute(query)
        
        # Verify cache was populated
        assert rag_pipeline.cache.set.call_count == 2  # retrieve + generate
        
        # Second call - use cache
        rag_pipeline.cache.get.side_effect = [context, "Cached answer"]
        mock_retrieve.reset_mock()
        mock_generate.reset_mock()
        
        result2 = rag_pipeline.execute(query)
        
        # Verify cached results were used (no expensive operations)
        mock_retrieve.assert_not_called()
        mock_generate.assert_not_called()
        
        assert result2["answer"] == "Cached answer"
        assert result2["sources"] == context