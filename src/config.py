"""
RAG3 Configuration Management
Cloud-Ready Database Configuration with Fallback Support
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

class RAGConfig:
    """Centralized configuration management with cloud support"""
    
    def __init__(self):
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.is_cloud = self.environment == 'production'
        self.logger = logging.getLogger(__name__)
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        
        # Database Configuration
        self._setup_database_config()
        
        # Storage Configuration
        self._setup_storage_config()
        
        # Model Configuration
        self._setup_model_config()
        
        # Microservices Configuration
        self._setup_microservices_config()
        
        # Phase 1: Advanced Semantic Chunking Configuration
        self._setup_semantic_chunking_config()
        
        # Model Definitions
        self._setup_model_definitions()
        
    def _setup_database_config(self):
        """Setup database configuration with cloud support"""
        
        if self.is_cloud:
            # Cloud SQL or external database for production
            self.database_config = {
                'type': os.getenv('DB_TYPE', 'sqlite'),  # 'postgresql', 'sqlite'
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT', '5432'),
                'name': os.getenv('DB_NAME', 'rag3_sessions'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'ssl': os.getenv('DB_SSL', 'true') == 'true',
                # Cloud SQL connection for Google Cloud Run
                'cloud_sql_connection': os.getenv('CLOUD_SQL_CONNECTION_NAME'),
                # Fallback to Cloud Storage-backed SQLite
                'fallback_storage': 'gcs',  # 'gcs', 'local'
                'storage_bucket': os.getenv('GCS_BUCKET_NAME', 'rag3-data'),
                'storage_path': 'databases/'
            }
        else:
            # Local SQLite for development
            self.database_config = {
                'type': 'sqlite',
                'path': 'data/analytics/sessions.db',
                'fallback_storage': 'local'
            }
    
    def _setup_storage_config(self):
        """Setup file storage configuration"""
        
        if self.is_cloud:
            self.storage_config = {
                'type': 'gcs',  # Google Cloud Storage
                'bucket': os.getenv('GCS_BUCKET_NAME', 'rag3-data'),
                'vector_path': 'vector_stores/',
                'backup_path': 'backups/',
                'export_path': 'exports/',
                'markdown_path': 'markdown/',
                'temp_path': '/tmp/rag3/',  # Cloud Run temp space
            }
        else:
            self.storage_config = {
                'type': 'local',
                'vector_path': 'data/vector_db/sessions/',
                'backup_path': 'data/backups/sessions/',
                'export_path': 'data/exports/sessions/',
                'markdown_path': 'data/markdown/',
                'temp_path': 'data/temp/'
            }
    
    def _setup_model_config(self):
        """Setup model and caching configuration"""
        
        self.model_config = {
            'embedding_provider': os.getenv('EMBEDDING_PROVIDER', 'sentence_transformers'),
            'llm_provider': os.getenv('LLM_PROVIDER', 'groq'),
            'cache_models': os.getenv('CACHE_MODELS', 'true') == 'true',
            'model_cache_path': '/app/models' if self.is_cloud else 'models/',
            'max_memory_mb': int(os.getenv('MARKER_MAX_MEMORY_MB', '3500')),
            'timeout_seconds': int(os.getenv('MARKER_TIMEOUT_SECONDS', '900')),
            'embedding_batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', '25')),  # Default batch size of 25
            
            # CRAG Retrieval Evaluator Settings (2025 Best Practice)
            'enable_retrieval_evaluation': os.getenv('ENABLE_RETRIEVAL_EVALUATION', 'true').lower() == 'true',
            'retrieval_evaluator_model': os.getenv(
                'RETRIEVAL_EVALUATOR_MODEL',
                'cross-encoder/ms-marco-MiniLM-L-6-v2'
            ),
            'retrieval_correct_threshold': float(os.getenv('RETRIEVAL_CORRECT_THRESHOLD', '0.7')),
            'retrieval_incorrect_threshold': float(os.getenv('RETRIEVAL_INCORRECT_THRESHOLD', '0.3')),
            'retrieval_filter_threshold': float(os.getenv('RETRIEVAL_FILTER_THRESHOLD', '0.5'))
        }
    
    def _setup_microservices_config(self):
        """Setup microservices configuration"""
        
        # Cloud vs Local environment handling
        if self.is_cloud:
            # Production: Use full Cloud Run service URLs
            default_pdf_url = 'https://pdf-processor-service-url.run.app'
            default_model_inference_url = 'https://model-inference-service-url.run.app'
        else:
            # Local development: Use docker-compose service names
            default_pdf_url = 'http://pdf-processor:8001'
            default_model_inference_url = 'http://model-inferencer:8002'
            
        self.microservices_config = {
            'pdf_processor_url': os.getenv('PDF_PROCESSOR_URL', default_pdf_url),
            'model_inference_url': os.getenv('MODEL_INFERENCE_URL', default_model_inference_url)
        }

    def _setup_semantic_chunking_config(self):
        """Setup Phase 1 Advanced Semantic Chunking Architecture Configuration"""
        
        self.semantic_chunking_config = {
            # Core Chunking Parameters
            'chunk_size': int(os.getenv('CHUNK_SIZE', '1000')),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '200')),
            'min_chunk_size': int(os.getenv('MIN_CHUNK_SIZE', '100')),
            'max_chunk_size': int(os.getenv('MAX_CHUNK_SIZE', '1500')),
            
            # Semantic Analysis Settings
            'default_embedding_model': os.getenv(
                'SEMANTIC_EMBEDDING_MODEL',
                'paraphrase-multilingual-MiniLM-L12-v2'
            ),
            'embedding_cache_size': int(os.getenv('EMBEDDING_CACHE_SIZE', '10000')),
            'batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', '32')),
            'use_embedding_refinement': os.getenv('USE_EMBEDDING_REFINEMENT', 'true').lower() == 'true',
            
            # Language Support
            'supported_languages': ['tr', 'en', 'auto'],
            'default_language': os.getenv('DEFAULT_LANGUAGE', 'auto'),
            'turkish_sentence_patterns_enabled': os.getenv('TURKISH_PATTERNS_ENABLED', 'true').lower() == 'true',
            
            # Advanced Chunk Validation Configuration
            'validation_enabled': os.getenv('CHUNK_VALIDATION_ENABLED', 'true').lower() == 'true',
            'semantic_coherence_threshold': float(os.getenv('SEMANTIC_COHERENCE_THRESHOLD', '0.75')),
            'topic_consistency_threshold': float(os.getenv('TOPIC_CONSISTENCY_THRESHOLD', '0.70')),
            'length_quality_threshold': float(os.getenv('LENGTH_QUALITY_THRESHOLD', '0.80')),
            'overall_quality_threshold': float(os.getenv('OVERALL_QUALITY_THRESHOLD', '0.70')),
            
            # Validation Weights
            'coherence_weight': float(os.getenv('COHERENCE_WEIGHT', '0.40')),  # 40%
            'topic_weight': float(os.getenv('TOPIC_WEIGHT', '0.35')),          # 35%
            'length_weight': float(os.getenv('LENGTH_WEIGHT', '0.25')),        # 25%
            
            # AST Markdown Parser Configuration
            'ast_parser_enabled': os.getenv('AST_PARSER_ENABLED', 'true').lower() == 'true',
            'header_hierarchy_preservation': os.getenv('HEADER_HIERARCHY_PRESERVATION', 'true').lower() == 'true',
            'table_semantic_context': os.getenv('TABLE_SEMANTIC_CONTEXT', 'true').lower() == 'true',
            'code_block_protection': os.getenv('CODE_BLOCK_PROTECTION', 'true').lower() == 'true',
            'math_formula_protection': os.getenv('MATH_FORMULA_PROTECTION', 'true').lower() == 'true',
            'cross_reference_resolution': os.getenv('CROSS_REFERENCE_RESOLUTION', 'true').lower() == 'true',
            
            # Performance Optimization
            'cache_enabled': os.getenv('SEMANTIC_CACHE_ENABLED', 'true').lower() == 'true',
            'cache_ttl_hours': int(os.getenv('SEMANTIC_CACHE_TTL_HOURS', '24')),
            'memory_limit_mb': int(os.getenv('SEMANTIC_MEMORY_LIMIT_MB', '2048')),
            'parallel_processing': os.getenv('PARALLEL_PROCESSING', 'true').lower() == 'true',
            'max_workers': int(os.getenv('MAX_WORKERS', '4')),
            
            # Fallback Strategy Configuration
            'fallback_strategy': os.getenv('FALLBACK_STRATEGY', 'lightweight'),  # lightweight, ast_markdown, markdown, adaptive
            'enable_fallback_logging': os.getenv('ENABLE_FALLBACK_LOGGING', 'true').lower() == 'true',
            
            # Turkish Language Specific Optimizations
            'turkish_abbreviations': [
                'Dr.', 'Prof.', 'Doç.', 'Yrd.', 'Öğr.', 'Arş.', 'Uzm.',
                'vs.', 'vb.', 'örn.', 'yani', 'bkz.', 'bk.'
            ],
            'turkish_sentence_endings': ['.', '!', '?', '...'],
            'turkish_conjunctions': [
                've', 'veya', 'ya da', 'ama', 'ancak', 'fakat', 'lakin',
                'çünkü', 'zira', 'nitekim', 'dolayısıyla', 'sonuç olarak'
            ],
            
            # Quality Metrics Thresholds
            'min_sentences_per_chunk': int(os.getenv('MIN_SENTENCES_PER_CHUNK', '2')),
            'max_sentences_per_chunk': int(os.getenv('MAX_SENTENCES_PER_CHUNK', '20')),
            'average_sentence_length_threshold': int(os.getenv('AVG_SENTENCE_LENGTH_THRESHOLD', '80')),
            
            # Error Handling
            'max_retry_attempts': int(os.getenv('MAX_RETRY_ATTEMPTS', '3')),
            'error_logging_enabled': os.getenv('ERROR_LOGGING_ENABLED', 'true').lower() == 'true',
            'graceful_degradation': os.getenv('GRACEFUL_DEGRADATION', 'true').lower() == 'true'
        }

    def _setup_model_definitions(self):
        """Setup model definitions for cloud and local providers"""
        
        # Cloud models (Groq) - all confirmed working via API testing 2025-11-02
        self.cloud_models = {
            "llama-3.1-8b-instant": {
                "name": "Llama 3.1 8B Instant",
                "description": "Fast and efficient Llama model for general tasks",
                "provider": "groq",
                "size": "8B parameters",
                "performance": "High",
                "language": "Multi-language"
            },
            "llama-3.3-70b-versatile": {
                "name": "Llama 3.3 70B Versatile",
                "description": "Large versatile model for complex reasoning",
                "provider": "groq",
                "size": "70B parameters",
                "performance": "Very High",
                "language": "Multi-language"
            },
            "openai/gpt-oss-20b": {
                "name": "GPT OSS 20B",
                "description": "Open source GPT-style model",
                "provider": "groq",
                "size": "20B parameters",
                "performance": "High",
                "language": "Multi-language"
            },
            "qwen/qwen3-32b": {
                "name": "Qwen 3 32B",
                "description": "Advanced Chinese-English bilingual model",
                "provider": "groq",
                "size": "32B parameters",
                "performance": "High",
                "language": "Chinese/English"
            }
        }
        
        # OpenRouter models - FREE models only for cost-effective usage
        self.openrouter_models = {
            "meta-llama/llama-3.1-8b-instruct:free": {
                "name": "Llama 3.1 8B (Free)",
                "description": "Meta's instruction-tuned Llama model - Free tier",
                "provider": "openrouter",
                "size": "8B parameters",
                "performance": "High",
                "language": "Multi-language"
            },
            "mistralai/mistral-7b-instruct:free": {
                "name": "Mistral 7B (Free)",
                "description": "Mistral's efficient instruction-following model - Free tier",
                "provider": "openrouter",
                "size": "7B parameters",
                "performance": "High",
                "language": "Multi-language"
            },
            "microsoft/phi-3-mini-4k-instruct:free": {
                "name": "Phi-3 Mini (Free)",
                "description": "Microsoft's compact and efficient model - Free tier",
                "provider": "openrouter",
                "size": "3.8B parameters",
                "performance": "Medium-High",
                "language": "Multi-language"
            },
            "google/gemma-2-9b-it:free": {
                "name": "Gemma 2 9B (Free)",
                "description": "Google's open model with good performance - Free tier",
                "provider": "openrouter",
                "size": "9B parameters",
                "performance": "High",
                "language": "Multi-language"
            },
            "nousresearch/hermes-3-llama-3.1-8b:free": {
                "name": "Hermes 3 Llama 8B (Free)",
                "description": "Nous Research's fine-tuned Llama model - Free tier",
                "provider": "openrouter",
                "size": "8B parameters",
                "performance": "High",
                "language": "Multi-language"
            }
        }
        
        # Merge OpenRouter models into cloud models
        self.cloud_models.update(self.openrouter_models)
        
        # Local Ollama models
        self.available_models = {
            "mistral:7b": {
                "name": "Mistral 7B",
                "description": "Efficient local model for general tasks",
                "provider": "ollama",
                "size": "7B parameters",
                "performance": "Medium",
                "language": "Multi-language"
            },
            "llama2:13b": {
                "name": "Llama 2 13B", 
                "description": "Meta's Llama 2 model for advanced reasoning",
                "provider": "ollama",
                "size": "13B parameters",
                "performance": "High",
                "language": "Multi-language"
            }
        }
    
    def get_database_url(self) -> str:
        """Get database connection URL/path"""
        
        if self.database_config['type'] == 'postgresql':
            if self.database_config.get('cloud_sql_connection'):
                # Cloud SQL via Unix socket
                return (f"postgresql://{self.database_config['user']}:"
                       f"{self.database_config['password']}@/"
                       f"{self.database_config['name']}"
                       f"?host=/cloudsql/{self.database_config['cloud_sql_connection']}")
            else:
                # Standard PostgreSQL connection
                return (f"postgresql://{self.database_config['user']}:"
                       f"{self.database_config['password']}@"
                       f"{self.database_config['host']}:"
                       f"{self.database_config['port']}/"
                       f"{self.database_config['name']}")
        else:
            # SQLite - use cloud storage path if in production
            if self.is_cloud and self.database_config['fallback_storage'] == 'gcs':
                # This will be handled by the storage manager
                return 'cloud_storage'  
            else:
                return self.database_config.get('path', 'data/analytics/sessions.db')
    
    def get_storage_path(self, path_type: str) -> str:
        """Get storage path for different resource types"""
        return self.storage_config.get(f'{path_type}_path', '')
    
    def is_cloud_environment(self) -> bool:
        """Check if running in cloud environment"""
        return self.is_cloud
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging"""
        return {
            'environment': self.environment,
            'is_cloud': self.is_cloud,
            'database_type': self.database_config['type'],
            'storage_type': self.storage_config['type'],
            'embedding_provider': self.model_config['embedding_provider'],
            'llm_provider': self.model_config['llm_provider'],
            'pdf_processor_url': self.microservices_config['pdf_processor_url']
        }

# Global configuration instance
config = RAGConfig()

def get_config():
    """Get the global configuration instance"""
    return config

# Helper functions for backward compatibility
def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    return config.database_config

def get_storage_config() -> Dict[str, Any]:
    """Get storage configuration"""
    return config.storage_config

def get_microservices_config() -> Dict[str, Any]:
    """Get microservices configuration"""
    return config.microservices_config

def get_pdf_processor_url() -> str:
    """Get PDF processor service URL"""
    return config.microservices_config['pdf_processor_url']

def get_model_inference_url() -> str:
    """Get Model Inference service URL"""
    return config.microservices_config['model_inference_url']

def is_cloud_environment() -> bool:
    """Check if running in cloud environment"""
    return config.is_cloud_environment()

def get_database_url() -> str:
    """Get database URL/path"""
    return config.get_database_url()

# Phase 1: Advanced Semantic Chunking Configuration Helper Functions
def get_semantic_chunking_config() -> Dict[str, Any]:
    """Get Phase 1 semantic chunking configuration"""
    return config.semantic_chunking_config

def get_chunk_size() -> int:
    """Get default chunk size"""
    return config.semantic_chunking_config['chunk_size']

def get_chunk_overlap() -> int:
    """Get default chunk overlap"""
    return config.semantic_chunking_config['chunk_overlap']

def get_embedding_model() -> str:
    """Get default embedding model for semantic analysis"""
    return config.semantic_chunking_config['default_embedding_model']

def is_embedding_refinement_enabled() -> bool:
    """Check if embedding-based refinement is enabled"""
    return config.semantic_chunking_config['use_embedding_refinement']

def is_ast_parser_enabled() -> bool:
    """Check if AST markdown parser is enabled"""
    return config.semantic_chunking_config['ast_parser_enabled']

def get_validation_thresholds() -> Dict[str, float]:
    """Get chunk validation quality thresholds"""
    return {
        'semantic_coherence': config.semantic_chunking_config['semantic_coherence_threshold'],
        'topic_consistency': config.semantic_chunking_config['topic_consistency_threshold'],
        'length_quality': config.semantic_chunking_config['length_quality_threshold'],
        'overall_quality': config.semantic_chunking_config['overall_quality_threshold']
    }

def get_validation_weights() -> Dict[str, float]:
    """Get validation metric weights"""
    return {
        'coherence': config.semantic_chunking_config['coherence_weight'],
        'topic': config.semantic_chunking_config['topic_weight'],
        'length': config.semantic_chunking_config['length_weight']
    }

def get_turkish_language_config() -> Dict[str, Any]:
    """Get Turkish language optimization configuration"""
    return {
        'patterns_enabled': config.semantic_chunking_config['turkish_sentence_patterns_enabled'],
        'abbreviations': config.semantic_chunking_config['turkish_abbreviations'],
        'sentence_endings': config.semantic_chunking_config['turkish_sentence_endings'],
        'conjunctions': config.semantic_chunking_config['turkish_conjunctions']
    }

def get_performance_config() -> Dict[str, Any]:
    """Get performance optimization configuration"""
    return {
        'cache_enabled': config.semantic_chunking_config['cache_enabled'],
        'batch_size': config.semantic_chunking_config['batch_size'],
        'memory_limit_mb': config.semantic_chunking_config['memory_limit_mb'],
        'parallel_processing': config.semantic_chunking_config['parallel_processing'],
        'max_workers': config.semantic_chunking_config['max_workers']
    }

def setup_directories():
    """Setup necessary directories for local development"""
    if not config.is_cloud:
        directories = [
            'data/analytics',
            'data/vector_db/sessions',
            'data/backups/sessions',
            'data/exports/sessions',
            'data/markdown',
            'data/temp'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# Global model constants for backward compatibility
CLOUD_MODELS = config.cloud_models
AVAILABLE_MODELS = config.available_models
OLLAMA_BASE_URL = "http://localhost:11434"

# Phase 1: Global semantic chunking constants for backward compatibility
CHUNK_SIZE = config.semantic_chunking_config['chunk_size']
CHUNK_OVERLAP = config.semantic_chunking_config['chunk_overlap']
MIN_CHUNK_SIZE = config.semantic_chunking_config['min_chunk_size']
MAX_CHUNK_SIZE = config.semantic_chunking_config['max_chunk_size']

# Semantic analysis constants
EMBEDDING_MODEL = config.semantic_chunking_config['default_embedding_model']
EMBEDDING_CACHE_SIZE = config.semantic_chunking_config['embedding_cache_size']
BATCH_SIZE = config.semantic_chunking_config['batch_size']

# Validation thresholds
SEMANTIC_COHERENCE_THRESHOLD = config.semantic_chunking_config['semantic_coherence_threshold']
TOPIC_CONSISTENCY_THRESHOLD = config.semantic_chunking_config['topic_consistency_threshold']
OVERALL_QUALITY_THRESHOLD = config.semantic_chunking_config['overall_quality_threshold']

# Turkish language patterns
TURKISH_ABBREVIATIONS = config.semantic_chunking_config['turkish_abbreviations']
TURKISH_SENTENCE_ENDINGS = config.semantic_chunking_config['turkish_sentence_endings']
TURKISH_CONJUNCTIONS = config.semantic_chunking_config['turkish_conjunctions']

# Initialize directories on import
setup_directories()
