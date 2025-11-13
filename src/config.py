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

# Initialize directories on import
setup_directories()
