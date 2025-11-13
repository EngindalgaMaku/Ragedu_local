"""
Cloud LLM Client - Model Inference Service'e yönlendiren API client.
Artık tüm model çıkarım görevleri Model Inference Service mikroservisi üzerinden yapılır.
"""

import requests
import os
from typing import Dict, Any, Optional, List

# Use absolute imports to avoid relative import issues
try:
    from src.config import get_config, get_model_inference_url
    from src.utils.logger import get_logger
except ImportError:
    # Fallback for when running from different contexts
    from config import get_config, get_model_inference_url
    from utils.logger import get_logger

class CloudLLMClient:
    """Model Inference Service'e yönlendiren client."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__, self.config)
        self.model_inference_url = get_model_inference_url()
        
    def generate_response(self, prompt: str, model_name: str, provider: str = "groq") -> str:
        """Model Inference Service'ten cevap üret."""
        
        try:
            return self._call_model_inference_service(prompt, model_name)
                
        except Exception as e:
            self.logger.error(f"Model Inference Service hatası: {e}")
            return f"❌ Model inference hatası: {str(e)}"
    
    def generate(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """OpenAI-style chat completions - Model Inference Service'e yönlendirir."""
        
        # Convert messages to single prompt
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
        
        return self._call_model_inference_service(prompt.strip(), model, temperature, max_tokens)
    
    def _call_model_inference_service(self, prompt: str, model_name: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Model Inference Service'e HTTP isteği gönder."""
        try:
            request_data = {
                "prompt": prompt,
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                f"{self.model_inference_url}/models/generate",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                self.logger.error(f"Model Inference Service API error: {response.status_code}")
                return "❌ Model servisi hatası."
                
        except requests.exceptions.ConnectionError:
            self.logger.error("Model Inference Service ile bağlantı kurulamadı")
            return "❌ Model servisi ile bağlantı kurulamadı."
        except requests.exceptions.Timeout:
            self.logger.error("Model Inference Service isteği zaman aşımına uğradı")
            return "❌ İstek zaman aşımına uğradı."
        except Exception as e:
            self.logger.error(f"Model Inference Service call error: {e}")
            return "❌ Model servisi hatası."
    
def get_cloud_llm_client():
    """Global cloud LLM client instance."""
    return CloudLLMClient()