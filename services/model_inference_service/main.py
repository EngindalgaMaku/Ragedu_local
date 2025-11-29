import os
import time
import requests
import urllib3
import traceback
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import ollama
from groq import Groq
from huggingface_hub import InferenceClient
# sentence-transformers is optional - only imported when needed (lazy loading)
# from sentence_transformers import CrossEncoder  # Moved to get_rerank_model() function
from openai import OpenAI as OpenAIClient

# Disable SSL warnings for HuggingFace API (common in corporate/proxy environments)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Configuration ---
# Google Cloud Run compatible - all URLs from environment variables
# For Docker: use service name or localhost (e.g., http://ollama-service:11434)
# For Cloud Run: use full URL if Ollama is deployed separately
OLLAMA_HOST = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_URL", "http://localhost:11434"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ALIBABA_API_KEY = os.getenv("ALIBABA_API_KEY", os.getenv("DASHSCOPE_API_KEY"))
ALIBABA_API_BASE = os.getenv("ALIBABA_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Model Inference Service",
    description="A microservice to interact with various LLM providers like Ollama and Groq.",
    version="1.0.0"
)

# --- Pydantic Models for API requests ---
class GenerationRequest(BaseModel):
    prompt: str
    model: str
    # Optional parameters
    temperature: float = 0.7
    max_tokens: int = 1024

class GenerationResponse(BaseModel):
    response: str
    model_used: str

class EmbedRequest(BaseModel):
    texts: list[str]
    model: Optional[str] = None

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model_used: str

class PullModelRequest(BaseModel):
    model_name: str

class PullModelResponse(BaseModel):
    status: str
    message: str

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class RerankResponse(BaseModel):
    results: List[dict]

class GenerateAnswerRequest(BaseModel):
    query: str
    docs: List[dict]
    model: Optional[str] = "llama-3.1-8b-instant"
    max_context_chars: Optional[int] = 8000

# Add the RAG query models
class RAGQueryRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5
    use_rerank: bool = True
    min_score: float = 0.1
    max_context_chars: int = 8000
    model: Optional[str] = None

class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[str] = []

class OutputCleanerRequest(BaseModel):
    raw_output: str
    original_query: str
    model: Optional[str] = "llama-3.1-8b-instant"

class OutputCleanerResponse(BaseModel):
    cleaned_answer: str
    model_used: str

# --- LLM Clients ---
ollama_client = None
OLLAMA_AVAILABLE = False

def get_ollama_client():
    """Get or initialize Ollama client with fast timeout to avoid blocking."""
    global ollama_client, OLLAMA_AVAILABLE
    
    if ollama_client is not None and OLLAMA_AVAILABLE:
        try:
            # Quick check if still available (with timeout)
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Ollama check timeout")
            
            # Use a very short timeout for quick check (1 second)
            try:
                # Try to list models with timeout
                import threading
                result = [None]
                exception = [None]
                
                def check_ollama():
                    try:
                        result[0] = ollama_client.list()
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=check_ollama)
                thread.daemon = True
                thread.start()
                thread.join(timeout=1.0)  # 1 second timeout
                
                if thread.is_alive():
                    # Timeout - Ollama is not responding
                    ollama_client = None
                    OLLAMA_AVAILABLE = False
                    return None
                
                if exception[0]:
                    raise exception[0]
                
                return ollama_client
            except:
                # Client became unavailable, reset
                ollama_client = None
                OLLAMA_AVAILABLE = False
        except:
            # Client became unavailable, reset
            ollama_client = None
            OLLAMA_AVAILABLE = False
    
    # Try to initialize/reconnect with fast timeout
    max_retries = 2  # Reduced from 5 to 2
    for attempt in range(max_retries):
        try:
            print(f"🔵 [DIAGNOSTIC] Attempting to connect to Ollama at: {OLLAMA_HOST} (attempt {attempt + 1}/{max_retries})")
            
            # Use threading with timeout to avoid blocking
            import threading
            client_result = [None]
            client_exception = [None]
            
            def try_connect():
                try:
                    client = ollama.Client(host=OLLAMA_HOST, timeout=2.0)  # 2 second timeout
                    client.list()  # Quick check
                    client_result[0] = client
                except Exception as e:
                    client_exception[0] = e
            
            thread = threading.Thread(target=try_connect)
            thread.daemon = True
            thread.start()
            thread.join(timeout=3.0)  # Max 3 seconds per attempt
            
            if thread.is_alive():
                # Timeout - Ollama is not responding
                print(f"⚠️ [DIAGNOSTIC] Ollama connection timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Reduced wait time from 2-8 seconds to 0.5 seconds
                continue
            
            if client_exception[0]:
                raise client_exception[0]
            
            if client_result[0]:
                ollama_client = client_result[0]
                OLLAMA_AVAILABLE = True
                print("✅ [DIAGNOSTIC] Successfully connected to Ollama and listed models.")
                return ollama_client
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 0.5  # Reduced from (attempt + 1) * 2 to 0.5 seconds
                print(f"⚠️ [DIAGNOSTIC] Could not connect to Ollama (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time} seconds...")
                print(f"   Error: {e}")
                time.sleep(wait_time)
            else:
                print(f"❌ [CRITICAL] Could not connect to Ollama at {OLLAMA_HOST} after {max_retries} attempts.")
                print(f"   Error Type: {type(e).__name__}")
                print(f"   Error Details: {e}")
                print(f"   ℹ️ [INFO] Ollama is unavailable. Service will use HuggingFace/Alibaba embeddings as fallback.")
                ollama_client = None
                OLLAMA_AVAILABLE = False
                return None
    
    return None

# Initial connection attempt - DISABLED to avoid blocking startup
# Ollama will be connected lazily on first use
print("ℹ️ [INFO] Ollama connection will be attempted on first use (lazy loading)")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
if not groq_client:
    print("⚠️ Warning: GROQ_API_KEY not set. Groq models will not be available.")

# OpenRouter client setup
openrouter_client = None
if OPENROUTER_API_KEY:
    openrouter_client = True  # We'll use requests directly for OpenRouter
    print("✅ OpenRouter API key found.")
else:
    print("⚠️ Warning: OPENROUTER_API_KEY not set. OpenRouter models will not be available.")

# Initialize HuggingFace client with explicit provider
try:
    huggingface_client = InferenceClient(
        token=HUGGINGFACE_API_KEY,
        # Don't specify provider, let it auto-detect or use router
    ) if HUGGINGFACE_API_KEY else None
    if not huggingface_client:
        print("⚠️ Warning: HUGGINGFACE_API_KEY not set. HuggingFace models will not be available.")
    else:
        print("✅ HuggingFace Inference API client initialized.")
except Exception as e:
    print(f"⚠️ Warning: Failed to initialize HuggingFace client: {e}")
    huggingface_client = None

# DeepSeek client setup (OpenAI-compatible API)
deepseek_client = None
if DEEPSEEK_API_KEY:
    try:
        deepseek_client = OpenAIClient(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        print("✅ DeepSeek API client initialized.")
    except Exception as e:
        print(f"⚠️ Warning: Failed to initialize DeepSeek client: {e}")
        deepseek_client = None
else:
    print("⚠️ Warning: DEEPSEEK_API_KEY not set. DeepSeek models will not be available.")

# Alibaba DashScope client setup (OpenAI-compatible API)
alibaba_client = None
if ALIBABA_API_KEY:
    try:
        alibaba_client = OpenAIClient(
            api_key=ALIBABA_API_KEY,
            base_url=ALIBABA_API_BASE
        )
        print(f"✅ Alibaba DashScope API client initialized with endpoint: {ALIBABA_API_BASE}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to initialize Alibaba client: {e}")
        alibaba_client = None
else:
    print("⚠️ Warning: ALIBABA_API_KEY not set. Alibaba models will not be available.")

# --- Reranking ---
# NOTE: Reranking is now handled by reranker-service (Alibaba DashScope API)
# No local sentence-transformers dependency needed
# The /rerank endpoint proxies requests to reranker-service

# --- Output Cleaning Functions ---
def clean_llm_output(raw_output: str, original_query: str = "") -> str:
    """
    DEBUG modda çıktı temizleyici.
    Her adımı logluyor.
    """
    print(f"🔧 [CLEANER] Raw input length: {len(raw_output)}")
    print(f"🔧 [CLEANER] Raw input: '{raw_output[:100]}...'")
    
    if not raw_output or not raw_output.strip():
        print("🔧 [CLEANER] Empty input, returning default")
        return "Cevap oluşturulamadı."
    
    try:
        original = raw_output.strip()
        
        # YAKLAŞIM 1: "Cevap:" sonrasını bul - EN ÖNCELİKLİ
        cevap_match = re.search(r'(?i)cevap\s*:?\s*(.+?)$', original, re.MULTILINE | re.DOTALL)
        if cevap_match:
            answer = cevap_match.group(1).strip()
            print(f"🔧 [CLEANER] Found 'Cevap:' match: '{answer}'")
            if len(answer) > 2:
                clean_answer = answer + ('.' if not answer.endswith('.') else '')
                print(f"🔧 [CLEANER] Returning cleaned answer: '{clean_answer}'")
                return clean_answer
        
        # YAKLAŞIM 2: Son satırları kontrol et
        lines = [line.strip() for line in original.split('\n') if line.strip()]
        print(f"🔧 [CLEANER] Total lines: {len(lines)}")
        
        # Son 3 satırı kontrol et
        for i, line in enumerate(reversed(lines[-3:])):
            print(f"🔧 [CLEANER] Checking line {3-i}: '{line}'")
            if (len(line) > 10 and
                not line.startswith(('1.', '2.', '3.', '4.', 'Adım', 'ÖNCE', 'SONRA')) and
                not any(word in line.lower() for word in ['analiz', 'kontrol', 'cevaplayacağım', 'bağlam'])):
                clean_line = line + ('.' if not line.endswith('.') else '')
                print(f"🔧 [CLEANER] Returning clean line: '{clean_line}'")
                return clean_line
        
        # YAKLAŞIM 3: Son noktalı cümleyi bul
        sentences = re.findall(r'[^.!?]*[.!?]', original)
        if sentences:
            last_sentence = sentences[-1].strip()
            print(f"🔧 [CLEANER] Last sentence: '{last_sentence}'")
            if len(last_sentence) > 5:
                return last_sentence
        
        # Son çare
        print("🔧 [CLEANER] No clean match found, returning last 50 chars")
        return original[-50:] if len(original) > 50 else original
        
    except Exception as e:
        print(f"🔧 [CLEANER] ERROR: {e}")
        return raw_output.strip()


# --- Helper Functions ---
def is_groq_model(model_name: str) -> bool:
    """Check if the model is intended for Groq based on tested working models."""
    groq_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b",
        # All 4 models above confirmed working via direct Groq API testing 2025-11-02
        # Keep other potentially working models for future testing
        "allam-2-7b",
        "groq/compound",
        "groq/compound-mini",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-guard-4-12b",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-safeguard-20b"
    ]
    return model_name in groq_models

def is_openrouter_model(model_name: str) -> bool:
    """Check if the model is intended for OpenRouter based on free models."""
    # Free OpenRouter models - cost-effective options
    openrouter_models = [
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "microsoft/phi-3-mini-4k-instruct:free",
        "google/gemma-2-9b-it:free",
        "google/gemini-2.5-flash-lite",
        "nousresearch/hermes-3-llama-3.1-8b:free",
        "qwen/qwen3-32b",
        "x-ai/grok-4.1-fast:free"
    ]
    return model_name in openrouter_models

def is_huggingface_model(model_name: str) -> bool:
    """Check if the model is a HuggingFace model."""
    # HuggingFace models typically have format: "organization/model-name"
    # Exclude models that are NOT available on Inference API
    excluded_models = [
        "openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-safeguard-20b",
        "teknium/OpenHermes-2.5-Mistral-7B",  # Not available on Inference API
        "meta-llama/Meta-Llama-3-8B-Instruct",  # Not available on Inference API
        "meta-llama/Llama-3-8B-Instruct",  # Not available on Inference API
        "meta-llama/Llama-2-7b-chat-hf",  # May not be available
    ]
    
    # If model is explicitly excluded, it's not a HuggingFace Inference API model
    if model_name in excluded_models:
        return False
    
    # Check if it has the format of a HuggingFace model (contains "/")
    # and is not in the excluded list
    return "/" in model_name and model_name not in excluded_models

def is_deepseek_model(model_name: str) -> bool:
    """Check if the model is a DeepSeek model."""
    deepseek_models = [
        "deepseek-chat",
        "deepseek-reasoner"
    ]
    return model_name in deepseek_models

def is_alibaba_model(model_name: str) -> bool:
    """Check if the model is an Alibaba DashScope Qwen model."""
    alibaba_models = [
        "qwen-plus",
        "qwen-turbo",
        "qwen-max",
        "qwen-max-longcontext",
        "qwen-7b-chat",
        "qwen-14b-chat",
        "qwen-72b-chat",
        "qwen-vl-plus",
        "qwen-vl-max",
        "qwen-flash"
    ]
    return model_name in alibaba_models

def is_alibaba_embedding_model(model_name: str) -> bool:
    """Check if the model is an Alibaba DashScope embedding model."""
    if not model_name:
        return False
    # Alibaba embedding models start with "text-embedding-"
    return model_name.startswith("text-embedding-") or "alibaba" in model_name.lower() or "dashscope" in model_name.lower()

def is_huggingface_embedding_model(model_name: str) -> bool:
    """Check if the model is a HuggingFace embedding model."""
    if not model_name:
        return False
    hf_orgs = ["sentence-transformers/", "intfloat/", "BAAI/"]
    return "/" in model_name and any(model_name.startswith(org) for org in hf_orgs)

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
def health_check():
    """Check the operational status of the service and its connections."""
    return {
        "status": "ok",
        "ollama_available": OLLAMA_AVAILABLE,
        "groq_available": bool(groq_client),
        "huggingface_available": bool(huggingface_client),
        "openrouter_available": bool(openrouter_client and OPENROUTER_API_KEY),
        "deepseek_available": bool(deepseek_client and DEEPSEEK_API_KEY),
        "alibaba_available": bool(alibaba_client and ALIBABA_API_KEY),
        "ollama_host": OLLAMA_HOST
    }

@app.post("/models/generate", response_model=GenerationResponse, summary="Generate Response from a Model")
async def generate_response(request: GenerationRequest):
    """
    Receives a prompt and a model name, and returns a generated response.
    It dynamically selects the provider (Ollama or Groq) based on the model name.
    """
    model_name = request.model
    prompt = request.prompt

    try:
        if is_alibaba_model(model_name):
            if not alibaba_client or not ALIBABA_API_KEY:
                raise HTTPException(status_code=503, detail="Alibaba client is not available. Check ALIBABA_API_KEY.")

            try:
                chat_completion = alibaba_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=False
                )
                response_content = chat_completion.choices[0].message.content or ""
                return GenerationResponse(response=response_content, model_used=model_name)
            except Exception as e:
                error_str = str(e)
                # Check for authentication errors specifically
                if "401" in error_str or "authentication" in error_str.lower() or "invalid" in error_str.lower():
                    error_detail = f"Alibaba API authentication failed. Please check your ALIBABA_API_KEY environment variable. Error: {error_str}"
                    print(f"❌ {error_detail}")
                    raise HTTPException(status_code=401, detail=error_detail)
                else:
                    error_detail = f"Alibaba API error: {error_str}"
                    print(f"❌ {error_detail}")
                    raise HTTPException(status_code=500, detail=error_detail)

        elif is_deepseek_model(model_name):
            if not deepseek_client or not DEEPSEEK_API_KEY:
                raise HTTPException(status_code=503, detail="DeepSeek client is not available. Check DEEPSEEK_API_KEY.")

            try:
                chat_completion = deepseek_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=False
                )
                response_content = chat_completion.choices[0].message.content or ""
                return GenerationResponse(response=response_content, model_used=model_name)
            except Exception as e:
                error_str = str(e)
                # Check for authentication errors specifically
                if "401" in error_str or "authentication" in error_str.lower() or "invalid" in error_str.lower():
                    error_detail = f"DeepSeek API authentication failed. Please check your DEEPSEEK_API_KEY environment variable. Error: {error_str}"
                    print(f"❌ {error_detail}")
                    raise HTTPException(status_code=401, detail=error_detail)
                else:
                    error_detail = f"DeepSeek API error: {error_str}"
                    print(f"❌ {error_detail}")
                    raise HTTPException(status_code=500, detail=error_detail)

        elif is_openrouter_model(model_name):
            if not openrouter_client or not OPENROUTER_API_KEY:
                raise HTTPException(status_code=503, detail="OpenRouter client is not available. Check OPENROUTER_API_KEY.")

            # OpenRouter API call
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",  # Optional: for analytics
                "X-Title": "RAG3 Local System"  # Optional: for analytics
            }
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
            
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        response_content = result["choices"][0]["message"]["content"] or ""
                        return GenerationResponse(response=response_content, model_used=model_name)
                    else:
                        raise HTTPException(status_code=500, detail="Invalid response format from OpenRouter")
                else:
                    error_detail = f"OpenRouter API error: {response.status_code}"
                    if response.text:
                        error_detail += f" - {response.text[:500]}"
                    raise HTTPException(status_code=response.status_code, detail=error_detail)
                    
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=503, detail=f"OpenRouter API request failed: {str(e)}")

        elif is_groq_model(model_name):
            if not groq_client:
                raise HTTPException(status_code=503, detail="Groq client is not available. Check GROQ_API_KEY.")

            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            response_content = chat_completion.choices[0].message.content or ""
            return GenerationResponse(response=response_content, model_used=model_name)

        elif is_huggingface_model(model_name):
            if not huggingface_client:
                raise HTTPException(status_code=503, detail="HuggingFace client is not available. Check HUGGINGFACE_API_KEY.")

            # Use direct HTTP request to HuggingFace API with multiple endpoint formats
            try:
                print(f"Using direct HTTP request to HuggingFace API for model: {model_name}")
                
                # Try multiple endpoint formats
                # Note: Spaces API format is complex and model-specific, so we skip it
                api_endpoints = [
                    f"https://router.huggingface.co/hf-inference/models/{model_name}",
                    # Removed deprecated endpoint
                ]
                
                headers = {"Content-Type": "application/json"}
                if HUGGINGFACE_API_KEY:
                    headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"
                
                # Payload for text generation
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "return_full_text": False
                    }
                }
                
                last_error = None
                for api_url in api_endpoints:
                    try:
                        print(f"Trying HuggingFace API endpoint: {api_url}")
                        response = requests.post(api_url, json=payload, headers=headers, timeout=120, verify=False)
                        print(f"Response status: {response.status_code}")
                        
                        if response.status_code == 200:
                            result = response.json()
                            # Handle different response formats
                            if isinstance(result, list) and len(result) > 0:
                                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                                    response_content = result[0]['generated_text'].strip()
                                else:
                                    response_content = str(result[0]).strip()
                            elif isinstance(result, dict):
                                if 'generated_text' in result:
                                    response_content = result['generated_text'].strip()
                                elif 'data' in result and len(result['data']) > 0:
                                    # Spaces API format
                                    response_content = str(result['data'][0]).strip()
                                else:
                                    response_content = str(result).strip()
                            else:
                                response_content = str(result).strip()
                            
                            print(f"✅ Successfully generated response from {api_url}, length: {len(response_content)}")
                            return GenerationResponse(response=response_content, model_used=model_name)
                            
                        elif response.status_code == 503:
                            # Model is loading
                            error_data = response.json() if response.text else {}
                            wait_time = error_data.get('estimated_time', 20)
                            print(f"Model {model_name} is loading on {api_url}, waiting {wait_time} seconds...")
                            time.sleep(min(wait_time, 60))
                            
                            # Retry
                            response = requests.post(api_url, json=payload, headers=headers, timeout=120, verify=False)
                            if response.status_code == 200:
                                result = response.json()
                                if isinstance(result, list) and len(result) > 0:
                                    response_content = result[0].get('generated_text', str(result[0])).strip()
                                else:
                                    response_content = str(result).strip()
                                print(f"✅ Successfully generated response after retry, length: {len(response_content)}")
                                return GenerationResponse(response=response_content, model_used=model_name)
                            else:
                                error_msg = response.text[:500] if response.text else f"HTTP {response.status_code}"
                                last_error = f"HTTP {response.status_code} after retry: {error_msg}"
                                print(f"⚠️ Retry failed: {last_error}")
                        elif response.status_code == 410:
                            # Deprecated endpoint, skip
                            print(f"⚠️ Endpoint {api_url} is deprecated (410), trying next...")
                            continue
                        elif response.status_code == 404:
                            # Model not found on this endpoint, try next
                            print(f"⚠️ Model not found on {api_url} (404), trying next endpoint...")
                            last_error = f"HTTP 404: Model not found on {api_url}"
                            continue
                        else:
                            error_msg = response.text[:500] if response.text else f"HTTP {response.status_code}"
                            last_error = f"HTTP {response.status_code}: {error_msg}"
                            print(f"⚠️ Error on {api_url}: {last_error}")
                            continue
                            
                    except requests.RequestException as req_error:
                        last_error = f"Request failed: {str(req_error)}"
                        print(f"⚠️ Request error on {api_url}: {req_error}")
                        continue
                
                # All endpoints failed
                error_detail = (
                    f"HuggingFace API error for model '{model_name}': All endpoints failed. "
                    f"Last error: {last_error or 'Unknown error'}. "
                    f"This model may not be available on HuggingFace Inference API. "
                    f"Some models exist on HuggingFace but are only available via local transformers library, "
                    f"not via Inference API. Some models (like Meta Llama) may require special access or approval. "
                    f"Please check: https://huggingface.co/{model_name} "
                    f"to see if it supports Inference API. "
                    f"Recommended alternatives that work with Inference API: "
                    f"mistralai/Mistral-7B-Instruct-v0.3, Qwen/Qwen2-7B-Instruct, or google/gemma-7b-it"
                )
                print(f"❌ {error_detail}")
                raise HTTPException(status_code=500, detail=error_detail)
            except HTTPException:
                raise
            except Exception as e:
                error_detail = f"HuggingFace API error for model '{model_name}': {str(e)}"
                print(f"❌ {error_detail}")
                raise HTTPException(status_code=500, detail=error_detail)

        else: # Default to Ollama
            client = get_ollama_client()
            if client is None:
                raise HTTPException(status_code=503, detail="Ollama client is not available. Check connection to Ollama.")

            # Map common model names to the ones we know are available
            model_mapping = {
                "llama3": "llama3:8b",
                "llama3-8b": "llama3:8b",
                "llama3-70b": "llama3:8b",  # Using 8b as fallback
                "mistral": "mistral:7b",
                "mistral-7b": "mistral:7b",
                "llama3.2": "llama3:8b",  # Using 8b as fallback
                "llama3-8b-8192": "llama3:8b",  # Using 8b as fallback
                "llama3:latest": "llama3:8b",  # Using 8b as fallback
                "mistral:latest": "mistral:7b"  # Using 7b as fallback
            }
            
            # Use the mapped model name if it exists, otherwise use the original
            actual_model_name = model_mapping.get(model_name, model_name)
            
            print(f"Attempting to use model: {actual_model_name}")  # Debug print
            
            response = client.chat(
                model=actual_model_name,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False
            )
            response_content = response['message']['content'] if response.get('message') and response['message'].get('content') else ""
            return GenerationResponse(response=response_content, model_used=actual_model_name)

    except Exception as e:
        # Log the exception details here in a real application
        print(f"❌ Error during model generation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the response: {str(e)}")

@app.get("/models/available", summary="List Available Models")
def get_available_models():
    """Returns a list of available models from all configured providers."""
    models = {"groq": [], "ollama": [], "huggingface": [], "openrouter": [], "deepseek": [], "alibaba": []}

    if groq_client:
        # Only tested and confirmed working models
        models["groq"] = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
            # All 4 models above confirmed working via direct Groq API testing 2025-11-02
            # Keep other models for future testing when they become stable
            "groq/compound",
            "groq/compound-mini",
            "allam-2-7b",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-safeguard-20b",
            "moonshotai/kimi-k2-instruct",
            "moonshotai/kimi-k2-instruct-0905"
        ]

    if openrouter_client and OPENROUTER_API_KEY:
        # Free OpenRouter models - cost-effective and reliable
        models["openrouter"] = [
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "microsoft/phi-3-mini-4k-instruct:free",
            "google/gemma-2-9b-it:free",
            "google/gemini-2.5-flash-lite",
            "nousresearch/hermes-3-llama-3.1-8b:free",
            "qwen/qwen3-32b",
            "x-ai/grok-4.1-fast:free"  # Grok 4.1 Fast - Free tier
        ]

    if deepseek_client and DEEPSEEK_API_KEY:
        # DeepSeek models - OpenAI-compatible API
        models["deepseek"] = [
            "deepseek-chat",  # Non-thinking mode (DeepSeek-V3.2-Exp)
            "deepseek-reasoner"  # Thinking mode (DeepSeek-V3.2-Exp)
        ]

    # Try to get Ollama models with timeout (DISABLED - causes unnecessary connection attempts)
    # Ollama models will be empty by default to avoid blocking and connection errors
    # If Ollama is needed, it will be connected lazily on first use
    models["ollama"] = []
    # DISABLED: Ollama connection check to prevent unnecessary connection attempts
    # client = get_ollama_client()
    # if client is not None:
    #     try:
    #         # Use threading to add timeout to Ollama list() call
    #         import threading
    #         installed_models = None
    #         exception_occurred = None
    #         
    #         def fetch_models():
    #             nonlocal installed_models, exception_occurred
    #             try:
    #                 installed_models = client.list()
    #             except Exception as e:
    #                 exception_occurred = e
    #         
    #         thread = threading.Thread(target=fetch_models)
    #         thread.daemon = True
    #         thread.start()
    #         thread.join(timeout=3)  # 3 second timeout
    #         
    #         if thread.is_alive():
    #             print("⚠️ Ollama list() call timed out after 3 seconds")
    #             models["ollama"] = []
    #         elif exception_occurred:
    #             print(f"Could not fetch Ollama models: {exception_occurred}")
    #             models["ollama"] = []
    #         elif installed_models:
    #             print(f"Ollama list() response: {installed_models}")  # Debug print
    #             # Fix the model name extraction
    #             if 'models' in installed_models and isinstance(installed_models['models'], list):
    #                 model_names = []
    #                 for model in installed_models['models']:
    #                     if isinstance(model, dict):
    #                         # Check for both 'name' and 'model' fields for compatibility
    #                         if 'name' in model:
    #                             model_names.append(model['name'])
    #                         elif 'model' in model:
    #                             model_names.append(model['model'])
    #                     elif isinstance(model, str):
    #                         model_names.append(model)
    #                     else:
    #                         # Handle Model objects with 'model' attribute
    #                         if hasattr(model, 'model'):
    #                             model_names.append(model.model)
    #                         elif hasattr(model, 'name'):
    #                             model_names.append(model.name)
    #                 models["ollama"] = model_names
    #             else:
    #                 # Handle case where the response format is different
    #                 models["ollama"] = []
    #                 print(f"Unexpected response format from Ollama list(): {installed_models}")
    #         else:
    #             models["ollama"] = []
    #     except Exception as e:
    #         print(f"Could not fetch Ollama models: {e}")
    #         # Return empty list for ollama if it fails during the call
    #         models["ollama"] = []

    # HuggingFace models - Only models confirmed to work with Inference API
    # Note: Meta Llama models are NOT available on Inference API (require local transformers)
    # Note: Some models exist on HuggingFace but are only available via local transformers
    models["huggingface"] = [
        # Mistral models - Most reliable and confirmed working
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
        # Microsoft models
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        # Qwen models
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        # Google models
        "google/gemma-7b-it",
        "google/gemma-2-7b-it",
        # Other models
        "tiiuae/falcon-7b-instruct",
        "HuggingFaceH4/zephyr-7b-beta",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "Intel/neural-chat-7b-v3-1",
        # Meta Llama models - REMOVED: Not available on Inference API
        # Use mistralai/Mistral-7B-Instruct-v0.3 or Ollama llama3:8b as alternatives
    ]

    # Alibaba DashScope Qwen models - OpenAI-compatible API
    # Always include models in list (API key check happens during actual usage)
    # Alibaba DashScope Qwen models - OpenAI-compatible API (Güncel 2025 modelleri)
    # Always include models in list (API key check happens during actual usage)
    models["alibaba"] = [
        # Güncel chat modelleri (2025)
        "qwen-plus",              # Qwen Plus - Genel amaçlı
        "qwen-turbo",             # Qwen Turbo - Hızlı yanıt
        "qwen-flash",             # Qwen Flash - Çok hızlı yanıt
        "qwen-max",               # Qwen Max - En güçlü model
        "qwen-max-longcontext",   # Qwen Max Long Context - Uzun bağlam desteği
        "qwen2.5-max",            # Qwen 2.5 Max - 2025 güncel model
    ]

    return models

@app.get("/models/embedding", summary="List Available Embedding Models")
def get_available_embedding_models():
    """Returns a list of available embedding models from Ollama, HuggingFace, and Alibaba."""
    embedding_models = {
        "ollama": [],
        "huggingface": [],
        "alibaba": []
    }
    
    # Get Ollama embedding models (DISABLED - causes unnecessary connection attempts)
    # Ollama embedding models will be empty by default to avoid blocking and connection errors
    # If Ollama is needed, it will be connected lazily on first use
    embedding_models["ollama"] = []
    # DISABLED: Ollama connection check to prevent unnecessary connection attempts
    # client = get_ollama_client()
    # if client is not None:
    #     try:
    #         installed_models = client.list()
    #         print(f"[Embedding] Ollama list response type: {type(installed_models)}")
    #         print(f"[Embedding] Ollama list response: {installed_models}")
    #         
    #         # Handle different response formats
    #         models_list = []
    #         if isinstance(installed_models, dict) and 'models' in installed_models:
    #             models_list = installed_models['models']
    #         elif isinstance(installed_models, list):
    #             models_list = installed_models
    #         elif hasattr(installed_models, 'models'):
    #             models_list = installed_models.models
    #         else:
    #             # Try to extract models from the response
    #             models_list = installed_models if isinstance(installed_models, list) else []
    #         
    #         # Filter for embedding models (common embedding model names)
    #         embedding_keywords = ['embed', 'bge', 'nomic', 'instructor', 'e5']
    #         for model in models_list:
    #             model_name = ""
    #             # Handle Model objects (from ollama library)
    #             if hasattr(model, 'model'):
    #                 model_name = model.model
    #             elif hasattr(model, 'name'):
    #                 model_name = model.name
    #             # Handle dict format
    #             elif isinstance(model, dict):
    #                 model_name = model.get('name', model.get('model', ''))
    #             # Handle string format
    #             elif isinstance(model, str):
    #                 model_name = model
    #             
    #             # Check if it's an embedding model
    #             if model_name and any(keyword in model_name.lower() for keyword in embedding_keywords):
    #                 embedding_models["ollama"].append(model_name)
    #                 print(f"[Embedding] Added Ollama embedding model: {model_name}")
    #         
    #         print(f"[Embedding] Found {len(embedding_models['ollama'])} Ollama embedding models")
    #     except Exception as e:
    #         print(f"[Embedding] Could not fetch Ollama embedding models: {e}")
    #         import traceback
    #         traceback.print_exc()
    
    # Popular free HuggingFace embedding models
    embedding_models["huggingface"] = [
        {
            "id": "sentence-transformers/all-MiniLM-L6-v2",
            "name": "all-MiniLM-L6-v2",
            "description": "En popüler, hızlı ve küçük (384 boyut)",
            "dimensions": 384,
            "language": "en"
        },
        {
            "id": "sentence-transformers/all-mpnet-base-v2",
            "name": "all-mpnet-base-v2",
            "description": "Yüksek kalite (768 boyut)",
            "dimensions": 768,
            "language": "en"
        },
        {
            "id": "BAAI/bge-small-en-v1.5",
            "name": "bge-small-en-v1.5",
            "description": "BAAI BGE - İyi kalite, küçük boyut",
            "dimensions": 384,
            "language": "en"
        },
        {
            "id": "BAAI/bge-base-en-v1.5",
            "name": "bge-base-en-v1.5",
            "description": "BAAI BGE - Daha iyi kalite (768 boyut)",
            "dimensions": 768,
            "language": "en"
        },
        {
            "id": "intfloat/multilingual-e5-small",
            "name": "multilingual-e5-small",
            "description": "Çok dilli destek (384 boyut)",
            "dimensions": 384,
            "language": "multilingual"
        },
        {
            "id": "intfloat/multilingual-e5-base",
            "name": "multilingual-e5-base",
            "description": "Çok dilli destek - Yüksek kalite (768 boyut)",
            "dimensions": 768,
            "language": "multilingual"
        },
        {
            "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "name": "paraphrase-multilingual-MiniLM-L12-v2",
            "description": "Çok dilli parafraz (384 boyut)",
            "dimensions": 384,
            "language": "multilingual"
        },
        {
            "id": "sentence-transformers/all-MiniLM-L12-v2",
            "name": "all-MiniLM-L12-v2",
            "description": "Daha büyük MiniLM (384 boyut)",
            "dimensions": 384,
            "language": "en"
        },
        {
            "id": "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "name": "paraphrase-MiniLM-L6-v2",
            "description": "Parafraz için optimize (384 boyut)",
            "dimensions": 384,
            "language": "en"
        },
        {
            "id": "sentence-transformers/distiluse-base-multilingual-cased-v1",
            "name": "distiluse-base-multilingual-cased-v1",
            "description": "Çok dilli DistilUSE (512 boyut)",
            "dimensions": 512,
            "language": "multilingual"
        }
    ]
    
    # Alibaba DashScope embedding models (sadece v4 - en güncel)
    embedding_models["alibaba"] = [
        {
            "id": "text-embedding-v4",
            "name": "text-embedding-v4",
            "description": "Alibaba DashScope Text Embedding v4 - Qwen3-Embedding (1024 boyut, 100+ dil)",
            "dimensions": 1024,
            "language": "multilingual"
        }
    ]
    
    return embedding_models

@app.get("/debug/models", summary="Debug: List All Models with Details")
def debug_get_all_models():
    """Returns detailed information about all available models."""
    result = {
        "ollama_available": OLLAMA_AVAILABLE,
        "groq_available": bool(groq_client),
        "models": {}
    }

    if OLLAMA_AVAILABLE and ollama_client is not None:
        try:
            raw_response = ollama_client.list()
            result["models"]["ollama_raw"] = raw_response
            result["models"]["ollama_processed"] = []
            
            if 'models' in raw_response and isinstance(raw_response['models'], list):
                for model in raw_response['models']:
                    result["models"]["ollama_processed"].append(model)
        except Exception as e:
            result["ollama_error"] = str(e)

    if groq_client:
        result["models"]["groq"] = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
            # All 4 models above confirmed working via direct Groq API testing 2025-11-02
            "groq/compound",
            "groq/compound-mini",
            "allam-2-7b",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-safeguard-20b",
            "moonshotai/kimi-k2-instruct",
            "moonshotai/kimi-k2-instruct-0905"
        ]

    return result

@app.post("/models/pull", response_model=PullModelResponse, summary="Pull a Model into Ollama")
async def pull_model(request: PullModelRequest):
    """
    Pulls a model into Ollama if it's not already available.
    """
    model_name = request.model_name
    
    client = get_ollama_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Ollama client is not available. Check connection to Ollama.")
    
    try:
        # Check if model is already available
        installed_models = client.list()
        model_names = []
        for model in installed_models['models']:
            if isinstance(model, dict):
                if 'name' in model:
                    model_names.append(model['name'])
                elif 'model' in model:
                    model_names.append(model['model'])
            elif hasattr(model, 'model'):
                model_names.append(model.model)
            elif hasattr(model, 'name'):
                model_names.append(model.name)
        
        if model_name in model_names:
            return PullModelResponse(
                status="success", 
                message=f"Model {model_name} is already available"
            )
        
        # Pull the model
        print(f"Pulling model {model_name}...")
        # Use the client's pull method
        response = client.pull(model_name)
        print(f"Model {model_name} pulled successfully")
        
        return PullModelResponse(
            status="success", 
            message=f"Model {model_name} pulled successfully"
        )
        
    except Exception as e:
        print(f"❌ Error during model pull: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while pulling the model: {str(e)}")

def is_huggingface_embedding_model(model_name: str) -> bool:
    """Check if the model is a HuggingFace embedding model."""
    # HuggingFace models typically have format: "organization/model-name" or contain "/"
    hf_embedding_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "intfloat/multilingual-e5-small",
        "intfloat/multilingual-e5-base",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v1",
    ]
    # Check if it's in the list or contains "/" (typical HuggingFace format)
    return model_name in hf_embedding_models or ("/" in model_name and not model_name.startswith("openai/"))

def is_alibaba_embedding_model(model_name: str) -> bool:
    """Check if the model is an Alibaba DashScope embedding model."""
    alibaba_embedding_models = [
        "text-embedding-v4"
    ]
    return model_name in alibaba_embedding_models or model_name.startswith("text-embedding-")

@app.post("/embeddings", response_model=EmbedResponse, summary="Generate Embeddings for Texts (OpenAI-compatible endpoint)")
async def generate_embeddings_openai_compatible(request: EmbedRequest):
    """
    OpenAI-compatible embeddings endpoint.
    Receives a list of texts and returns their embeddings.
    Supports Ollama, HuggingFace, and Alibaba DashScope embedding models.
    This endpoint is provided for compatibility with OpenAI API format.
    """
    # Delegate to the main /embed endpoint
    return await generate_embeddings(request)

@app.post("/embed", response_model=EmbedResponse, summary="Generate Embeddings for Texts")
async def generate_embeddings(request: EmbedRequest):
    """
    Receives a list of texts and returns their embeddings.
    Supports Ollama, HuggingFace, and Alibaba DashScope embedding models.
    """
    texts = request.texts
    
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided for embedding.")
    
    try:
        start_time = time.time()
        # Get model from request, if not provided use Alibaba DashScope embedding (default in system)
        # This prevents unnecessary Ollama connection attempts when session embedding model is not passed
        default_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")
        model_name = getattr(request, "model", None) or default_model
        
        # Clean model name (remove :latest, :v1, etc.)
        if ":" in model_name:
            model_name = model_name.split(":")[0]
        
        print(f"🔵 [EMBEDDING] Generating embeddings for {len(texts)} texts using model: {model_name}")
        print(f"🔵 [EMBEDDING] Default model: {default_model}, Request model: {getattr(request, 'model', None)}")
        
        embeddings = []
        
        # Check for Alibaba embedding model first (PRIORITY 1)
        is_alibaba_embedding = is_alibaba_embedding_model(model_name)
        alibaba_failed = False
        if is_alibaba_embedding:
            print(f"🔵 [EMBEDDING] Detected Alibaba embedding model: {model_name}")
            if not alibaba_client or not ALIBABA_API_KEY:
                error_msg = f"❌ [CRITICAL] Alibaba client not available for embedding model '{model_name}'. "
                if not ALIBABA_API_KEY:
                    error_msg += "ALIBABA_API_KEY is not set. "
                if not alibaba_client:
                    error_msg += "Alibaba client initialization failed. "
                error_msg += "Falling back to HuggingFace."
                print(error_msg)
                # Skip Ollama check entirely for Alibaba models, go directly to HuggingFace
                alibaba_failed = True
            else:
                try:
                    print(f"✅ Using Alibaba DashScope embedding model: {model_name}")
                    
                    # Process all texts individually (no batch to avoid 8192 char limit)
                    # Individual chunks should be small enough, batch causes total char limit issues
                    embeddings = []
                    print(f"Processing {len(texts)} texts individually for Alibaba embedding")
                    
                    for i, text in enumerate(texts):
                        try:
                            print(f"Processing text {i+1}/{len(texts)} (length: {len(text)} chars)")
                            
                            # Send individual text to avoid batch character limit
                            embedding_response = alibaba_client.embeddings.create(
                                model=model_name,
                                input=text  # Send single text directly (not as list)
                            )
                            
                            # Process individual response
                            if hasattr(embedding_response, 'data') and len(embedding_response.data) > 0:
                                embedding = embedding_response.data[0].embedding
                            elif isinstance(embedding_response, dict) and 'data' in embedding_response:
                                embedding = embedding_response['data'][0]['embedding']
                            else:
                                raise Exception(f"Unexpected Alibaba embedding response format for text {i+1}")
                            
                            embeddings.append(embedding)
                            
                        except Exception as individual_error:
                            print(f"⚠️ Individual Alibaba embedding failed for text {i+1}: {str(individual_error)}")
                            # Add a zero vector as fallback for failed embeddings
                            embeddings.append([0.0] * 1024)  # Assuming 1024-dimensional embeddings for text-embedding-v4
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    print(f"✅ Successfully generated {len(embeddings)} embeddings using Alibaba DashScope in {processing_time:.2f} seconds.")
                    return EmbedResponse(embeddings=embeddings, model_used=model_name)
                except Exception as alibaba_error:
                    print(f"⚠️ Alibaba embedding failed: {alibaba_error}. Trying HuggingFace fallback...")
                    alibaba_failed = True
        
        # Check model type and skip Ollama entirely unless explicitly an Ollama model
        # This prevents unnecessary connection attempts
        hf_orgs = ["sentence-transformers/", "intfloat/", "BAAI/"]
        is_hf_model = "/" in model_name and any(model_name.startswith(org) for org in hf_orgs)
        ollama_models = ["nomic-embed-text", "nomic-embed", "embeddinggemma", "bge-m3"]
        is_ollama_model = model_name in ollama_models
        
        # Skip Ollama for all non-Ollama models (Alibaba, HuggingFace, or unknown)
        # If Alibaba was detected but failed, skip Ollama entirely
        if is_alibaba_embedding and alibaba_failed:
            print(f"⚠️ Alibaba model '{model_name}' failed, skipping Ollama and using HuggingFace fallback.")
        elif is_huggingface_embedding_model(model_name) or is_hf_model:
            print(f"Detected HuggingFace model '{model_name}', skipping Ollama")
        elif not is_ollama_model:
            # Unknown model or default - if it's not explicitly an Ollama model, skip Ollama
            # Note: If it's an Alibaba model, we should have handled it above
            if is_alibaba_embedding_model(model_name):
                print(f"⚠️ Model '{model_name}' is Alibaba embedding model, but Alibaba check failed earlier. Using HuggingFace fallback...")
            else:
                print(f"Model '{model_name}' is not an Ollama model, skipping Ollama and using HuggingFace directly")
        else:
            # Only try Ollama if it's explicitly an Ollama model
            if is_ollama_model:
                print(f"Detected Ollama model '{model_name}', attempting Ollama connection...")
                client = get_ollama_client()
                if client is not None:
                    try:
                        for text in texts:
                            response = client.embeddings(model=model_name, prompt=text)
                            
                            if isinstance(response, dict):
                                embedding = response.get('embedding')
                            else:
                                embedding = getattr(response, 'embedding', None)

                            if embedding is not None:
                                embeddings.append(embedding)
                            else:
                                raise Exception(f"Could not extract embedding from Ollama response.")
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        print(f"✅ Successfully generated {len(embeddings)} embeddings using Ollama in {processing_time:.2f} seconds.")
                        return EmbedResponse(embeddings=embeddings, model_used=model_name)
                    except Exception as ollama_error:
                        print(f"⚠️ Ollama embedding failed: {ollama_error}. Trying HuggingFace fallback...")
                else:
                    print(f"⚠️ Ollama client not available, skipping Ollama and using HuggingFace/Alibaba fallback")
        
        # Use HuggingFace if Ollama is not available, failed, or if a HuggingFace model was explicitly requested
        # Map Ollama embedding models to HuggingFace equivalents
        ollama_to_hf_mapping = {
            "nomic-embed-text": "sentence-transformers/all-MiniLM-L6-v2",
            "nomic-embed": "sentence-transformers/all-MiniLM-L6-v2",
            "embeddinggemma": "sentence-transformers/all-MiniLM-L6-v2",  # Use reliable model
            "bge-m3": "BAAI/bge-small-en-v1.5",  # Similar to BGE
            "default": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        # Popular free HuggingFace embedding models (ordered by popularity and quality)
        hf_embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",  # Most popular, 384 dim, fast
            "sentence-transformers/all-mpnet-base-v2",  # Better quality, 768 dim
            "BAAI/bge-small-en-v1.5",  # Popular, good quality
            "intfloat/multilingual-e5-small",  # Multilingual support
            "intfloat/multilingual-e5-base",  # Multilingual support, better quality
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual
        ]
        
        # Determine which HuggingFace model to use
        if model_name in ollama_to_hf_mapping:
            hf_embedding_model = ollama_to_hf_mapping[model_name]
        elif model_name in hf_embedding_models:
            hf_embedding_model = model_name
        else:
            # Default to most popular and reliable model
            hf_embedding_model = hf_embedding_models[0]
        
        print(f"Using HuggingFace embedding model: {hf_embedding_model}")
        
        # Try models in order of preference until one works
        last_error = None
        for attempt_model in [hf_embedding_model] + [m for m in hf_embedding_models if m != hf_embedding_model]:
            try:
                print(f"Attempting to use HuggingFace model: {attempt_model}")
                
                # Use direct API calls to HuggingFace Inference API with BATCH processing
                # This is more reliable than using InferenceClient for embeddings
                # Process all texts in a single batch request instead of individual requests
                api_endpoints = [
                    f"https://router.huggingface.co/hf-inference/models/{attempt_model}/pipeline/feature-extraction",
                    # Old router format as fallback
                    f"https://router.huggingface.co/hf-inference/pipeline/feature-extraction/{attempt_model}",
                    # Removed deprecated legacy API endpoint
                ]
                
                # Try with API key first, then without if it fails with 403 or 401
                use_api_key = bool(HUGGINGFACE_API_KEY)
                last_request_error = None
                embedding_response = None
                response = None
                
                for api_url in api_endpoints:
                    try:
                        headers = {"Content-Type": "application/json"}
                        if use_api_key and HUGGINGFACE_API_KEY:
                            headers["Authorization"] = f"Bearer {HUGGINGFACE_API_KEY}"
                        
                        # HuggingFace feature-extraction API supports batch processing
                        # Send all texts in one request for better performance
                        payload = {"inputs": texts}
                        
                        print(f"Calling HuggingFace API: {api_url} (batch size: {len(texts)}, with API key: {use_api_key})")
                        # Use increased timeout for batch processing - 5 minutes for large batches
                        response = requests.post(api_url, json=payload, headers=headers, timeout=300, verify=False)
                        print(f"Response status: {response.status_code} for {api_url}")
                        
                        # Handle 410 (deprecated endpoint) - skip old endpoints immediately
                        if response.status_code == 410:
                            error_msg = response.text[:500] if response.text else "Endpoint deprecated"
                            print(f"⚠️ Endpoint deprecated (410): {error_msg}")
                            last_request_error = f"HTTP {response.status_code}: {error_msg}"
                            continue  # Try next endpoint
                        
                        # Track if we retried without API key
                        retried_without_key = False
                        
                        # Handle 401 (unauthorized) - try without API key
                        if response.status_code == 401 and use_api_key:
                            print(f"⚠️ Unauthorized (401) with API key, trying without API key...")
                            headers_no_key = {"Content-Type": "application/json"}
                            response = requests.post(api_url, json=payload, headers=headers_no_key, timeout=300, verify=False)
                            print(f"Response without API key (401->): {response.status_code}")
                            retried_without_key = True
                        
                        # Handle 403 (permission denied) - try without API key
                        if response.status_code == 403 and use_api_key:
                            print(f"⚠️ API key permission denied (403), trying without API key...")
                            headers_no_key = {"Content-Type": "application/json"}
                            response = requests.post(api_url, json=payload, headers=headers_no_key, timeout=300, verify=False)
                            print(f"Response without API key (403->): {response.status_code}")
                            retried_without_key = True
                        
                        # Handle 503 (model loading) with retry - after auth handling
                        retry_count = 0
                        current_headers = headers_no_key if retried_without_key else headers
                        while response.status_code == 503 and retry_count < 3:
                            print(f"Model {attempt_model} is loading, waiting 10 seconds... (attempt {retry_count + 1}/3)")
                            time.sleep(10)
                            response = requests.post(api_url, json=payload, headers=current_headers, timeout=300, verify=False)
                            print(f"Retry response status: {response.status_code}")
                            retry_count += 1
                        
                        if response.status_code == 200:
                            embedding_response = response.json()
                            print(f"✅ Successfully got batch embeddings from {api_url}")
                            
                            # Process batch response - should be list of embeddings
                            if isinstance(embedding_response, list):
                                # Batch response: list of embeddings for each input
                                for emb in embedding_response:
                                    if isinstance(emb, list):
                                        embeddings.append([float(x) for x in emb])
                                    else:
                                        print(f"⚠️ Unexpected embedding format: {type(emb)}")
                                        embeddings.append([float(emb)] if isinstance(emb, (int, float)) else [0.0])
                            else:
                                print(f"⚠️ Unexpected batch response format: {type(embedding_response)}")
                                # Fallback: treat as single embedding
                                if hasattr(embedding_response, 'tolist'):
                                    embeddings.append(embedding_response.tolist())
                                else:
                                    embeddings.append(list(embedding_response) if hasattr(embedding_response, '__iter__') else [0.0])
                            break  # Success, exit endpoint loop
                        
                        elif response.status_code == 404:
                            error_msg = response.text[:500] if response.text else "No error message"
                            print(f"❌ Failed with status {response.status_code}: {error_msg}")
                            last_request_error = f"HTTP {response.status_code}: {error_msg}"
                            continue
                        else:
                            error_msg = response.text[:500] if response.text else "No error message"
                            print(f"❌ Failed with status {response.status_code}: {error_msg}")
                            last_request_error = f"HTTP {response.status_code}: {error_msg}"
                            continue  # Try next endpoint
                            
                    except requests.RequestException as req_error:
                        last_request_error = f"Request failed: {str(req_error)}"
                        continue  # Try next endpoint
                
                # Check if we got successful batch response
                if len(embeddings) == 0 or len(embeddings) != len(texts):
                    error_msg = f"Batch processing failed: expected {len(texts)} embeddings, got {len(embeddings)}"
                    if last_request_error:
                        error_msg += f". Last error: {last_request_error}"
                    raise Exception(error_msg)
                
                # Success - update the model name used
                hf_embedding_model = attempt_model
                break
                
            except Exception as hf_error:
                last_error = hf_error
                print(f"⚠️ HuggingFace model {attempt_model} failed: {hf_error}. Trying next model...")
                embeddings = []  # Reset for next attempt
                continue
        
        if len(embeddings) == 0:
            error_detail = f"Failed to generate embeddings with all HuggingFace models."
            if last_error:
                error_detail += f" Last error: {str(last_error)}"
            if skip_ollama:
                error_detail += f" (Ollama was skipped because HuggingFace model '{model_name}' was requested. ALL embeddings must use the same model to ensure consistent similarity scores. Please retry or choose a different model.)"
            raise HTTPException(status_code=500, detail=error_detail)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"✅ Successfully generated {len(embeddings)} embeddings using HuggingFace in {processing_time:.2f} seconds.")
        
        return EmbedResponse(embeddings=embeddings, model_used=hf_embedding_model)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during embedding generation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating embeddings: {str(e)}")

# Replace the query endpoint implementation with a proper one
@app.post("/query", response_model=RAGQueryResponse, summary="Handle RAG Query")
async def handle_rag_query(request: RAGQueryRequest):
    """
    Handle RAG queries by generating a response based on the query.
    Since this service doesn't have access to vector stores, it will generate
    a response indicating that RAG functionality should be handled by a dedicated service.
    """
    # Generate a response using the model inference capabilities
    try:
        # Use the existing generation endpoint to create a response
        generation_request = GenerationRequest(
            prompt=f"Answer the following question: {request.query}",
            model=request.model or "llama3:8b",
            temperature=0.7,
            max_tokens=512
        )
        
        # Call the generation function directly instead of using await
        response = await generate_response(generation_request)
        
        return RAGQueryResponse(
            answer=response.response,
            sources=[]  # No sources since we don't have RAG capabilities here
        )
    except Exception as e:
        # Fallback response
        return RAGQueryResponse(
            answer="I'm a lightweight model inference service and don't have access to the full RAG pipeline. "
                   "Please ensure you're connecting to the correct RAG service.",
            sources=[]
        )

@app.post("/rerank", response_model=RerankResponse, summary="Rerank Documents")
async def rerank_documents(request: RerankRequest):
    """
    Reranks a list of documents based on their relevance to a query.
    Proxies to reranker-service which uses Alibaba DashScope API (no heavy dependencies).
    """
    # Proxy to reranker-service instead of using local sentence-transformers
    RERANKER_SERVICE_URL = os.getenv("RERANKER_SERVICE_URL", "http://reranker-service:8008")
    
    try:
        # Forward request to reranker-service
        response = requests.post(
            f"{RERANKER_SERVICE_URL}/rerank",
            json={
                "query": request.query,
                "documents": request.documents
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            # Convert reranker-service response format to our format
            rerank_results = result.get("results", [])
            results = []
            for item in rerank_results:
                results.append({
                    "document": item.get("document", ""),
                    "index": item.get("index", 0),
                    "relevance_score": item.get("relevance_score", 0.0)
                })
            return RerankResponse(results=results)
        else:
            error_detail = f"Reranker service error: {response.status_code}"
            if response.text:
                error_detail += f" - {response.text[:500]}"
            print(f"❌ {error_detail}")
            raise HTTPException(status_code=response.status_code, detail=error_detail)
            
    except requests.exceptions.RequestException as e:
        error_detail = f"Failed to connect to reranker-service: {str(e)}"
        print(f"❌ {error_detail}")
        raise HTTPException(status_code=503, detail=error_detail)
    except Exception as e:
        print(f"❌ Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during reranking: {str(e)}")


@app.post("/generate-answer", response_model=GenerationResponse, summary="Generate Answer from Documents")
async def generate_answer_from_docs(request: GenerateAnswerRequest):
    """
    Generates an answer to a query based on a provided list of documents.
    """
    try:
        context_texts = [doc.get('content', '') or doc.get('text', '') for doc in request.docs]
        context_str = "\n\n".join(context_texts)

        if len(context_str) > request.max_context_chars:
            context_str = context_str[:request.max_context_chars] + "..."

        system_prompt = (
            "Sen öğrencilere yardımcı olan bir eğitim asistanısın.\n\n"
            "KURALLAR:\n"
            "• Verilen ders materyalindeki bilgileri kullanarak soruyu TÜRKÇE cevapla\n"
            "• Ders materyalinde bilgi varsa: Madde madde, net ve anlaşılır şekilde açıkla\n"
            "• Ders materyalinde bilgi yoksa: 'Bu konu, verilen ders materyallerinde bulunmuyor' de\n"
            "• Kısa ve öğretici ol, gereksiz açıklama yapma\n"
        )
        
        full_prompt = f"""DERS MATERYALİ:
{context_str}

SORU: {request.query}

CEVAP:"""

        generation_request = GenerationRequest(
            prompt=full_prompt,
            model=request.model,
            temperature=0.3,  # Reduced from 0.7 to 0.3 for more accurate, context-faithful answers
            max_tokens=1024
        )
        
        # Ham cevabı al
        raw_response = await generate_response(generation_request)
        
        # CLEANER DEVRE DIŞI - Ham cevabı olduğu gibi döndür
        # Çünkü cleaner yanlış satırları seçiyor ve cevabı bozuyor
        # cleaned_answer = clean_llm_output(raw_response.response, request.query)
        
        return GenerationResponse(response=raw_response.response, model_used=raw_response.model_used)

    except Exception as e:
        print(f"❌ Error in generate_answer_from_docs: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)