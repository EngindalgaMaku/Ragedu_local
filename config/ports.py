"""
Merkezi Port Konfigürasyon Dosyası
Tüm servislerin port bilgileri burada tanımlanır
"""

import os

# Ana Servis Portları
API_GATEWAY_PORT = int(os.getenv("API_GATEWAY_PORT", 8000))
AUTH_SERVICE_PORT = int(os.getenv("AUTH_SERVICE_PORT", 8006))
APRAG_SERVICE_PORT = int(os.getenv("APRAG_SERVICE_PORT", 8007))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 3000))

# Mikroservis Portları  
DOCUMENT_PROCESSOR_PORT = int(os.getenv("DOCUMENT_PROCESSOR_PORT", 8003))
MODEL_INFERENCE_PORT = int(os.getenv("MODEL_INFERENCE_PORT", 8002))
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", 8004))
MARKER_API_PORT = int(os.getenv("MARKER_API_PORT", 8090))

# URL Builder Functions
def get_service_url(service_name: str, host: str = "localhost", use_docker_names: bool = False) -> str:
    """Servis URL'lerini oluştur"""
    
    if use_docker_names:
        # Docker compose içinde servis isimleri kullan
        docker_names = {
            "api_gateway": "api-gateway",
            "auth_service": "auth-service",
            "aprag_service": "aprag-service",
            "document_processor": "document-processing-service",
            "model_inference": "model-inference-service",
            "chromadb": "chromadb-service",
            "marker_api": "marker-api"
        }
        host = docker_names.get(service_name, service_name)
    
    ports = {
        "api_gateway": API_GATEWAY_PORT,
        "auth_service": AUTH_SERVICE_PORT,
        "aprag_service": APRAG_SERVICE_PORT,
        "frontend": FRONTEND_PORT,
        "document_processor": DOCUMENT_PROCESSOR_PORT,
        "model_inference": MODEL_INFERENCE_PORT,
        "chromadb": CHROMADB_PORT,
        "marker_api": MARKER_API_PORT
    }
    
    port = ports.get(service_name)
    if not port:
        raise ValueError(f"Unknown service: {service_name}")
        
    return f"http://{host}:{port}"

# Sık kullanılan URL'ler
API_GATEWAY_URL = get_service_url("api_gateway")
AUTH_SERVICE_URL = get_service_url("auth_service")
APRAG_SERVICE_URL = get_service_url("aprag_service")
FRONTEND_URL = get_service_url("frontend")

# Docker için URL'ler
API_GATEWAY_DOCKER_URL = get_service_url("api_gateway", use_docker_names=True)
AUTH_SERVICE_DOCKER_URL = get_service_url("auth_service", use_docker_names=True)
APRAG_SERVICE_DOCKER_URL = get_service_url("aprag_service", use_docker_names=True)

# Test için tüm servislerin health check URL'leri
HEALTH_URLS = {
    "api_gateway": f"{API_GATEWAY_URL}/health",
    "auth_service": f"{AUTH_SERVICE_URL}/health",
    "aprag_service": f"{APRAG_SERVICE_URL}/health",
    "frontend": FRONTEND_URL
}

# CORS için allowed origins - Environment variable'dan override edilebilir
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
_cors_origins_list = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()] if _cors_origins_env else []

CORS_ORIGINS = _cors_origins_list + [
    API_GATEWAY_URL,
    AUTH_SERVICE_URL,
    APRAG_SERVICE_URL,
    FRONTEND_URL,
    f"http://127.0.0.1:{FRONTEND_PORT}",
    f"http://host.docker.internal:{FRONTEND_PORT}",
    f"http://host.docker.internal:{API_GATEWAY_PORT}",
    f"http://host.docker.internal:{APRAG_SERVICE_PORT}",
    API_GATEWAY_DOCKER_URL,
    APRAG_SERVICE_DOCKER_URL,
    f"http://frontend:{FRONTEND_PORT}",
    API_GATEWAY_DOCKER_URL
]

if __name__ == "__main__":
    print("🔧 Port Configuration:")
    print(f"API Gateway: {API_GATEWAY_URL}")
    print(f"Auth Service: {AUTH_SERVICE_URL}")
    print(f"APRAG Service: {APRAG_SERVICE_URL}")
    print(f"Frontend: {FRONTEND_URL}")
    print(f"CORS Origins: {', '.join(CORS_ORIGINS)}")