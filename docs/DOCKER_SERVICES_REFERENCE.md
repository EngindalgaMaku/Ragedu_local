# Docker Services Reference Guide

**RAG Education Assistant - Complete Service Configuration**

_Generated: 2025-11-17_  
_Purpose: Prevent time waste during debugging and deployments_

## 🔧 Network Configuration

**Primary Network:** `rag-education-assistant_rag-network`

```bash
# List networks
docker network ls

# Expected output should show:
# rag-education-assistant_rag-network   bridge    local
```

## 📋 Complete Service Overview

| Service             | Container Name              | Image                                           | External Port | Internal Port | Status |
| ------------------- | --------------------------- | ----------------------------------------------- | ------------- | ------------- | ------ |
| Frontend            | rag3-frontend               | rag-education-assistant-frontend                | 3000          | 3000          | ✅     |
| API Gateway         | api-gateway                 | rag-education-assistant-api-gateway             | 8000          | 8000          | ✅     |
| Model Inference     | model-inference-service     | rag-education-assistant-model-inference-service | 8002          | 8002          | ✅     |
| Document Processing | document-processing-service | document-processing-service                     | 8003          | 8080          | ✅     |
| ChromaDB            | chromadb-service            | chromadb/chroma:1.3.0                           | 8004          | 8000          | ✅     |
| DocStrange          | docstrange-service          | rag-education-assistant-docstrange-service      | 8005          | 80            | ✅     |
| Auth Service        | auth-service                | rag-education-assistant-auth-service            | 8006          | 8006          | ✅     |
| APRAG Service       | aprag-service               | rag-education-assistant-aprag-service           | 8007          | 8007          | ✅     |
| Marker API          | marker-api                  | wirawan/marker-api:latest                       | 8090          | 8080          | ✅     |

## 🐳 Docker Build Commands

### 1. Document Processing Service

```bash
# Build (from rag3_for_local/ directory)
cd rag3_for_local
docker build --no-cache -t document-processing-service -f services/document_processing_service/Dockerfile .

# Quick build (with cache)
docker build -t document-processing-service -f services/document_processing_service/Dockerfile .
```

### 2. API Gateway Service

```bash
# Build (from rag3_for_local/ directory)
docker build --no-cache -t rag-education-assistant-api-gateway -f src/api/Dockerfile .
```

### 3. Auth Service

```bash
# Build (from rag3_for_local/ directory)
docker build --no-cache -t rag-education-assistant-auth-service -f services/auth_service/Dockerfile .
```

### 4. Model Inference Service

```bash
# Build (from rag3_for_local/ directory)
docker build --no-cache -t rag-education-assistant-model-inference-service -f services/model_inference_service/Dockerfile .
```

### 5. Frontend Service

```bash
# Build (from rag3_for_local/ directory)
docker build --no-cache -t rag-education-assistant-frontend -f frontend/Dockerfile .
```

## 🚀 Docker Run Commands

### 1. Document Processing Service (CRITICAL CONFIGURATION)

```bash
# Stop and remove existing
docker stop document-processing-service
docker rm document-processing-service

# Run with CORRECT configuration
docker run -d \
  --name document-processing-service \
  --network rag-education-assistant_rag-network \
  -p 8003:8080 \
  -e MODEL_INFERENCER_URL=http://model-inference-service:8002 \
  -e CHROMADB_URL=http://chromadb-service:8000 \
  --env-file rag3_for_local/.env \
  document-processing-service:latest
```

### 2. Model Inference Service

```bash
docker run -d \
  --name model-inference-service \
  --network rag-education-assistant_rag-network \
  -p 8002:8002 \
  --env-file rag3_for_local/.env \
  rag-education-assistant-model-inference-service:latest
```

### 3. ChromaDB Service

```bash
docker run -d \
  --name chromadb-service \
  --network rag-education-assistant_rag-network \
  -p 8004:8000 \
  -v chromadb_data:/chroma/chroma \
  chromadb/chroma:1.3.0
```

### 4. API Gateway Service

```bash
docker run -d \
  --name api-gateway \
  --network rag-education-assistant_rag-network \
  -p 8000:8000 \
  --env-file rag3_for_local/.env \
  rag-education-assistant-api-gateway:latest
```

### 5. Auth Service

```bash
docker run -d \
  --name auth-service \
  --network rag-education-assistant_rag-network \
  -p 8006:8006 \
  --env-file rag3_for_local/.env \
  -v auth_db_data:/app/data \
  rag-education-assistant-auth-service:latest
```

### 6. Frontend Service

```bash
docker run -d \
  --name rag3-frontend \
  --network rag-education-assistant_rag-network \
  -p 3000:3000 \
  --env-file rag3_for_local/.env \
  rag-education-assistant-frontend:latest
```

### 7. DocStrange Service

```bash
docker run -d \
  --name docstrange-service \
  --network rag-education-assistant_rag-network \
  -p 8005:80 \
  --env-file rag3_for_local/.env \
  rag-education-assistant-docstrange-service:latest
```

### 8. APRAG Service

```bash
docker run -d \
  --name aprag-service \
  --network rag-education-assistant_rag-network \
  -p 8007:8007 \
  --env-file rag3_for_local/.env \
  rag-education-assistant-aprag-service:latest
```

### 9. Marker API Service

```bash
docker run -d \
  --name marker-api \
  --network rag-education-assistant_rag-network \
  -p 8090:8080 \
  wirawan/marker-api:latest
```

## ⚙️ Critical Environment Variables

**File: `rag3_for_local/.env`**

```env
# Service URLs (Internal Docker Network)
MODEL_INFERENCE_URL=http://model-inference-service:8002
CHROMADB_URL=http://chromadb-service:8000
PDF_PROCESSOR_URL=http://docstrange-service:80
DOC_PROCESSOR_URL=http://document-processing-service:8080
AUTH_SERVICE_URL=http://auth-service:8006

# API Keys
GROQ_API_KEY=your_groq_api_key_here
DOCSTRANGE_API_KEY=your_docstrange_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_AUTH_URL=http://localhost:8006
```

## 🔍 Service Health Check Commands

```bash
# Check all running containers
docker ps

# Check specific service logs
docker logs document-processing-service
docker logs model-inference-service
docker logs api-gateway
docker logs chromadb-service

# Test service endpoints
curl -I http://localhost:3000        # Frontend
curl -I http://localhost:8000        # API Gateway
curl -I http://localhost:8002        # Model Inference
curl -I http://localhost:8003        # Document Processing
curl -I http://localhost:8004        # ChromaDB
curl -I http://localhost:8005        # DocStrange
curl -I http://localhost:8006        # Auth Service
curl -I http://localhost:8007        # APRAG Service
```

## 🚨 Common Issues & Quick Fixes

### 1. Document Processing Service Connection Issues

**Problem:** `HTTPConnectionPool(host='model-inference-service', port=8003): Connection refused`
**Solution:**

```bash
# The service is trying to connect to wrong port (8003 instead of 8002)
docker stop document-processing-service
docker rm document-processing-service

# Restart with correct MODEL_INFERENCER_URL
docker run -d --name document-processing-service \
  --network rag-education-assistant_rag-network \
  -p 8003:8080 \
  -e MODEL_INFERENCER_URL=http://model-inference-service:8002 \
  -e CHROMADB_URL=http://chromadb-service:8000 \
  --env-file rag3_for_local/.env \
  document-processing-service:latest
```

### 2. ChromaDB Collection Not Found

**Problem:** `Collection 'session_xxx' not found`
**Solution:** Collections use timestamped format (`collection_name_timestamp`). The enhanced search logic handles this automatically.

### 3. HuggingFace API Authentication

**Problem:** `Failed to get local embeddings: 401`
**Solution:** Update API key in `.env` file and restart services.

### 4. Network Issues

**Problem:** `network rag-education-assistant_default not found`
**Solution:** Use correct network name: `rag-education-assistant_rag-network`

## 🔄 Complete System Restart Sequence

```bash
# 1. Stop all services
docker stop rag3-frontend api-gateway model-inference-service document-processing-service \
           chromadb-service docstrange-service auth-service aprag-service marker-api

# 2. Remove containers (optional - preserves volumes)
docker rm rag3-frontend api-gateway model-inference-service document-processing-service \
         chromadb-service docstrange-service auth-service aprag-service marker-api

# 3. Start essential services first
docker run -d --name chromadb-service --network rag-education-assistant_rag-network -p 8004:8000 -v chromadb_data:/chroma/chroma chromadb/chroma:1.3.0
docker run -d --name model-inference-service --network rag-education-assistant_rag-network -p 8002:8002 --env-file rag3_for_local/.env rag-education-assistant-model-inference-service:latest

# 4. Wait 10 seconds for services to start
sleep 10

# 5. Start dependent services
docker run -d --name document-processing-service --network rag-education-assistant_rag-network -p 8003:8080 -e MODEL_INFERENCER_URL=http://model-inference-service:8002 -e CHROMADB_URL=http://chromadb-service:8000 --env-file rag3_for_local/.env document-processing-service:latest
docker run -d --name auth-service --network rag-education-assistant_rag-network -p 8006:8006 --env-file rag3_for_local/.env -v auth_db_data:/app/data rag-education-assistant-auth-service:latest
docker run -d --name docstrange-service --network rag-education-assistant_rag-network -p 8005:80 --env-file rag3_for_local/.env rag-education-assistant-docstrange-service:latest

# 6. Start API Gateway and Frontend
docker run -d --name api-gateway --network rag-education-assistant_rag-network -p 8000:8000 --env-file rag3_for_local/.env rag-education-assistant-api-gateway:latest
docker run -d --name rag3-frontend --network rag-education-assistant_rag-network -p 3000:3000 --env-file rag3_for_local/.env rag-education-assistant-frontend:latest

# 7. Start additional services
docker run -d --name aprag-service --network rag-education-assistant_rag-network -p 8007:8007 --env-file rag3_for_local/.env rag-education-assistant-aprag-service:latest
docker run -d --name marker-api --network rag-education-assistant_rag-network -p 8090:8080 wirawan/marker-api:latest
```

## 📊 Port Reference Quick Guide

| Port | Service             | Purpose                    |
| ---- | ------------------- | -------------------------- |
| 3000 | Frontend            | React/Next.js UI           |
| 8000 | API Gateway         | Main API routing           |
| 8002 | Model Inference     | Ollama/LLM models          |
| 8003 | Document Processing | Text chunking & embeddings |
| 8004 | ChromaDB            | Vector database            |
| 8005 | DocStrange          | PDF processing             |
| 8006 | Auth Service        | Authentication             |
| 8007 | APRAG Service       | Advanced RAG features      |
| 8090 | Marker API          | Document conversion        |

## 📝 Notes for Future Debugging

1. **Always check network name first:** `docker network ls`
2. **Document Processing Service is the most critical** - it connects to both Model Inference (8002) and ChromaDB (8000)
3. **Environment variables override defaults** - always use `--env-file rag3_for_local/.env`
4. **Service startup order matters** - start ChromaDB and Model Inference first
5. **Use container names for internal communication** - not localhost or external ports
6. **Timestamped collections are normal** - enhanced search logic handles them automatically

---

_This document should be referenced for all future Docker operations to prevent time waste and configuration errors._
