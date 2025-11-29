# 🏗️ RAG Altyapısı - Detaylı Görsel Analiz ve Şematizasyon

## 📋 İçindekiler
1. [Genel Sistem Mimarisi](#genel-sistem-mimarisi)
2. [Mikroservis Yapısı](#mikroservis-yapısı)
3. [RAG Query Akış Diyagramı](#rag-query-akış-diyagramı)
4. [Document Processing Pipeline](#document-processing-pipeline)
5. [Hybrid Retrieval Sistemi](#hybrid-retrieval-sistemi)
6. [APRAG (Adaptive Personalized RAG) Mimarisi](#aprag-mimarisi)
7. [Veri Akış Şeması](#veri-akış-şeması)

---

## 🏛️ Genel Sistem Mimarisi

```mermaid
graph TB
    subgraph "Frontend Layer"
        FE[Next.js Frontend<br/>Port: 3000<br/>TypeScript/React]
    end
    
    subgraph "API Gateway Layer"
        AG[API Gateway<br/>Port: 8000<br/>FastAPI<br/>Routing & Session Management]
    end
    
    subgraph "Core Services"
        APRAG[APRAG Service<br/>Port: 8007<br/>Adaptive RAG Orchestrator]
        DOC[Document Processing Service<br/>Port: 8080<br/>Chunking & Hybrid Search]
        MODEL[Model Inference Service<br/>Port: 8002<br/>LLM & Embeddings]
        RERANK[Reranker Service<br/>Port: 8008<br/>Document Reranking]
        AUTH[Auth Service<br/>Port: 8006<br/>JWT & RBAC]
    end
    
    subgraph "Data Storage"
        CHROMA[(ChromaDB<br/>Port: 8004<br/>Vector Database)]
        SQLITE[(SQLite DB<br/>Session & User Data)]
    end
    
    subgraph "External Services"
        DOCSTRANGE[DocStrange Service<br/>Port: 8005<br/>PDF Processing]
        MARKER[Marker API<br/>Port: 8090<br/>PDF to Markdown]
    end
    
    FE -->|HTTP/HTTPS| AG
    AG -->|Proxy| APRAG
    AG -->|Proxy| DOC
    AG -->|Proxy| AUTH
    
    APRAG -->|Query| DOC
    APRAG -->|Generate| MODEL
    APRAG -->|Rerank| RERANK
    APRAG -->|Read/Write| SQLITE
    
    DOC -->|Embed| MODEL
    DOC -->|Query| CHROMA
    DOC -->|Rerank| RERANK
    
    MODEL -->|External APIs| EXT[Groq/Alibaba<br/>Ollama/HuggingFace]
    
    DOC -->|Process PDF| DOCSTRANGE
    DOC -->|Process PDF| MARKER
    
    style FE fill:#e1f5ff
    style AG fill:#fff4e1
    style APRAG fill:#e8f5e9
    style DOC fill:#f3e5f5
    style MODEL fill:#fff3e0
    style RERANK fill:#fce4ec
    style CHROMA fill:#e0f2f1
    style SQLITE fill:#e0f2f1
```

---

## 🔧 Mikroservis Yapısı

```mermaid
graph LR
    subgraph "Service Communication"
        AG[API Gateway<br/>:8000]
        
        subgraph "RAG Pipeline Services"
            APRAG[APRAG Service<br/>:8007<br/>Orchestration]
            DOC[Document Processing<br/>:8080<br/>Retrieval]
            MODEL[Model Inference<br/>:8002<br/>LLM/Embeddings]
            RERANK[Reranker<br/>:8008<br/>Relevance Scoring]
        end
        
        subgraph "Supporting Services"
            AUTH[Auth Service<br/>:8006<br/>Authentication]
            DOCSTRANGE[DocStrange<br/>:8005<br/>PDF Processing]
            MARKER[Marker API<br/>:8090<br/>PDF→MD]
        end
        
        subgraph "Storage Layer"
            CHROMA[(ChromaDB<br/>:8004)]
            SQLITE[(SQLite<br/>File-based)]
        end
    end
    
    AG --> APRAG
    AG --> DOC
    AG --> AUTH
    
    APRAG --> DOC
    APRAG --> MODEL
    APRAG --> RERANK
    APRAG --> SQLITE
    
    DOC --> MODEL
    DOC --> CHROMA
    DOC --> RERANK
    DOC --> DOCSTRANGE
    DOC --> MARKER
    
    MODEL --> EXT[External APIs]
    
    style APRAG fill:#4caf50,color:#fff
    style DOC fill:#9c27b0,color:#fff
    style MODEL fill:#ff9800,color:#fff
    style RERANK fill:#e91e63,color:#fff
```

---

## 🔄 RAG Query Akış Diyagramı

```mermaid
sequenceDiagram
    participant User as 👤 Kullanıcı
    participant Frontend as 🖥️ Frontend<br/>(Next.js)
    participant Gateway as 🚪 API Gateway<br/>(:8000)
    participant APRAG as 🎯 APRAG Service<br/>(:8007)
    participant DocProc as 📄 Document Processing<br/>(:8080)
    participant Model as 🤖 Model Inference<br/>(:8002)
    participant Rerank as 🔄 Reranker<br/>(:8008)
    participant Chroma as 💾 ChromaDB<br/>(:8004)
    participant KB as 📚 Knowledge Base<br/>(SQLite)
    
    User->>Frontend: Soru gönder
    Frontend->>Gateway: POST /api/aprag/hybrid-rag/query
    Gateway->>APRAG: Proxy request
    
    Note over APRAG: 1. Topic Classification
    APRAG->>KB: Topic matching
    KB-->>APRAG: Matched topics
    
    Note over APRAG: 2. Hybrid Retrieval
    APRAG->>DocProc: POST /query<br/>(query, top_k, use_rerank)
    
    Note over DocProc: 2.1 Embedding Generation
    DocProc->>Model: POST /embed<br/>(query text)
    Model-->>DocProc: Query embedding (1024D)
    
    Note over DocProc: 2.2 Vector Search
    DocProc->>Chroma: Query collection<br/>(top_k * 2 if rerank)
    Chroma-->>DocProc: Retrieved chunks
    
    Note over DocProc: 2.3 Hybrid Search (BM25 + Semantic)
    DocProc->>DocProc: BM25 scoring
    DocProc->>DocProc: Weighted fusion
    
    Note over DocProc: 2.4 Reranking (if enabled)
    DocProc->>Rerank: POST /rerank<br/>(query, documents)
    Rerank->>Model: Alibaba Reranker API
    Model-->>Rerank: Rerank scores
    Rerank-->>DocProc: Reranked chunks (top_k)
    
    DocProc-->>APRAG: Chunks + scores
    
    Note over APRAG: 3. Knowledge Base Retrieval
    APRAG->>KB: Get KB content<br/>(matched topics)
    KB-->>APRAG: KB items
    
    Note over APRAG: 4. Merge Results
    APRAG->>APRAG: Merge chunks + KB + QA pairs
    APRAG->>APRAG: Build context
    
    Note over APRAG: 5. LLM Generation
    APRAG->>Model: POST /models/generate<br/>(context + query)
    Model->>EXT: External API call<br/>(Groq/Alibaba)
    EXT-->>Model: Generated answer
    Model-->>APRAG: Answer text
    
    Note over APRAG: 6. Personalization (Optional)
    APRAG->>APRAG: CACS scoring<br/>ZPD adaptation
    
    APRAG-->>Gateway: HybridRAGQueryResponse
    Gateway-->>Frontend: JSON response
    Frontend-->>User: Display answer + sources
```

---

## 📥 Document Processing Pipeline

```mermaid
graph TD
    subgraph "Document Upload Flow"
        UPLOAD[📤 Document Upload<br/>PDF/DOCX/PPTX]
        UPLOAD --> GATEWAY[API Gateway<br/>/upload]
        GATEWAY --> DOCSTRANGE[DocStrange Service<br/>PDF → Markdown]
        DOCSTRANGE --> STORE[Store Markdown]
    end
    
    subgraph "Document Processing Pipeline"
        STORE --> CHUNK[Text Chunking<br/>Lightweight Strategy<br/>chunk_size: 1000<br/>overlap: 200]
        CHUNK --> EMBED[Generate Embeddings<br/>Model: text-embedding-v4<br/>Dimension: 1024D]
        EMBED --> STORE_VEC[Store in ChromaDB<br/>Collection per Session<br/>Metadata: filename, chunk_index]
    end
    
    subgraph "Query Processing Pipeline"
        QUERY[User Query] --> Q_EMBED[Query Embedding<br/>Same model: 1024D]
        Q_EMBED --> VEC_SEARCH[Vector Search<br/>k-NN in ChromaDB<br/>top_k * 2 if rerank]
        VEC_SEARCH --> BM25[BM25 Keyword Search<br/>Optional]
        BM25 --> FUSION[Hybrid Fusion<br/>Weighted: 0.7 semantic<br/>0.3 BM25]
        FUSION --> RERANK_STEP[Reranking<br/>Alibaba GTE-Rerank-v2<br/>Select top_k]
        RERANK_STEP --> CONTEXT[Build Context<br/>Max: 8000 chars]
        CONTEXT --> LLM[LLM Generation]
    end
    
    style CHUNK fill:#e3f2fd
    style EMBED fill:#f3e5f5
    style VEC_SEARCH fill:#e8f5e9
    style RERANK_STEP fill:#fff3e0
```

---

## 🔍 Hybrid Retrieval Sistemi

```mermaid
graph TB
    subgraph "Retrieval Strategy"
        QUERY[User Query:<br/>'mitoz ve mayoz farkları']
        
        subgraph "Semantic Search"
            Q_EMBED[Query Embedding<br/>text-embedding-v4<br/>1024D]
            VEC_DB[(ChromaDB<br/>Vector Store)]
            Q_EMBED --> VEC_DB
            VEC_DB --> SEM_RES[Semantic Results<br/>15 chunks<br/>Similarity scores]
        end
        
        subgraph "Keyword Search (BM25)"
            TOKENIZE[Tokenize Query<br/>Remove stopwords]
            BM25_SEARCH[BM25 Scoring<br/>Term frequency<br/>Inverse doc frequency]
            TOKENIZE --> BM25_SEARCH
            BM25_SEARCH --> BM25_RES[BM25 Results<br/>Keyword matches]
        end
        
        subgraph "Hybrid Fusion"
            FUSION[Weighted Fusion<br/>Semantic: 0.7<br/>BM25: 0.3]
            SEM_RES --> FUSION
            BM25_RES --> FUSION
            FUSION --> TOP_CHUNKS[Top 10 Chunks<br/>Before Reranking]
        end
        
        subgraph "Reranking"
            RERANK[Reranker Service<br/>Alibaba GTE-Rerank-v2<br/>Cross-encoder]
            TOP_CHUNKS --> RERANK
            RERANK --> FINAL[Top 5 Chunks<br/>After Reranking]
        end
    end
    
    QUERY --> Q_EMBED
    QUERY --> TOKENIZE
    
    style SEM_RES fill:#e3f2fd
    style BM25_RES fill:#fff3e0
    style FUSION fill:#e8f5e9
    style RERANK fill:#fce4ec
    style FINAL fill:#4caf50,color:#fff
```

---

## 🎯 APRAG (Adaptive Personalized RAG) Mimarisi

```mermaid
graph TB
    subgraph "APRAG Service Architecture"
        INPUT[User Query + Session ID]
        
        subgraph "Topic Classification"
            TC[Topic Classifier<br/>Keyword + LLM<br/>Confidence Score]
            INPUT --> TC
            TC --> TOPICS[Matched Topics<br/>+ Confidence]
        end
        
        subgraph "Hybrid Knowledge Retriever"
            HKR[Hybrid Knowledge Retriever]
            TOPICS --> HKR
            
            subgraph "Retrieval Sources"
                CHUNKS[Document Chunks<br/>from ChromaDB]
                KB[Knowledge Base<br/>from SQLite]
                QA[QA Pairs<br/>from SQLite]
            end
            
            HKR --> CHUNKS
            HKR --> KB
            HKR --> QA
        end
        
        subgraph "Reranking"
            RERANK[Reranker<br/>Alibaba API]
            CHUNKS --> RERANK
            RERANK --> RERANKED[Reranked Chunks<br/>top_k]
        end
        
        subgraph "Context Building"
            MERGE[Merge Results<br/>Chunks + KB + QA]
            RERANKED --> MERGE
            KB --> MERGE
            QA --> MERGE
            MERGE --> CONTEXT[Context String<br/>Max 8000 chars]
        end
        
        subgraph "LLM Generation"
            LLM[LLM Call<br/>Groq/Alibaba<br/>qwen-plus/llama-3.1]
            CONTEXT --> LLM
            LLM --> ANSWER[Generated Answer]
        end
        
        subgraph "Personalization (Optional)"
            CACS[CACS Scoring<br/>Conversation-Aware]
            ZPD[ZPD Adaptation<br/>Zone of Proximal Dev]
            BLOOM[Bloom Level<br/>Classification]
            COG[Cognitive Load<br/>Assessment]
            
            ANSWER --> CACS
            CACS --> ZPD
            ZPD --> BLOOM
            BLOOM --> COG
            COG --> PERSONALIZED[Personalized Response]
        end
        
        subgraph "Response"
            RESPONSE[HybridRAGQueryResponse<br/>Answer + Sources + Debug]
            PERSONALIZED --> RESPONSE
            ANSWER --> RESPONSE
        end
    end
    
    style TC fill:#e3f2fd
    style HKR fill:#f3e5f5
    style RERANK fill:#fff3e0
    style LLM fill:#e8f5e9
    style CACS fill:#fce4ec
    style RESPONSE fill:#4caf50,color:#fff
```

---

## 📊 Veri Akış Şeması

```mermaid
flowchart TD
    subgraph "Data Flow: Document Processing"
        PDF[PDF Document] --> MARKDOWN[Markdown Text]
        MARKDOWN --> CHUNKS[Text Chunks<br/>1000 chars each<br/>200 overlap]
        CHUNKS --> EMBEDDINGS[Embeddings<br/>1024D vectors]
        EMBEDDINGS --> CHROMA[(ChromaDB<br/>Collection per Session)]
    end
    
    subgraph "Data Flow: Query Processing"
        QUERY[User Query] --> Q_EMBED[Query Embedding<br/>1024D]
        Q_EMBED --> RETRIEVE[Retrieve from ChromaDB<br/>top_k * 2]
        RETRIEVE --> RERANK[Rerank to top_k]
        RERANK --> MERGE[Merge with KB + QA]
        MERGE --> CONTEXT[Context String]
        CONTEXT --> LLM_GEN[LLM Generation]
        LLM_GEN --> RESPONSE[Final Answer]
    end
    
    subgraph "Data Storage"
        CHROMA
        SQLITE[(SQLite<br/>Sessions<br/>Users<br/>KB<br/>QA Pairs)]
    end
    
    style CHROMA fill:#e0f2f1
    style SQLITE fill:#e0f2f1
    style RESPONSE fill:#4caf50,color:#fff
```

---

## 🔑 Önemli Teknik Detaylar

### Embedding Model
- **Model**: `text-embedding-v4` (Alibaba)
- **Dimension**: 1024D
- **Provider**: Alibaba DashScope API
- **Collection Dimension**: 1024D (ChromaDB)

### LLM Models
- **Primary**: `qwen-plus` (Alibaba)
- **Fallback**: `llama-3.1-8b-instant` (Groq)
- **Provider**: Groq API / Alibaba DashScope

### Reranker
- **Model**: `gte-rerank-v2` (Alibaba)
- **Type**: Cross-encoder
- **Service**: Dedicated Reranker Service (Port 8008)

### Chunking Strategy
- **Method**: Lightweight (regex-based)
- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters
- **Strategy**: Semantic-aware splitting

### Hybrid Search
- **Semantic Weight**: 0.7
- **BM25 Weight**: 0.3
- **Top K Before Rerank**: `top_k * 2`
- **Top K After Rerank**: `top_k` (default: 5)

### Knowledge Base Integration
- **Storage**: SQLite database
- **Topics**: Topic-based organization
- **Retrieval**: Topic classification → KB matching
- **Confidence Threshold**: 0.7

---

## 📈 Performans Metrikleri

### Response Times (Typical)
- **Topic Classification**: ~100-200ms
- **Vector Search**: ~200-500ms
- **Reranking**: ~500-1000ms
- **LLM Generation**: ~1000-3000ms
- **Total Query Time**: ~2-5 seconds

### Throughput
- **Concurrent Queries**: Supports multiple concurrent users
- **Rate Limiting**: 300 RPM (Auth Service)
- **Caching**: No explicit caching (stateless design)

---

## 🛠️ Deployment Architecture

```mermaid
graph TB
    subgraph "Docker Compose Services"
        subgraph "Frontend"
            FE[Frontend Container<br/>Next.js<br/>:3000]
        end
        
        subgraph "Backend Services"
            AG[API Gateway<br/>:8000]
            APRAG[APRAG Service<br/>:8007]
            DOC[Document Processing<br/>:8080]
            MODEL[Model Inference<br/>:8002]
            RERANK[Reranker<br/>:8008]
            AUTH[Auth Service<br/>:8006]
        end
        
        subgraph "Data Services"
            CHROMA[ChromaDB<br/>:8004]
        end
        
        subgraph "External Services"
            DOCSTRANGE[DocStrange<br/>:8005]
            MARKER[Marker API<br/>:8090]
        end
    end
    
    subgraph "Docker Network"
        NET[rag-network<br/>Bridge Network]
    end
    
    subgraph "Volumes"
        VOL1[session_data]
        VOL2[chroma_data]
        VOL3[database_data]
        VOL4[reranker_models]
    end
    
    FE -.->|Network| AG
    AG -.->|Network| APRAG
    AG -.->|Network| DOC
    AG -.->|Network| AUTH
    
    CHROMA -.->|Volume| VOL2
    APRAG -.->|Volume| VOL3
    
    style NET fill:#e3f2fd
    style VOL1 fill:#fff3e0
    style VOL2 fill:#fff3e0
    style VOL3 fill:#fff3e0
    style VOL4 fill:#fff3e0
```

---

## 📝 Sonuç

Bu görselleştirme, mevcut RAG altyapısının tüm bileşenlerini, veri akışını ve servisler arası iletişimi detaylı bir şekilde göstermektedir. Sistem, mikroservis mimarisi üzerine kurulu, ölçeklenebilir ve modüler bir yapıya sahiptir.

### Ana Özellikler:
- ✅ **Mikroservis Mimarisi**: Her servis bağımsız çalışır
- ✅ **Hybrid Retrieval**: Semantic + BM25 + Reranking
- ✅ **Knowledge Base Integration**: Topic-based KB retrieval
- ✅ **Adaptive Personalization**: CACS, ZPD, Bloom, Cognitive Load
- ✅ **Multi-Model Support**: Groq, Alibaba, Ollama, HuggingFace
- ✅ **Scalable Design**: Docker Compose ile kolay deployment

---

*Son Güncelleme: 26 Kasım 2025*



