/**
 * RAG Types for Session Detail Page
 * Comprehensive type definitions for RAG system configuration and operations
 */

// Embedding Models
export interface EmbeddingModel {
  id: string;
  name: string;
  provider: "huggingface" | "openai" | "alibaba" | "local";
  dimensions: number;
  max_tokens: number;
  description?: string;
  is_available: boolean;
  performance_score?: number;
  language_support: string[];
}

export const EMBEDDING_MODELS: Record<string, EmbeddingModel> = {
  "sentence-transformers/all-MiniLM-L6-v2": {
    id: "sentence-transformers/all-MiniLM-L6-v2",
    name: "All-MiniLM-L6-v2",
    provider: "huggingface",
    dimensions: 384,
    max_tokens: 512,
    description: "Fast, lightweight embedding model",
    is_available: true,
    performance_score: 85,
    language_support: ["en", "tr"],
  },
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
    id: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    name: "Paraphrase Multilingual MiniLM-L12-v2",
    provider: "huggingface",
    dimensions: 384,
    max_tokens: 512,
    description: "Multilingual embedding model",
    is_available: true,
    performance_score: 88,
    language_support: ["en", "tr", "de", "fr", "es"],
  },
  "gte-large-en-v1.5": {
    id: "gte-large-en-v1.5",
    name: "GTE Large EN v1.5",
    provider: "alibaba",
    dimensions: 1024,
    max_tokens: 8192,
    description: "High-performance Alibaba embedding model",
    is_available: true,
    performance_score: 92,
    language_support: ["en", "tr"],
  },
};

// Reranker Models
export interface RerankerModel {
  id: string;
  name: string;
  provider: "cohere" | "huggingface" | "alibaba" | "local";
  max_query_length: number;
  max_document_length: number;
  description?: string;
  is_available: boolean;
  performance_score?: number;
  language_support: string[];
}

export const RERANKER_MODELS: Record<string, RerankerModel> = {
  "gte-reranker-base": {
    id: "gte-reranker-base",
    name: "GTE Reranker Base",
    provider: "alibaba",
    max_query_length: 512,
    max_document_length: 8192,
    description: "Alibaba GTE reranker model",
    is_available: true,
    performance_score: 90,
    language_support: ["en", "tr"],
  },
  "ms-marco-MiniLM-L-6-v2": {
    id: "ms-marco-MiniLM-L-6-v2",
    name: "MS-MARCO MiniLM-L-6-v2",
    provider: "huggingface",
    max_query_length: 512,
    max_document_length: 4096,
    description: "Microsoft MARCO reranker",
    is_available: true,
    performance_score: 87,
    language_support: ["en"],
  },
};

// RAG Configuration
export interface RagConfig {
  // Basic RAG settings
  retrieval_method: "semantic" | "hybrid" | "keyword";
  top_k: number;
  similarity_threshold: number;

  // Embedding settings
  embedding_model: string;
  embedding_provider: "huggingface" | "openai" | "alibaba" | "local";

  // Reranking settings
  use_reranker: boolean;
  reranker_model?: string;
  reranker_provider?: "cohere" | "huggingface" | "alibaba" | "local";
  rerank_top_k?: number;

  // Generation settings
  llm_provider: "ollama" | "grok" | "openai";
  llm_model: string;
  temperature: number;
  max_tokens: number;

  // Context settings
  context_window_size: number;
  max_context_chunks: number;

  // Advanced settings
  use_query_expansion: boolean;
  use_context_compression: boolean;
  response_language: "tr" | "en" | "auto";
}

// RAG Processing States
export interface RagProcessingState {
  step:
    | "idle"
    | "embedding"
    | "retrieving"
    | "reranking"
    | "generating"
    | "complete"
    | "error";
  progress: number;
  message: string;
  startTime?: string;
  processingTime?: number;
  error?: string;
}

// RAG Query and Response
export interface RagQuery {
  query: string;
  session_id: string;
  config?: Partial<RagConfig>;
  context?: string[];
  metadata?: Record<string, any>;
}

export interface RagContext {
  chunk_id: string;
  text: string;
  score: number;
  metadata: Record<string, any>;
  source_file?: string;
  document_name?: string;
}

export interface RagResponse {
  success: boolean;
  response: string;
  contexts: RagContext[];
  query: string;
  processing_time_ms: number;
  token_usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  metadata: {
    embedding_model: string;
    reranker_model?: string;
    llm_model: string;
    retrieval_method: string;
    contexts_used: number;
    similarity_scores: number[];
  };
  error?: string;
}

// RAG Metrics and Analytics
export interface RagMetrics {
  session_id: string;
  total_queries: number;
  average_response_time: number;
  average_context_relevance: number;
  context_utilization_rate: number;
  model_performance: {
    embedding_model: string;
    reranker_model?: string;
    llm_model: string;
    accuracy_score: number;
  };
  query_patterns: {
    most_common_topics: string[];
    query_length_distribution: Record<string, number>;
    response_satisfaction: number;
  };
}

// Model Management
export interface ModelStatus {
  model_id: string;
  status: "available" | "loading" | "unavailable" | "error";
  load_time?: number;
  memory_usage?: string;
  last_used?: string;
  error_message?: string;
}

export interface ModelProvider {
  id: string;
  name: string;
  type: "embedding" | "reranker" | "llm";
  status: "online" | "offline" | "error";
  models: string[];
  capabilities: string[];
  rate_limits?: {
    requests_per_minute: number;
    tokens_per_minute: number;
  };
}

// RAG Store State
export interface RagStoreState {
  // Configuration
  config: RagConfig;
  availableEmbeddingModels: EmbeddingModel[];
  availableRerankerModels: RerankerModel[];
  availableProviders: ModelProvider[];

  // Processing state
  processingState: RagProcessingState;
  isProcessing: boolean;

  // Query and response
  currentQuery: string;
  queryHistory: RagQuery[];
  lastResponse: RagResponse | null;

  // Performance and metrics
  metrics: RagMetrics | null;
  modelStatuses: Record<string, ModelStatus>;

  // UI state
  loading: boolean;
  error: string | null;
  success: string | null;

  // Actions - Configuration
  setConfig: (config: Partial<RagConfig>) => void;
  updateConfigField: <K extends keyof RagConfig>(
    field: K,
    value: RagConfig[K]
  ) => void;
  resetConfig: () => void;

  // Actions - Models
  setAvailableModels: (
    embeddingModels: EmbeddingModel[],
    rerankerModels: RerankerModel[]
  ) => void;
  setModelStatuses: (statuses: Record<string, ModelStatus>) => void;
  updateModelStatus: (modelId: string, status: ModelStatus) => void;

  // Actions - Processing
  setProcessingState: (state: RagProcessingState) => void;
  setIsProcessing: (processing: boolean) => void;
  updateProcessingStep: (
    step: RagProcessingState["step"],
    message: string,
    progress: number
  ) => void;

  // Actions - Query/Response
  setCurrentQuery: (query: string) => void;
  addToQueryHistory: (query: RagQuery) => void;
  setLastResponse: (response: RagResponse | null) => void;

  // Actions - UI
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setSuccess: (success: string | null) => void;
  clearMessages: () => void;

  // Actions - Metrics
  setMetrics: (metrics: RagMetrics) => void;
  updateMetrics: (updates: Partial<RagMetrics>) => void;

  // Utility actions
  resetState: () => void;
}

// API Request/Response Types
export interface ConfigureRagRequest {
  session_id: string;
  config: RagConfig;
}

export interface ConfigureRagResponse {
  success: boolean;
  config: RagConfig;
  message?: string;
}

export interface QueryRagRequest extends RagQuery {}

export interface QueryRagResponse extends RagResponse {}

export interface RagHealthCheck {
  embedding_service: boolean;
  reranker_service: boolean;
  llm_service: boolean;
  database: boolean;
  overall_status: "healthy" | "degraded" | "down";
}

// Presets and Templates
export interface RagPreset {
  id: string;
  name: string;
  description: string;
  config: RagConfig;
  use_case: string;
  performance_level: "fast" | "balanced" | "accurate";
  language: "tr" | "en" | "multilingual";
}

export const RAG_PRESETS: Record<string, RagPreset> = {
  fast_turkish: {
    id: "fast_turkish",
    name: "Hızlı Türkçe",
    description: "Türkçe içerik için hızlı yanıt",
    config: {
      retrieval_method: "semantic",
      top_k: 5,
      similarity_threshold: 0.7,
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
      embedding_provider: "huggingface",
      use_reranker: false,
      llm_provider: "ollama",
      llm_model: "llama3.2:3b",
      temperature: 0.3,
      max_tokens: 500,
      context_window_size: 4000,
      max_context_chunks: 3,
      use_query_expansion: false,
      use_context_compression: true,
      response_language: "tr",
    },
    use_case: "Hızlı soru-cevap",
    performance_level: "fast",
    language: "tr",
  },
  accurate_multilingual: {
    id: "accurate_multilingual",
    name: "Hassas Çok Dilli",
    description: "Yüksek doğruluk için optimize edilmiş",
    config: {
      retrieval_method: "hybrid",
      top_k: 10,
      similarity_threshold: 0.6,
      embedding_model: "gte-large-en-v1.5",
      embedding_provider: "alibaba",
      use_reranker: true,
      reranker_model: "gte-reranker-base",
      reranker_provider: "alibaba",
      rerank_top_k: 5,
      llm_provider: "grok",
      llm_model: "grok-beta",
      temperature: 0.1,
      max_tokens: 1000,
      context_window_size: 8000,
      max_context_chunks: 7,
      use_query_expansion: true,
      use_context_compression: false,
      response_language: "auto",
    },
    use_case: "Akademik araştırma ve detaylı analiz",
    performance_level: "accurate",
    language: "multilingual",
  },
};

// Export utility functions
export const getDefaultRagConfig = (): RagConfig => ({
  retrieval_method: "semantic",
  top_k: 7,
  similarity_threshold: 0.75,
  embedding_model:
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  embedding_provider: "huggingface",
  use_reranker: true,
  reranker_model: "gte-reranker-base",
  reranker_provider: "alibaba",
  rerank_top_k: 5,
  llm_provider: "ollama",
  llm_model: "llama3.2:3b",
  temperature: 0.3,
  max_tokens: 800,
  context_window_size: 6000,
  max_context_chunks: 5,
  use_query_expansion: false,
  use_context_compression: true,
  response_language: "tr",
});
