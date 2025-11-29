/**
 * Chunks Types for Session Detail Page
 * Comprehensive type definitions for document chunks and related data structures
 */

export interface ChunkMetadata {
  chunk_id?: string;
  source_file?: string;
  filename?: string;
  document_name?: string;
  source_files?: string[] | string;
  llm_improved?: boolean;
  improvement_timestamp?: string;
  improvement_model?: string;
  original_length?: number;
  improved_length?: number;
  quality_score?: number;
  processing_time_ms?: number;
}

export interface Chunk {
  id: string;
  session_id: string;
  document_name: string;
  chunk_index: number;
  chunk_text: string;
  chunk_metadata?: ChunkMetadata;
  embedding?: number[];
  created_at: string;
  updated_at: string;
  token_count?: number;
  character_count: number;
  language?: string;
  quality_metrics?: ChunkQualityMetrics;
}

export interface ChunkQualityMetrics {
  coherence_score?: number;
  completeness_score?: number;
  readability_score?: number;
  information_density?: number;
  semantic_clarity?: number;
}

export interface ChunkProcessingConfig {
  chunk_strategy: "lightweight" | "semantic" | "fixed";
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  use_llm_post_processing: boolean;
  llm_model_name?: string;
  llm_provider?: "ollama" | "grok" | "openai";
}

export interface ChunkProcessingResult {
  success: boolean;
  processed_count: number;
  total_chunks_added: number;
  failed_count?: number;
  processing_time_ms: number;
  errors?: string[];
  chunks?: Chunk[];
}

export interface ChunkFilter {
  document_name?: string;
  search_term?: string;
  min_length?: number;
  max_length?: number;
  has_llm_improvement?: boolean;
  language?: string;
  quality_threshold?: number;
}

export interface ChunkStoreState {
  // Chunks data
  chunks: Chunk[];
  filteredChunks: Chunk[];
  totalChunks: number;

  // Pagination
  currentPage: number;
  chunksPerPage: number;
  totalPages: number;

  // Loading states
  loading: boolean;
  processing: boolean;
  error: string | null;
  success: string | null;

  // Filters
  activeFilters: ChunkFilter;
  selectedDocumentFilter: string | null;
  searchTerm: string;
  showAllFiles: boolean;

  // Document management
  availableDocuments: string[];
  processedFiles: Set<string>;

  // Processing states
  processingStep: string;
  llmProcessingEnabled: boolean;

  // Actions
  setChunks: (chunks: Chunk[]) => void;
  setFilteredChunks: (chunks: Chunk[]) => void;
  addChunks: (chunks: Chunk[]) => void;
  updateChunk: (chunkId: string, updates: Partial<Chunk>) => void;
  deleteChunk: (chunkId: string) => void;

  // Loading actions
  setLoading: (loading: boolean) => void;
  setProcessing: (processing: boolean) => void;
  setError: (error: string | null) => void;
  setSuccess: (success: string | null) => void;
  setProcessingStep: (step: string) => void;

  // Pagination actions
  setCurrentPage: (page: number) => void;
  setChunksPerPage: (perPage: number) => void;

  // Filter actions
  setActiveFilters: (filters: ChunkFilter) => void;
  updateFilter: (key: keyof ChunkFilter, value: any) => void;
  clearFilters: () => void;
  setSelectedDocumentFilter: (document: string | null) => void;
  setSearchTerm: (term: string) => void;
  setShowAllFiles: (show: boolean) => void;

  // Document actions
  setAvailableDocuments: (documents: string[]) => void;
  setProcessedFiles: (files: Set<string>) => void;
  addProcessedFile: (file: string) => void;

  // Utility actions
  applyFilters: () => void;
  clearMessages: () => void;
  resetChunkState: () => void;
}

// Chunk API Response Types
export interface ChunksResponse {
  success: boolean;
  chunks: Chunk[];
  total?: number;
  page?: number;
  per_page?: number;
  message?: string;
}

export interface ChunkResponse {
  success: boolean;
  chunk: Chunk;
  message?: string;
}

export interface ChunkUploadRequest {
  session_id: string;
  markdown_files: string[];
  chunk_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  use_llm_post_processing: boolean;
  llm_model_name?: string;
}

export interface ChunkImprovementRequest {
  chunk_id: string;
  llm_provider: "ollama" | "grok" | "openai";
  llm_model: string;
  improvement_type: "clarity" | "completeness" | "coherence" | "all";
}

export interface ChunkImprovementResponse {
  success: boolean;
  original_chunk: Chunk;
  improved_chunk: Chunk;
  improvements: {
    clarity_score: number;
    completeness_score: number;
    coherence_score: number;
    overall_improvement: number;
  };
  processing_time_ms: number;
  message?: string;
}

// Chunk Statistics
export interface ChunkStats {
  session_id: string;
  total_chunks: number;
  total_documents: number;
  average_chunk_size: number;
  total_characters: number;
  llm_improved_chunks: number;
  improvement_rate: number;
  languages: Record<string, number>;
  quality_distribution: {
    high: number;
    medium: number;
    low: number;
  };
}

// Document-related types
export interface DocumentInfo {
  name: string;
  chunk_count: number;
  total_characters: number;
  processing_date: string;
  status: "processed" | "processing" | "failed";
  language?: string;
  llm_improved_chunks: number;
}

export interface MarkdownFile {
  filename: string;
  size: number;
  last_modified: string;
  is_processed: boolean;
  chunk_count?: number;
}

export interface FileUploadModal {
  isOpen: boolean;
  selectedFiles: string[];
  availableFiles: MarkdownFile[];
  processingConfig: ChunkProcessingConfig;
  isProcessing: boolean;
  processingStep: string;
  error: string | null;
}
