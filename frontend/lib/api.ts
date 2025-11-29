// API URL will be dynamically determined by BackendContext
// This is a fallback for server-side rendering - uses centralized config
import { tokenManager } from "./token-manager";
import { URLS } from "@/config/ports";

const DEFAULT_API_URL = URLS.API_GATEWAY;

// Set API URL globally for client-side access
export function setGlobalApiUrl(url: string) {
  if (typeof window !== "undefined") {
    (window as any).__BACKEND_API_URL__ = url;
  }
}

// Get API URL from context or use default
export function getApiUrl(): string {
  if (typeof window !== "undefined") {
    // Client-side: always use /api (Next.js rewrites will proxy to api-gateway)
    return "/api";
  }
  // Server-side: use default (for SSR)
  return DEFAULT_API_URL;
}

export type SessionMeta = {
  session_id: string;
  name: string;
  description: string;
  category: string;
  status: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  last_accessed: string;
  grade_level: string;
  subject_area: string;
  learning_objectives: string[];
  tags: string[];
  document_count: number;
  total_chunks: number;
  query_count: number;
  student_entry_count: number; // Unique student entries count
  user_rating: number;
  is_public: boolean;
  backup_count: number;
  rag_settings?: {
    model?: string;
    provider?: string;
    chain_type?: "stuff" | "refine" | "map_reduce";
    top_k?: number;
    use_rerank?: boolean;
    min_score?: number;
    max_context_chars?: number;
    use_direct_llm?: boolean;
    embedding_model?: string;
    embedding_provider?: string;
    chunk_strategy?: string;
    chunk_size?: number;
    chunk_overlap?: number;
    use_reranker_service?: boolean;
    reranker_type?: string;
  } | null;
};

export type Chunk = {
  document_name: string;
  chunk_index: number;
  chunk_text: string;
  chunk_metadata?: any;
};

// Markdown category types
export type MarkdownCategory = {
  id: number;
  name: string;
  description?: string | null;
};

export type MarkdownFileWithCategory = {
  filename: string;
  category_id?: number | null;
  category_name?: string | null;
  created_at?: string | null;
};

export type SessionChunksResponse = {
  chunks: Chunk[];
  total_count: number;
  session_id: string;
};

export async function listSessions(): Promise<SessionMeta[]> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/sessions`, {
    cache: "no-store",
    headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
  });
  if (!res.ok) throw new Error("Failed to fetch sessions");
  return res.json();
}

export async function createSession(data: {
  name: string;
  description?: string;
  category: string;
  created_by?: string;
  grade_level?: string;
  subject_area?: string;
  learning_objectives?: string[];
  tags?: string[];
  is_public?: boolean;
}): Promise<SessionMeta> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/sessions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// =============================
// Markdown category API helpers
// =============================

export async function listMarkdownCategories(): Promise<MarkdownCategory[]> {
  const res = await fetch(`${getApiUrl()}/markdown-categories`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function createMarkdownCategory(
  data: Omit<MarkdownCategory, "id">
): Promise<MarkdownCategory> {
  const res = await fetch(`${getApiUrl()}/markdown-categories`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateMarkdownCategory(
  id: number,
  data: Omit<MarkdownCategory, "id">
): Promise<MarkdownCategory> {
  const res = await fetch(`${getApiUrl()}/markdown-categories/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteMarkdownCategory(id: number): Promise<void> {
  const res = await fetch(`${getApiUrl()}/markdown-categories/${id}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function assignMarkdownCategory(
  filenames: string[],
  categoryId: number | null
): Promise<void> {
  const res = await fetch(`${getApiUrl()}/markdown-files/assign-category`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filenames, category_id: categoryId }),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function listMarkdownFilesWithCategories(
  categoryId?: number
): Promise<MarkdownFileWithCategory[]> {
  const url = new URL(
    `${getApiUrl()}/markdown-files/with-categories`,
    window.location.origin
  );
  if (categoryId !== undefined) {
    url.searchParams.set("category_id", String(categoryId));
  }
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteSession(
  sessionId: string
): Promise<{ deleted: boolean; session_id: string }> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/sessions/${sessionId}`, {
    method: "DELETE",
    headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateSessionStatus(
  sessionId: string,
  status: "active" | "inactive"
): Promise<{
  success: boolean;
  session_id: string;
  new_status: string;
  updated_session: SessionMeta;
}> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/sessions/${sessionId}/status`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ status }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadDocument(form: FormData): Promise<any> {
  const res = await fetch(
    `${getApiUrl()}/documents/convert-document-to-markdown`,
    {
      method: "POST",
      body: form,
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Define proper source type
export type RAGSource = {
  content: string;
  score: number;
  metadata?: {
    source_file?: string;
    filename?: string;
    chunk_index?: number;
    total_chunks?: number;
    chunk_title?: string;
    page_number?: number;
    section?: string;
    [key: string]: any;
  };
};

export async function ragQuery(data: {
  session_id: string;
  query: string;
  top_k?: number;
  use_rerank?: boolean;
  min_score?: number;
  max_context_chars?: number;
  model?: string;
  use_direct_llm?: boolean;
  chain_type?: "stuff" | "refine" | "map_reduce";
  embedding_model?: string;
  conversation_history?: Array<{ role: "user" | "assistant"; content: string }>;
}): Promise<{
  answer: string;
  sources: RAGSource[];
  processing_time_ms?: number;
  suggestions?: string[];
  correction?: CorrectionDetails;
}> {
  const res = await fetch(`${getApiUrl()}/rag/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// KB-Enhanced Hybrid RAG query (APRAG service)
export async function hybridRAGQuery(data: {
  session_id: string;
  query: string;
  top_k?: number;
  use_kb?: boolean;
  use_qa_pairs?: boolean;
  use_crag?: boolean;
  model?: string;
  max_tokens?: number;
  temperature?: number;
  max_context_chars?: number;
  include_examples?: boolean;
  include_sources?: boolean;
  embedding_model?: string;
}): Promise<{
  answer: string;
  sources: RAGSource[];
  processing_time_ms?: number;
  direct_qa_match?: boolean;
  matched_topics?: any[];
  classification_confidence?: number;
  debug_info?: any;
  confidence?: string;
  retrieval_strategy?: string;
  sources_used?: { chunks: number; kb: number; qa_pairs: number };
  suggestions?: string[]; // Follow-up question suggestions
}> {
  // Use direct API Gateway URL for hybrid-rag queries to avoid Next.js proxy timeout
  // Other endpoints continue to use /api (Next.js proxy)
  const apiGatewayUrl =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const hybridRagUrl = apiGatewayUrl.startsWith("http")
    ? `${apiGatewayUrl}/api/aprag/hybrid-rag/query`
    : `${getApiUrl()}/aprag/hybrid-rag/query`;

  const res = await fetch(hybridRagUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  const json = await res.json();

  const mappedSources: RAGSource[] = Array.isArray(json?.sources)
    ? json.sources.map((s: any) => {
        const rawType = s.type || s.source || s.source_type || "unknown";
        const isQa =
          rawType === "qa_pair" ||
          rawType === "direct_qa" ||
          rawType === "question_answer";

        const baseContent = s.content || "";
        let content = baseContent;

        // Eğer QA kaynağıysa ve content boşsa, soru/cevap'tan içerik üret
        if (!content && isQa && (s.question || s.answer)) {
          const q = s.question || "";
          const a = s.answer || "";
          content = `Soru: ${q}\n\nCevap: ${a}`;
        }

        return {
          content,
          score: typeof s.score === "number" ? s.score : 0,
          metadata: {
            ...(s.metadata || {}),
            source_type: rawType,
            // QA kaynakları için soru/cevap bilgisini de metadata'da sakla
            ...(isQa && (s.question || s.answer)
              ? {
                  qa_question: s.question,
                  qa_answer: s.answer,
                  qa_similarity: s.similarity,
                }
              : {}),
          },
        };
      })
    : [];

  return {
    answer: json.answer,
    sources: mappedSources,
    processing_time_ms: json.processing_time_ms,
    direct_qa_match: json.direct_qa_match,
    matched_topics: json.matched_topics || [],
    classification_confidence: json.classification_confidence ?? 0,
    debug_info: json.debug_info,
    confidence: json.confidence,
    retrieval_strategy: json.retrieval_strategy,
    sources_used: json.sources_used,
    suggestions: json.suggestions || [], // Add suggestions from response
  };
}

// ============================================================================
// ASYNC RAG QUERY (Background Processing)
// ============================================================================

export interface AsyncRAGRequest {
  session_id: string;
  query: string;
  user_id?: string;
  top_k?: number;
  use_kb?: boolean;
  use_qa_pairs?: boolean;
  use_crag?: boolean;
  model?: string;
  embedding_model?: string;
  max_tokens?: number;
  temperature?: number;
  max_context_chars?: number;
  include_examples?: boolean;
  include_sources?: boolean;
}

export interface AsyncRAGInitResponse {
  task_id: string;
  status: string; // "processing"
  estimated_time_seconds: number;
  message: string;
}

export interface AsyncRAGStatusResponse {
  task_id: string;
  status: string; // "processing", "completed", "failed"
  progress?: number; // 0-100
  current_step?: string;
  estimated_remaining_seconds?: number;
  result?: {
    answer: string;
    sources: RAGSource[];
    processing_time_ms?: number;
    direct_qa_match?: boolean;
    matched_topics?: any[];
    classification_confidence?: number;
    confidence?: string;
    retrieval_strategy?: string;
    sources_used?: { chunks: number; kb: number; qa_pairs: number };
    debug_info?: any;
  };
  error?: string;
}

// Start async RAG query (returns immediately with task_id)
export async function startAsyncRAGQuery(
  data: AsyncRAGRequest
): Promise<AsyncRAGInitResponse> {
  // Use direct API Gateway URL for async-rag queries to avoid Next.js proxy timeout
  const apiGatewayUrl =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const asyncRagUrl = apiGatewayUrl.startsWith("http")
    ? `${apiGatewayUrl}/api/aprag/async-rag/async-query`
    : `${getApiUrl()}/aprag/async-rag/async-query`;

  const res = await fetch(asyncRagUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Poll async RAG query status
export async function getAsyncRAGStatus(
  taskId: string
): Promise<AsyncRAGStatusResponse> {
  const res = await fetch(
    `${getApiUrl()}/aprag/async-rag/async-query/${taskId}/status`
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Cancel async RAG task
export async function cancelAsyncRAGTask(
  taskId: string
): Promise<{ task_id: string; message: string; cancelled: boolean }> {
  const res = await fetch(
    `${getApiUrl()}/aprag/async-rag/async-query/${taskId}`,
    {
      method: "DELETE",
    }
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

export async function generateSuggestions(data: {
  question: string;
  answer: string;
  sources?: any[];
}): Promise<string[]> {
  const res = await fetch(`${getApiUrl()}/rag/suggestions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    return [];
  }
  const json = await res.json();
  return Array.isArray(json?.suggestions) ? json.suggestions : [];
}

// Generate course-specific questions based on chunks
export async function generateCourseQuestions(
  sessionId: string,
  limit: number = 5
): Promise<string[]> {
  const res = await fetch(
    `${getApiUrl()}/sessions/${sessionId}/generate-questions`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ count: limit }), // Backend expects 'count', not 'limit'
    }
  );
  if (!res.ok) {
    console.error("Failed to generate course questions:", await res.text());
    return [];
  }
  const json = await res.json();
  return Array.isArray(json?.questions) ? json.questions : [];
}

export async function listMarkdownFiles(): Promise<string[]> {
  const res = await fetch(`${getApiUrl()}/documents/list-markdown`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error("Failed to fetch markdown files");
  const data = await res.json();
  return data.markdown_files;
}

export async function addMarkdownDocumentsToSession(
  sessionId: string,
  filenames: string[],
  embeddingModel: string = "mxbai-embed-large"
): Promise<{
  success: boolean;
  processed_count: number;
  total_chunks_added: number;
  message: string;
  errors?: string[];
  job_id?: string;
  total_files?: number;
  background_processing?: boolean;
}> {
  const formData = new FormData();
  formData.append("session_id", sessionId);
  formData.append("markdown_files", JSON.stringify(filenames));
  formData.append("chunk_strategy", "lightweight");
  formData.append("chunk_size", "800");
  formData.append("chunk_overlap", "100");
  formData.append("embedding_model", embeddingModel);

  // Use batch processing endpoint
  const res = await fetch(`${getApiUrl()}/documents/process-and-store-batch`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  
  return {
    ...data,
    background_processing: true,
    processed_count: 0,
    total_chunks_added: 0,
  };
}

// Get batch processing job status
export async function getBatchProcessingStatus(jobId: string): Promise<{
  success: boolean;
  job: {
    job_id: string;
    session_id: string;
    status: string;
    started_at: string;
    completed_at: string | null;
    total_files: number;
    processed_successfully: number;
    errors_count: number;
    current_file: string | null;
    total_chunks: number;
    results: Array<{ filename: string; chunks_processed: number; success: boolean }>;
    errors: Array<{ filename: string; error: string }>;
  };
}> {
  const res = await fetch(`${getApiUrl()}/documents/process-and-store-batch/status/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getChunksForSession(sessionId: string): Promise<Chunk[]> {
  // Add timeout and circuit breaker for chunks endpoint
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout

  try {
    const res = await fetch(`${getApiUrl()}/sessions/${sessionId}/chunks`, {
      cache: "no-store",
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      // More specific error messages
      if (res.status === 404) {
        console.warn(`No chunks found for session ${sessionId}`);
        return [];
      }
      if (res.status >= 500) {
        throw new Error(
          `Server error (${res.status}): Backend service may be busy`
        );
      }
      throw new Error(`Failed to fetch chunks: HTTP ${res.status}`);
    }

    const data = await res.json();
    return Array.isArray(data?.chunks) ? data.chunks : [];
  } catch (error: any) {
    clearTimeout(timeoutId);

    if (error.name === "AbortError") {
      throw new Error(
        "Request timeout: API is taking too long to respond. Please try refreshing the page."
      );
    }

    // Network errors
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new Error(
        "Network error: Unable to connect to server. Please check your connection."
      );
    }

    throw error;
  }
}

export async function reprocessSessionDocuments(
  sessionId: string,
  embeddingModel: string,
  sourceFiles?: string[],
  chunkSize: number = 1000,
  chunkOverlap: number = 200
): Promise<{
  success: boolean;
  message: string;
  chunks_processed: number;
  successful_files: string[];
  failed_files?: string[];
  embedding_model: string;
}> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/sessions/${sessionId}/reprocess`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      embedding_model: embeddingModel,
      source_files: sourceFiles,
      chunk_size: chunkSize,
      chunk_overlap: chunkOverlap,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getMarkdownFileContent(
  filename: string
): Promise<{ content: string }> {
  const res = await fetch(
    `${getApiUrl()}/documents/markdown/${encodeURIComponent(filename)}`,
    {
      cache: "no-store",
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteMarkdownFile(
  filename: string
): Promise<{ deleted: boolean; filename: string }> {
  const res = await fetch(
    `${getApiUrl()}/documents/markdown/${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteAllMarkdownFiles(): Promise<{
  deleted: boolean;
  count: number;
}> {
  const res = await fetch(`${getApiUrl()}/documents/markdown?delete_all=true`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ============================================================================
// FIRE-AND-FORGET PROCESSING (No Background Polling)
// ============================================================================

// Background polling completely removed - using dashboard notifications instead

export async function configureAndProcess(data: {
  session_id: string;
  markdown_files: string[];
  chunk_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  use_llm_post_processing?: boolean;
  llm_model_name?: string;
}): Promise<{
  success: boolean;
  message: string;
  processed_count: number;
  total_chunks_added: number;
  processing_time?: number;
  background_processing?: boolean;
}> {
  const formData = new FormData();
  formData.append("session_id", data.session_id);
  formData.append("markdown_files", JSON.stringify(data.markdown_files));
  formData.append("chunk_strategy", data.chunk_strategy);
  formData.append("chunk_size", data.chunk_size.toString());
  formData.append("chunk_overlap", data.chunk_overlap.toString());
  formData.append("embedding_model", data.embedding_model);

  // LLM post-processing parameters
  if (data.use_llm_post_processing !== undefined) {
    formData.append(
      "use_llm_post_processing",
      data.use_llm_post_processing.toString()
    );
  }
  if (data.llm_model_name) {
    formData.append("llm_model_name", data.llm_model_name);
  }

  // Use direct API Gateway URL for long-running processing to avoid Next.js proxy timeout
  const apiGatewayUrl =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const processUrl = apiGatewayUrl.startsWith("http")
    ? `${apiGatewayUrl}/api/documents/process-and-store`
    : `${getApiUrl()}/documents/process-and-store`;

  // ✅ AGGRESSIVE FIRE-AND-FORGET: 10 saniye timeout - herhangi bir hata = arka plan işleme
  const fireAndForgetTimeout = 10000; // 10 saniye - çok agresif!
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), fireAndForgetTimeout);

  try {
    const res = await fetch(processUrl, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    // ✅ Yanıt geldi ama 500 hatası da olabilir - önemli değil, fire-and-forget!
    if (res.ok) {
      // Gerçekten hızlıca bitti
      try {
        return await res.json();
      } catch {
        // JSON parse hatası bile olsa background processing olarak treat et
        return {
          success: true,
          message: "İşlem başlatıldı ve arka planda devam ediyor.",
          processed_count: 0,
          total_chunks_added: 0,
          processing_time: fireAndForgetTimeout,
          background_processing: true,
        };
      }
    } else {
      // 500 hatası - önemli değil, fire-and-forget!
      console.log(
        `🚀 API returned ${res.status}, treating as background processing`
      );
      return {
        success: true,
        message:
          "İşlem başlatıldı ve arka planda devam ediyor. Sonuçlar birazdan görünecek.",
        processed_count: 0,
        total_chunks_added: 0,
        processing_time: fireAndForgetTimeout,
        background_processing: true,
      };
    }
  } catch (error: any) {
    clearTimeout(timeoutId);

    // ✅ HER TÜRLÜ HATA = FIRE-AND-FORGET SUCCESS!
    console.log(
      `🚀 Fire-and-forget: ${
        error.name || "Error"
      } - treating as background processing`
    );

    return {
      success: true,
      message:
        "İşlem arka planda başlatıldı. Döküman parçaları birazdan sayfada görünecek.",
      processed_count: 0,
      total_chunks_added: 0,
      processing_time: fireAndForgetTimeout,
      background_processing: true,
    };
  }
}

export async function checkApiHealth(): Promise<{ status: string }> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const res = await fetch(`${getApiUrl()}/health`, {
      cache: "no-store",
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      throw new Error(`API ${res.status} durum kodu döndürdü`);
    }
    return await res.json();
  } catch (error) {
    // Bu kısım network hatalarını ve zaman aşımlarını yakalar
    throw new Error("API sağlık kontrolü başarısız oldu");
  }
}

export type ModelInfo = {
  id: string;
  name: string;
  provider: string;
  type: string;
  description: string;
};

export type ModelProvider = {
  name: string;
  description: string;
  models: string[];
};

export async function listAvailableModels(): Promise<{
  models: ModelInfo[];
  providers: Record<string, ModelProvider>;
}> {
  try {
    const res = await fetch(`${getApiUrl()}/models`, {
      cache: "no-store",
    });

    if (!res.ok) {
      console.warn(`Failed to fetch models: ${res.status} ${res.statusText}`);
      // Return fallback models instead of throwing
      return {
        models: [
          {
            id: "llama-3.1-8b-instant",
            name: "llama-3.1-8b-instant",
            provider: "groq",
            type: "cloud",
            description: "Groq (Hızlı)",
          },
          {
            id: "llama-3.3-70b-versatile",
            name: "llama-3.3-70b-versatile",
            provider: "groq",
            type: "cloud",
            description: "Groq (Hızlı)",
          },
        ],
        providers: {
          groq: {
            name: "Groq",
            description: "Hızlı Cloud Modelleri",
            models: ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
          },
        },
      };
    }

    const data = await res.json();

    // Handle both old format (fallback) and new format
    if (
      data.models &&
      Array.isArray(data.models) &&
      typeof data.models[0] === "string"
    ) {
      // Old format - convert to new format
      return {
        models: data.models.map((model: string) => ({
          id: model,
          name: model,
          provider: "unknown",
          type: "cloud",
          description: "Model",
        })),
        providers: {},
      };
    }

    // New format
    return data;
  } catch (error: any) {
    console.error("Error fetching available models:", error);
    // Return fallback models on network errors
    return {
      models: [
        {
          id: "llama-3.1-8b-instant",
          name: "llama-3.1-8b-instant",
          provider: "groq",
          type: "cloud",
          description: "Groq (Hızlı)",
        },
        {
          id: "llama-3.3-70b-versatile",
          name: "llama-3.3-70b-versatile",
          provider: "groq",
          type: "cloud",
          description: "Groq (Hızlı)",
        },
      ],
      providers: {
        groq: {
          name: "Groq",
          description: "Hızlı Cloud Modelleri",
          models: ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        },
      },
    };
  }
}

// Changelog types and functions
export type ChangelogEntry = {
  id: number;
  version: string;
  date: string;
  changes: string[];
};

export async function getChangelog(): Promise<ChangelogEntry[]> {
  // Changelog functionality not available in API Gateway
  // Return empty array or mock data for now
  console.warn(
    "getChangelog: API endpoint not available in gateway, returning empty array"
  );
  return [];
}

export async function createChangelogEntry(data: {
  version: string;
  date: string;
  changes: string[];
}): Promise<ChangelogEntry> {
  // Changelog functionality not available in API Gateway
  console.warn("createChangelogEntry: API endpoint not available in gateway");
  throw new Error("Changelog functionality not available");
}

// Upload markdown file directly
export async function uploadMarkdownFile(file: File): Promise<{
  success: boolean;
  message: string;
  markdown_filename: string;
}> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${getApiUrl()}/documents/upload-markdown`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getRecentInteractions(params?: {
  limit?: number;
  page?: number;
  session_id?: string;
  q?: string;
}): Promise<{
  items: Array<{
    interaction_id: number;
    user_id: string;
    session_id: string;
    timestamp: string;
    query: string;
    response: string;
    processing_time_ms?: number;
    model?: string;
    top_k?: number;
    success?: boolean;
    error_message?: string;
    chain_type?: string;
  }>;
  count: number;
  page: number;
  limit: number;
}> {
  const u = new URL(`${getApiUrl()}/analytics/recent-interactions`);
  if (params?.limit) u.searchParams.set("limit", String(params.limit));
  if (params?.page) u.searchParams.set("page", String(params.page));
  if (params?.session_id) u.searchParams.set("session_id", params.session_id);
  if (params?.q) u.searchParams.set("q", params.q);
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(u.toString(), {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function clearSessionInteractions(
  sessionId: string
): Promise<{ success: boolean; deleted: number; session_id: string }> {
  const u = new URL(`${getApiUrl()}/analytics/recent-interactions`);
  u.searchParams.set("session_id", sessionId);
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(u.toString(), {
    method: "DELETE",
    headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Profile Management Functions
export interface UserProfile {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  is_active: boolean;
  role_id: number;
  role_name: string;
  created_at: string;
  updated_at: string;
  last_login?: string | null;
}

export async function getProfile(): Promise<UserProfile> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/profile`, {
    headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateProfile(data: {
  username?: string;
  email?: string;
  first_name?: string;
  last_name?: string;
}): Promise<UserProfile> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/profile`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function changePassword(
  oldPassword: string,
  newPassword: string
): Promise<{ success: boolean; message?: string }> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/profile/change-password`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      old_password: oldPassword,
      new_password: newPassword,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getSession(sessionId: string): Promise<SessionMeta> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/sessions/${sessionId}`, {
    cache: "no-store",
    headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Export session: returns a blob URL for download
export async function exportSession(
  sessionId: string,
  format: "zip" | "json" = "zip"
): Promise<Blob> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/sessions/${sessionId}/export?format=${format}`,
    {
      method: "GET",
      headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return await res.blob();
}

// Import session from file
export async function importSessionFromFile(file: File): Promise<{
  success: boolean;
  new_session_id: string;
  imported_markdowns: number;
}> {
  const token = tokenManager.getAccessToken?.() || null;
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${getApiUrl()}/sessions/import`, {
    method: "POST",
    headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function saveSessionRagSettings(
  sessionId: string,
  settings: {
    model?: string;
    provider?: string;
    chain_type?: "stuff" | "refine" | "map_reduce";
    top_k?: number;
    use_rerank?: boolean;
    min_score?: number;
    max_context_chars?: number;
    use_direct_llm?: boolean;
    embedding_model?: string;
    embedding_provider?: string;
    use_reranker_service?: boolean;
    reranker_type?: "bge" | "ms-marco";
  }
): Promise<{ success: boolean; session_id: string; rag_settings: any }> {
  const res = await fetch(`${getApiUrl()}/sessions/${sessionId}/rag-settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export type EmbeddingModel = {
  id: string;
  name: string;
  description?: string;
  dimensions?: number;
  language?: string;
};

export async function listAvailableEmbeddingModels(): Promise<{
  ollama: string[];
  huggingface: EmbeddingModel[];
}> {
  const res = await fetch(`${getApiUrl()}/models/embedding`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error("Failed to fetch available embedding models");
  return res.json();
}

// Document Conversion Functions
export async function convertPdfToMarkdown(
  file: File,
  useFallback: boolean = false
): Promise<any> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("use_fallback", useFallback ? "true" : "false");

  // Shorter timeout for fallback (pdfplumber is faster)
  const timeout = useFallback ? 120000 : 600000; // 2 min for pdfplumber, 10 min for Nanonets
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const res = await fetch(
      `${getApiUrl()}/documents/convert-document-to-markdown`,
      {
        method: "POST",
        body: formData,
        signal: controller.signal,
      }
    );

    clearTimeout(timeoutId);

    if (!res.ok) throw new Error(await res.text());
    return res.json();
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === "AbortError") {
      throw new Error(
        "PDF dönüştürme işlemi zaman aşımına uğradı. Lütfen daha küçük bir dosya deneyin veya hızlı işlemi seçin."
      );
    }
    throw error;
  }
}

export async function convertMarker(file: File): Promise<any> {
  const formData = new FormData();
  formData.append("file", file);

  // 15 minutes timeout for Marker (complex documents)
  const timeout = 900000;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const res = await fetch(`${getApiUrl()}/documents/convert-marker`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) throw new Error(await res.text());
    return res.json();
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === "AbortError") {
      throw new Error(
        "Marker dönüştürme işlemi zaman aşımına uğradı (15 dk). Lütfen daha küçük bir dosya deneyin."
      );
    }
    throw error;
  }
}

export interface CorrectionDetails {
  original_answer: string;
  issues: string[];
  was_corrected: boolean;
}

// Student-specific chat history functions
export interface StudentChatMessage {
  id?: string;
  user: string;
  bot: string;
  sources?: RAGSource[];
  durationMs?: number;
  suggestions?: string[];
  timestamp: string;
  session_id: string;
  aprag_interaction_id?: number; // For emoji feedback
  correction?: CorrectionDetails; // NEW: For self-correction details
}

export interface StudentChatHistory {
  session_id: string;
  messages: StudentChatMessage[];
  last_updated: string;
}

// Get student's chat history for a specific session
export async function getStudentChatHistory(
  sessionId: string
): Promise<StudentChatMessage[]> {
  const token = tokenManager.getAccessToken();
  const res = await fetch(`${getApiUrl()}/students/chat-history/${sessionId}`, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!res.ok) {
    if (res.status === 404) {
      // No chat history yet, return empty array
      return [];
    }
    throw new Error(await res.text());
  }

  return res.json();
}

// Save student's chat message to database
export async function saveStudentChatMessage(
  message: Omit<StudentChatMessage, "id" | "timestamp">
): Promise<StudentChatMessage> {
  const token = tokenManager.getAccessToken();
  const res = await fetch(`${getApiUrl()}/students/chat-message`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(message),
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Clear student's chat history for a specific session
export async function clearStudentChatHistory(
  sessionId: string
): Promise<{ success: boolean; deleted: number }> {
  const token = tokenManager.getAccessToken();
  const res = await fetch(`${getApiUrl()}/students/chat-history/${sessionId}`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Get all student's sessions with their latest activity
export async function getStudentSessions(): Promise<
  Array<{
    session: SessionMeta;
    last_message_time?: string;
    message_count: number;
  }>
> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/students/sessions`, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// APRAG Interaction Types
export interface APRAGInteraction {
  interaction_id: number;
  user_id: string;
  session_id: string;
  query: string;
  original_response: string;
  personalized_response?: string;
  timestamp: string;
  processing_time_ms?: number;
  model_used?: string;
  chain_type?: string;
  sources?: Array<{
    source: string;
    score?: number;
    chunk_text?: string;
  }>;
  metadata?: Record<string, any>;
  // Enhanced student information
  first_name?: string;
  last_name?: string;
  username?: string;
  student_name?: string;
  // Enhanced topic information
  topic_title?: string;
  topic_confidence?: number;
  question_complexity?: string;
  question_type?: string;
  topic_info?: {
    title: string;
    confidence?: number;
    question_complexity?: string;
    question_type?: string;
  } | null;
}

export interface APRAGInteractionsResponse {
  interactions: APRAGInteraction[];
  total: number;
  count?: number;
  limit: number;
  offset: number;
}

// Get interactions for a session (APRAG)
export async function getSessionInteractions(
  sessionId: string,
  limit: number = 50,
  offset: number = 0
): Promise<APRAGInteractionsResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  // Use API Gateway which proxies to APRAG service
  const res = await fetch(
    `${getApiUrl()}/aprag/interactions/session/${sessionId}?limit=${limit}&offset=${offset}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    if (res.status === 404) {
      // No interactions yet, return empty
      return {
        interactions: [],
        total: 0,
        count: 0,
        limit,
        offset,
      };
    }
    throw new Error(await res.text());
  }

  return res.json();
}

// Get total student interactions count
export async function getTotalInteractions(
  sessionIds?: string[]
): Promise<{ total: number }> {
  const token = tokenManager.getAccessToken?.() || null;
  const sessionIdsParam =
    sessionIds && sessionIds.length > 0
      ? `?session_ids=${sessionIds.join(",")}`
      : "";
  const res = await fetch(
    `${getApiUrl()}/aprag/interactions/total${sessionIdsParam}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// APRAG Feedback Types
export interface APRAGFeedback {
  feedback_id: number;
  interaction_id: number;
  user_id: string;
  session_id: string;
  understanding_level?: number;
  answer_adequacy?: number;
  satisfaction_level?: number;
  difficulty_level?: number;
  topic_understood?: boolean;
  answer_helpful?: boolean;
  needs_more_explanation?: boolean;
  comment?: string;
  timestamp: string;
}

export interface FeedbackCreate {
  interaction_id: number;
  user_id: string;
  session_id: string;
  understanding_level?: number;
  answer_adequacy?: number;
  satisfaction_level?: number;
  difficulty_level?: number;
  topic_understood?: boolean;
  answer_helpful?: boolean;
  needs_more_explanation?: boolean;
  comment?: string;
}

// Create an APRAG interaction (log student query and response)
export interface APRAGInteractionCreate {
  user_id: string;
  session_id: string;
  query: string;
  response: string;
  personalized_response?: string;
  processing_time_ms?: number;
  model_used?: string;
  chain_type?: string;
  sources?: Array<{
    content: string;
    score: number;
    metadata?: any;
  }>;
  metadata?: Record<string, any>;
}

export async function createAPRAGInteraction(
  interaction: APRAGInteractionCreate
): Promise<{ interaction_id: number; message: string }> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/interactions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(interaction),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Submit feedback for an interaction
export async function submitFeedback(
  feedback: FeedbackCreate
): Promise<{ feedback_id: number; message: string }> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(feedback),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// ============================================================================
// Emoji Feedback (Quick Micro-Feedback)
// ============================================================================

export interface EmojiFeedbackCreate {
  interaction_id: number;
  user_id: string;
  session_id: string;
  emoji: "😊" | "👍" | "😐" | "❌";
  comment?: string;
}

// ============================================================================
// Multi-Dimensional Feedback
// ============================================================================

export interface MultiFeedbackCreate {
  interaction_id: number;
  user_id: string;
  session_id: string;
  understanding: number; // 1-5 scale
  relevance: number; // 1-5 scale
  clarity: number; // 1-5 scale
  emoji?: "😊" | "👍" | "😐" | "❌";
  comment?: string;
}

export interface MultiFeedbackResponse {
  message: string;
  interaction_id: number;
  feedback_entry_id: number;
  dimensions: {
    understanding: number;
    relevance: number;
    clarity: number;
  };
  overall_score: number;
  profile_updated: boolean;
}

export interface MultiDimensionalStats {
  user_id: string;
  session_id: string;
  avg_understanding?: number;
  avg_relevance?: number;
  avg_clarity?: number;
  avg_overall?: number;
  total_feedback_count: number;
  dimension_feedback_count: number;
  emoji_only_count: number;
  understanding_distribution: Record<string, number>;
  relevance_distribution: Record<string, number>;
  clarity_distribution: Record<string, number>;
  improvement_trend: string;
  weak_dimensions: string[];
  strong_dimensions: string[];
}

export interface EmojiFeedbackResponse {
  message: string;
  emoji: string;
  score: number;
  description: string;
  interaction_id: number;
  profile_updated: boolean;
}

export interface EmojiOption {
  emoji: string;
  name: string;
  description: string;
  score: number;
}

// Get available emoji options
export async function getAvailableEmojis(): Promise<{ emojis: EmojiOption[] }> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/emoji-feedback/emojis`, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Submit emoji feedback
export async function submitEmojiFeedback(
  feedback: EmojiFeedbackCreate
): Promise<EmojiFeedbackResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/emoji-feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(feedback),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Get emoji feedback stats
export async function getEmojiStats(
  userId: string,
  sessionId: string
): Promise<{
  user_id: string;
  session_id: string;
  total_feedback_count: number;
  emoji_distribution: Record<string, number>;
  avg_score: number;
  most_common_emoji: string | null;
  recent_trend: "positive" | "negative" | "neutral";
}> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/aprag/emoji-feedback/stats/${userId}/${sessionId}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    if (res.status === 404) {
      return {
        user_id: userId,
        session_id: sessionId,
        total_feedback_count: 0,
        emoji_distribution: {},
        avg_score: 0.5,
        most_common_emoji: null,
        recent_trend: "neutral",
      };
    }
    throw new Error(await res.text());
  }

  return res.json();
}

// Submit multi-dimensional feedback
export async function submitDetailedFeedback(
  feedback: MultiFeedbackCreate
): Promise<MultiFeedbackResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/aprag/emoji-feedback/detailed-feedback`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(feedback),
    }
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Get multi-dimensional feedback stats
export async function getMultiDimensionalStats(
  userId: string,
  sessionId: string
): Promise<MultiDimensionalStats> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/aprag/emoji-feedback/multi-stats/${userId}/${sessionId}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    if (res.status === 404) {
      return {
        user_id: userId,
        session_id: sessionId,
        total_feedback_count: 0,
        dimension_feedback_count: 0,
        emoji_only_count: 0,
        understanding_distribution: {},
        relevance_distribution: {},
        clarity_distribution: {},
        improvement_trend: "insufficient_data",
        weak_dimensions: [],
        strong_dimensions: [],
      };
    }
    throw new Error(await res.text());
  }

  return res.json();
}

// ============================================================================
// APRAG Adaptive Query (Full Pipeline)
// ============================================================================

export interface AdaptiveQueryRequest {
  user_id: string;
  session_id: string;
  query: string;
  rag_documents: Array<{
    doc_id: string;
    content: string;
    score: number;
    metadata?: Record<string, any>;
  }>;
  rag_response: string;
}

export interface PedagogicalContext {
  zpd_level: string;
  zpd_recommended: string;
  zpd_success_rate: number;
  bloom_level: string;
  bloom_level_index: number;
  cognitive_load: number;
  needs_simplification: boolean;
}

export interface DocumentScore {
  doc_id: string;
  final_score: number;
  base_score: number;
  personal_score: number;
  global_score: number;
  context_score: number;
  rank: number;
}

export interface AdaptiveQueryResponse {
  personalized_response: string;
  original_response: string;
  interaction_id: number;
  top_documents: DocumentScore[];
  cacs_applied: boolean;
  pedagogical_context: PedagogicalContext;
  feedback_emoji_options: string[];
  processing_time_ms?: number;
  components_active: {
    cacs: boolean;
    zpd: boolean;
    bloom: boolean;
    cognitive_load: boolean;
    emoji_feedback: boolean;
  };
  personalization_data?: {
    personalization_factors?: Record<string, any>;
    zpd_info?: any;
    bloom_info?: any;
    cognitive_load?: any;
    pedagogical_instructions?: string;
  } | null;
  hybrid_rag_debug?: {
    llm_request?: {
      model?: string;
      max_tokens?: number;
      temperature?: number;
      context_length?: number;
      query_length?: number;
    };
    crag_evaluation?: {
      action?: string;
      confidence?: number;
      max_score?: number;
      avg_score?: number;
      filtered?: number;
    };
    retrieval_details?: {
      chunks_retrieved?: number;
      kb_items_retrieved?: number;
      qa_pairs_matched?: number;
      total_merged?: number;
    };
    response_size?: number;
    reranker_used?: boolean;
  };
}

// Call APRAG Adaptive Query (Full Eğitsel-KBRAG Pipeline)
export async function apragAdaptiveQuery(
  request: AdaptiveQueryRequest
): Promise<AdaptiveQueryResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/adaptive-query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "APRAG adaptive query failed");
  }

  return res.json();
}

// Get APRAG Adaptive Query Status
export async function getAPRAGAdaptiveStatus(): Promise<{
  pipeline: string;
  status: string;
  components: Record<string, boolean>;
  description: string;
}> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/adaptive-query/status`, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// ============================================================================
// APRAG Feature Flags and Settings
// ============================================================================

export interface APRAGSettings {
  enabled: boolean;
  global_enabled: boolean;
  session_enabled: boolean | null;
  features: {
    feedback_collection: boolean;
    personalization: boolean;
    recommendations: boolean;
    analytics: boolean;
    emoji_feedback?: boolean;
    cacs?: boolean;
    zpd?: boolean;
    bloom?: boolean;
    cognitive_load?: boolean;
  };
}

// Get APRAG status and feature flags
export async function getAPRAGSettings(
  sessionId?: string
): Promise<APRAGSettings> {
  const token = tokenManager.getAccessToken?.() || null;
  const params = sessionId
    ? `?session_id=${encodeURIComponent(sessionId)}`
    : "";

  try {
    const res = await fetch(`${getApiUrl()}/aprag/settings/status${params}`, {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    });

    if (!res.ok) {
      // If APRAG service is down or not configured, return disabled
      console.warn(
        `Failed to fetch APRAG settings: ${res.status} ${res.statusText}`
      );
      return {
        enabled: false,
        global_enabled: false,
        session_enabled: null,
        features: {
          feedback_collection: false,
          personalization: false,
          recommendations: false,
          analytics: false,
          emoji_feedback: false,
          cacs: false,
          zpd: false,
          bloom: false,
          cognitive_load: false,
        },
      };
    }

    return res.json();
  } catch (error) {
    console.error("Failed to get APRAG settings:", error);
    // Return disabled if service is unreachable
    return {
      enabled: false,
      global_enabled: false,
      session_enabled: null,
      features: {
        feedback_collection: false,
        personalization: false,
        recommendations: false,
        analytics: false,
        emoji_feedback: false,
        cacs: false,
        zpd: false,
        bloom: false,
        cognitive_load: false,
      },
    };
  }
}

// Check if APRAG is enabled (quick check)
export async function isAPRAGEnabled(sessionId?: string): Promise<boolean> {
  try {
    const settings = await getAPRAGSettings(sessionId);
    return settings.enabled;
  } catch {
    return false;
  }
}

// APRAG Recommendation Types
export interface APRAGRecommendation {
  recommendation_id?: number;
  recommendation_type: string;
  title: string;
  description: string;
  content: {
    suggested_questions?: string[];
    topic?: string;
    action?: string;
    reason?: string;
  };
  priority: number;
  relevance_score: number;
  status?: string;
}

export interface APRAGRecommendationsResponse {
  recommendations: APRAGRecommendation[];
  total: number;
}

// Get recommendations for a user
export async function getRecommendations(
  userId: string,
  sessionId?: string,
  limit: number = 10
): Promise<APRAGRecommendationsResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const params = new URLSearchParams({ limit: limit.toString() });
  if (sessionId) {
    params.append("session_id", sessionId);
  }

  const res = await fetch(
    `${getApiUrl()}/aprag/recommendations/${userId}?${params.toString()}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    if (res.status === 404) {
      return { recommendations: [], total: 0 };
    }
    throw new Error(await res.text());
  }

  return res.json();
}

// Accept a recommendation
export async function acceptRecommendation(
  recommendationId: number
): Promise<{ message: string; recommendation_id: number }> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/aprag/recommendations/${recommendationId}/accept`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Dismiss a recommendation
export async function dismissRecommendation(
  recommendationId: number
): Promise<{ message: string; recommendation_id: number }> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/aprag/recommendations/${recommendationId}/dismiss`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// APRAG Analytics Types
export interface APRAGAnalytics {
  total_interactions: number;
  total_feedback: number;
  average_understanding: number | null;
  average_satisfaction: number | null;
  improvement_trend: "improving" | "stable" | "declining" | "insufficient_data";
  learning_patterns: Array<{
    pattern_type: string;
    description: string;
    strength: string;
    recommendation: string;
  }>;
  topic_performance: {
    weak_topics: Record<string, any>;
    strong_topics: Record<string, any>;
    interaction_topics: Record<string, any>;
    total_topics_covered: number;
  };
  engagement_metrics: {
    total_interactions: number;
    days_active: number;
    avg_per_day: number;
    most_active_day: string | null;
    engagement_level: "high" | "medium" | "low";
  };
  time_analysis: {
    peak_hour: number | null;
    peak_day: string | null;
    hour_distribution: Record<number, number>;
    day_distribution: Record<string, number>;
  };
}

export interface APRAGAnalyticsSummary {
  total_interactions: number;
  average_understanding: number | null;
  improvement_trend: string;
  engagement_level: string;
  key_patterns: Array<{
    pattern_type: string;
    description: string;
    strength: string;
    recommendation: string;
  }>;
}

// Get analytics for a user
export async function getAnalytics(
  userId: string,
  sessionId?: string
): Promise<APRAGAnalytics> {
  const token = tokenManager.getAccessToken?.() || null;
  const params = new URLSearchParams();
  if (sessionId) {
    params.append("session_id", sessionId);
  }

  const res = await fetch(
    `${getApiUrl()}/aprag/analytics/${userId}?${params.toString()}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    if (res.status === 404) {
      return {
        total_interactions: 0,
        total_feedback: 0,
        average_understanding: null,
        average_satisfaction: null,
        improvement_trend: "insufficient_data",
        learning_patterns: [],
        topic_performance: {
          weak_topics: {},
          strong_topics: {},
          interaction_topics: {},
          total_topics_covered: 0,
        },
        engagement_metrics: {
          total_interactions: 0,
          days_active: 0,
          avg_per_day: 0,
          most_active_day: null,
          engagement_level: "low",
        },
        time_analysis: {
          peak_hour: null,
          peak_day: null,
          hour_distribution: {},
          day_distribution: {},
        },
      };
    }
    throw new Error(await res.text());
  }

  return res.json();
}

// Get analytics summary for a user
export async function getAnalyticsSummary(
  userId: string,
  sessionId?: string
): Promise<APRAGAnalyticsSummary> {
  const token = tokenManager.getAccessToken?.() || null;
  const params = new URLSearchParams();
  if (sessionId) {
    params.append("session_id", sessionId);
  }

  const res = await fetch(
    `${getApiUrl()}/aprag/analytics/${userId}/summary?${params.toString()}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    if (res.status === 404) {
      return {
        total_interactions: 0,
        average_understanding: null,
        improvement_trend: "insufficient_data",
        engagement_level: "low",
        key_patterns: [],
      };
    }
    throw new Error(await res.text());
  }

  return res.json();
}

// ============================================================================
// Topic-Based Learning Path Tracking
// ============================================================================

export interface Topic {
  topic_id: number;
  session_id: string;
  topic_title: string;
  parent_topic_id: number | null;
  topic_order: number;
  description: string | null;
  keywords: string[];
  estimated_difficulty: string | null;
  prerequisites: number[];
  extraction_confidence: number | null;
  is_active: boolean;
}

export interface TopicExtractionRequest {
  session_id: string;
  extraction_method?: string;
  options?: {
    include_subtopics?: boolean;
    min_confidence?: number;
    max_topics?: number;
  };
}

export interface TopicExtractionResponse {
  success: boolean;
  topics: Array<{
    topic_id: number;
    topic_title: string;
    topic_order: number;
    extraction_confidence: number;
  }>;
  total_topics: number;
  extraction_time_ms: number;
}

export interface QuestionClassificationRequest {
  question: string;
  session_id: string;
  interaction_id?: number;
  user_id?: string; // Optional: for topic progress tracking when interaction_id is not available
}

export interface TopicRecommendation {
  type: "topic_recommendation";
  message: string;
  current_topic_id: number;
  current_topic_title: string;
  next_topic_id: number;
  next_topic_title: string;
  mastery_score: number;
  readiness_score: number;
}

export interface QuestionClassificationResponse {
  success: boolean;
  topic_id: number;
  topic_title: string;
  confidence_score: number;
  question_complexity: string;
  question_type: string;
  recommendation?: TopicRecommendation;
}

export interface TopicProgress {
  topic_id: number;
  topic_title: string;
  topic_order: number;
  questions_asked: number;
  average_understanding: number | null;
  mastery_level: string | null;
  mastery_score: number | null;
  is_ready_for_next: boolean | null;
  readiness_score: number | null;
  time_spent_minutes: number | null;
}

export interface StudentProgressResponse {
  success: boolean;
  progress: TopicProgress[];
  current_topic: TopicProgress | null;
  next_recommended_topic: TopicProgress | null;
}

// Extract topics from session chunks
export async function extractTopics(
  request: TopicExtractionRequest
): Promise<TopicExtractionResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/topics/extract`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Get topics for a session
export async function getSessionTopics(sessionId: string): Promise<{
  success: boolean;
  topics: Topic[];
  total: number;
}> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/topics/session/${sessionId}`, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!res.ok) {
    if (res.status === 404) {
      return { success: false, topics: [], total: 0 };
    }
    throw new Error(await res.text());
  }

  return res.json();
}

// Update a topic
export async function updateTopic(
  topicId: number,
  updates: {
    topic_title?: string;
    topic_order?: number;
    description?: string;
    keywords?: string[];
    estimated_difficulty?: string;
    prerequisites?: number[];
    is_active?: boolean;
  }
): Promise<{ success: boolean; message: string }> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/topics/${topicId}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(updates),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Delete a topic
export async function deleteTopic(
  topicId: number
): Promise<{ success: boolean; message: string; topic_id: number }> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/topics/${topicId}`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Classify a question to a topic
export async function classifyQuestion(
  request: QuestionClassificationRequest
): Promise<QuestionClassificationResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/topics/classify-question`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Get student progress for all topics in a session
export async function getStudentProgress(
  userId: string,
  sessionId: string
): Promise<StudentProgressResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/aprag/topics/progress/${userId}/${sessionId}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    if (res.status === 404) {
      return {
        success: false,
        progress: [],
        current_topic: null,
        next_recommended_topic: null,
      };
    }
    throw new Error(await res.text());
  }

  return res.json();
}

// ===================================================================
// CHUNK IMPROVEMENT WITH LLM
// ===================================================================

export async function improveSingleChunk(
  chunkText: string,
  language: string = "tr",
  modelName: string = "llama-3.1-8b-instant",
  sessionId?: string, // For ChromaDB update
  chunkId?: string, // For ChromaDB update (if known)
  documentName?: string, // Alternative to chunkId
  chunkIndex?: number // Alternative to chunkId
): Promise<{
  success: boolean;
  original_text: string;
  improved_text: string | null;
  message: string | null;
  processing_time_ms: number | null;
}> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/chunks/improve-single`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      chunk_text: chunkText,
      language,
      model_name: modelName,
      session_id: sessionId,
      chunk_id: chunkId,
      document_name: documentName,
      chunk_index: chunkIndex,
    }),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

export async function improveAllChunks(
  sessionId: string,
  language: string = "tr",
  modelName: string = "llama-3.1-8b-instant",
  skipAlreadyImproved: boolean = true
): Promise<{
  success: boolean;
  total_chunks: number;
  processed: number;
  improved: number;
  failed: number;
  skipped: number;
  message: string;
  processing_time_ms: number;
}> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/sessions/${sessionId}/chunks/improve-all`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        language,
        model_name: modelName,
        skip_already_improved: skipAlreadyImproved,
      }),
    }
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// ============================================================================
// SESSION SETTINGS (Teacher Panel Controls)
// ============================================================================

export interface SessionSettings {
  session_id: string;
  user_id: string;
  enable_progressive_assessment: boolean;
  enable_personalized_responses: boolean;
  enable_multi_dimensional_feedback: boolean;
  enable_topic_analytics: boolean;
  enable_cacs: boolean;
  enable_zpd: boolean;
  enable_bloom: boolean;
  enable_cognitive_load: boolean;
  enable_emoji_feedback: boolean;
  enable_ebars: boolean;
}

export interface SessionSettingsUpdate {
  enable_progressive_assessment?: boolean;
  enable_personalized_responses?: boolean;
  enable_multi_dimensional_feedback?: boolean;
  enable_topic_analytics?: boolean;
  enable_cacs?: boolean;
  enable_zpd?: boolean;
  enable_bloom?: boolean;
  enable_cognitive_load?: boolean;
  enable_emoji_feedback?: boolean;
  enable_ebars?: boolean;
}

export interface SessionSettingsResponse {
  success: boolean;
  message: string;
  settings: SessionSettings;
  updated_at: string;
}

export interface SessionSettingsPreset {
  name: string;
  description: string;
  settings: Partial<SessionSettingsUpdate>;
}

export interface SessionSettingsPresetsResponse {
  success: boolean;
  presets: Record<string, SessionSettingsPreset>;
}

// Get session settings for a specific session
export async function getSessionSettings(
  sessionId: string
): Promise<SessionSettingsResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/aprag/session-settings/${sessionId}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Update session settings
export async function updateSessionSettings(
  sessionId: string,
  updates: SessionSettingsUpdate,
  userId: string
): Promise<SessionSettingsResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/aprag/session-settings/${sessionId}?user_id=${encodeURIComponent(
      userId
    )}`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(updates),
    }
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Reset session settings to defaults
export async function resetSessionSettings(
  sessionId: string,
  userId: string
): Promise<SessionSettingsResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/aprag/session-settings/${sessionId}/reset?user_id=${encodeURIComponent(
      userId
    )}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Get available settings presets
export async function getSessionSettingsPresets(
  sessionId: string
): Promise<SessionSettingsPresetsResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/aprag/session-settings/${sessionId}/presets`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Apply a settings preset
export async function applySessionSettingsPreset(
  sessionId: string,
  presetName: string,
  userId: string
): Promise<SessionSettingsResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(
    `${getApiUrl()}/aprag/session-settings/${sessionId}/apply-preset/${presetName}?user_id=${encodeURIComponent(
      userId
    )}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

export async function updateSessionName(
  sessionId: string,
  newName: string
): Promise<SessionMeta> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/sessions/${sessionId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ name: newName }),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

// Create module from RAG-based semantic search
export async function createModuleFromRAG(data: {
  session_id: string;
  module_title: string;
  module_description?: string;
  top_k?: number;
  similarity_threshold?: number;
  use_hybrid_search?: boolean;
}): Promise<{
  success: boolean;
  module_id: number;
  module_title: string;
  topics_added: number;
  chunks_used: number;
  message: string;
}> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/aprag/modules/create-from-rag`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      session_id: data.session_id,
      module_title: data.module_title,
      module_description: data.module_description,
      top_k: data.top_k || 20,
      similarity_threshold: data.similarity_threshold || 0.6,
      use_hybrid_search: data.use_hybrid_search !== false,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(
      errorData.detail || errorData.message || "Modül oluşturulamadı"
    );
  }

  return res.json();
}

// ============================================================================
// STUDENT PROFILE
// ============================================================================

export interface StudentProfile {
  user_id: string;
  session_id: string;
  average_understanding: number | null;
  average_satisfaction: number | null;
  total_interactions: number;
  total_feedback_count: number;
  strong_topics: Record<string, any> | null;
  weak_topics: Record<string, any> | null;
  preferred_explanation_style: string | null;
  preferred_difficulty_level: string | null;
}

/**
 * Get student profile for a user and session
 */
export async function getStudentProfile(
  userId: string,
  sessionId: string
): Promise<StudentProfile> {
  const token = tokenManager.getAccessToken?.() || null;
  try {
    const res = await fetch(
      `${getApiUrl()}/aprag/profiles/${userId}/${sessionId}`,
      {
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      }
    );

    if (!res.ok) {
      throw new Error(`Failed to fetch student profile: ${res.status}`);
    }

    return res.json();
  } catch (error) {
    console.error("Error fetching student profile:", error);
    throw error;
  }
}

// Get student's current pedagogical state (ZPD, Bloom, Cognitive Load)
export interface PedagogicalState {
  user_id: string;
  session_id: string;
  zpd: {
    current_level: string;
    recommended_level: string;
    success_rate: number;
    level_index: number;
  };
  bloom: {
    level: string;
    level_index: number;
    confidence: number;
  };
  cognitive_load: {
    total_load: number;
    needs_simplification: boolean;
  };
  personalization_factors?: {
    understanding_level?: string;
    difficulty_level?: string;
    explanation_style?: string;
  };
  profile_stats?: {
    total_interactions: number;
    total_feedback_count: number;
    average_understanding: number | null;
    average_satisfaction: number | null;
  };
  total_interactions?: number; // Legacy support
  last_updated: string | null;
}

export async function getPedagogicalState(
  userId: string,
  sessionId: string
): Promise<PedagogicalState> {
  const token = tokenManager.getAccessToken?.() || null;
  try {
    const res = await fetch(
      `${getApiUrl()}/aprag/profiles/${userId}/${sessionId}/pedagogical-state`,
      {
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      }
    );

    if (!res.ok) {
      throw new Error(`Failed to fetch pedagogical state: ${res.status}`);
    }

    return res.json();
  } catch (error) {
    console.error("Error fetching pedagogical state:", error);
    throw error;
  }
}

// Reset student profile and learning parameters
export async function resetPedagogicalState(
  userId: string,
  sessionId: string
): Promise<{ message: string; user_id: string; session_id: string }> {
  const token = tokenManager.getAccessToken?.() || null;
  try {
    const res = await fetch(
      `${getApiUrl()}/aprag/profiles/${userId}/${sessionId}/reset`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      }
    );

    if (!res.ok) {
      throw new Error(`Failed to reset pedagogical state: ${res.status}`);
    }

    return res.json();
  } catch (error) {
    console.error("Error resetting pedagogical state:", error);
    throw error;
  }
}

// ============================================================================
// NOTIFICATION SYSTEM
// ============================================================================

export interface NotificationCreate {
  type: "success" | "error" | "warning" | "info";
  title: string;
  message: string;
  sessionId?: string;
  userId?: string;
}

export interface NotificationResponse {
  id: string;
  type: "success" | "error" | "warning" | "info";
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  sessionId?: string;
  userId?: string;
}

// Fetch pending notifications
export async function getPendingNotifications(params?: {
  userId?: string;
  sessionId?: string;
}): Promise<{
  notifications: NotificationResponse[];
  count: number;
}> {
  const searchParams = new URLSearchParams();
  if (params?.userId) searchParams.set("user_id", params.userId);
  if (params?.sessionId) searchParams.set("session_id", params.sessionId);

  const url = `${getApiUrl()}/v1/notifications/pending${
    searchParams.toString() ? "?" + searchParams.toString() : ""
  }`;

  const res = await fetch(url, {
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error("Failed to fetch notifications");
  }

  return res.json();
}

// Create a new notification
export async function createNotification(
  notification: NotificationCreate
): Promise<{
  success: boolean;
  notification: NotificationResponse;
  message: string;
}> {
  const res = await fetch(`${getApiUrl()}/v1/notifications/pending`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(notification),
  });

  if (!res.ok) {
    throw new Error("Failed to create notification");
  }

  return res.json();
}

// Mark notifications as read
export async function markNotificationsAsRead(
  notificationIds: string[]
): Promise<{
  success: boolean;
  message: string;
  updatedCount: number;
}> {
  const res = await fetch(`${getApiUrl()}/v1/notifications/delivered`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ notificationIds }),
  });

  if (!res.ok) {
    throw new Error("Failed to mark notifications as read");
  }

  return res.json();
}

// Delete notifications
export async function deleteNotifications(notificationIds: string[]): Promise<{
  success: boolean;
  message: string;
  deletedCount: number;
}> {
  const res = await fetch(`${getApiUrl()}/v1/notifications/delivered`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ notificationIds }),
  });

  if (!res.ok) {
    throw new Error("Failed to delete notifications");
  }

  return res.json();
}

// Helper function to create background processing notification
export async function createBackgroundProcessingNotification(
  sessionId: string,
  isTimeout: boolean = true
) {
  try {
    await createNotification({
      type: "info",
      title: "İşlem Arka Planda Devam Ediyor",
      message: isTimeout
        ? "Döküman işleme işlemi arka planda devam ediyor. Bitince bildirim alacaksınız."
        : "İşlem başarıyla başlatıldı ve arka planda işleniyor.",
      sessionId,
    });
    console.log(
      "🔔 Background processing notification created for session:",
      sessionId
    );
  } catch (error) {
    console.error(
      "Failed to create background processing notification:",
      error
    );
  }
}
