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
    // Client-side: try to get from global state
    return (window as any).__BACKEND_API_URL__ || DEFAULT_API_URL;
  }
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
    chain_type?: "stuff" | "refine" | "map_reduce";
    top_k?: number;
    use_rerank?: boolean;
    min_score?: number;
    max_context_chars?: number;
    use_direct_llm?: boolean;
    embedding_model?: string;
  } | null;
};

export type Chunk = {
  document_name: string;
  chunk_index: number;
  chunk_text: string;
  chunk_metadata?: any;
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
}> {
  const res = await fetch(`${getApiUrl()}/rag/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
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
}> {
  const formData = new FormData();
  formData.append("session_id", sessionId);
  formData.append("markdown_files", JSON.stringify(filenames));
  formData.append("chunk_strategy", "semantic");
  formData.append("chunk_size", "1500");
  formData.append("chunk_overlap", "150");
  formData.append("embedding_model", embeddingModel);

  const res = await fetch(`${getApiUrl()}/documents/process-and-store`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getChunksForSession(sessionId: string): Promise<Chunk[]> {
  const res = await fetch(`${getApiUrl()}/sessions/${sessionId}/chunks`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error("Failed to fetch chunks");
  const data = await res.json();
  return data.chunks || [];
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

export async function configureAndProcess(data: {
  session_id: string;
  markdown_files: string[];
  chunk_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
}): Promise<{
  success: boolean;
  message: string;
  processed_count: number;
  total_chunks_added: number;
  processing_time?: number;
}> {
  const formData = new FormData();
  formData.append("session_id", data.session_id);
  formData.append("markdown_files", JSON.stringify(data.markdown_files));
  formData.append("chunk_strategy", data.chunk_strategy);
  formData.append("chunk_size", data.chunk_size.toString());
  formData.append("chunk_overlap", data.chunk_overlap.toString());
  formData.append("embedding_model", data.embedding_model);

  const res = await fetch(`${getApiUrl()}/documents/process-and-store`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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
  const res = await fetch(`${getApiUrl()}/models`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error("Failed to fetch available models");
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
    chain_type?: "stuff" | "refine" | "map_reduce";
    top_k?: number;
    use_rerank?: boolean;
    min_score?: number;
    max_context_chars?: number;
    use_direct_llm?: boolean;
    embedding_model?: string;
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
  const token = tokenManager.getAccessToken?.() || null;
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
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/students/chat-history`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      ...message,
      timestamp: new Date().toISOString(),
    }),
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Clear student's chat history for a specific session
export async function clearStudentChatHistory(
  sessionId: string
): Promise<{ success: boolean; deleted: number }> {
  const token = tokenManager.getAccessToken?.() || null;
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
    `${getApiUrl()}/api/aprag/interactions/session/${sessionId}?limit=${limit}&offset=${offset}`,
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

// Submit feedback for an interaction
export async function submitFeedback(
  feedback: FeedbackCreate
): Promise<{ feedback_id: number; message: string }> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/api/aprag/feedback`, {
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
    `${getApiUrl()}/api/aprag/recommendations/${userId}?${params.toString()}`,
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
    `${getApiUrl()}/api/aprag/recommendations/${recommendationId}/accept`,
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
    `${getApiUrl()}/api/aprag/recommendations/${recommendationId}/dismiss`,
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
    `${getApiUrl()}/api/aprag/analytics/${userId}?${params.toString()}`,
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
    `${getApiUrl()}/api/aprag/analytics/${userId}/summary?${params.toString()}`,
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
}

export interface QuestionClassificationResponse {
  success: boolean;
  topic_id: number;
  topic_title: string;
  confidence_score: number;
  question_complexity: string;
  question_type: string;
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
  const res = await fetch(`${getApiUrl()}/api/aprag/topics/extract`, {
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
  const res = await fetch(`${getApiUrl()}/api/aprag/topics/session/${sessionId}`, {
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
  const res = await fetch(`${getApiUrl()}/api/aprag/topics/${topicId}`, {
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

// Classify a question to a topic
export async function classifyQuestion(
  request: QuestionClassificationRequest
): Promise<QuestionClassificationResponse> {
  const token = tokenManager.getAccessToken?.() || null;
  const res = await fetch(`${getApiUrl()}/api/aprag/topics/classify-question`, {
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
    `${getApiUrl()}/api/aprag/topics/progress/${userId}/${sessionId}`,
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
