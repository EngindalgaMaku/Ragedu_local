/**
 * Session Types for Session Detail Page
 * Comprehensive type definitions for session-related data structures
 */

export interface SessionMeta {
  session_id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  user_id: string;
  rag_settings?: RagSettings;
  document_count?: number;
  chunk_count?: number;
  status: "active" | "inactive" | "processing";
}

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
  created_at: string;
  updated_at: string;
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
}

export interface SessionSettingsPreset {
  name: string;
  description: string;
  settings: SessionSettingsUpdate;
}

export interface SessionSettingsPresetsResponse {
  success: boolean;
  presets: {
    [key: string]: SessionSettingsPreset;
  };
}

export interface RagSettings {
  provider: string;
  model: string;
  embedding_provider: string;
  embedding_model: string;
  use_reranker_service: boolean;
  reranker_type: string;
  top_k: number;
  use_rerank: boolean;
  min_score: number;
  max_context_chars: number;
}

export interface SessionInteraction {
  interaction_id: string;
  session_id: string;
  user_id: string;
  query: string;
  original_response: string;
  personalized_response?: string;
  timestamp: string;
  processing_time_ms?: number;
  sources?: Array<{
    source: string;
    score?: number;
  }>;
  topic_info?: {
    title: string;
    confidence?: number;
  };
  student_name?: string;
  first_name?: string;
  last_name?: string;
}

export type APRAGInteraction = SessionInteraction;

export interface SessionStoreState {
  // Current session data
  currentSession: SessionMeta | null;
  sessions: SessionMeta[];

  // Loading states
  loading: boolean;
  error: string | null;
  success: string | null;

  // Session settings
  sessionSettings: SessionSettings | null;
  settingsLoading: boolean;

  // Session interactions
  interactions: SessionInteraction[];
  interactionsLoading: boolean;
  interactionsPage: number;
  interactionsTotal: number;

  // Actions
  setCurrentSession: (session: SessionMeta | null) => void;
  setSessions: (sessions: SessionMeta[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setSuccess: (success: string | null) => void;

  // Session settings actions
  setSessionSettings: (settings: SessionSettings | null) => void;
  setSettingsLoading: (loading: boolean) => void;
  updateSessionSetting: (
    key: keyof SessionSettingsUpdate,
    value: boolean
  ) => void;

  // Interactions actions
  setInteractions: (interactions: SessionInteraction[]) => void;
  setInteractionsLoading: (loading: boolean) => void;
  setInteractionsPage: (page: number) => void;
  setInteractionsTotal: (total: number) => void;

  // Utility actions
  clearMessages: () => void;
  resetSessionState: () => void;
}

// Session API Response Types
export interface SessionResponse {
  success: boolean;
  session: SessionMeta;
  message?: string;
}

export interface SessionsListResponse {
  success: boolean;
  sessions: SessionMeta[];
  total?: number;
  message?: string;
}

export interface SessionSettingsResponse {
  success: boolean;
  settings: SessionSettings;
  message?: string;
}

export interface SessionInteractionsResponse {
  success: boolean;
  interactions: SessionInteraction[];
  total: number;
  page: number;
  per_page: number;
  message?: string;
}

// Session Configuration Types
export interface SessionConfig {
  id: string;
  name: string;
  description?: string;
  rag_settings?: Partial<RagSettings>;
  session_settings?: Partial<SessionSettingsUpdate>;
}

export interface SessionCreateRequest {
  name: string;
  description?: string;
  rag_settings?: Partial<RagSettings>;
  session_settings?: Partial<SessionSettingsUpdate>;
}

export interface SessionUpdateRequest {
  name?: string;
  description?: string;
  rag_settings?: Partial<RagSettings>;
  session_settings?: Partial<SessionSettingsUpdate>;
}

// Session Statistics
export interface SessionStats {
  session_id: string;
  total_chunks: number;
  total_documents: number;
  total_interactions: number;
  average_response_time: number;
  last_activity: string;
  active_users: number;
}
