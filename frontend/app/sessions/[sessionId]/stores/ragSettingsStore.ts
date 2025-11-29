/**
 * RAG Settings Store - Zustand state management for RAG configuration
 * Manages RAG settings, model selection, processing states, and operations
 */

import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type {
  RagStoreState,
  RagConfig,
  RagProcessingState,
  RagQuery,
  RagResponse,
  EmbeddingModel,
  RerankerModel,
  ModelProvider,
  ModelStatus,
  RagMetrics,
} from "../types/rag.types";
import {
  getDefaultRagConfig,
  EMBEDDING_MODELS,
  RERANKER_MODELS,
  RAG_PRESETS,
} from "../types/rag.types";

// API base URL helper
const getApiUrl = () => {
  if (typeof window === "undefined") return "http://localhost:8000";
  return window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "https://api-gateway:8000";
};

// Default processing state
const DEFAULT_PROCESSING_STATE: RagProcessingState = {
  step: "idle",
  progress: 0,
  message: "Ready",
  startTime: undefined,
  processingTime: undefined,
  error: undefined,
};

export const useRagSettingsStore = create<RagStoreState>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Configuration
        config: getDefaultRagConfig(),
        availableEmbeddingModels: Object.values(EMBEDDING_MODELS),
        availableRerankerModels: Object.values(RERANKER_MODELS),
        availableProviders: [],

        // Processing state
        processingState: DEFAULT_PROCESSING_STATE,
        isProcessing: false,

        // Query and response
        currentQuery: "",
        queryHistory: [],
        lastResponse: null,

        // Performance and metrics
        metrics: null,
        modelStatuses: {},

        // UI state
        loading: false,
        error: null,
        success: null,

        // Actions - Configuration
        setConfig: (config: Partial<RagConfig>) =>
          set((state) => {
            state.config = { ...state.config, ...config };
          }),

        updateConfigField: <K extends keyof RagConfig>(
          field: K,
          value: RagConfig[K]
        ) =>
          set((state) => {
            state.config[field] = value;
          }),

        resetConfig: () =>
          set((state) => {
            state.config = getDefaultRagConfig();
          }),

        // Actions - Models
        setAvailableModels: (
          embeddingModels: EmbeddingModel[],
          rerankerModels: RerankerModel[]
        ) =>
          set((state) => {
            state.availableEmbeddingModels = embeddingModels;
            state.availableRerankerModels = rerankerModels;
          }),

        setModelStatuses: (statuses: Record<string, ModelStatus>) =>
          set((state) => {
            state.modelStatuses = statuses;
          }),

        updateModelStatus: (modelId: string, status: ModelStatus) =>
          set((state) => {
            state.modelStatuses[modelId] = status;
          }),

        // Actions - Processing
        setProcessingState: (processingState: RagProcessingState) =>
          set((state) => {
            state.processingState = processingState;
          }),

        setIsProcessing: (processing: boolean) =>
          set((state) => {
            state.isProcessing = processing;
          }),

        updateProcessingStep: (
          step: RagProcessingState["step"],
          message: string,
          progress: number
        ) =>
          set((state) => {
            state.processingState = {
              ...state.processingState,
              step,
              message,
              progress,
              startTime:
                state.processingState.startTime || new Date().toISOString(),
            };
          }),

        // Actions - Query/Response
        setCurrentQuery: (query: string) =>
          set((state) => {
            state.currentQuery = query;
          }),

        addToQueryHistory: (query: RagQuery) =>
          set((state) => {
            state.queryHistory.unshift(query);
            // Keep only last 50 queries
            if (state.queryHistory.length > 50) {
              state.queryHistory = state.queryHistory.slice(0, 50);
            }
          }),

        setLastResponse: (response: RagResponse | null) =>
          set((state) => {
            state.lastResponse = response;
            if (response) {
              state.processingState = {
                ...state.processingState,
                step: "complete",
                progress: 100,
                message: "Query completed successfully",
                processingTime: response.processing_time_ms,
              };
            }
          }),

        // Actions - UI
        setLoading: (loading: boolean) =>
          set((state) => {
            state.loading = loading;
          }),

        setError: (error: string | null) =>
          set((state) => {
            state.error = error;
            if (error) {
              state.loading = false;
              state.isProcessing = false;
              state.processingState = {
                ...state.processingState,
                step: "error",
                message: error,
              };
            }
          }),

        setSuccess: (success: string | null) =>
          set((state) => {
            state.success = success;
          }),

        clearMessages: () =>
          set((state) => {
            state.error = null;
            state.success = null;
          }),

        // Actions - Metrics
        setMetrics: (metrics: RagMetrics) =>
          set((state) => {
            state.metrics = metrics;
          }),

        updateMetrics: (updates: Partial<RagMetrics>) =>
          set((state) => {
            if (state.metrics) {
              Object.assign(state.metrics, updates);
            }
          }),

        // Utility actions
        resetState: () =>
          set((state) => {
            state.config = getDefaultRagConfig();
            state.processingState = DEFAULT_PROCESSING_STATE;
            state.isProcessing = false;
            state.currentQuery = "";
            state.queryHistory = [];
            state.lastResponse = null;
            state.metrics = null;
            state.modelStatuses = {};
            state.loading = false;
            state.error = null;
            state.success = null;
          }),
      })),
      {
        name: "rag-settings-store",
        partialize: (state) => ({
          config: state.config,
          queryHistory: state.queryHistory.slice(0, 10), // Persist only last 10 queries
        }),
      }
    ),
    {
      name: "RagSettingsStore",
    }
  )
);

// Extended store with API operations
interface ExtendedRagStore extends RagStoreState {
  // API operations
  fetchModelStatuses: () => Promise<void>;
  saveRagConfig: (sessionId: string) => Promise<void>;
  loadRagConfig: (sessionId: string) => Promise<void>;
  testRagQuery: (
    query: string,
    sessionId: string
  ) => Promise<RagResponse | null>;
  applyPreset: (presetId: string) => void;
  validateConfig: () => { isValid: boolean; errors: string[] };

  // Health and diagnostics
  checkRagHealth: () => Promise<void>;
  getPerformanceMetrics: (sessionId: string) => Promise<void>;
}

// Create extended store
export const useExtendedRagStore = create<ExtendedRagStore>()(
  devtools(
    immer((set, get) => ({
      ...useRagSettingsStore.getState(),

      // API Operations
      fetchModelStatuses: async () => {
        set((state) => {
          state.loading = true;
          state.error = null;
        });

        try {
          const response = await fetch(`${getApiUrl()}/api/models/status`);
          if (!response.ok) throw new Error("Failed to fetch model statuses");

          const data = await response.json();

          set((state) => {
            state.modelStatuses = data.model_statuses || {};
            state.availableProviders = data.providers || [];
            state.loading = false;
            state.success = "Model statuses updated successfully";
          });
        } catch (error) {
          set((state) => {
            state.loading = false;
            state.error =
              error instanceof Error
                ? error.message
                : "Failed to fetch model statuses";
          });
        }
      },

      saveRagConfig: async (sessionId: string) => {
        const { config } = get();

        set((state) => {
          state.loading = true;
          state.error = null;
        });

        try {
          const response = await fetch(
            `${getApiUrl()}/api/sessions/${sessionId}/rag-config`,
            {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ config }),
            }
          );

          if (!response.ok) throw new Error("Failed to save RAG configuration");

          set((state) => {
            state.loading = false;
            state.success = "RAG configuration saved successfully";
          });
        } catch (error) {
          set((state) => {
            state.loading = false;
            state.error =
              error instanceof Error
                ? error.message
                : "Failed to save RAG configuration";
          });
        }
      },

      loadRagConfig: async (sessionId: string) => {
        set((state) => {
          state.loading = true;
          state.error = null;
        });

        try {
          const response = await fetch(
            `${getApiUrl()}/api/sessions/${sessionId}/rag-config`
          );
          if (!response.ok) throw new Error("Failed to load RAG configuration");

          const data = await response.json();

          set((state) => {
            if (data.config) {
              state.config = { ...getDefaultRagConfig(), ...data.config };
            }
            state.loading = false;
            state.success = "RAG configuration loaded successfully";
          });
        } catch (error) {
          set((state) => {
            state.loading = false;
            state.error =
              error instanceof Error
                ? error.message
                : "Failed to load RAG configuration";
          });
        }
      },

      testRagQuery: async (query: string, sessionId: string) => {
        const { config } = get();

        set((state) => {
          state.isProcessing = true;
          state.currentQuery = query;
          state.error = null;
          state.processingState = {
            step: "embedding",
            progress: 10,
            message: "Generating embeddings...",
            startTime: new Date().toISOString(),
          };
        });

        try {
          const ragQuery: RagQuery = {
            query,
            session_id: sessionId,
            config,
            metadata: {
              test_query: true,
              timestamp: new Date().toISOString(),
            },
          };

          // Add to query history
          get().addToQueryHistory(ragQuery);

          // Update processing state - retrieving
          set((state) => {
            state.processingState = {
              ...state.processingState,
              step: "retrieving",
              progress: 30,
              message: "Retrieving relevant contexts...",
            };
          });

          const response = await fetch(`${getApiUrl()}/api/rag/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(ragQuery),
          });

          if (!response.ok) throw new Error("RAG query failed");

          // Update processing state - reranking
          set((state) => {
            state.processingState = {
              ...state.processingState,
              step: "reranking",
              progress: 60,
              message: "Reranking contexts...",
            };
          });

          // Update processing state - generating
          set((state) => {
            state.processingState = {
              ...state.processingState,
              step: "generating",
              progress: 80,
              message: "Generating response...",
            };
          });

          const ragResponse: RagResponse = await response.json();

          set((state) => {
            state.lastResponse = ragResponse;
            state.isProcessing = false;
            state.success = "Query completed successfully";
          });

          return ragResponse;
        } catch (error) {
          set((state) => {
            state.isProcessing = false;
            state.error =
              error instanceof Error ? error.message : "RAG query failed";
          });
          return null;
        }
      },

      applyPreset: (presetId: string) => {
        const preset = RAG_PRESETS[presetId];
        if (preset) {
          set((state) => {
            state.config = { ...preset.config };
            state.success = `Applied preset: ${preset.name}`;
          });
        } else {
          set((state) => {
            state.error = `Preset not found: ${presetId}`;
          });
        }
      },

      validateConfig: () => {
        const { config, availableEmbeddingModels, availableRerankerModels } =
          get();
        const errors: string[] = [];

        // Validate embedding model
        const embeddingModel = availableEmbeddingModels.find(
          (m) => m.id === config.embedding_model
        );
        if (!embeddingModel) {
          errors.push(`Invalid embedding model: ${config.embedding_model}`);
        } else if (!embeddingModel.is_available) {
          errors.push(
            `Embedding model not available: ${config.embedding_model}`
          );
        }

        // Validate reranker model if enabled
        if (config.use_reranker && config.reranker_model) {
          const rerankerModel = availableRerankerModels.find(
            (m) => m.id === config.reranker_model
          );
          if (!rerankerModel) {
            errors.push(`Invalid reranker model: ${config.reranker_model}`);
          } else if (!rerankerModel.is_available) {
            errors.push(
              `Reranker model not available: ${config.reranker_model}`
            );
          }
        }

        // Validate numeric values
        if (config.top_k <= 0 || config.top_k > 50) {
          errors.push("top_k must be between 1 and 50");
        }

        if (
          config.similarity_threshold < 0 ||
          config.similarity_threshold > 1
        ) {
          errors.push("similarity_threshold must be between 0 and 1");
        }

        if (config.temperature < 0 || config.temperature > 2) {
          errors.push("temperature must be between 0 and 2");
        }

        if (config.max_tokens <= 0 || config.max_tokens > 8192) {
          errors.push("max_tokens must be between 1 and 8192");
        }

        return {
          isValid: errors.length === 0,
          errors,
        };
      },

      checkRagHealth: async () => {
        set((state) => {
          state.loading = true;
          state.error = null;
        });

        try {
          const response = await fetch(`${getApiUrl()}/api/rag/health`);
          if (!response.ok) throw new Error("Health check failed");

          const healthData = await response.json();

          set((state) => {
            state.loading = false;
            if (healthData.overall_status === "healthy") {
              state.success = "All RAG services are healthy";
            } else {
              state.error = `RAG services status: ${healthData.overall_status}`;
            }
          });
        } catch (error) {
          set((state) => {
            state.loading = false;
            state.error =
              error instanceof Error ? error.message : "Health check failed";
          });
        }
      },

      getPerformanceMetrics: async (sessionId: string) => {
        set((state) => {
          state.loading = true;
          state.error = null;
        });

        try {
          const response = await fetch(
            `${getApiUrl()}/api/sessions/${sessionId}/rag-metrics`
          );
          if (!response.ok)
            throw new Error("Failed to fetch performance metrics");

          const metrics: RagMetrics = await response.json();

          set((state) => {
            state.metrics = metrics;
            state.loading = false;
            state.success = "Performance metrics loaded successfully";
          });
        } catch (error) {
          set((state) => {
            state.loading = false;
            state.error =
              error instanceof Error
                ? error.message
                : "Failed to load performance metrics";
          });
        }
      },
    })),
    {
      name: "ExtendedRagStore",
    }
  )
);

// Export individual hooks for better performance
export const useRagConfig = () => useRagSettingsStore((state) => state.config);
export const useRagLoading = () =>
  useRagSettingsStore((state) => state.loading);
export const useRagProcessing = () =>
  useRagSettingsStore((state) => state.isProcessing);
export const useRagError = () => useRagSettingsStore((state) => state.error);
export const useRagProcessingState = () =>
  useRagSettingsStore((state) => state.processingState);
export const useLastRagResponse = () =>
  useRagSettingsStore((state) => state.lastResponse);
export const useAvailableEmbeddingModels = () =>
  useRagSettingsStore((state) => state.availableEmbeddingModels);
export const useAvailableRerankerModels = () =>
  useRagSettingsStore((state) => state.availableRerankerModels);
export const useRagMetrics = () =>
  useRagSettingsStore((state) => state.metrics);
