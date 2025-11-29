/**
 * Chunks Store - Zustand state management for document chunks
 * Manages chunk data, filtering, pagination, and processing states
 */

import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { getChunksForSession } from "@/lib/api";
import type {
  ChunkStoreState,
  Chunk,
  ChunkFilter,
  ChunkProcessingConfig,
  ChunkProcessingResult,
  ChunkStats,
  DocumentInfo,
} from "../types/chunks.types";

// Default processing configuration
const DEFAULT_PROCESSING_CONFIG: ChunkProcessingConfig = {
  chunk_strategy: "semantic",
  chunk_size: 1000,
  chunk_overlap: 200,
  embedding_model:
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  use_llm_post_processing: true,
  llm_model_name: "llama3.2:3b",
  llm_provider: "ollama",
};

// API base URL helper
const getApiUrl = () => {
  if (typeof window === "undefined") return "http://localhost:8000";
  return window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "https://api-gateway:8000";
};

export const useChunksStore = create<ChunkStoreState>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Chunks data
        chunks: [],
        filteredChunks: [],
        totalChunks: 0,

        // Pagination
        currentPage: 1,
        chunksPerPage: 20,
        totalPages: 1,

        // Loading states
        loading: false,
        processing: false,
        error: null,
        success: null,

        // Filters
        activeFilters: {},
        selectedDocumentFilter: null,
        searchTerm: "",
        showAllFiles: true,

        // Document management
        availableDocuments: [],
        processedFiles: new Set(),

        // Processing states
        processingStep: "",
        llmProcessingEnabled: true,

        // Actions - Chunks Management
        setChunks: (chunks: Chunk[]) =>
          set((state) => {
            state.chunks = chunks;
            state.totalChunks = chunks.length;
            state.totalPages = Math.ceil(chunks.length / state.chunksPerPage);
            state.availableDocuments = [
              ...new Set(chunks.map((c) => c.document_name)),
            ];
          }),

        setFilteredChunks: (chunks: Chunk[]) =>
          set((state) => {
            state.filteredChunks = chunks;
          }),

        addChunks: (chunks: Chunk[]) =>
          set((state) => {
            const newChunks = [...state.chunks, ...chunks];
            state.chunks = newChunks;
            state.totalChunks = newChunks.length;
            state.totalPages = Math.ceil(
              newChunks.length / state.chunksPerPage
            );

            // Update available documents
            const newDocs = chunks.map((c) => c.document_name);
            const allDocs = [
              ...new Set([...state.availableDocuments, ...newDocs]),
            ];
            state.availableDocuments = allDocs;
          }),

        updateChunk: (chunkId: string, updates: Partial<Chunk>) =>
          set((state) => {
            const index = state.chunks.findIndex(
              (c: Chunk) => c.id === chunkId
            );
            if (index !== -1) {
              Object.assign(state.chunks[index], updates);
              state.chunks[index].updated_at = new Date().toISOString();
            }

            // Update filtered chunks if they exist
            const filteredIndex = state.filteredChunks.findIndex(
              (c: Chunk) => c.id === chunkId
            );
            if (filteredIndex !== -1) {
              Object.assign(state.filteredChunks[filteredIndex], updates);
            }
          }),

        deleteChunk: (chunkId: string) =>
          set((state) => {
            state.chunks = state.chunks.filter((c: Chunk) => c.id !== chunkId);
            state.filteredChunks = state.filteredChunks.filter(
              (c: Chunk) => c.id !== chunkId
            );
            state.totalChunks = state.chunks.length;
            state.totalPages = Math.ceil(
              state.chunks.length / state.chunksPerPage
            );
          }),

        // Actions - Loading States
        setLoading: (loading: boolean) =>
          set((state) => {
            state.loading = loading;
          }),

        setProcessing: (processing: boolean) =>
          set((state) => {
            state.processing = processing;
          }),

        setError: (error: string | null) =>
          set((state) => {
            state.error = error;
            if (error) {
              state.loading = false;
              state.processing = false;
            }
          }),

        setSuccess: (success: string | null) =>
          set((state) => {
            state.success = success;
          }),

        setProcessingStep: (step: string) =>
          set((state) => {
            state.processingStep = step;
          }),

        // Actions - Pagination
        setCurrentPage: (page: number) =>
          set((state) => {
            state.currentPage = page;
          }),

        setChunksPerPage: (perPage: number) =>
          set((state) => {
            state.chunksPerPage = perPage;
            state.totalPages = Math.ceil(state.chunks.length / perPage);
            state.currentPage = 1; // Reset to first page
          }),

        // Actions - Filtering
        setActiveFilters: (filters: ChunkFilter) =>
          set((state) => {
            state.activeFilters = filters;
            state.currentPage = 1; // Reset to first page when filtering
          }),

        updateFilter: (key: keyof ChunkFilter, value: any) =>
          set((state) => {
            state.activeFilters[key] = value;
            state.currentPage = 1; // Reset to first page when filtering
          }),

        clearFilters: () =>
          set((state) => {
            state.activeFilters = {};
            state.selectedDocumentFilter = null;
            state.searchTerm = "";
            state.currentPage = 1;
          }),

        setSelectedDocumentFilter: (document: string | null) =>
          set((state) => {
            state.selectedDocumentFilter = document;
            state.currentPage = 1; // Reset to first page when filtering
          }),

        setSearchTerm: (term: string) =>
          set((state) => {
            state.searchTerm = term;
            state.currentPage = 1; // Reset to first page when searching
          }),

        setShowAllFiles: (show: boolean) =>
          set((state) => {
            state.showAllFiles = show;
          }),

        // Actions - Document Management
        setAvailableDocuments: (documents: string[]) =>
          set((state) => {
            state.availableDocuments = documents;
          }),

        setProcessedFiles: (files: Set<string>) =>
          set((state) => {
            state.processedFiles = files;
          }),

        addProcessedFile: (file: string) =>
          set((state) => {
            state.processedFiles.add(file);
          }),

        // Actions - Utility
        applyFilters: () => {
          const state = get();
          let filtered = [...state.chunks];

          // Apply document filter
          if (state.selectedDocumentFilter) {
            filtered = filtered.filter(
              (chunk: Chunk) =>
                chunk.document_name === state.selectedDocumentFilter
            );
          }

          // Apply search term filter
          if (state.searchTerm) {
            const term = state.searchTerm.toLowerCase();
            filtered = filtered.filter(
              (chunk: Chunk) =>
                chunk.chunk_text.toLowerCase().includes(term) ||
                chunk.document_name.toLowerCase().includes(term) ||
                chunk.chunk_metadata?.source_file
                  ?.toLowerCase()
                  .includes(term) ||
                false
            );
          }

          // Apply active filters
          if (state.activeFilters.document_name) {
            filtered = filtered.filter(
              (chunk: Chunk) =>
                chunk.document_name === state.activeFilters.document_name
            );
          }

          if (state.activeFilters.min_length) {
            filtered = filtered.filter(
              (chunk: Chunk) =>
                chunk.character_count >= (state.activeFilters.min_length || 0)
            );
          }

          if (state.activeFilters.max_length) {
            filtered = filtered.filter(
              (chunk: Chunk) =>
                chunk.character_count <=
                (state.activeFilters.max_length || Infinity)
            );
          }

          if (state.activeFilters.has_llm_improvement !== undefined) {
            filtered = filtered.filter(
              (chunk: Chunk) =>
                Boolean(chunk.chunk_metadata?.llm_improved) ===
                state.activeFilters.has_llm_improvement
            );
          }

          if (state.activeFilters.language) {
            filtered = filtered.filter(
              (chunk: Chunk) => chunk.language === state.activeFilters.language
            );
          }

          set((state) => {
            state.filteredChunks = filtered;
          });
        },

        clearMessages: () =>
          set((state) => {
            state.error = null;
            state.success = null;
          }),

        resetChunkState: () =>
          set((state) => {
            state.chunks = [];
            state.filteredChunks = [];
            state.totalChunks = 0;
            state.currentPage = 1;
            state.totalPages = 1;
            state.loading = false;
            state.processing = false;
            state.error = null;
            state.success = null;
            state.activeFilters = {};
            state.selectedDocumentFilter = null;
            state.searchTerm = "";
            state.showAllFiles = true;
            state.availableDocuments = [];
            state.processedFiles = new Set();
            state.processingStep = "";
          }),
      })),
      {
        name: "chunks-store",
        partialize: (state) => ({
          chunksPerPage: state.chunksPerPage,
          showAllFiles: state.showAllFiles,
          llmProcessingEnabled: state.llmProcessingEnabled,
        }),
      }
    ),
    {
      name: "ChunksStore",
    }
  )
);

// Extended store with API operations
interface ExtendedChunksStore extends ChunkStoreState {
  // API operations
  fetchChunks: (sessionId: string) => Promise<void>;
  processFiles: (
    sessionId: string,
    files: string[],
    config: ChunkProcessingConfig
  ) => Promise<ChunkProcessingResult>;
  improveChunk: (chunkId: string, config: any) => Promise<void>;
  deleteChunksForDocument: (documentName: string) => Promise<void>;

  // Stats and analytics
  getChunkStats: (sessionId: string) => Promise<ChunkStats | null>;
  getDocumentInfo: () => DocumentInfo[];

  // Computed getters
  getPaginatedChunks: () => Chunk[];
  getFilteredChunksCount: () => number;
}

// Create extended store
export const useExtendedChunksStore = create<ExtendedChunksStore>()(
  devtools(
    immer((set, get) => ({
      ...useChunksStore.getState(),

      // API Operations
      fetchChunks: async (sessionId: string) => {
        set((state) => {
          state.loading = true;
          state.error = null;
        });

        try {
          // Use resilient API function with timeout and error handling
          const chunks = await getChunksForSession(sessionId);

          set((state) => {
            // Transform API chunks to store format with required fields
            const transformedChunks = (chunks || []).map(
              (apiChunk: any, index: number) => ({
                id: `${sessionId}-${apiChunk.chunk_index || index}`,
                session_id: sessionId,
                document_name: apiChunk.document_name,
                chunk_index: apiChunk.chunk_index || index,
                chunk_text: apiChunk.chunk_text || "",
                chunk_metadata: apiChunk.chunk_metadata || {},
                character_count: apiChunk.chunk_text?.length || 0,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                language: "tr", // Default language
              })
            );

            state.chunks = transformedChunks;
            state.totalChunks = transformedChunks.length;
            state.totalPages = Math.ceil(
              transformedChunks.length / state.chunksPerPage
            );
            state.availableDocuments = [
              ...new Set(transformedChunks.map((c) => c.document_name)),
            ];
            state.loading = false;
            state.success =
              transformedChunks.length > 0
                ? `${transformedChunks.length} chunks loaded successfully`
                : "No chunks found for this session";
          });

          // Apply current filters
          get().applyFilters();
        } catch (error) {
          set((state) => {
            state.loading = false;
            state.error =
              error instanceof Error
                ? error.message
                : "Failed to load chunks - please try refreshing the page";
          });
        }
      },

      processFiles: async (
        sessionId: string,
        files: string[],
        config: ChunkProcessingConfig
      ) => {
        set((state) => {
          state.processing = true;
          state.error = null;
          state.processingStep = "Initializing...";
        });

        try {
          const response = await fetch(
            `${getApiUrl()}/api/sessions/${sessionId}/process-files`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                markdown_files: files,
                ...config,
              }),
            }
          );

          if (!response.ok) throw new Error("Failed to process files");

          const result: ChunkProcessingResult = await response.json();

          if (result.success && result.chunks) {
            set((state) => {
              // Add new chunks to existing ones
              const newChunks = [...state.chunks, ...result.chunks!];
              state.chunks = newChunks;
              state.totalChunks = newChunks.length;
              state.totalPages = Math.ceil(
                newChunks.length / state.chunksPerPage
              );

              // Update processed files
              files.forEach((file) => state.processedFiles.add(file));

              // Update available documents
              const newDocs = result.chunks!.map((c) => c.document_name);
              const allDocs = [
                ...new Set([...state.availableDocuments, ...newDocs]),
              ];
              state.availableDocuments = allDocs;

              state.processing = false;
              state.success = `Successfully processed ${result.processed_count} chunks`;
            });

            // Apply current filters
            get().applyFilters();
          }

          return result;
        } catch (error) {
          set((state) => {
            state.processing = false;
            state.error =
              error instanceof Error
                ? error.message
                : "Failed to process files";
          });

          return {
            success: false,
            processed_count: 0,
            total_chunks_added: 0,
            processing_time_ms: 0,
            errors: [error instanceof Error ? error.message : "Unknown error"],
          };
        }
      },

      improveChunk: async (chunkId: string, config: any) => {
        set((state) => {
          state.processing = true;
          state.error = null;
          state.processingStep = "Improving chunk with LLM...";
        });

        try {
          const response = await fetch(
            `${getApiUrl()}/api/chunks/${chunkId}/improve`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(config),
            }
          );

          if (!response.ok) throw new Error("Failed to improve chunk");

          const data = await response.json();

          if (data.success && data.improved_chunk) {
            get().updateChunk(chunkId, data.improved_chunk);

            set((state) => {
              state.processing = false;
              state.success = "Chunk improved successfully";
            });
          }
        } catch (error) {
          set((state) => {
            state.processing = false;
            state.error =
              error instanceof Error
                ? error.message
                : "Failed to improve chunk";
          });
        }
      },

      deleteChunksForDocument: async (documentName: string) => {
        set((state) => {
          state.loading = true;
          state.error = null;
        });

        try {
          const response = await fetch(
            `${getApiUrl()}/api/documents/${encodeURIComponent(
              documentName
            )}/chunks`,
            {
              method: "DELETE",
            }
          );

          if (!response.ok) throw new Error("Failed to delete document chunks");

          set((state) => {
            // Remove chunks for this document
            state.chunks = state.chunks.filter(
              (c: Chunk) => c.document_name !== documentName
            );
            state.filteredChunks = state.filteredChunks.filter(
              (c: Chunk) => c.document_name !== documentName
            );

            // Update totals
            state.totalChunks = state.chunks.length;
            state.totalPages = Math.ceil(
              state.chunks.length / state.chunksPerPage
            );

            // Update available documents
            state.availableDocuments = state.availableDocuments.filter(
              (doc: string) => doc !== documentName
            );

            state.loading = false;
            state.success = `Deleted chunks for document: ${documentName}`;
          });
        } catch (error) {
          set((state) => {
            state.loading = false;
            state.error =
              error instanceof Error
                ? error.message
                : "Failed to delete document chunks";
          });
        }
      },

      getChunkStats: async (sessionId: string) => {
        try {
          const response = await fetch(
            `${getApiUrl()}/api/sessions/${sessionId}/chunks/stats`
          );
          if (!response.ok) throw new Error("Failed to fetch chunk stats");

          const stats: ChunkStats = await response.json();
          return stats;
        } catch (error) {
          set((state) => {
            state.error =
              error instanceof Error
                ? error.message
                : "Failed to load chunk stats";
          });
          return null;
        }
      },

      getDocumentInfo: () => {
        const state = get();
        const documentStats: Record<string, DocumentInfo> = {};

        state.chunks.forEach((chunk: Chunk) => {
          if (!documentStats[chunk.document_name]) {
            documentStats[chunk.document_name] = {
              name: chunk.document_name,
              chunk_count: 0,
              total_characters: 0,
              processing_date: chunk.created_at,
              status: "processed",
              llm_improved_chunks: 0,
            };
          }

          documentStats[chunk.document_name].chunk_count++;
          documentStats[chunk.document_name].total_characters +=
            chunk.character_count;

          if (chunk.chunk_metadata?.llm_improved) {
            documentStats[chunk.document_name].llm_improved_chunks++;
          }
        });

        return Object.values(documentStats);
      },

      // Computed getters
      getPaginatedChunks: () => {
        const state = get();
        const chunks =
          state.filteredChunks.length > 0 ? state.filteredChunks : state.chunks;
        const startIndex = (state.currentPage - 1) * state.chunksPerPage;
        const endIndex = startIndex + state.chunksPerPage;
        return chunks.slice(startIndex, endIndex);
      },

      getFilteredChunksCount: () => {
        const state = get();
        return state.filteredChunks.length > 0
          ? state.filteredChunks.length
          : state.chunks.length;
      },
    })),
    {
      name: "ExtendedChunksStore",
    }
  )
);

// Export individual hooks for better performance
export const useChunks = () => useChunksStore((state) => state.chunks);
export const useChunksLoading = () => useChunksStore((state) => state.loading);
export const useChunksProcessing = () =>
  useChunksStore((state) => state.processing);
export const useChunksError = () => useChunksStore((state) => state.error);
export const useAvailableDocuments = () =>
  useChunksStore((state) => state.availableDocuments);
export const useProcessingStep = () =>
  useChunksStore((state) => state.processingStep);
