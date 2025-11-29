/**
 * useChunksData Hook - Advanced chunk data management with filtering, pagination, and modal handling
 * Provides centralized state management for the chunks tab with debounced search and real-time updates
 */

import { useState, useEffect, useCallback, useMemo } from "react";
import { useExtendedChunksStore } from "../stores/chunksStore";
import { getChunksForSession } from "@/lib/api";
import type { Chunk, ChunkFilter } from "../types/chunks.types";

// Custom debounce hook implementation
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

export interface ChunksDataConfig {
  sessionId: string;
  initialPageSize?: number;
  enableAutoRefresh?: boolean;
  autoRefreshInterval?: number;
}

export interface ChunksDataReturn {
  // Data
  chunks: Chunk[];
  filteredChunks: Chunk[];
  paginatedChunks: Chunk[];

  // Statistics
  totalChunks: number;
  filteredCount: number;
  improvedChunksCount: number;
  availableDocuments: string[];

  // Loading states
  loading: boolean;
  processing: boolean;
  error: string | null;
  success: string | null;

  // Pagination
  currentPage: number;
  chunksPerPage: number;
  totalPages: number;

  // Filtering
  searchTerm: string;
  debouncedSearchTerm: string;
  selectedDocumentFilter: string | null;
  statusFilter: "all" | "improved" | "original";
  showAllFiles: boolean;

  // Modal state
  selectedChunk: Chunk | null;
  isModalOpen: boolean;

  // Actions
  setSearchTerm: (term: string) => void;
  setSelectedDocumentFilter: (document: string | null) => void;
  setStatusFilter: (status: "all" | "improved" | "original") => void;
  setCurrentPage: (page: number) => void;
  setChunksPerPage: (perPage: number) => void;
  setShowAllFiles: (show: boolean) => void;

  // Modal actions
  openChunkModal: (chunk: Chunk) => void;
  closeChunkModal: () => void;

  // Data actions
  refreshChunks: () => Promise<void>;
  clearFilters: () => void;
  clearMessages: () => void;
}

export function useChunksData({
  sessionId,
  initialPageSize = 10,
  enableAutoRefresh = false,
  autoRefreshInterval = 30000,
}: ChunksDataConfig): ChunksDataReturn {
  // Store integration
  const chunksStore = useExtendedChunksStore();

  // Local state
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedDocumentFilter, setSelectedDocumentFilter] = useState<
    string | null
  >(null);
  const [statusFilter, setStatusFilter] = useState<
    "all" | "improved" | "original"
  >("all");
  const [showAllFiles, setShowAllFiles] = useState(false);
  const [selectedChunk, setSelectedChunk] = useState<Chunk | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Debounced search term for performance
  const debouncedSearchTerm = useDebounce(searchTerm, 300);

  // Initialize chunks per page
  useEffect(() => {
    if (chunksStore.chunksPerPage !== initialPageSize) {
      chunksStore.setChunksPerPage(initialPageSize);
    }
  }, [initialPageSize, chunksStore]);

  // Memoized filtered chunks
  const filteredChunks = useMemo(() => {
    let filtered = [...chunksStore.chunks];

    // Apply document filter
    if (selectedDocumentFilter) {
      filtered = filtered.filter(
        (chunk) => chunk.document_name === selectedDocumentFilter
      );
    }

    // Apply search term filter
    if (debouncedSearchTerm) {
      const term = debouncedSearchTerm.toLowerCase();
      filtered = filtered.filter(
        (chunk) =>
          chunk.chunk_text.toLowerCase().includes(term) ||
          chunk.document_name.toLowerCase().includes(term) ||
          (chunk.chunk_metadata?.source_file?.toLowerCase().includes(term) ??
            false)
      );
    }

    // Apply status filter
    if (statusFilter === "improved") {
      filtered = filtered.filter(
        (chunk) => chunk.chunk_metadata?.llm_improved === true
      );
    } else if (statusFilter === "original") {
      filtered = filtered.filter(
        (chunk) => !chunk.chunk_metadata?.llm_improved
      );
    }

    return filtered;
  }, [
    chunksStore.chunks,
    selectedDocumentFilter,
    debouncedSearchTerm,
    statusFilter,
  ]);

  // Update store with filtered chunks
  useEffect(() => {
    chunksStore.setFilteredChunks(filteredChunks);
  }, [filteredChunks, chunksStore]);

  // Memoized paginated chunks
  const paginatedChunks = useMemo(() => {
    const startIndex =
      (chunksStore.currentPage - 1) * chunksStore.chunksPerPage;
    const endIndex = startIndex + chunksStore.chunksPerPage;
    return filteredChunks.slice(startIndex, endIndex);
  }, [filteredChunks, chunksStore.currentPage, chunksStore.chunksPerPage]);

  // Memoized statistics
  const totalPages = useMemo(
    () => Math.ceil(filteredChunks.length / chunksStore.chunksPerPage),
    [filteredChunks.length, chunksStore.chunksPerPage]
  );

  const improvedChunksCount = useMemo(
    () =>
      chunksStore.chunks.filter((chunk) => chunk.chunk_metadata?.llm_improved)
        .length,
    [chunksStore.chunks]
  );

  const availableDocuments = useMemo(
    () =>
      [
        ...new Set(chunksStore.chunks.map((chunk) => chunk.document_name)),
      ].sort(),
    [chunksStore.chunks]
  );

  // Fetch chunks function
  const refreshChunks = useCallback(async () => {
    if (!sessionId) return;

    try {
      await chunksStore.fetchChunks(sessionId);
    } catch (error) {
      console.error("Failed to refresh chunks:", error);
      chunksStore.setError(
        error instanceof Error ? error.message : "Failed to load chunks"
      );
    }
  }, [sessionId, chunksStore]);

  // Initial data loading
  useEffect(() => {
    if (sessionId && chunksStore.chunks.length === 0 && !chunksStore.loading) {
      refreshChunks();
    }
  }, [
    sessionId,
    chunksStore.chunks.length,
    chunksStore.loading,
    refreshChunks,
  ]);

  // Auto-refresh setup
  useEffect(() => {
    if (!enableAutoRefresh || !sessionId) return;

    const interval = setInterval(refreshChunks, autoRefreshInterval);
    return () => clearInterval(interval);
  }, [enableAutoRefresh, sessionId, autoRefreshInterval, refreshChunks]);

  // Reset page when filters change
  useEffect(() => {
    if (chunksStore.currentPage !== 1) {
      chunksStore.setCurrentPage(1);
    }
  }, [selectedDocumentFilter, debouncedSearchTerm, statusFilter, chunksStore]);

  // Modal actions
  const openChunkModal = useCallback((chunk: Chunk) => {
    setSelectedChunk(chunk);
    setIsModalOpen(true);
  }, []);

  const closeChunkModal = useCallback(() => {
    setSelectedChunk(null);
    setIsModalOpen(false);
  }, []);

  // Filter actions
  const clearFilters = useCallback(() => {
    setSearchTerm("");
    setSelectedDocumentFilter(null);
    setStatusFilter("all");
    chunksStore.setCurrentPage(1);
  }, [chunksStore]);

  const clearMessages = useCallback(() => {
    chunksStore.setError(null);
    chunksStore.setSuccess(null);
  }, [chunksStore]);

  // Page change handler with validation
  const setCurrentPage = useCallback(
    (page: number) => {
      const validPage = Math.max(1, Math.min(page, totalPages));
      chunksStore.setCurrentPage(validPage);
    },
    [totalPages, chunksStore]
  );

  return {
    // Data
    chunks: chunksStore.chunks,
    filteredChunks,
    paginatedChunks,

    // Statistics
    totalChunks: chunksStore.chunks.length,
    filteredCount: filteredChunks.length,
    improvedChunksCount,
    availableDocuments,

    // Loading states
    loading: chunksStore.loading,
    processing: chunksStore.processing,
    error: chunksStore.error,
    success: chunksStore.success,

    // Pagination
    currentPage: chunksStore.currentPage,
    chunksPerPage: chunksStore.chunksPerPage,
    totalPages,

    // Filtering
    searchTerm,
    debouncedSearchTerm,
    selectedDocumentFilter,
    statusFilter,
    showAllFiles,

    // Modal state
    selectedChunk,
    isModalOpen,

    // Actions
    setSearchTerm,
    setSelectedDocumentFilter,
    setStatusFilter,
    setCurrentPage,
    setChunksPerPage: chunksStore.setChunksPerPage,
    setShowAllFiles,

    // Modal actions
    openChunkModal,
    closeChunkModal,

    // Data actions
    refreshChunks,
    clearFilters,
    clearMessages,
  };
}

// Hook for chunk improvement functionality
export function useChunkImprovement(sessionId: string) {
  const chunksStore = useExtendedChunksStore();
  const [improvingChunks, setImprovingChunks] = useState<Set<string>>(
    new Set()
  );

  const improveChunk = useCallback(
    async (
      chunkId: string,
      config: {
        llm_provider: "ollama" | "grok" | "openai";
        llm_model: string;
        improvement_type: "clarity" | "completeness" | "coherence" | "all";
      }
    ) => {
      if (improvingChunks.has(chunkId)) return;

      setImprovingChunks((prev) => new Set(prev).add(chunkId));

      try {
        await chunksStore.improveChunk(chunkId, config);
      } catch (error) {
        console.error("Failed to improve chunk:", error);
      } finally {
        setImprovingChunks((prev) => {
          const newSet = new Set(prev);
          newSet.delete(chunkId);
          return newSet;
        });
      }
    },
    [improvingChunks, chunksStore]
  );

  return {
    improveChunk,
    improvingChunks,
    isImproving: (chunkId: string) => improvingChunks.has(chunkId),
  };
}
