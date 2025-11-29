/**
 * useInteractions Hook - Interactions data management
 * Manages APRAG interactions with pagination, filtering, and modal state
 */

import { useState, useEffect, useCallback } from "react";
import { getSessionInteractions, type APRAGInteraction } from "@/lib/api";

export interface UseInteractionsState {
  interactions: APRAGInteraction[];
  loading: boolean;
  error: string | null;

  // Pagination
  currentPage: number;
  totalPages: number;
  totalInteractions: number;
  perPage: number;

  // Filtering and search
  searchTerm: string;
  feedbackFilter: "all" | "positive" | "negative" | "neutral" | null;
  dateRange: { start?: Date; end?: Date } | null;

  // Modal state
  selectedInteraction: APRAGInteraction | null;
  modalOpen: boolean;

  // Statistics
  stats: {
    totalQueries: number;
    feedbackCounts: {
      positive: number;
      negative: number;
      neutral: number;
      none: number;
    };
    averageResponseTime: number;
    topicsDistribution: Record<string, number>;
  };
}

export interface UseInteractionsActions {
  // Data operations
  fetchInteractions: (page?: number) => Promise<void>;
  refreshInteractions: () => Promise<void>;

  // Pagination
  setCurrentPage: (page: number) => void;
  setPerPage: (perPage: number) => void;

  // Filtering
  setSearchTerm: (term: string) => void;
  setFeedbackFilter: (filter: UseInteractionsState["feedbackFilter"]) => void;
  setDateRange: (range: UseInteractionsState["dateRange"]) => void;
  clearFilters: () => void;

  // Modal operations
  openModal: (interaction: APRAGInteraction) => void;
  closeModal: () => void;

  // Utility
  clearError: () => void;
}

export interface UseInteractionsOptions {
  sessionId: string;
  initialPerPage?: number;
  autoRefresh?: boolean;
  autoRefreshInterval?: number;
}

export function useInteractions(
  options: UseInteractionsOptions
): UseInteractionsState & UseInteractionsActions {
  const {
    sessionId,
    initialPerPage = 10,
    autoRefresh = false,
    autoRefreshInterval = 30000,
  } = options;

  const [state, setState] = useState<UseInteractionsState>({
    interactions: [],
    loading: false,
    error: null,

    currentPage: 1,
    totalPages: 0,
    totalInteractions: 0,
    perPage: initialPerPage,

    searchTerm: "",
    feedbackFilter: "all",
    dateRange: null,

    selectedInteraction: null,
    modalOpen: false,

    stats: {
      totalQueries: 0,
      feedbackCounts: {
        positive: 0,
        negative: 0,
        neutral: 0,
        none: 0,
      },
      averageResponseTime: 0,
      topicsDistribution: {},
    },
  });

  // Calculate statistics from interactions
  const calculateStats = useCallback((interactions: APRAGInteraction[]) => {
    const stats = {
      totalQueries: interactions.length,
      feedbackCounts: {
        positive: 0,
        negative: 0,
        neutral: 0,
        none: 0,
      },
      averageResponseTime: 0,
      topicsDistribution: {} as Record<string, number>,
    };

    if (interactions.length === 0) return stats;

    let totalResponseTime = 0;
    let responseTimeCount = 0;

    interactions.forEach((interaction) => {
      // Count response times
      if (interaction.processing_time_ms) {
        totalResponseTime += interaction.processing_time_ms;
        responseTimeCount++;
      }

      // Count topics
      if (interaction.topic_info?.title) {
        const topic = interaction.topic_info.title;
        stats.topicsDistribution[topic] =
          (stats.topicsDistribution[topic] || 0) + 1;
      }

      // Note: Feedback counting would need to be implemented if feedback data is available
      // For now, we'll just count as 'none'
      stats.feedbackCounts.none++;
    });

    stats.averageResponseTime =
      responseTimeCount > 0 ? totalResponseTime / responseTimeCount : 0;

    return stats;
  }, []);

  // Fetch interactions from API
  const fetchInteractions = useCallback(
    async (page: number = 1) => {
      setState((prev) => ({ ...prev, loading: true, error: null }));

      try {
        const offset = (page - 1) * state.perPage;
        const response = await getSessionInteractions(
          sessionId,
          state.perPage,
          offset
        );

        const interactions = response.interactions || [];
        const total = response.total || 0;
        const totalPages = Math.ceil(total / state.perPage);
        const stats = calculateStats(interactions);

        setState((prev) => ({
          ...prev,
          interactions,
          totalInteractions: total,
          totalPages,
          currentPage: page,
          loading: false,
          stats,
        }));
      } catch (error: any) {
        console.error("Failed to fetch interactions:", error);
        setState((prev) => ({
          ...prev,
          loading: false,
          error: error.message || "Etkileşimler yüklenemedi",
          interactions: [],
          totalInteractions: 0,
          totalPages: 0,
        }));
      }
    },
    [sessionId, state.perPage, calculateStats]
  );

  // Refresh current page
  const refreshInteractions = useCallback(async () => {
    await fetchInteractions(state.currentPage);
  }, [fetchInteractions, state.currentPage]);

  // Pagination actions
  const setCurrentPage = useCallback(
    (page: number) => {
      setState((prev) => ({ ...prev, currentPage: page }));
      fetchInteractions(page);
    },
    [fetchInteractions]
  );

  const setPerPage = useCallback(
    (perPage: number) => {
      setState((prev) => ({
        ...prev,
        perPage,
        currentPage: 1, // Reset to first page when changing per page
      }));
      // Re-fetch with new per page setting
      fetchInteractions(1);
    },
    [fetchInteractions]
  );

  // Filtering actions
  const setSearchTerm = useCallback((searchTerm: string) => {
    setState((prev) => ({ ...prev, searchTerm, currentPage: 1 }));
  }, []);

  const setFeedbackFilter = useCallback(
    (feedbackFilter: UseInteractionsState["feedbackFilter"]) => {
      setState((prev) => ({ ...prev, feedbackFilter, currentPage: 1 }));
    },
    []
  );

  const setDateRange = useCallback(
    (dateRange: UseInteractionsState["dateRange"]) => {
      setState((prev) => ({ ...prev, dateRange, currentPage: 1 }));
    },
    []
  );

  const clearFilters = useCallback(() => {
    setState((prev) => ({
      ...prev,
      searchTerm: "",
      feedbackFilter: "all",
      dateRange: null,
      currentPage: 1,
    }));
  }, []);

  // Modal operations
  const openModal = useCallback((interaction: APRAGInteraction) => {
    setState((prev) => ({
      ...prev,
      selectedInteraction: interaction,
      modalOpen: true,
    }));
  }, []);

  const closeModal = useCallback(() => {
    setState((prev) => ({
      ...prev,
      selectedInteraction: null,
      modalOpen: false,
    }));
  }, []);

  // Utility actions
  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  // Auto refresh effect
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      refreshInteractions();
    }, autoRefreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, autoRefreshInterval, refreshInteractions]);

  // Initial data load
  useEffect(() => {
    fetchInteractions(1);
  }, [sessionId]); // Only depend on sessionId to avoid infinite loops

  // Apply client-side filtering
  const filteredInteractions = state.interactions.filter((interaction) => {
    // Search filter
    if (state.searchTerm) {
      const searchLower = state.searchTerm.toLowerCase();
      const matchesQuery = interaction.query
        .toLowerCase()
        .includes(searchLower);
      const matchesResponse =
        interaction.original_response?.toLowerCase().includes(searchLower) ||
        interaction.personalized_response?.toLowerCase().includes(searchLower);
      const matchesTopic = interaction.topic_info?.title
        .toLowerCase()
        .includes(searchLower);

      if (!matchesQuery && !matchesResponse && !matchesTopic) {
        return false;
      }
    }

    // Date range filter
    if (state.dateRange) {
      const interactionDate = new Date(interaction.timestamp);
      if (state.dateRange.start && interactionDate < state.dateRange.start) {
        return false;
      }
      if (state.dateRange.end && interactionDate > state.dateRange.end) {
        return false;
      }
    }

    // Note: Feedback filter would be implemented when feedback data is available

    return true;
  });

  return {
    // State
    ...state,
    interactions: filteredInteractions,

    // Actions
    fetchInteractions,
    refreshInteractions,
    setCurrentPage,
    setPerPage,
    setSearchTerm,
    setFeedbackFilter,
    setDateRange,
    clearFilters,
    openModal,
    closeModal,
    clearError,
  };
}
