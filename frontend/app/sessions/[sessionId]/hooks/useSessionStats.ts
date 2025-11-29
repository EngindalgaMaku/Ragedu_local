/**
 * useSessionStats Hook - Lightweight session statistics without loading full chunks
 * Provides basic statistics immediately when session loads, independent of chunks loading
 */

import { useState, useEffect, useCallback } from "react";
import { getApiUrl } from "@/lib/api";

export interface SessionStats {
  total_documents: number;
  total_chunks: number;
  total_characters: number;
  llm_improved: number;
  session_id: string;
  source:
    | "document_processor_stats"
    | "session_metadata"
    | "empty_session"
    | "error";
}

export interface UseSessionStatsReturn {
  stats: SessionStats | null;
  loading: boolean;
  error: string | null;
  refreshStats: () => Promise<void>;
}

export function useSessionStats(sessionId: string): UseSessionStatsReturn {
  const [stats, setStats] = useState<SessionStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(async () => {
    if (!sessionId) return;

    setLoading(true);
    setError(null);

    try {
      console.log(
        `🔍 [SESSION STATS] Fetching lightweight stats for session ${sessionId}`
      );

      const response = await fetch(
        `${getApiUrl()}/api/sessions/${sessionId}/stats`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error(
          `Failed to fetch stats: ${response.status} ${response.statusText}`
        );
      }

      const data: SessionStats = await response.json();

      console.log(`✅ [SESSION STATS] Got lightweight stats:`, data);
      setStats(data);
    } catch (err) {
      console.error(`❌ [SESSION STATS] Error fetching stats:`, err);
      setError(
        err instanceof Error ? err.message : "Failed to fetch statistics"
      );

      // Fallback: Set empty stats
      setStats({
        total_documents: 0,
        total_chunks: 0,
        total_characters: 0,
        llm_improved: 0,
        session_id: sessionId,
        source: "error",
      });
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  const refreshStats = useCallback(async () => {
    await fetchStats();
  }, [fetchStats]);

  // Initial fetch when sessionId changes
  useEffect(() => {
    if (sessionId) {
      fetchStats();
    }
  }, [sessionId, fetchStats]);

  return {
    stats,
    loading,
    error,
    refreshStats,
  };
}
