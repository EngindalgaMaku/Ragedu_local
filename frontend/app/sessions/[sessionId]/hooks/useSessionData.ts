/**
 * useSessionData Hook - Unified session data management
 * Coordinates between session, chunks, and RAG stores for comprehensive session management
 */

import { useEffect, useCallback, useMemo } from "react";
import { useParams } from "next/navigation";
import {
  useSessionStore,
  useCurrentSession,
  useSessionSettings,
  useInteractions,
  useSessionLoading,
  useSessionError,
} from "../stores/sessionStore";
import {
  useChunksStore,
  useExtendedChunksStore,
  useChunks,
  useChunksLoading,
  useChunksError,
} from "../stores/chunksStore";
import {
  useRagSettingsStore,
  useExtendedRagStore,
  useRagConfig,
  useRagLoading,
  useRagError,
} from "../stores/ragSettingsStore";

// Combined loading and error states
interface SessionDataState {
  // Loading states
  loading: boolean;
  sessionLoading: boolean;
  chunksLoading: boolean;
  ragLoading: boolean;

  // Error states
  error: string | null;
  sessionError: string | null;
  chunksError: string | null;
  ragError: string | null;

  // Data availability
  hasSession: boolean;
  hasChunks: boolean;
  hasRagConfig: boolean;

  // Data counts
  sessionCount: number;
  chunksCount: number;
  interactionsCount: number;
}

// Session operations interface
interface SessionOperations {
  // Data fetching
  loadSessionData: () => Promise<void>;
  refreshSessionData: () => Promise<void>;
  loadChunks: () => Promise<void>;
  loadRagConfig: () => Promise<void>;

  // Data saving
  saveSession: () => Promise<boolean>;
  saveSessionSettings: () => Promise<boolean>;
  saveRagConfig: () => Promise<boolean>;
  saveAllSettings: () => Promise<boolean>;

  // Session management
  deleteSession: () => Promise<boolean>;
  resetSession: () => void;

  // Error handling
  clearAllErrors: () => void;
  clearSessionError: () => void;
  clearChunksError: () => void;
  clearRagError: () => void;

  // Utility methods
  isDataComplete: () => boolean;
  getDataStatus: () => SessionDataState;
  validateConfiguration: () => { isValid: boolean; errors: string[] };
}

// Main hook
export const useSessionData = (): SessionOperations & SessionDataState => {
  const params = useParams();
  const sessionId = params?.sessionId as string;

  // Session store hooks
  const currentSession = useCurrentSession();
  const sessionSettings = useSessionSettings();
  const interactions = useInteractions();
  const sessionLoading = useSessionLoading();
  const sessionError = useSessionError();

  // Chunks store hooks
  const chunks = useChunks();
  const chunksLoading = useChunksLoading();
  const chunksError = useChunksError();

  // RAG store hooks
  const ragConfig = useRagConfig();
  const ragLoading = useRagLoading();
  const ragError = useRagError();

  // Store actions
  const sessionActions = useSessionStore();
  const chunksActions = useExtendedChunksStore();
  const ragActions = useExtendedRagStore();

  // Computed state
  const state: SessionDataState = useMemo(() => {
    const hasError = !!(sessionError || chunksError || ragError);
    const anyLoading = sessionLoading || chunksLoading || ragLoading;

    return {
      // Loading states
      loading: anyLoading,
      sessionLoading,
      chunksLoading,
      ragLoading,

      // Error states
      error: hasError ? sessionError || chunksError || ragError : null,
      sessionError,
      chunksError,
      ragError,

      // Data availability
      hasSession: !!currentSession,
      hasChunks: chunks.length > 0,
      hasRagConfig: !!ragConfig,

      // Data counts
      sessionCount: currentSession ? 1 : 0,
      chunksCount: chunks.length,
      interactionsCount: interactions.length,
    };
  }, [
    currentSession,
    sessionSettings,
    chunks,
    ragConfig,
    sessionLoading,
    chunksLoading,
    ragLoading,
    sessionError,
    chunksError,
    ragError,
    interactions.length,
  ]);

  // Operations
  const loadSessionData = useCallback(async () => {
    if (!sessionId) {
      console.warn(`⚠️ [USE SESSION DATA] No sessionId provided`);
      return;
    }

    console.log(
      `🔄 [USE SESSION DATA] Loading session data for sessionId: ${sessionId}`
    );

    try {
      await sessionActions.fetchSessionData(sessionId);
      console.log(`✅ [USE SESSION DATA] Session data loaded successfully`);
    } catch (error) {
      console.error(
        `💥 [USE SESSION DATA] Failed to load session data:`,
        error
      );
    }
  }, [sessionId, sessionActions]);

  const refreshSessionData = useCallback(async () => {
    if (!sessionId) return;

    // Clear existing data first
    sessionActions.resetSessionState();
    chunksActions.resetChunkState();
    ragActions.resetState();

    // Reload all data
    await Promise.allSettled([
      loadSessionData(),
      loadChunks(),
      loadRagConfig(),
    ]);
  }, [sessionId, sessionActions, chunksActions, ragActions, loadSessionData]);

  const loadChunks = useCallback(async () => {
    if (!sessionId) return;

    try {
      await chunksActions.fetchChunks(sessionId);
    } catch (error) {
      console.error("Failed to load chunks:", error);
    }
  }, [sessionId, chunksActions]);

  const loadRagConfig = useCallback(async () => {
    if (!sessionId) return;

    try {
      await ragActions.loadRagConfig(sessionId);
    } catch (error) {
      console.error("Failed to load RAG config:", error);
    }
  }, [sessionId, ragActions]);

  const saveSession = useCallback(async () => {
    if (!sessionId || !currentSession) return false;

    try {
      // Implementation would depend on session update API
      // For now, return true as placeholder
      return true;
    } catch (error) {
      console.error("Failed to save session:", error);
      return false;
    }
  }, [sessionId, currentSession]);

  const saveSessionSettings = useCallback(async () => {
    if (!sessionId) return false;

    try {
      await sessionActions.saveSessionSettings(sessionId);
      return !sessionError; // Use the error state from hook instead of getState()
    } catch (error) {
      console.error("Failed to save session settings:", error);
      return false;
    }
  }, [sessionId, sessionActions]);

  const saveRagConfig = useCallback(async () => {
    if (!sessionId) return false;

    try {
      await ragActions.saveRagConfig(sessionId);
      return !ragError; // Use the error state from hook instead of getState()
    } catch (error) {
      console.error("Failed to save RAG config:", error);
      return false;
    }
  }, [sessionId, ragActions]);

  const saveAllSettings = useCallback(async () => {
    if (!sessionId) return false;

    const results = await Promise.allSettled([
      saveSession(),
      saveSessionSettings(),
      saveRagConfig(),
    ]);

    // Return true only if all saves succeeded
    return results.every(
      (result) => result.status === "fulfilled" && result.value === true
    );
  }, [saveSession, saveSessionSettings, saveRagConfig]);

  const deleteSession = useCallback(async () => {
    if (!sessionId) return false;

    try {
      const success = await sessionActions.deleteSession(sessionId);
      if (success) {
        // Reset all stores after successful deletion
        sessionActions.resetSessionState();
        chunksActions.resetChunkState();
        ragActions.resetState();
      }
      return success;
    } catch (error) {
      console.error("Failed to delete session:", error);
      return false;
    }
  }, [sessionId, sessionActions, chunksActions, ragActions]);

  const resetSession = useCallback(() => {
    sessionActions.resetSessionState();
    chunksActions.resetChunkState();
    ragActions.resetState();
  }, [sessionActions, chunksActions, ragActions]);

  const clearAllErrors = useCallback(() => {
    sessionActions.clearMessages();
    chunksActions.clearMessages();
    ragActions.clearMessages();
  }, [sessionActions, chunksActions, ragActions]);

  const clearSessionError = useCallback(() => {
    sessionActions.clearMessages();
  }, [sessionActions]);

  const clearChunksError = useCallback(() => {
    chunksActions.clearMessages();
  }, [chunksActions]);

  const clearRagError = useCallback(() => {
    ragActions.clearMessages();
  }, [ragActions]);

  const isDataComplete = useCallback(() => {
    return !!(currentSession && sessionSettings && ragConfig);
  }, [currentSession, sessionSettings, ragConfig]);

  const getDataStatus = useCallback(() => state, [state]);

  const validateConfiguration = useCallback(() => {
    const errors: string[] = [];

    // Validate session data
    if (!currentSession) {
      errors.push("Session data is missing");
    }

    if (!sessionSettings) {
      errors.push("Session settings are missing");
    }

    // Validate RAG configuration
    if (!ragConfig) {
      errors.push("RAG configuration is missing");
    } else {
      const ragValidation = ragActions.validateConfig();
      if (!ragValidation.isValid) {
        errors.push(...ragValidation.errors);
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }, [currentSession, sessionSettings, ragConfig, ragActions]);

  // Load basic session metadata (lightweight) - but NOT chunks automatically
  useEffect(() => {
    console.log(
      `🎣 [USE SESSION DATA] useEffect triggered - sessionId: ${sessionId}, loading: ${
        state.loading
      }, hasCurrentSession: ${!!currentSession}`
    );

    if (sessionId && !state.loading && !currentSession) {
      console.log(
        `🚀 [USE SESSION DATA] Conditions met, triggering loadSessionData`
      );
      loadSessionData();
    } else {
      console.log(`🚫 [USE SESSION DATA] Conditions not met - skipping load`);
    }
  }, [sessionId, state.loading, currentSession, loadSessionData]);

  // DISABLED: Auto-load chunks when session is loaded but chunks are missing
  // This was causing CORS errors and performance issues
  // Original system uses manual upload, not automatic chunk loading
  // useEffect(() => {
  //   if (
  //     sessionId &&
  //     currentSession &&
  //     !chunksLoading &&
  //     chunks.length === 0 &&
  //     !chunksError
  //   ) {
  //     loadChunks();
  //   }
  // }, [
  //   sessionId,
  //   currentSession,
  //   chunksLoading,
  //   chunks.length,
  //   chunksError,
  //   loadChunks,
  // ]);

  // DISABLED: Auto-load RAG config when session is loaded but config is missing
  // This was causing automatic API calls and CORS errors
  // Original system requires manual user action to configure RAG settings
  // useEffect(() => {
  //   if (sessionId && currentSession && !ragLoading && !ragConfig && !ragError) {
  //     loadRagConfig();
  //   }
  // }, [
  //   sessionId,
  //   currentSession,
  //   ragLoading,
  //   ragConfig,
  //   ragError,
  //   loadRagConfig,
  // ]);

  return {
    // State
    ...state,

    // Operations
    loadSessionData,
    refreshSessionData,
    loadChunks,
    loadRagConfig,
    saveSession,
    saveSessionSettings,
    saveRagConfig,
    saveAllSettings,
    deleteSession,
    resetSession,
    clearAllErrors,
    clearSessionError,
    clearChunksError,
    clearRagError,
    isDataComplete,
    getDataStatus,
    validateConfiguration,
  };
};

// Convenience hooks for specific data types
export const useSessionMetadata = () => {
  const currentSession = useCurrentSession();
  const sessionSettings = useSessionSettings();
  const sessionLoading = useSessionLoading();
  const sessionError = useSessionError();

  return {
    session: currentSession,
    settings: sessionSettings,
    loading: sessionLoading,
    error: sessionError,
  };
};

export const useSessionChunks = () => {
  const chunks = useChunks();
  const chunksLoading = useChunksLoading();
  const chunksError = useChunksError();
  const chunksActions = useExtendedChunksStore();

  return {
    chunks,
    loading: chunksLoading,
    error: chunksError,
    actions: chunksActions,
  };
};

export const useSessionRagConfig = () => {
  const ragConfig = useRagConfig();
  const ragLoading = useRagLoading();
  const ragError = useRagError();
  const ragActions = useExtendedRagStore();

  return {
    config: ragConfig,
    loading: ragLoading,
    error: ragError,
    actions: ragActions,
  };
};

// Hook for managing session lifecycle
export const useSessionLifecycle = () => {
  const {
    loadSessionData,
    refreshSessionData,
    saveAllSettings,
    deleteSession,
    resetSession,
    isDataComplete,
    validateConfiguration,
  } = useSessionData();

  return {
    load: loadSessionData,
    refresh: refreshSessionData,
    save: saveAllSettings,
    delete: deleteSession,
    reset: resetSession,
    isComplete: isDataComplete,
    validate: validateConfiguration,
  };
};

export default useSessionData;
