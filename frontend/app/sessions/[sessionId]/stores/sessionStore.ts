/**
 * Session Store - Zustand state management for session data
 * Manages session metadata, settings, interactions, and UI state
 */

import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type {
  SessionStoreState,
  SessionMeta,
  SessionSettings,
  SessionInteraction,
  SessionSettingsUpdate,
} from "../types/session.types";

// Default session settings based on the actual interface
const DEFAULT_SESSION_SETTINGS: SessionSettings = {
  session_id: "",
  user_id: "",
  enable_progressive_assessment: true,
  enable_personalized_responses: true,
  enable_multi_dimensional_feedback: true,
  enable_topic_analytics: true,
  enable_cacs: false,
  enable_zpd: false,
  enable_bloom: false,
  enable_cognitive_load: false,
  enable_emoji_feedback: true,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
};

// Extended store state for additional UI functionality
interface ExtendedSessionStoreState extends SessionStoreState {
  // Additional UI state
  activeTab: "chat" | "documents" | "settings" | "analytics";
  isSettingsPanelOpen: boolean;
  isTopicPanelOpen: boolean;

  // Pagination and filtering for interactions
  currentPage: number;
  interactionsPerPage: number;
  totalPages: number;
  searchTerm: string;
  filterByRating: number | null;
  sortBy: "timestamp" | "rating" | "response_time";
  sortOrder: "asc" | "desc";

  // Extended actions
  setActiveTab: (tab: "chat" | "documents" | "settings" | "analytics") => void;
  setSettingsPanelOpen: (open: boolean) => void;
  setTopicPanelOpen: (open: boolean) => void;

  // Pagination actions
  setCurrentPage: (page: number) => void;
  setInteractionsPerPage: (perPage: number) => void;
  setSearchTerm: (term: string) => void;
  setFilterByRating: (rating: number | null) => void;
  setSortBy: (sortBy: "timestamp" | "rating" | "response_time") => void;
  setSortOrder: (order: "asc" | "desc") => void;

  // API operations
  fetchSessionData: (sessionId: string) => Promise<void>;
  saveSessionSettings: (sessionId: string) => Promise<void>;
  deleteSession: (sessionId: string) => Promise<boolean>;

  // Computed getters
  getFilteredInteractions: () => SessionInteraction[];
  getPaginatedInteractions: () => SessionInteraction[];

  // Additional interaction management
  addInteraction: (interaction: SessionInteraction) => void;
  updateInteraction: (
    interactionId: string,
    updates: Partial<SessionInteraction>
  ) => void;
  deleteInteraction: (interactionId: string) => void;
}

// API base URL helper
const getApiUrl = () => {
  if (typeof window === "undefined") return "http://localhost:8000";
  return window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "https://api-gateway:8000";
};

export const useSessionStore = create<ExtendedSessionStoreState>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Core SessionStoreState implementation
        currentSession: null,
        sessions: [],
        loading: false,
        error: null,
        success: null,
        sessionSettings: null,
        settingsLoading: false,
        interactions: [],
        interactionsLoading: false,
        interactionsPage: 1,
        interactionsTotal: 0,

        // Extended UI state
        activeTab: "chat",
        isSettingsPanelOpen: false,
        isTopicPanelOpen: false,
        currentPage: 1,
        interactionsPerPage: 20,
        totalPages: 1,
        searchTerm: "",
        filterByRating: null,
        sortBy: "timestamp",
        sortOrder: "desc",

        // Core SessionStoreState actions
        setCurrentSession: (session: SessionMeta | null) =>
          set((state) => {
            state.currentSession = session;
          }),

        setSessions: (sessions: SessionMeta[]) =>
          set((state) => {
            state.sessions = sessions;
          }),

        setLoading: (loading: boolean) =>
          set((state) => {
            state.loading = loading;
          }),

        setError: (error: string | null) =>
          set((state) => {
            state.error = error;
            if (error) state.loading = false;
          }),

        setSuccess: (success: string | null) =>
          set((state) => {
            state.success = success;
          }),

        setSessionSettings: (settings: SessionSettings | null) =>
          set((state) => {
            state.sessionSettings = settings;
          }),

        setSettingsLoading: (loading: boolean) =>
          set((state) => {
            state.settingsLoading = loading;
          }),

        updateSessionSetting: (
          key: keyof SessionSettingsUpdate,
          value: boolean
        ) =>
          set((state) => {
            if (state.sessionSettings) {
              (state.sessionSettings as any)[key] = value;
              state.sessionSettings.updated_at = new Date().toISOString();
            }
          }),

        setInteractions: (interactions: SessionInteraction[]) =>
          set((state) => {
            state.interactions = interactions;
            state.interactionsTotal = interactions.length;
            state.totalPages = Math.ceil(
              interactions.length / state.interactionsPerPage
            );
          }),

        setInteractionsLoading: (loading: boolean) =>
          set((state) => {
            state.interactionsLoading = loading;
          }),

        setInteractionsPage: (page: number) =>
          set((state) => {
            state.interactionsPage = page;
          }),

        setInteractionsTotal: (total: number) =>
          set((state) => {
            state.interactionsTotal = total;
          }),

        clearMessages: () =>
          set((state) => {
            state.error = null;
            state.success = null;
          }),

        resetSessionState: () =>
          set((state) => {
            state.currentSession = null;
            state.sessions = [];
            state.sessionSettings = null;
            state.interactions = [];
            state.loading = false;
            state.error = null;
            state.success = null;
            state.settingsLoading = false;
            state.interactionsLoading = false;
            state.interactionsPage = 1;
            state.interactionsTotal = 0;
            state.activeTab = "chat";
            state.isSettingsPanelOpen = false;
            state.isTopicPanelOpen = false;
            state.currentPage = 1;
            state.totalPages = 1;
            state.searchTerm = "";
            state.filterByRating = null;
            state.sortBy = "timestamp";
            state.sortOrder = "desc";
          }),

        // Extended UI actions
        setActiveTab: (tab: "chat" | "documents" | "settings" | "analytics") =>
          set((state) => {
            state.activeTab = tab;
          }),

        setSettingsPanelOpen: (open: boolean) =>
          set((state) => {
            state.isSettingsPanelOpen = open;
          }),

        setTopicPanelOpen: (open: boolean) =>
          set((state) => {
            state.isTopicPanelOpen = open;
          }),

        // Pagination actions
        setCurrentPage: (page: number) =>
          set((state) => {
            state.currentPage = page;
          }),

        setInteractionsPerPage: (perPage: number) =>
          set((state) => {
            state.interactionsPerPage = perPage;
            state.totalPages = Math.ceil(state.interactions.length / perPage);
            state.currentPage = 1;
          }),

        setSearchTerm: (term: string) =>
          set((state) => {
            state.searchTerm = term;
            state.currentPage = 1;
          }),

        setFilterByRating: (rating: number | null) =>
          set((state) => {
            state.filterByRating = rating;
            state.currentPage = 1;
          }),

        setSortBy: (sortBy: "timestamp" | "rating" | "response_time") =>
          set((state) => {
            state.sortBy = sortBy;
          }),

        setSortOrder: (order: "asc" | "desc") =>
          set((state) => {
            state.sortOrder = order;
          }),

        // Additional interaction management
        addInteraction: (interaction: SessionInteraction) =>
          set((state) => {
            state.interactions.unshift(interaction);
            state.interactionsTotal = state.interactions.length;
            state.totalPages = Math.ceil(
              state.interactions.length / state.interactionsPerPage
            );
          }),

        updateInteraction: (
          interactionId: string,
          updates: Partial<SessionInteraction>
        ) =>
          set((state) => {
            const index = state.interactions.findIndex(
              (i: SessionInteraction) => i.interaction_id === interactionId
            );
            if (index !== -1) {
              Object.assign(state.interactions[index], updates);
            }
          }),

        deleteInteraction: (interactionId: string) =>
          set((state) => {
            state.interactions = state.interactions.filter(
              (i: SessionInteraction) => i.interaction_id !== interactionId
            );
            state.interactionsTotal = state.interactions.length;
            state.totalPages = Math.ceil(
              state.interactions.length / state.interactionsPerPage
            );
          }),

        // API operations - Using backup system's working approach
        fetchSessionData: async (sessionId: string) => {
          console.log(
            `🔍 [SESSION STORE] Starting fetchSessionData for sessionId: ${sessionId}`
          );

          set((state) => {
            state.loading = true;
            state.error = null;
          });

          try {
            const apiUrl = getApiUrl();

            // First try the original single session endpoint
            console.log(
              `📡 [SESSION STORE] Trying single session endpoint: ${apiUrl}/api/sessions/${sessionId}`
            );
            let response = await fetch(`${apiUrl}/api/sessions/${sessionId}`);

            if (!response.ok) {
              console.log(
                `⚠️ [SESSION STORE] Single session endpoint failed (${response.status}), falling back to listSessions approach`
              );

              // Fallback to backup system's working approach: listSessions
              console.log(
                `📡 [SESSION STORE] Using backup approach: ${apiUrl}/list_sessions`
              );
              response = await fetch(`${apiUrl}/list_sessions`);

              if (!response.ok) {
                const errorText = await response.text();
                console.error(
                  `❌ [SESSION STORE] Both endpoints failed. listSessions error: ${response.status} - ${errorText}`
                );
                throw new Error(
                  `Session API Error: ${response.status} - ${errorText}`
                );
              }

              const allSessions = await response.json();
              console.log(
                `📦 [SESSION STORE] Retrieved ${allSessions.length} sessions, searching for ${sessionId}`
              );

              const targetSession = allSessions.find(
                (s: any) => s.session_id === sessionId
              );

              if (!targetSession) {
                console.error(
                  `❌ [SESSION STORE] Session ${sessionId} not found in ${allSessions.length} sessions`
                );
                throw new Error(`Session ${sessionId} not found`);
              }

              console.log(
                `✅ [SESSION STORE] Found target session:`,
                targetSession
              );

              set((state) => {
                state.currentSession = targetSession;
                state.sessionSettings =
                  targetSession.session_settings || DEFAULT_SESSION_SETTINGS;
                state.interactions = [];
                state.interactionsTotal = 0;
                state.totalPages = 1;
                state.loading = false;
                state.success =
                  "Session data loaded successfully via listSessions";

                console.log(
                  `✅ [SESSION STORE] State updated via backup approach. Session name: ${targetSession?.name}`
                );
              });

              return;
            }

            // Original endpoint worked
            console.log(
              `📊 [SESSION STORE] Single session endpoint worked: ${response.status}`
            );
            const data = await response.json();
            console.log(`📦 [SESSION STORE] Single session data:`, data);

            set((state) => {
              state.currentSession = data.session || data;
              state.sessionSettings =
                data.settings ||
                data.session_settings ||
                DEFAULT_SESSION_SETTINGS;
              state.interactions = data.interactions || [];
              state.interactionsTotal = (data.interactions || []).length;
              state.totalPages = Math.ceil(
                (data.interactions || []).length / state.interactionsPerPage
              );
              state.loading = false;
              state.success = "Session data loaded successfully";

              console.log(
                `✅ [SESSION STORE] State updated via single endpoint. Session:`,
                state.currentSession
              );
            });
          } catch (error) {
            console.error(`💥 [SESSION STORE] Fetch error:`, error);
            set((state) => {
              state.loading = false;
              state.error =
                error instanceof Error
                  ? error.message
                  : "Failed to load session data";
              console.log(`❌ [SESSION STORE] Error state set:`, state.error);
            });
          }
        },

        saveSessionSettings: async (sessionId: string) => {
          const { sessionSettings } = get();

          set((state) => {
            state.settingsLoading = true;
            state.error = null;
          });

          try {
            const response = await fetch(
              `${getApiUrl()}/api/sessions/${sessionId}/settings`,
              {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(sessionSettings),
              }
            );

            if (!response.ok)
              throw new Error("Failed to save session settings");

            set((state) => {
              state.settingsLoading = false;
              state.success = "Settings saved successfully";
            });
          } catch (error) {
            set((state) => {
              state.settingsLoading = false;
              state.error =
                error instanceof Error
                  ? error.message
                  : "Failed to save settings";
            });
          }
        },

        deleteSession: async (sessionId: string) => {
          set((state) => {
            state.loading = true;
            state.error = null;
          });

          try {
            const response = await fetch(
              `${getApiUrl()}/api/sessions/${sessionId}`,
              {
                method: "DELETE",
              }
            );

            if (!response.ok) throw new Error("Failed to delete session");

            set((state) => {
              state.loading = false;
              state.success = "Session deleted successfully";
            });

            return true;
          } catch (error) {
            set((state) => {
              state.loading = false;
              state.error =
                error instanceof Error
                  ? error.message
                  : "Failed to delete session";
            });
            return false;
          }
        },

        // Computed getters
        getFilteredInteractions: () => {
          const state = get();
          let filtered = state.interactions;

          // Apply search filter
          if (state.searchTerm) {
            const term = state.searchTerm.toLowerCase();
            filtered = filtered.filter(
              (interaction) =>
                interaction.query.toLowerCase().includes(term) ||
                interaction.original_response?.toLowerCase().includes(term) ||
                false ||
                interaction.personalized_response
                  ?.toLowerCase()
                  .includes(term) ||
                false
            );
          }

          // Apply rating filter - note: SessionInteraction doesn't have user_rating in the type
          // This would need to be added to the SessionInteraction interface if needed
          if (state.filterByRating !== null) {
            // For now, skip this filter since user_rating is not in SessionInteraction
            // filtered = filtered.filter(interaction =>
            //   interaction.user_rating === state.filterByRating
            // );
          }

          // Apply sorting
          filtered.sort((a, b) => {
            let aValue: any, bValue: any;

            switch (state.sortBy) {
              case "timestamp":
                aValue = new Date(a.timestamp);
                bValue = new Date(b.timestamp);
                break;
              case "rating":
                // Skip rating sort since not available in type
                return 0;
              case "response_time":
                aValue = a.processing_time_ms || 0;
                bValue = b.processing_time_ms || 0;
                break;
              default:
                return 0;
            }

            if (state.sortOrder === "asc") {
              return aValue > bValue ? 1 : -1;
            } else {
              return aValue < bValue ? 1 : -1;
            }
          });

          return filtered;
        },

        getPaginatedInteractions: () => {
          const state = get();
          const filtered = state.getFilteredInteractions();
          const startIndex =
            (state.currentPage - 1) * state.interactionsPerPage;
          const endIndex = startIndex + state.interactionsPerPage;
          return filtered.slice(startIndex, endIndex);
        },
      })),
      {
        name: "session-store",
        partialize: (state) => ({
          sessionSettings: state.sessionSettings,
          activeTab: state.activeTab,
          interactionsPerPage: state.interactionsPerPage,
          sortBy: state.sortBy,
          sortOrder: state.sortOrder,
        }),
      }
    ),
    {
      name: "SessionStore",
    }
  )
);

// Export individual hooks for better performance
export const useCurrentSession = () =>
  useSessionStore((state) => state.currentSession);
export const useSessionSettings = () =>
  useSessionStore((state) => state.sessionSettings);
export const useInteractions = () =>
  useSessionStore((state) => state.interactions);
export const useSessionLoading = () =>
  useSessionStore((state) => state.loading);
export const useSessionError = () => useSessionStore((state) => state.error);
export const useActiveTab = () => useSessionStore((state) => state.activeTab);
export const useInteractionsLoading = () =>
  useSessionStore((state) => state.interactionsLoading);
