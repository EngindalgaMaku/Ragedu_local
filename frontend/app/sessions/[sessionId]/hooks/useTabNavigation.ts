/**
 * useTabNavigation Hook - Tab state management and URL synchronization
 * Manages tab switching, URL synchronization, and tab validation logic
 */

import { useState, useEffect, useCallback, useMemo } from "react";
import { useRouter, useSearchParams, useParams } from "next/navigation";
import { useSessionData } from "./useSessionData";

export type TabId =
  | "documents"
  | "chunks"
  | "rag-settings"
  | "session-settings"
  | "topics"
  | "interactions";

export interface TabState {
  activeTab: TabId;
  previousTab: TabId | null;
  visitedTabs: TabId[];
  tabHistory: TabId[];
}

export interface TabValidation {
  isValid: boolean;
  reason?: string;
  canAccess: boolean;
}

export interface TabNavigationHook {
  // Current state
  activeTab: TabId;
  previousTab: TabId | null;
  visitedTabs: TabId[];

  // Navigation actions
  setActiveTab: (tab: TabId) => void;
  goToNextTab: () => void;
  goToPreviousTab: () => void;
  goBack: () => void;

  // Validation
  validateTab: (tab: TabId) => TabValidation;
  getAvailableTabs: () => TabId[];

  // URL synchronization
  updateURL: (tab: TabId) => void;

  // Utility functions
  isTabValid: (tab: TabId) => boolean;
  isTabAccessible: (tab: TabId) => boolean;
  getTabOrder: () => TabId[];
  getNextAvailableTab: (currentTab?: TabId) => TabId | null;
  getPreviousAvailableTab: (currentTab?: TabId) => TabId | null;

  // State management
  resetTabState: () => void;
  clearHistory: () => void;
}

const TAB_ORDER: TabId[] = [
  "documents",
  "chunks",
  "rag-settings",
  "session-settings",
  "topics",
  "interactions",
];

const DEFAULT_TAB: TabId = "documents";

export const useTabNavigation = (): TabNavigationHook => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const params = useParams();
  const sessionId = params?.sessionId as string;

  // Get session data for validation
  const {
    hasSession,
    hasChunks,
    hasRagConfig,
    chunksCount,
    interactionsCount,
  } = useSessionData();

  // Initialize tab state from URL or default
  const getInitialTab = (): TabId => {
    const tabFromURL = searchParams?.get("tab") as TabId;
    if (tabFromURL && TAB_ORDER.includes(tabFromURL)) {
      return tabFromURL;
    }
    return DEFAULT_TAB;
  };

  const [tabState, setTabState] = useState<TabState>(() => {
    const initialTab = getInitialTab();
    return {
      activeTab: initialTab,
      previousTab: null,
      visitedTabs: [initialTab],
      tabHistory: [initialTab],
    };
  });

  // Update tab state when URL changes
  useEffect(() => {
    const tabFromURL = searchParams?.get("tab") as TabId;
    if (
      tabFromURL &&
      TAB_ORDER.includes(tabFromURL) &&
      tabFromURL !== tabState.activeTab
    ) {
      setTabState((prev) => ({
        activeTab: tabFromURL,
        previousTab: prev.activeTab,
        visitedTabs: prev.visitedTabs.includes(tabFromURL)
          ? prev.visitedTabs
          : [...prev.visitedTabs, tabFromURL],
        tabHistory: [...prev.tabHistory, tabFromURL],
      }));
    }
  }, [searchParams, tabState.activeTab]);

  // Tab validation logic
  const validateTab = useCallback(
    (tab: TabId): TabValidation => {
      switch (tab) {
        case "documents":
          return {
            isValid: true,
            canAccess: true,
          };

        case "chunks":
          return {
            isValid: hasSession,
            canAccess: true, // Always accessible to allow file upload
            reason: !hasSession ? "Oturum yüklenmemiş" : undefined,
          };

        case "rag-settings":
          return {
            isValid: hasSession,
            canAccess: hasSession,
            reason: !hasSession ? "Oturum yüklenmemiş" : undefined,
          };

        case "session-settings":
          return {
            isValid: hasSession,
            canAccess: hasSession,
            reason: !hasSession ? "Oturum yüklenmemiş" : undefined,
          };

        case "topics":
          return {
            isValid: hasSession && hasChunks,
            canAccess: hasSession,
            reason: !hasSession
              ? "Oturum yüklenmemiş"
              : !hasChunks
              ? "Henüz belge işlenmemiş"
              : undefined,
          };

        case "interactions":
          return {
            isValid: hasSession,
            canAccess: hasSession,
            reason: !hasSession ? "Oturum yüklenmemiş" : undefined,
          };

        default:
          return {
            isValid: false,
            canAccess: false,
            reason: "Geçersiz tab",
          };
      }
    },
    [hasSession, hasChunks]
  );

  // Get available tabs based on current state
  const getAvailableTabs = useCallback((): TabId[] => {
    return TAB_ORDER.filter((tab) => validateTab(tab).canAccess);
  }, [validateTab]);

  // Update URL when tab changes
  const updateURL = useCallback(
    (tab: TabId) => {
      if (!sessionId) return;

      const url = new URL(window.location.href);
      url.searchParams.set("tab", tab);

      // Use replace to avoid adding to browser history for every tab change
      router.replace(url.pathname + url.search, { scroll: false });
    },
    [router, sessionId]
  );

  // Set active tab
  const setActiveTab = useCallback(
    (tab: TabId) => {
      const validation = validateTab(tab);

      if (!validation.canAccess) {
        console.warn(`Cannot access tab: ${tab}. Reason: ${validation.reason}`);
        return;
      }

      setTabState((prev) => ({
        activeTab: tab,
        previousTab: prev.activeTab,
        visitedTabs: prev.visitedTabs.includes(tab)
          ? prev.visitedTabs
          : [...prev.visitedTabs, tab],
        tabHistory: [...prev.tabHistory, tab],
      }));

      updateURL(tab);
    },
    [validateTab, updateURL]
  );

  // Get next available tab in sequence
  const getNextAvailableTab = useCallback(
    (currentTab?: TabId): TabId | null => {
      const current = currentTab || tabState.activeTab;
      const currentIndex = TAB_ORDER.indexOf(current);
      const availableTabs = getAvailableTabs();

      for (let i = currentIndex + 1; i < TAB_ORDER.length; i++) {
        const nextTab = TAB_ORDER[i];
        if (availableTabs.includes(nextTab)) {
          return nextTab;
        }
      }

      return null;
    },
    [tabState.activeTab, getAvailableTabs]
  );

  // Get previous available tab in sequence
  const getPreviousAvailableTab = useCallback(
    (currentTab?: TabId): TabId | null => {
      const current = currentTab || tabState.activeTab;
      const currentIndex = TAB_ORDER.indexOf(current);
      const availableTabs = getAvailableTabs();

      for (let i = currentIndex - 1; i >= 0; i--) {
        const prevTab = TAB_ORDER[i];
        if (availableTabs.includes(prevTab)) {
          return prevTab;
        }
      }

      return null;
    },
    [tabState.activeTab, getAvailableTabs]
  );

  // Navigation functions
  const goToNextTab = useCallback(() => {
    const nextTab = getNextAvailableTab();
    if (nextTab) {
      setActiveTab(nextTab);
    }
  }, [getNextAvailableTab, setActiveTab]);

  const goToPreviousTab = useCallback(() => {
    const prevTab = getPreviousAvailableTab();
    if (prevTab) {
      setActiveTab(prevTab);
    }
  }, [getPreviousAvailableTab, setActiveTab]);

  const goBack = useCallback(() => {
    if (tabState.previousTab && validateTab(tabState.previousTab).canAccess) {
      setActiveTab(tabState.previousTab);
    } else {
      goToPreviousTab();
    }
  }, [tabState.previousTab, validateTab, setActiveTab, goToPreviousTab]);

  // Utility functions
  const isTabValid = useCallback(
    (tab: TabId): boolean => {
      return validateTab(tab).isValid;
    },
    [validateTab]
  );

  const isTabAccessible = useCallback(
    (tab: TabId): boolean => {
      return validateTab(tab).canAccess;
    },
    [validateTab]
  );

  const getTabOrder = useCallback((): TabId[] => {
    return TAB_ORDER;
  }, []);

  // State management
  const resetTabState = useCallback(() => {
    const initialTab = DEFAULT_TAB;
    setTabState({
      activeTab: initialTab,
      previousTab: null,
      visitedTabs: [initialTab],
      tabHistory: [initialTab],
    });
    updateURL(initialTab);
  }, [updateURL]);

  const clearHistory = useCallback(() => {
    setTabState((prev) => ({
      ...prev,
      visitedTabs: [prev.activeTab],
      tabHistory: [prev.activeTab],
      previousTab: null,
    }));
  }, []);

  // Auto-redirect to available tab if current tab becomes inaccessible
  useEffect(() => {
    const currentValidation = validateTab(tabState.activeTab);

    if (!currentValidation.canAccess) {
      const availableTabs = getAvailableTabs();
      const fallbackTab = availableTabs.includes(DEFAULT_TAB)
        ? DEFAULT_TAB
        : availableTabs[0];

      if (fallbackTab && fallbackTab !== tabState.activeTab) {
        console.warn(
          `Current tab ${tabState.activeTab} is not accessible. Redirecting to ${fallbackTab}.`
        );
        setActiveTab(fallbackTab);
      }
    }
  }, [tabState.activeTab, validateTab, getAvailableTabs, setActiveTab]);

  return {
    // Current state
    activeTab: tabState.activeTab,
    previousTab: tabState.previousTab,
    visitedTabs: tabState.visitedTabs,

    // Navigation actions
    setActiveTab,
    goToNextTab,
    goToPreviousTab,
    goBack,

    // Validation
    validateTab,
    getAvailableTabs,

    // URL synchronization
    updateURL,

    // Utility functions
    isTabValid,
    isTabAccessible,
    getTabOrder,
    getNextAvailableTab,
    getPreviousAvailableTab,

    // State management
    resetTabState,
    clearHistory,
  };
};

// Convenience hook for tab navigation with keyboard shortcuts
export const useTabNavigationWithKeyboard = () => {
  const tabNavigation = useTabNavigationCore();

  useEffect(() => {
    const handleKeydown = (event: KeyboardEvent) => {
      // Ctrl/Cmd + Arrow keys for tab navigation
      if ((event.ctrlKey || event.metaKey) && !event.shiftKey) {
        switch (event.key) {
          case "ArrowLeft":
            event.preventDefault();
            tabNavigation.goToPreviousTab();
            break;
          case "ArrowRight":
            event.preventDefault();
            tabNavigation.goToNextTab();
            break;
        }
      }

      // Ctrl/Cmd + Number keys for direct tab access
      if (
        (event.ctrlKey || event.metaKey) &&
        event.key >= "1" &&
        event.key <= "6"
      ) {
        event.preventDefault();
        const tabIndex = parseInt(event.key) - 1;
        const availableTabs = tabNavigation.getAvailableTabs();
        const targetTab = tabNavigation.getTabOrder()[tabIndex];

        if (targetTab && availableTabs.includes(targetTab)) {
          tabNavigation.setActiveTab(targetTab);
        }
      }
    };

    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [tabNavigation]);

  return tabNavigation;
};

// Hook alias for backwards compatibility
const useTabNavigationCore = useTabNavigation;

export default useTabNavigation;

// Hook for getting tab metadata
export const useTabMetadata = () => {
  const {
    chunksCount,
    interactionsCount,
    hasChunks,
    sessionError,
    chunksError,
    ragError,
  } = useSessionData();

  return useMemo(() => {
    const tabs = [
      {
        id: "documents" as const,
        label: "Belgeler",
        hasError: !!sessionError,
        badge: undefined,
      },
      {
        id: "chunks" as const,
        label: "Parçalar",
        hasError: !!chunksError,
        badge: chunksCount > 0 ? chunksCount : undefined,
      },
      {
        id: "rag-settings" as const,
        label: "RAG Ayarları",
        hasError: !!ragError,
        badge: undefined,
      },
      {
        id: "session-settings" as const,
        label: "Oturum Ayarları",
        hasError: false,
        badge: undefined,
      },
      {
        id: "topics" as const,
        label: "Konular",
        hasError: false,
        badge: undefined,
        disabled: !hasChunks,
      },
      {
        id: "interactions" as const,
        label: "Etkileşimler",
        hasError: false,
        badge: interactionsCount > 0 ? interactionsCount : undefined,
      },
    ];

    return tabs;
  }, [
    chunksCount,
    interactionsCount,
    hasChunks,
    sessionError,
    chunksError,
    ragError,
  ]);
};
