"use client";

import { useState, useEffect } from "react";
import { getAPRAGSettings, APRAGSettings } from "@/lib/api";

interface UseAPRAGSettingsReturn {
  settings: APRAGSettings | null;
  isEnabled: boolean;
  isLoading: boolean;
  features: APRAGSettings["features"];
  refresh: () => Promise<void>;
}

export function useAPRAGSettings(sessionId?: string): UseAPRAGSettingsReturn {
  const [settings, setSettings] = useState<APRAGSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const loadSettings = async () => {
    try {
      setIsLoading(true);
      const apragSettings = await getAPRAGSettings(sessionId);
      setSettings(apragSettings);
    } catch (error) {
      console.error("Failed to load APRAG settings:", error);
      // Set to disabled state on error
      setSettings({
        enabled: false,
        global_enabled: false,
        session_enabled: null,
        features: {
          feedback_collection: false,
          personalization: false,
          recommendations: false,
          analytics: false,
        },
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadSettings();
  }, [sessionId]);

  return {
    settings,
    isEnabled: settings?.enabled ?? false,
    isLoading,
    features: settings?.features ?? {
      feedback_collection: false,
      personalization: false,
      recommendations: false,
      analytics: false,
    },
    refresh: loadSettings,
  };
}

export default useAPRAGSettings;













