/**
 * useRagSettings Hook - RAG configuration management
 * Handles RAG settings, model loading and validation, settings persistence
 * Integrates with existing API and session management
 */

import { useState, useEffect } from "react";
import { useSessionStore } from "../stores/sessionStore";
import {
  listAvailableModels,
  listAvailableEmbeddingModels,
  saveSessionRagSettings,
  getApiUrl,
} from "@/lib/api";

// Types matching the backup page implementation
interface ModelInfo {
  id: string;
  name?: string;
  provider: string;
  description?: string;
  dimensions?: number;
  language?: string;
}

interface AvailableEmbeddingModels {
  ollama: string[];
  huggingface: Array<{
    id: string;
    name: string;
    description?: string;
    dimensions?: number;
    language?: string;
  }>;
  alibaba?: Array<{
    id: string;
    name: string;
    description?: string;
    dimensions?: number;
    language?: string;
  }>;
}

interface RagSettings {
  provider?: string;
  model?: string;
  embedding_provider?: string;
  embedding_model?: string;
  use_reranker_service?: boolean;
  reranker_type?: "bge" | "ms-marco";
  top_k?: number;
  use_rerank?: boolean;
  min_score?: number;
  max_context_chars?: number;
}

export const useRagSettings = (sessionId: string) => {
  const { currentSession, fetchSessionData } = useSessionStore();

  // Model states
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [availableEmbeddingModels, setAvailableEmbeddingModels] =
    useState<AvailableEmbeddingModels>({
      ollama: [],
      huggingface: [],
      alibaba: [],
    });

  // Settings states
  const [selectedProvider, setSelectedProvider] = useState<string>("groq");
  const [selectedQueryModel, setSelectedQueryModel] = useState<string>("");
  const [selectedEmbeddingProvider, setSelectedEmbeddingProvider] =
    useState<string>("alibaba"); // Alibaba first as per requirement
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] =
    useState<string>("");
  const [useRerankerService, setUseRerankerService] = useState<boolean>(false);
  const [selectedRerankerType, setSelectedRerankerType] =
    useState<string>("bge-reranker-v2-m3");

  // Loading states
  const [modelsLoading, setModelsLoading] = useState(false);
  const [embeddingModelsLoading, setEmbeddingModelsLoading] = useState(false);
  const [savingSettings, setSavingSettings] = useState(false);

  // UI states
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Provider options matching backup page
  const PROVIDER_OPTIONS = [
    { value: "groq", label: "🌐 Groq (Cloud - Hızlı)" },
    { value: "alibaba", label: "🛒 Alibaba (Cloud - Qwen)" },
    { value: "deepseek", label: "🔮 DeepSeek (Cloud - Premium)" },
    { value: "openrouter", label: "🚀 OpenRouter (Cloud - Güçlü)" },
    { value: "huggingface", label: "🤗 HuggingFace (Ücretsiz)" },
    { value: "ollama", label: "🏠 Ollama (Yerel)" },
  ];

  const EMBEDDING_PROVIDER_OPTIONS = [
    { value: "alibaba", label: "🛒 Alibaba (Cloud - Qwen)" },
    { value: "huggingface", label: "🤗 HuggingFace (Ücretsiz)" },
    { value: "ollama", label: "🏠 Ollama (Yerel)" },
  ];

  const RERANKER_TYPE_OPTIONS = [
    {
      value: "bge-reranker-v2-m3",
      label: "🔮 BGE-Reranker-V2-M3 (Türkçe Optimize)",
    },
    {
      value: "ms-marco-minilm-l6",
      label: "⚡ MS-MARCO MiniLM-L6 (Hafif, Hızlı)",
    },
    {
      value: "gte-rerank-v2",
      label: "🛒 GTE-Rerank-V2 (Alibaba - 50+ Dil)",
    },
  ];

  // Load settings from session
  useEffect(() => {
    if (currentSession?.rag_settings) {
      const settings = currentSession.rag_settings;

      if (settings.provider) setSelectedProvider(settings.provider);
      if (settings.model) setSelectedQueryModel(settings.model);
      if (settings.embedding_provider)
        setSelectedEmbeddingProvider(settings.embedding_provider);
      if (settings.embedding_model)
        setSelectedEmbeddingModel(settings.embedding_model);
      if (settings.use_reranker_service !== undefined)
        setUseRerankerService(settings.use_reranker_service);
      if (settings.reranker_type)
        setSelectedRerankerType(settings.reranker_type);

      setHasUnsavedChanges(false);
    } else {
      // Set defaults
      setSelectedProvider("groq");
      setSelectedQueryModel("");
      setSelectedEmbeddingProvider("alibaba");
      setSelectedEmbeddingModel("");
      setUseRerankerService(false);
      setSelectedRerankerType("bge-reranker-v2-m3");
      setHasUnsavedChanges(false);
    }
  }, [currentSession?.rag_settings]);

  // Track unsaved changes
  useEffect(() => {
    if (!currentSession?.rag_settings) return;

    const currentSettings = currentSession.rag_settings;
    const hasChanges =
      currentSettings.provider !== selectedProvider ||
      currentSettings.model !== selectedQueryModel ||
      currentSettings.embedding_provider !== selectedEmbeddingProvider ||
      currentSettings.embedding_model !== selectedEmbeddingModel ||
      currentSettings.use_reranker_service !== useRerankerService ||
      currentSettings.reranker_type !== selectedRerankerType;

    setHasUnsavedChanges(hasChanges);
  }, [
    selectedProvider,
    selectedQueryModel,
    selectedEmbeddingProvider,
    selectedEmbeddingModel,
    useRerankerService,
    selectedRerankerType,
    currentSession?.rag_settings,
  ]);

  // Fetch available models when provider changes
  const fetchModels = async () => {
    if (!selectedProvider) return;

    setModelsLoading(true);
    setError(null);

    try {
      const response = await listAvailableModels();
      const models = Array.isArray(response) ? response : response.models || [];

      const filtered = models.filter((m: any) => {
        const provider = m.provider || "";
        return provider.toLowerCase() === selectedProvider.toLowerCase();
      });

      setAvailableModels(filtered);
    } catch (e: any) {
      setError(`Modeller yüklenemedi: ${e.message}`);
    } finally {
      setModelsLoading(false);
    }
  };

  // Fetch embedding models
  const fetchEmbeddingModels = async () => {
    setEmbeddingModelsLoading(true);
    setError(null);

    try {
      const data = await listAvailableEmbeddingModels();
      // Handle the data properly with type assertion
      const embeddingData = data as any;
      setAvailableEmbeddingModels({
        ollama: embeddingData.ollama || [],
        huggingface: embeddingData.huggingface || [],
        alibaba: embeddingData.alibaba || [],
      });

      // Set default embedding model if not set
      if (!selectedEmbeddingModel) {
        if (
          selectedEmbeddingProvider === "alibaba" &&
          embeddingData.alibaba &&
          embeddingData.alibaba.length > 0
        ) {
          setSelectedEmbeddingModel(embeddingData.alibaba[0].id);
        } else if (
          selectedEmbeddingProvider === "ollama" &&
          embeddingData.ollama &&
          embeddingData.ollama.length > 0
        ) {
          setSelectedEmbeddingModel(embeddingData.ollama[0]);
        } else if (
          embeddingData.huggingface &&
          embeddingData.huggingface.length > 0
        ) {
          setSelectedEmbeddingModel(embeddingData.huggingface[0].id);
        }
      }
    } catch (e: any) {
      setError(`Embedding modelleri yüklenemedi: ${e.message}`);
    } finally {
      setEmbeddingModelsLoading(false);
    }
  };

  // Save RAG settings
  const saveSettings = async () => {
    setSavingSettings(true);
    setError(null);
    setSuccess(null);

    try {
      // Prepare settings object
      const settingsToSave: RagSettings = {
        top_k: 5,
        use_rerank: currentSession?.rag_settings?.use_rerank ?? false,
        min_score: 0.5,
        max_context_chars: 8000,
        use_reranker_service: useRerankerService,
      };

      // Add selected values if they exist
      if (selectedQueryModel?.trim()) {
        settingsToSave.model = selectedQueryModel;
      }
      if (selectedProvider?.trim()) {
        settingsToSave.provider = selectedProvider;
      }
      if (selectedEmbeddingModel?.trim()) {
        settingsToSave.embedding_model = selectedEmbeddingModel;
      }
      if (selectedEmbeddingProvider?.trim()) {
        settingsToSave.embedding_provider = selectedEmbeddingProvider;
      }
      if (useRerankerService && selectedRerankerType) {
        // Map reranker type values to expected API format
        const rerankerTypeMapping: Record<string, "bge" | "ms-marco"> = {
          "bge-reranker-v2-m3": "bge",
          "ms-marco-minilm-l6": "ms-marco",
          "gte-rerank-v2": "bge",
        };
        settingsToSave.reranker_type =
          rerankerTypeMapping[selectedRerankerType] || "bge";
      }

      const response = await saveSessionRagSettings(
        sessionId,
        settingsToSave as any
      );

      setSuccess("✅ RAG ayarları başarıyla kaydedildi");
      setHasUnsavedChanges(false);

      // Refresh session data
      await fetchSessionData(sessionId);
    } catch (e: any) {
      setError(`Ayarlar kaydedilemedi: ${e.message}`);
    } finally {
      setSavingSettings(false);
    }
  };

  // Reset to defaults
  const resetSettings = () => {
    setSelectedProvider("groq");
    setSelectedQueryModel("");
    setSelectedEmbeddingProvider("alibaba");
    setSelectedEmbeddingModel("");
    setUseRerankerService(false);
    setSelectedRerankerType("bge-reranker-v2-m3");
    setError(null);
    setSuccess(null);
  };

  // Validation
  const validateSettings = () => {
    const errors: string[] = [];

    if (!selectedProvider) errors.push("AI Provider seçilmeli");
    if (!selectedQueryModel) errors.push("AI Model seçilmeli");
    if (!selectedEmbeddingProvider) errors.push("Embedding Provider seçilmeli");
    if (!selectedEmbeddingModel) errors.push("Embedding Model seçilmeli");

    return {
      isValid: errors.length === 0,
      errors,
    };
  };

  // Clear messages
  const clearMessages = () => {
    setError(null);
    setSuccess(null);
  };

  // Auto-clear messages after delay
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 8000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 8000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  // Initial data fetching
  useEffect(() => {
    fetchEmbeddingModels();
  }, []);

  useEffect(() => {
    fetchModels();
  }, [selectedProvider]);

  return {
    // State
    availableModels,
    availableEmbeddingModels,
    selectedProvider,
    selectedQueryModel,
    selectedEmbeddingProvider,
    selectedEmbeddingModel,
    useRerankerService,
    selectedRerankerType,

    // Loading states
    modelsLoading,
    embeddingModelsLoading,
    savingSettings,

    // UI states
    error,
    success,
    hasUnsavedChanges,

    // Actions
    setSelectedProvider,
    setSelectedQueryModel,
    setSelectedEmbeddingProvider,
    setSelectedEmbeddingModel,
    setUseRerankerService,
    setSelectedRerankerType,

    // Operations
    saveSettings,
    resetSettings,
    validateSettings,
    clearMessages,
    fetchModels,
    fetchEmbeddingModels,

    // Constants
    PROVIDER_OPTIONS,
    EMBEDDING_PROVIDER_OPTIONS,
    RERANKER_TYPE_OPTIONS,
  };
};
