"use client";

import { useState, useEffect, useCallback, useMemo, FormEvent } from "react";
import {
  listSessions,
  ragQuery,
  generateSuggestions,
  listAvailableModels,
  listAvailableEmbeddingModels,
  getSession,
  saveSessionRagSettings,
  SessionMeta,
  RAGSource,
  CorrectionDetails,
} from "@/lib/api";
import { useStudentChat } from "@/hooks/useStudentChat";
import { useAuth } from "@/contexts/AuthContext";

// Extended RAG source interface
interface ExtendedRAGSource extends Omit<RAGSource, "content"> {
  content?: string;
  text?: string;
  metadata?: any;
  score: number;
  crag_score?: number;
}

// Chat history item interface
interface ChatHistoryItem {
  user: string;
  bot: string;
  sources?: ExtendedRAGSource[];
  durationMs?: number;
  suggestions?: string[];
  timestamp?: string;
  interactionId?: number;
  correction?: CorrectionDetails;
  aprag_interaction_id?: number;
}

export function useEducationAssistant() {
  const { user } = useAuth();

  // Session Management
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string>("");
  const [sessionRagSettings, setSessionRagSettings] = useState<any>(null);

  // Chat & Query States
  const [query, setQuery] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);

  // localStorage utility functions
  const loadChatHistory = useCallback((): ChatHistoryItem[] => {
    try {
      const saved = localStorage.getItem("education-assistant-chat-history");
      if (saved) {
        const parsed = JSON.parse(saved);
        return Array.isArray(parsed) ? parsed : [];
      }
    } catch (error) {
      console.log("Chat history load error:", error);
    }
    return [];
  }, []);

  const saveChatHistory = useCallback((history: ChatHistoryItem[]) => {
    try {
      localStorage.setItem(
        "education-assistant-chat-history",
        JSON.stringify(history)
      );
    } catch (error) {
      console.log("Chat history save error:", error);
    }
  }, []);

  // Clear chat history function
  const clearChatHistory = useCallback(() => {
    setChatHistory([]);
    localStorage.removeItem("education-assistant-chat-history");
  }, []);

  // Model Selection States
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>("groq");
  const [selectedQueryModel, setSelectedQueryModel] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelProviders, setModelProviders] = useState<Record<string, any>>({});

  // Embedding Model States
  const [selectedEmbeddingProvider, setSelectedEmbeddingProvider] =
    useState<string>("alibaba");
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] =
    useState<string>("nomad-embed");
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState<{
    ollama: string[];
    huggingface: Array<{
      id: string;
      name: string;
      description?: string;
      dimensions?: number;
      language?: string;
    }>;
    alibaba: Array<{
      id: string;
      name: string;
      description?: string;
      dimensions?: number;
      language?: string;
    }>;
  }>({ ollama: [], huggingface: [], alibaba: [] });
  const [embeddingModelsLoading, setEmbeddingModelsLoading] = useState(false);

  // Advanced Settings
  const [useDirectLLM, setUseDirectLLM] = useState<boolean>(false);
  const [chainType, setChainType] = useState<"stuff" | "refine" | "map_reduce">(
    "stuff"
  );
  const [answerLength, setAnswerLength] = useState<
    "short" | "normal" | "detailed"
  >("normal");
  const [selectedRerankerType, setSelectedRerankerType] =
    useState<string>("ms-marco");
  const [useRerankerService, setUseRerankerService] = useState<boolean>(false);

  // UI States
  const [isChatOpen, setIsChatOpen] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Source Modal States
  const [sourceModalOpen, setSourceModalOpen] = useState(false);
  const [selectedSource, setSelectedSource] = useState<RAGSource | null>(null);

  // Settings Save States
  const [savingSettings, setSavingSettings] = useState(false);
  const [savedSettingsInfo, setSavedSettingsInfo] = useState<string | null>(
    null
  );

  // Normalize user role
  const userRole = user?.role_name?.toLowerCase() || "teacher";
  const isStudent =
    userRole === "student" ||
    user?.role_name === "student" ||
    user?.role_name === "Student";

  // Student chat integration
  const studentChatResult = useStudentChat({
    sessionId: isStudent && selectedSessionId ? selectedSessionId : "",
    autoSave: isStudent,
    maxMessages: 50,
  });

  const {
    messages: studentMessages = [],
    sendMessage: sendStudentMessage = async () => {},
    isLoading: studentChatLoading = false,
    clearHistory: clearStudentHistory = async () => {},
    error: studentChatError = null,
  } = isStudent && selectedSessionId ? studentChatResult : {};

  // Auto-clear messages
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 8000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Filtered models by provider
  const filteredModels = useMemo(() => {
    return availableModels.filter((model: any) => {
      if (typeof model === "string") return false;
      const modelProvider = (model.provider || "").toLowerCase().trim();
      const selectedProviderLower = selectedProvider.toLowerCase().trim();

      return (
        modelProvider === selectedProviderLower ||
        (selectedProviderLower === "huggingface" &&
          (modelProvider === "hf" || modelProvider === "huggingface"))
      );
    });
  }, [availableModels, selectedProvider]);

  // Session Management Functions
  const refreshSessions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await listSessions();
      setSessions(data);

      // Auto-select session only for teachers
      if (!selectedSessionId && data.length > 0 && !isStudent) {
        const sortedSessions = [...data].sort(
          (a, b) =>
            new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
        );
        setSelectedSessionId(sortedSessions[0].session_id);
      }
    } catch (e: any) {
      setError(e.message || "Oturumlar yüklenemedi");
    } finally {
      setLoading(false);
    }
  }, [selectedSessionId, isStudent]);

  // Load session RAG settings
  const loadSessionRagSettings = useCallback(async () => {
    if (!selectedSessionId) {
      setSessionRagSettings(null);
      return;
    }

    try {
      const session = await getSession(selectedSessionId);
      if (session?.rag_settings) {
        setSessionRagSettings(session.rag_settings);

        // Load settings into state
        if (session.rag_settings.embedding_model) {
          setSelectedEmbeddingModel(session.rag_settings.embedding_model);

          // Determine provider based on model name
          const modelName = session.rag_settings.embedding_model.toLowerCase();
          if (
            modelName.includes("huggingface") ||
            modelName.includes("sentence-transformers") ||
            modelName.includes("intfloat") ||
            modelName.includes("multilingual")
          ) {
            setSelectedEmbeddingProvider("huggingface");
          } else {
            setSelectedEmbeddingProvider("ollama");
          }
        }

        if (session.rag_settings.model) {
          setSelectedQueryModel(session.rag_settings.model);
        }

        if (session.rag_settings.chain_type) {
          setChainType(
            session.rag_settings.chain_type as "stuff" | "refine" | "map_reduce"
          );
        }

        if (session.rag_settings.use_direct_llm !== undefined) {
          setUseDirectLLM(session.rag_settings.use_direct_llm);
        }

        if (session.rag_settings.use_reranker_service !== undefined) {
          setUseRerankerService(session.rag_settings.use_reranker_service);
        }

        if (session.rag_settings.reranker_type) {
          setSelectedRerankerType(session.rag_settings.reranker_type);
        }
      } else {
        setSessionRagSettings(null);
      }
    } catch (e: any) {
      console.error("Failed to load session rag settings:", e);
      setSessionRagSettings(null);
    }
  }, [selectedSessionId]);

  // Model Management Functions
  const fetchAvailableModels = useCallback(async () => {
    try {
      setModelsLoading(true);
      setError(null);
      const data = await listAvailableModels();

      console.log("API Response:", data); // Debug log

      // Ensure we have models and providers data
      const models = data?.models || [];
      const providers = data?.providers || {};

      // If providers is empty but models exist, create providers from models
      if (Object.keys(providers).length === 0 && models.length > 0) {
        const createdProviders: Record<string, any> = {};
        models.forEach((model: any) => {
          if (model && typeof model === "object" && model.provider) {
            if (!createdProviders[model.provider]) {
              createdProviders[model.provider] = {
                name:
                  model.provider.charAt(0).toUpperCase() +
                  model.provider.slice(1),
                models: [],
              };
            }
            createdProviders[model.provider].models.push(model);
          }
        });
        setModelProviders(createdProviders);
      } else {
        setModelProviders(providers);
      }

      setAvailableModels(models);

      // Set default model based on selected provider
      if (models.length > 0) {
        const providerModels = models.filter((m: any) => {
          if (!m || typeof m === "string") return false;
          const modelProvider = (m.provider || "").toLowerCase();
          return modelProvider === selectedProvider.toLowerCase();
        });

        if (providerModels.length > 0) {
          const firstModel = providerModels[0];
          const modelId =
            typeof firstModel === "string"
              ? firstModel
              : firstModel.id || firstModel.name;
          if (modelId) {
            setSelectedQueryModel(modelId);
          }
        }
      }

      console.log(
        "Set models:",
        models.length,
        "providers:",
        Object.keys(providers).length
      ); // Debug log
    } catch (e: any) {
      console.error("Error fetching models:", e);
      setError(e.message || "Modeller yüklenemedi");
      // Set fallback providers if API fails
      setModelProviders({
        groq: { name: "Groq", models: [] },
        ollama: { name: "Ollama", models: [] },
        huggingface: { name: "HuggingFace", models: [] },
      });
    } finally {
      setModelsLoading(false);
    }
  }, [selectedProvider]);

  const fetchAvailableEmbeddingModels = useCallback(async () => {
    try {
      setEmbeddingModelsLoading(true);
      const data = await listAvailableEmbeddingModels();
      const dataWithAlibaba = {
        ollama: data.ollama || [],
        huggingface: data.huggingface || [],
        alibaba: (data as any).alibaba || [],
      };
      setAvailableEmbeddingModels(dataWithAlibaba);

      // Set default embedding model based on provider
      if (!selectedEmbeddingModel) {
        if (
          selectedEmbeddingProvider === "alibaba" &&
          dataWithAlibaba.alibaba?.length > 0
        ) {
          setSelectedEmbeddingModel(dataWithAlibaba.alibaba[0].id);
        } else if (
          selectedEmbeddingProvider === "ollama" &&
          dataWithAlibaba.ollama.length > 0
        ) {
          // Prefer nomad-embed if available, otherwise use first model
          const nomadEmbed = dataWithAlibaba.ollama.find((model) =>
            model.includes("nomad-embed")
          );
          setSelectedEmbeddingModel(nomadEmbed || dataWithAlibaba.ollama[0]);
        } else if (
          selectedEmbeddingProvider === "huggingface" &&
          dataWithAlibaba.huggingface.length > 0
        ) {
          setSelectedEmbeddingModel(dataWithAlibaba.huggingface[0].id);
        }
      }
    } catch (e: any) {
      console.error("Error fetching embedding models:", e);
      setError(e.message || "Embedding modelleri yüklenemedi");
    } finally {
      setEmbeddingModelsLoading(false);
    }
  }, [selectedEmbeddingModel]);

  // Provider change handler
  const handleProviderChange = useCallback(
    (provider: string) => {
      setSelectedProvider(provider);

      const providerModels = availableModels.filter((model: any) => {
        if (typeof model === "string") return false;
        const modelProvider = (model.provider || "").toLowerCase().trim();
        return modelProvider === provider.toLowerCase().trim();
      });

      if (providerModels.length > 0) {
        const firstModel = providerModels[0];
        const modelId =
          typeof firstModel === "string"
            ? firstModel
            : firstModel.id || firstModel.name;
        setSelectedQueryModel(modelId);
      } else {
        setSelectedQueryModel("");
      }
    },
    [availableModels]
  );

  // Query handling
  const handleQuery = useCallback(
    async (e: FormEvent, suggestionText?: string) => {
      e.preventDefault();

      if (!useDirectLLM && !selectedSessionId) return;
      const queryText = suggestionText || query;
      if (!queryText.trim()) return;

      if (!isStudent && !selectedQueryModel) {
        setError("Lütfen bir model seçin");
        return;
      }

      const userMessage = queryText;
      if (!suggestionText) setQuery("");
      setError(null);

      // For students, use the database-backed chat system
      if (isStudent && selectedSessionId && sendStudentMessage) {
        try {
          await sendStudentMessage(userMessage, sessionRagSettings);
        } catch (e: any) {
          setError(e.message || "Sorgu başarısız oldu");
        }
        return;
      }

      // Teacher/traditional flow
      setIsQuerying(true);

      // Build conversation history
      const conversationHistory = chatHistory.slice(-4).flatMap((msg) => {
        const messages: Array<{ role: "user" | "assistant"; content: string }> =
          [];
        if (msg.user && msg.user.trim() && msg.user !== "...") {
          messages.push({ role: "user", content: msg.user });
        }
        if (
          msg.bot &&
          msg.bot.trim() &&
          msg.bot !== "..." &&
          !msg.bot.startsWith("Hata:")
        ) {
          messages.push({ role: "assistant", content: msg.bot });
        }
        return messages;
      });

      setChatHistory((prev) => [
        ...prev,
        {
          user: userMessage,
          bot: "...",
          timestamp: new Date().toISOString(),
        },
      ]);

      const startTime = Date.now();

      try {
        const maxTokensMap = {
          short: 1024,
          normal: 2048,
          detailed: 4096,
        };
        const maxTokens = maxTokensMap[answerLength];

        const payload: any = {
          session_id: selectedSessionId || "direct-llm-session",
          query: userMessage,
          top_k: 5,
          use_rerank: sessionRagSettings?.use_rerank ?? false,
          min_score: sessionRagSettings?.min_score ?? 0.5,
          max_context_chars: 8000,
          use_direct_llm: useDirectLLM,
          max_tokens: maxTokens,
          conversation_history:
            conversationHistory.length > 0 ? conversationHistory : undefined,
        };

        if (!isStudent) {
          payload.model =
            sessionRagSettings?.model || selectedQueryModel || undefined;
          payload.chain_type =
            sessionRagSettings?.chain_type || chainType || "stuff";
        }

        if (sessionRagSettings?.embedding_model) {
          payload.embedding_model = sessionRagSettings.embedding_model;
        } else if (!isStudent && selectedEmbeddingModel) {
          payload.embedding_model = selectedEmbeddingModel;
        }

        const result = await ragQuery(payload);
        const actualDurationMs = Date.now() - startTime;

        setChatHistory((prev) => {
          const newHistory = [...prev];
          newHistory[newHistory.length - 1].bot = result.answer;
          newHistory[newHistory.length - 1].sources = result.sources || [];
          newHistory[newHistory.length - 1].suggestions = [];
          newHistory[newHistory.length - 1].durationMs = actualDurationMs;
          newHistory[newHistory.length - 1].correction = result.correction;
          return newHistory;
        });

        // Fetch suggestions asynchronously
        (async () => {
          try {
            const sugg = await generateSuggestions({
              question: userMessage,
              answer: result.answer,
              sources: result.sources || [],
            });
            if (Array.isArray(sugg) && sugg.length > 0) {
              setChatHistory((prev) => {
                const updated = [...prev];
                updated[updated.length - 1].suggestions = sugg;
                return updated;
              });
            }
          } catch (_err) {
            // Silently ignore suggestion errors
          }
        })();
      } catch (e: any) {
        const errorMessage = e.message || "Sorgu başarısız oldu";
        setError(errorMessage);
        setChatHistory((prev) => {
          const newHistory = [...prev];
          newHistory[newHistory.length - 1].bot = `Hata: ${errorMessage}`;
          return newHistory;
        });
      } finally {
        setIsQuerying(false);
      }
    },
    [
      useDirectLLM,
      selectedSessionId,
      query,
      isStudent,
      selectedQueryModel,
      sendStudentMessage,
      sessionRagSettings,
      chatHistory,
      answerLength,
      chainType,
      selectedEmbeddingModel,
    ]
  );

  // Suggestion click handler
  const handleSuggestionClick = useCallback(
    async (text: string) => {
      const fakeEvent = { preventDefault: () => {} } as unknown as FormEvent;
      await handleQuery(fakeEvent, text);
    },
    [handleQuery]
  );

  // Source modal handlers
  const handleOpenSourceModal = useCallback((source: RAGSource) => {
    setSelectedSource(source);
    setSourceModalOpen(true);
  }, []);

  const handleCloseSourceModal = useCallback(() => {
    setSelectedSource(null);
    setSourceModalOpen(false);
  }, []);

  // Load chat history from localStorage on component mount
  useEffect(() => {
    const savedHistory = loadChatHistory();
    if (savedHistory.length > 0) {
      setChatHistory(savedHistory);
    }
  }, [loadChatHistory]);

  // Save chat history to localStorage whenever it changes
  useEffect(() => {
    if (chatHistory.length > 0) {
      saveChatHistory(chatHistory);
    }
  }, [chatHistory, saveChatHistory]);

  // Initialize data on mount
  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    loadSessionRagSettings();
  }, [loadSessionRagSettings]);

  useEffect(() => {
    if (availableModels.length === 0) {
      fetchAvailableModels();
    }
  }, [availableModels.length, fetchAvailableModels]);

  useEffect(() => {
    if (
      availableEmbeddingModels.ollama.length === 0 &&
      availableEmbeddingModels.huggingface.length === 0 &&
      availableEmbeddingModels.alibaba.length === 0
    ) {
      fetchAvailableEmbeddingModels();
    }
  }, [
    availableEmbeddingModels.ollama.length,
    availableEmbeddingModels.huggingface.length,
    fetchAvailableEmbeddingModels,
  ]);

  return {
    // Session Management
    sessions,
    selectedSessionId,
    setSelectedSessionId,
    sessionRagSettings,
    refreshSessions,

    // Chat & Query
    query,
    setQuery,
    chatHistory,
    setChatHistory,
    clearChatHistory,
    isQuerying,
    handleQuery,
    handleSuggestionClick,

    // Student Chat Integration
    studentMessages,
    sendStudentMessage,
    studentChatLoading,
    clearStudentHistory,
    studentChatError,
    isStudent,

    // Model Management
    availableModels,
    filteredModels,
    selectedProvider,
    setSelectedProvider,
    selectedQueryModel,
    setSelectedQueryModel,
    modelsLoading,
    modelProviders,
    handleProviderChange,

    // Embedding Models
    selectedEmbeddingProvider,
    setSelectedEmbeddingProvider,
    selectedEmbeddingModel,
    setSelectedEmbeddingModel,
    availableEmbeddingModels,
    embeddingModelsLoading,

    // Advanced Settings
    useDirectLLM,
    setUseDirectLLM,
    chainType,
    setChainType,
    answerLength,
    setAnswerLength,
    selectedRerankerType,
    setSelectedRerankerType,
    useRerankerService,
    setUseRerankerService,

    // UI States
    isChatOpen,
    setIsChatOpen,
    loading,
    error,
    setError,
    success,
    setSuccess,

    // Source Modal
    sourceModalOpen,
    selectedSource,
    handleOpenSourceModal,
    handleCloseSourceModal,

    // Settings
    savingSettings,
    setSavingSettings,
    savedSettingsInfo,
    setSavedSettingsInfo,

    // User Info
    user,
    userRole,
  };
}

export type { ChatHistoryItem, ExtendedRAGSource };
