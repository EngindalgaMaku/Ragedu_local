"use client";
import React, { useState, useEffect } from "react";
import {
  getChunksForSession,
  listSessions,
  SessionMeta,
  Chunk,
  listAvailableEmbeddingModels,
  saveSessionRagSettings,
  getSessionInteractions,
  APRAGInteraction,
  getApiUrl,
  listAvailableModels,
  getSession,
} from "@/lib/api";
import TopicManagementPanel from "@/components/TopicManagementPanel";
import SessionSettingsPanel from "@/components/SessionSettingsPanel";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import FileUploadModal from "@/components/FileUploadModal";
import DocumentUploadModal from "@/components/DocumentUploadModal";
import TeacherLayout from "@/app/components/TeacherLayout";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  ArrowLeft,
  FileText,
  Settings,
  Target,
  BookOpen,
  MessageSquare,
  Upload,
  RefreshCw,
  Sparkles,
  Brain,
  Zap,
  Layers,
  Database,
  ChevronRight,
} from "lucide-react";

export default function SessionPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;

  // Handler for sidebar navigation
  type TabType = "dashboard" | "sessions" | "upload" | "analytics" | "modules" | "assistant" | "query";
  const handleTabChange = (tab: TabType) => {
    // Navigate to main page, which will handle the tab change
    router.push("/");
  };

  // State management
  const [session, setSession] = useState<SessionMeta | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [chunkPage, setChunkPage] = useState(1);
  const [showModal, setShowModal] = useState(false);
  const [showChunkModal, setShowChunkModal] = useState(false);
  const [selectedChunkForModal, setSelectedChunkForModal] =
    useState<Chunk | null>(null);
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] =
    useState<string>("nomic-embed-text");
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState<{
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
  }>({ ollama: [], huggingface: [], alibaba: [] });
  const [embeddingModelsLoading, setEmbeddingModelsLoading] = useState(false);

  // Note: Embedding model loading is now handled in the main RAG settings useEffect above
  const [interactions, setInteractions] = useState<APRAGInteraction[]>([]);
  const [interactionsLoading, setInteractionsLoading] = useState(false);
  const [showInteractions, setShowInteractions] = useState(false);
  const [apragEnabled, setApragEnabled] = useState<boolean>(false);
  const [interactionsPage, setInteractionsPage] = useState(1);
  const [interactionsTotal, setInteractionsTotal] = useState(0);
  const INTERACTIONS_PER_PAGE = 10;
  const [activeTab, setActiveTab] = useState<
    "genel" | "chunks" | "topics" | "interactions" | "session-settings"
  >("genel");
  
  // Batch processing job tracking
  const [batchJobId, setBatchJobId] = useState<string | null>(null);
  const [batchStatus, setBatchStatus] = useState<any | null>(null);

  // RAG Settings states
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>("groq");
  const [selectedQueryModel, setSelectedQueryModel] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelProviders, setModelProviders] = useState<Record<string, any>>({});
  const [savingSettings, setSavingSettings] = useState(false);
  const [savedSettingsInfo, setSavedSettingsInfo] = useState<string | null>(
    null
  );
  const [selectedRerankerType, setSelectedRerankerType] =
    useState<string>("ms-marco");
  const [useRerankerService, setUseRerankerService] = useState<boolean>(false);
  const [selectedEmbeddingProvider, setSelectedEmbeddingProvider] =
    useState<string>("ollama");
  const [selectedDocumentFilter, setSelectedDocumentFilter] = useState<
    string | null
  >(null); // Filter by document name
  const [searchTerm, setSearchTerm] = useState("");
  const [showAllFiles, setShowAllFiles] = useState(false);
  const CHUNKS_PER_PAGE = 10;

  // Fetch session details
  const fetchSessionDetails = async () => {
    try {
      const currentSession = await getSession(sessionId);
      if (currentSession) {
        setSession(currentSession);
      } else {
        setError("Ders oturumu bulunamadı");
      }
    } catch (e: any) {
      setError(e.message || "Ders oturumu bilgileri yüklenemedi");
    }
  };

  // Fetch chunks for the session
  const fetchChunks = async () => {
    try {
      setLoading(true);
      setError(null);
      const sessionChunks = await getChunksForSession(sessionId);

      // CRITICAL FIX: Remove duplicate chunks based on chunk_id
      const seenChunkIds = new Set<string>();
      const uniqueChunks = sessionChunks.filter((chunk: any) => {
        const chunkId =
          chunk.chunk_metadata?.chunk_id ||
          `${chunk.document_name}-${chunk.chunk_index}`;
        if (seenChunkIds.has(chunkId)) {
          console.warn(`⚠️ Duplicate chunk filtered: ${chunkId}`);
          return false;
        }
        seenChunkIds.add(chunkId);
        return true;
      });

      console.log(
        `📊 Fetched ${sessionChunks.length} chunks, ${uniqueChunks.length} unique after deduplication`
      );
      setChunks(uniqueChunks);
      // Reset filter when chunks are refreshed
      setSelectedDocumentFilter(null);
      setChunkPage(1);
    } catch (e: any) {
      setError(e.message || "Parçalar yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  // Handle modal success
  const handleModalSuccess = async (result: any) => {
    // Check if this is a batch processing job
    if (result.job_id && result.background_processing) {
      // Batch processing started - track job
      setBatchJobId(result.job_id);
      setProcessing(true);
      setSuccess(
        `Batch işlem başlatıldı: ${result.total_files || 0} dosya arka planda işleniyor...`
      );
      // Polling will start via useEffect
      return;
    }
    
    // Normal completion (non-batch or completed batch)
    const processedCount = result?.processed_count ?? 0;
    const totalChunks = result?.total_chunks_added ?? 0;

    setSuccess(
      `RAG işlemi tamamlandı! ${processedCount} dosya işlendi, ${totalChunks} parça oluşturuldu.`
    );
    setProcessing(false);
    setBatchJobId(null);
    setBatchStatus(null);
    // Refresh chunks after successful processing
    await fetchChunks();
    await fetchSessionDetails();
  };

  // Handle modal error
  const handleModalError = (error: string) => {
    setError(error);
    setProcessing(false);
  };

  const handleLogout = () => {
    localStorage.removeItem("isAuthenticated");
    localStorage.removeItem("userRole");
    router.push("/login");
  };

  // Poll batch processing status (similar to KB extraction)
  useEffect(() => {
    if (!batchJobId) return;

    let interval: NodeJS.Timeout | null = null;
    let consecutiveErrors = 0;
    const MAX_CONSECUTIVE_ERRORS = 5;

    const pollStatus = async () => {
      try {
        const res = await fetch(
          `/api/documents/process-and-store-batch/status/${batchJobId}`
        );

        if (!res.ok) {
          consecutiveErrors++;
          if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
            console.error("Batch status polling failed too many times");
            if (interval) clearInterval(interval);
            setBatchJobId(null);
            setProcessing(false);
            setError(
              "Batch işlem durumu alınamadı. İşlem arka planda devam ediyor olabilir."
            );
            return;
          }
          return;
        }

        consecutiveErrors = 0;
        const data = await res.json();
        setBatchStatus(data.job);

        if (data.job.status === "completed" || data.job.status === "completed_with_errors" || data.job.status === "failed") {
          if (interval) clearInterval(interval);
          setBatchJobId(null);
          setProcessing(false);

          if (data.job.status === "completed") {
            setSuccess(
              `Batch işlem tamamlandı! ${data.job.processed_successfully}/${data.job.total_files} dosya işlendi, ${data.job.total_chunks} chunk oluşturuldu.`
            );
            await fetchChunks();
            await fetchSessionDetails();
          } else if (data.job.status === "completed_with_errors") {
            setSuccess(
              `Batch işlem tamamlandı (bazı hatalar var): ${data.job.processed_successfully}/${data.job.total_files} dosya işlendi, ${data.job.total_chunks} chunk oluşturuldu.`
            );
            if (data.job.errors && data.job.errors.length > 0) {
              setError(`Hatalar: ${data.job.errors.map((e: any) => e.filename).join(", ")}`);
            }
            await fetchChunks();
            await fetchSessionDetails();
          } else {
            setError(
              `Batch işlem başarısız: ${data.job.errors?.[0]?.error || "Bilinmeyen hata"}`
            );
          }
        } else if (data.job.status === "running") {
          // Show progress with batch info (Batch 1, Batch 2, etc.)
          const batchInfo = data.job.current_batch && data.job.total_batches
            ? ` - Batch ${data.job.current_batch}/${data.job.total_batches}`
            : "";
          const currentFile = data.job.current_file 
            ? ` - İşleniyor: ${data.job.current_file.split('/').pop() || data.job.current_file}`
            : "";
          setSuccess(
            `Batch işlem devam ediyor... (${data.job.processed_successfully || 0}/${data.job.total_files || 0} dosya işlendi${batchInfo}${currentFile})`
          );
        }
      } catch (err: any) {
        consecutiveErrors++;
        console.error("Batch status polling error:", err);

        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          console.error("Batch status polling failed too many times, stopping");
          if (interval) clearInterval(interval);
          setBatchJobId(null);
          setProcessing(false);
          setError(
            "Batch işlem durumu alınamadı. İşlem arka planda devam ediyor olabilir."
          );
        }
      }
    };

    interval = setInterval(pollStatus, 3000);

    // Poll immediately
    pollStatus();

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [batchJobId, sessionId]);

  // Fetch available embedding models
  const fetchAvailableEmbeddingModels = async () => {
    try {
      setEmbeddingModelsLoading(true);
      const data = await listAvailableEmbeddingModels();
      setAvailableEmbeddingModels(data);
      // Set default if available
      if (data.ollama.length > 0 && !selectedEmbeddingModel) {
        setSelectedEmbeddingModel(data.ollama[0]);
      }
    } catch (e: any) {
      console.error("Failed to fetch embedding models:", e);
      // Fallback to default embedding models if API fails
      const fallbackModels = {
        ollama: ["nomic-embed-text", "mxbai-embed-large"],
        huggingface: [
          {
            id: "sentence-transformers/all-MiniLM-L6-v2",
            name: "all-MiniLM-L6-v2",
            description: "Lightweight multilingual",
            dimensions: 384,
          },
          {
            id: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            name: "paraphrase-multilingual-MiniLM-L12-v2",
            description: "Multilingual paraphrase",
            dimensions: 384,
          },
        ],
        alibaba: [],
      };
      setAvailableEmbeddingModels(fallbackModels);
      // Set default from fallback
      if (!selectedEmbeddingModel && fallbackModels.ollama.length > 0) {
        setSelectedEmbeddingModel(fallbackModels.ollama[0]);
      }
    } finally {
      setEmbeddingModelsLoading(false);
    }
  };

  // Fetch interactions for the session with pagination
  const fetchInteractions = async (page: number = 1) => {
    try {
      setInteractionsLoading(true);
      const offset = (page - 1) * INTERACTIONS_PER_PAGE;
      const data = await getSessionInteractions(
        sessionId,
        INTERACTIONS_PER_PAGE,
        offset
      );
      setInteractions(data.interactions || []);
      setInteractionsTotal(data.total || 0);
    } catch (e: any) {
      console.error("Failed to fetch interactions:", e);
      setInteractions([]);
      setInteractionsTotal(0);
    } finally {
      setInteractionsLoading(false);
    }
  };

  // Check APRAG status
  useEffect(() => {
    const checkApragStatus = async () => {
      try {
        const response = await fetch(`${getApiUrl()}/aprag/settings/status`);
        if (response.ok) {
          const data = await response.json();
          setApragEnabled(data.global_enabled || false);
        }
      } catch (err) {
        console.error("Failed to check APRAG status:", err);
        setApragEnabled(false);
      }
    };
    checkApragStatus();
  }, []);

  // Load RAG settings when session is loaded
  useEffect(() => {
    if (session?.rag_settings) {
      console.log(
        "[RAG SETTINGS LOAD] Loading settings from session:",
        session.rag_settings
      );

      // Load provider first (needed to load models)
      if (session.rag_settings.provider) {
        setSelectedProvider(session.rag_settings.provider);
      }
      // Load model (after provider is set, models will be fetched)
      if (session.rag_settings.model) {
        setSelectedQueryModel(session.rag_settings.model);
      }
      // Load embedding provider and model - IMPORTANT: Load provider first, then model
      if (session.rag_settings.embedding_provider) {
        console.log(
          "[RAG SETTINGS LOAD] Setting embedding provider:",
          session.rag_settings.embedding_provider
        );
        setSelectedEmbeddingProvider(session.rag_settings.embedding_provider);
      }
      if (session.rag_settings.embedding_model) {
        console.log(
          "[RAG SETTINGS LOAD] Setting embedding model:",
          session.rag_settings.embedding_model
        );
        setSelectedEmbeddingModel(session.rag_settings.embedding_model);
      }
      // Load reranker settings
      if (session.rag_settings.use_reranker_service !== undefined) {
        setUseRerankerService(session.rag_settings.use_reranker_service);
      }
      if (session.rag_settings.reranker_type) {
        setSelectedRerankerType(session.rag_settings.reranker_type);
      }
    } else {
      // Reset to defaults if no settings
      setSelectedProvider("groq");
      setSelectedQueryModel("");
      setSelectedEmbeddingProvider("ollama");
      setUseRerankerService(false);
      setSelectedRerankerType("ms-marco");
    }
  }, [session?.rag_settings]);

  // Fetch models when provider changes
  useEffect(() => {
    const fetchModels = async () => {
      if (!selectedProvider) return;
      setModelsLoading(true);
      try {
        const response = await listAvailableModels();
        // listAvailableModels returns { models: ModelInfo[], providers: ... }
        const models = Array.isArray(response)
          ? response
          : response.models || [];
        const filtered = models.filter((m: any) => {
          const provider = m.provider || "";
          return provider.toLowerCase() === selectedProvider.toLowerCase();
        });
        setAvailableModels(filtered);
        setModelProviders(
          models.reduce((acc: any, m: any) => {
            const provider = m.provider || "unknown";
            if (!acc[provider]) acc[provider] = [];
            acc[provider].push(m);
            return acc;
          }, {})
        );
      } catch (e: any) {
        console.error("Failed to fetch models:", e);
      } finally {
        setModelsLoading(false);
      }
    };
    fetchModels();
  }, [selectedProvider]);

  // Fetch embedding models on mount
  useEffect(() => {
    fetchAvailableEmbeddingModels();
  }, []);

  // Initial data loading
  useEffect(() => {
    if (sessionId) {
      fetchSessionDetails();
      fetchChunks();
      fetchAvailableEmbeddingModels();
      if (apragEnabled) {
        fetchInteractions();
      }
    }
  }, [sessionId, apragEnabled]);

  // Fetch interactions when interactions tab becomes active or page changes
  useEffect(() => {
    if (activeTab === "interactions" && apragEnabled) {
      fetchInteractions(interactionsPage);
    }
  }, [activeTab, apragEnabled, interactionsPage]);

  // Reset page to 1 when switching to interactions tab
  useEffect(() => {
    if (activeTab === "interactions" && interactionsPage !== 1) {
      setInteractionsPage(1);
    }
  }, [activeTab]);

  // Clear messages after some time
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

  if (!sessionId) {
    return (
      <div className="text-center py-12">
        <div className="text-red-600">Geçersiz ders oturumu ID</div>
      </div>
    );
  }

  return (
    <TeacherLayout activeTab="sessions" onTabChange={handleTabChange}>
      <div className="space-y-6">
        {/* Minimal Header */}
        <div className="border-b border-border pb-4">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <div className="flex-1 min-w-0">
              <h1 className="text-lg sm:text-xl lg:text-2xl font-semibold text-foreground mb-1">
                {session?.name || "Ders Oturumu Yükleniyor..."}
              </h1>
              {session?.description && (
                <p className="text-sm text-muted-foreground">
                  {session.description}
                </p>
              )}
            </div>
            <Link
              href="/"
              className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors min-h-[44px] px-2 py-2 rounded-md hover:bg-muted/50"
            >
              <ArrowLeft className="w-4 h-4" />
              <span className="sm:inline">Geri</span>
            </Link>
          </div>
        </div>

        {/* Error/Success Display */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3">
            <div className="text-sm text-red-800 dark:text-red-200">
              {error}
            </div>
          </div>
        )}

        {success && (
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md p-3">
            <div className="text-sm text-green-800 dark:text-green-200">
              {success}
            </div>
          </div>
        )}

        {/* Modern Tab Navigation with shadcn/ui */}
        <Card className="overflow-hidden">
          <Tabs
            value={activeTab}
            onValueChange={(value) => setActiveTab(value as typeof activeTab)}
            className="w-full"
          >
            <div className="border-b bg-muted/30">
              <TabsList className="h-auto w-full justify-start rounded-none bg-transparent p-2 gap-1">
                <TabsTrigger
                  value="genel"
                  className="data-[state=active]:bg-background data-[state=active]:shadow-sm gap-2"
                >
                  <Settings className="w-4 h-4" />
                  <span className="hidden sm:inline">Genel </span>Ayarlar
                </TabsTrigger>
                <TabsTrigger
                  value="chunks"
                  className="data-[state=active]:bg-background data-[state=active]:shadow-sm gap-2"
                >
                  <FileText className="w-4 h-4" />
                  <span className="hidden sm:inline">Döküman </span>Parçalar
                  <span className="ml-1.5 px-1.5 py-0.5 text-xs bg-muted rounded-full">
                    {chunks.length}
                  </span>
                </TabsTrigger>
                <TabsTrigger
                  value="session-settings"
                  className="data-[state=active]:bg-background data-[state=active]:shadow-sm gap-2"
                >
                  <Target className="w-4 h-4" />
                  <span className="hidden sm:inline">Öğretim </span>Ayarları
                </TabsTrigger>
                {apragEnabled && (
                  <>
                    <TabsTrigger
                      value="topics"
                      className="data-[state=active]:bg-background data-[state=active]:shadow-sm gap-2"
                    >
                      <BookOpen className="w-4 h-4" />
                      <span className="hidden sm:inline">Konu </span>Yönetimi
                    </TabsTrigger>
                    <TabsTrigger
                      value="interactions"
                      className="data-[state=active]:bg-background data-[state=active]:shadow-sm gap-2"
                    >
                      <MessageSquare className="w-4 h-4" />
                      <span className="hidden sm:inline">Öğrenci </span>Sorular
                      <span className="ml-1.5 px-1.5 py-0.5 text-xs bg-muted rounded-full">
                        {interactionsTotal || interactions.length}
                      </span>
                    </TabsTrigger>
                  </>
                )}
              </TabsList>
            </div>

            <TabsContent value="genel" className="mt-0">
              <div className="p-6 space-y-8">
                {/* Header */}
                <div className="text-center">
                  <div className="flex items-center justify-center gap-3 mb-4">
                    <div className="p-3 rounded-full bg-gradient-to-br from-blue-500 to-purple-600">
                      <Settings className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <h2 className="text-2xl font-bold text-foreground mb-2">
                    Genel Ayarlar
                  </h2>
                  <p className="text-muted-foreground max-w-2xl mx-auto">
                    Ders oturumunuz için döküman yönetimi ve RAG ayarlarını bu
                    bölümden yapılandırabilirsiniz
                  </p>
                </div>

                {/* Document Management Section */}
                <Card className="border-2 border-dashed border-primary/20 bg-gradient-to-r from-blue-50/30 to-indigo-50/30 dark:from-blue-950/10 dark:to-indigo-950/10">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-primary">
                      <Upload className="w-5 h-5" />
                      Döküman Yönetimi
                    </CardTitle>
                    <CardDescription>
                      Markdown dosyalarınızı yükleyin ve işleyin
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-col sm:flex-row gap-4 items-center">
                      <Button
                        onClick={() => setShowModal(true)}
                        disabled={processing}
                        size="lg"
                        className="gap-2 min-w-[200px]"
                      >
                        <Upload className="w-4 h-4" />
                        Markdown Yükle
                      </Button>

                      {processing && (
                        <div className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400">
                          <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent"></div>
                          <span>İşleme devam ediyor...</span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* RAG Settings Section */}
                <Card className="border-2 border-dashed border-purple-200/50 bg-gradient-to-r from-purple-50/30 to-pink-50/30 dark:from-purple-950/10 dark:to-pink-950/10">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-purple-700 dark:text-purple-300">
                      <Brain className="w-5 h-5" />
                      RAG Ayarları
                    </CardTitle>
                    <CardDescription>
                      Yapay zeka ve embedding modellerini yapılandırın
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* AI Provider */}
                      <div className="space-y-3">
                        <label className="text-sm font-medium text-foreground flex items-center gap-2">
                          <Zap className="w-4 h-4 text-primary" />
                          AI Provider
                        </label>
                        <select
                          value={selectedProvider}
                          onChange={(e) => {
                            setSelectedProvider(e.target.value);
                            setSelectedQueryModel("");
                          }}
                          className="w-full px-4 py-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all bg-background shadow-sm"
                          disabled={modelsLoading}
                        >
                          <option value="groq">🌐 Groq (Cloud - Hızlı)</option>
                          <option value="alibaba">
                            🛒 Alibaba (Cloud - Qwen)
                          </option>
                          <option value="deepseek">
                            🔮 DeepSeek (Cloud - Premium)
                          </option>
                          <option value="openrouter">
                            🚀 OpenRouter (Cloud - Güçlü)
                          </option>
                          <option value="huggingface">
                            🤗 HuggingFace (Ücretsiz)
                          </option>
                          <option value="ollama">🏠 Ollama (Yerel)</option>
                        </select>
                      </div>

                      {/* AI Model */}
                      <div className="space-y-3">
                        <label className="text-sm font-medium text-foreground flex items-center gap-2">
                          <Brain className="w-4 h-4 text-primary" />
                          AI Modeli
                        </label>
                        <select
                          value={selectedQueryModel}
                          onChange={(e) =>
                            setSelectedQueryModel(e.target.value)
                          }
                          className="w-full px-4 py-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all bg-background shadow-sm"
                          disabled={
                            modelsLoading || availableModels.length === 0
                          }
                        >
                          <option value="">
                            {modelsLoading
                              ? "Modeller yükleniyor..."
                              : availableModels.length === 0
                              ? "Bu provider için model bulunamadı"
                              : "Model seçin..."}
                          </option>
                          {availableModels.map((model: any) => (
                            <option
                              key={model.id || model}
                              value={model.id || model}
                            >
                              {typeof model === "string"
                                ? model
                                : model.name || model.id}
                            </option>
                          ))}
                        </select>
                        {availableModels.length === 0 && !modelsLoading && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={async () => {
                              setModelsLoading(true);
                              try {
                                const response = await listAvailableModels();
                                const models = Array.isArray(response)
                                  ? response
                                  : response.models || [];
                                setAvailableModels(models);
                                setModelProviders(
                                  models.reduce((acc: any, m: any) => {
                                    const provider = m.provider || "unknown";
                                    if (!acc[provider]) acc[provider] = [];
                                    acc[provider].push(m);
                                    return acc;
                                  }, {})
                                );
                              } catch (e: any) {
                                setError(e.message || "Modeller yüklenemedi");
                              } finally {
                                setModelsLoading(false);
                              }
                            }}
                            className="mt-2"
                          >
                            <RefreshCw className="w-3 h-3 mr-1" />
                            Modelleri Yükle
                          </Button>
                        )}
                      </div>

                      {/* Embedding Provider */}
                      <div className="space-y-3">
                        <label className="text-sm font-medium text-foreground flex items-center gap-2">
                          <Layers className="w-4 h-4 text-primary" />
                          Embedding Provider
                        </label>
                        <select
                          value={selectedEmbeddingProvider}
                          onChange={(e) => {
                            setSelectedEmbeddingProvider(e.target.value);
                            const providerModels =
                              e.target.value === "ollama"
                                ? availableEmbeddingModels.ollama
                                : e.target.value === "alibaba"
                                ? (availableEmbeddingModels.alibaba || []).map(
                                    (m) => m.id
                                  )
                                : availableEmbeddingModels.huggingface.map(
                                    (m) => m.id
                                  );
                            const currentModelInNewProvider =
                              providerModels.includes(selectedEmbeddingModel);
                            if (
                              providerModels.length > 0 &&
                              !currentModelInNewProvider
                            ) {
                              setSelectedEmbeddingModel(providerModels[0]);
                            }
                          }}
                          className="w-full px-4 py-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all bg-background shadow-sm"
                        >
                          <option value="ollama">🏠 Ollama (Yerel)</option>
                          <option value="huggingface">
                            🤗 HuggingFace (Ücretsiz)
                          </option>
                          <option value="alibaba">
                            🛒 Alibaba (Cloud - Qwen)
                          </option>
                        </select>
                      </div>

                      {/* Embedding Model */}
                      <div className="space-y-3">
                        <label className="text-sm font-medium text-foreground flex items-center gap-2">
                          <Database className="w-4 h-4 text-primary" />
                          Embedding Modeli
                        </label>
                        <select
                          value={selectedEmbeddingModel}
                          onChange={(e) =>
                            setSelectedEmbeddingModel(e.target.value)
                          }
                          className="w-full px-4 py-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all bg-background shadow-sm"
                          disabled={embeddingModelsLoading}
                        >
                          {embeddingModelsLoading ? (
                            <option value="">Modeller yükleniyor...</option>
                          ) : selectedEmbeddingProvider === "ollama" ? (
                            availableEmbeddingModels.ollama.length > 0 ? (
                              <>
                                {selectedEmbeddingModel &&
                                  !availableEmbeddingModels.ollama.includes(
                                    selectedEmbeddingModel
                                  ) && (
                                    <option
                                      key={selectedEmbeddingModel}
                                      value={selectedEmbeddingModel}
                                    >
                                      {selectedEmbeddingModel.replace(
                                        ":latest",
                                        ""
                                      )}{" "}
                                      (Kayıtlı)
                                    </option>
                                  )}
                                {availableEmbeddingModels.ollama.map(
                                  (model) => (
                                    <option key={model} value={model}>
                                      {model.replace(":latest", "")}
                                    </option>
                                  )
                                )}
                              </>
                            ) : (
                              <>
                                {selectedEmbeddingModel && (
                                  <option
                                    key={selectedEmbeddingModel}
                                    value={selectedEmbeddingModel}
                                  >
                                    {selectedEmbeddingModel.replace(
                                      ":latest",
                                      ""
                                    )}{" "}
                                    (Kayıtlı)
                                  </option>
                                )}
                                <option value="">
                                  Ollama model bulunamadı
                                </option>
                              </>
                            )
                          ) : selectedEmbeddingProvider === "alibaba" ? (
                            (availableEmbeddingModels.alibaba || []).length >
                            0 ? (
                              <>
                                {selectedEmbeddingModel &&
                                  !(
                                    availableEmbeddingModels.alibaba || []
                                  ).some(
                                    (m) => m.id === selectedEmbeddingModel
                                  ) && (
                                    <option
                                      key={selectedEmbeddingModel}
                                      value={selectedEmbeddingModel}
                                    >
                                      {selectedEmbeddingModel} (Kayıtlı)
                                    </option>
                                  )}
                                {(availableEmbeddingModels.alibaba || []).map(
                                  (model) => (
                                    <option key={model.id} value={model.id}>
                                      {model.name}{" "}
                                      {model.description
                                        ? `- ${model.description}`
                                        : ""}{" "}
                                      {model.dimensions
                                        ? `(${model.dimensions}D)`
                                        : ""}
                                    </option>
                                  )
                                )}
                              </>
                            ) : (
                              <>
                                {selectedEmbeddingModel && (
                                  <option
                                    key={selectedEmbeddingModel}
                                    value={selectedEmbeddingModel}
                                  >
                                    {selectedEmbeddingModel} (Kayıtlı)
                                  </option>
                                )}
                                <option value="">
                                  Bu provider için model bulunamadı
                                </option>
                              </>
                            )
                          ) : availableEmbeddingModels.huggingface.length >
                            0 ? (
                            <>
                              {selectedEmbeddingModel &&
                                !availableEmbeddingModels.huggingface.some(
                                  (m) => m.id === selectedEmbeddingModel
                                ) && (
                                  <option
                                    key={selectedEmbeddingModel}
                                    value={selectedEmbeddingModel}
                                  >
                                    {selectedEmbeddingModel} (Kayıtlı)
                                  </option>
                                )}
                              {availableEmbeddingModels.huggingface.map(
                                (model) => (
                                  <option key={model.id} value={model.id}>
                                    {model.name}{" "}
                                    {model.description
                                      ? `- ${model.description}`
                                      : ""}{" "}
                                    {model.dimensions
                                      ? `(${model.dimensions}D)`
                                      : ""}
                                  </option>
                                )
                              )}
                            </>
                          ) : (
                            <>
                              {selectedEmbeddingModel && (
                                <option
                                  key={selectedEmbeddingModel}
                                  value={selectedEmbeddingModel}
                                >
                                  {selectedEmbeddingModel} (Kayıtlı)
                                </option>
                              )}
                              <option value="">
                                HuggingFace model bulunamadı
                              </option>
                            </>
                          )}
                        </select>
                      </div>
                    </div>

                    {/* Reranker Section */}
                    <div className="mt-6 pt-6 border-t border-border">
                      <div className="space-y-4">
                        <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
                          <Sparkles className="w-4 h-4 text-primary" />
                          Reranker Ayarları
                        </h4>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={useRerankerService}
                            onChange={(e) =>
                              setUseRerankerService(e.target.checked)
                            }
                            className="w-4 h-4 text-primary border-border rounded focus:ring-primary"
                          />
                          <span className="text-sm text-foreground">
                            Gelişmiş Reranker Servisi Kullan
                          </span>
                        </div>
                        {useRerankerService && (
                          <select
                            value={selectedRerankerType}
                            onChange={(e) =>
                              setSelectedRerankerType(e.target.value)
                            }
                            className="w-full max-w-md px-4 py-2 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all bg-background shadow-sm text-sm"
                          >
                            <option value="bge-reranker-v2-m3">
                              🔮 BGE-Reranker-V2-M3 (Türkçe Optimize)
                            </option>
                            <option value="ms-marco-minilm-l6">
                              ⚡ MS-MARCO MiniLM-L6 (Hafif, Hızlı)
                            </option>
                            <option value="gte-rerank-v2">
                              🛒 GTE-Rerank-V2 (Alibaba - 50+ Dil)
                            </option>
                          </select>
                        )}
                      </div>
                    </div>

                    {/* Save Button */}
                    <div className="mt-8 flex items-center gap-3">
                      <Button
                        type="button"
                        onClick={async () => {
                          try {
                            setSavingSettings(true);
                            setSavedSettingsInfo(null);
                            const settingsToSave: any = {
                              top_k: 5,
                              use_rerank:
                                session?.rag_settings?.use_rerank ?? false,
                              min_score: 0.5,
                              max_context_chars: 8000,
                              use_reranker_service: useRerankerService,
                            };

                            if (
                              selectedQueryModel &&
                              selectedQueryModel.trim()
                            ) {
                              settingsToSave.model = selectedQueryModel;
                            }
                            if (selectedProvider && selectedProvider.trim()) {
                              settingsToSave.provider = selectedProvider;
                            }
                            if (
                              selectedEmbeddingModel &&
                              selectedEmbeddingModel.trim()
                            ) {
                              settingsToSave.embedding_model =
                                selectedEmbeddingModel;
                            }
                            if (
                              selectedEmbeddingProvider &&
                              selectedEmbeddingProvider.trim()
                            ) {
                              settingsToSave.embedding_provider =
                                selectedEmbeddingProvider;
                            }
                            if (useRerankerService && selectedRerankerType) {
                              settingsToSave.reranker_type =
                                selectedRerankerType as "bge" | "ms-marco";
                            }

                            const resp = await saveSessionRagSettings(
                              sessionId,
                              settingsToSave
                            );
                            setSavedSettingsInfo(
                              "✅ Ayarlar başarıyla kaydedildi"
                            );
                            await fetchSessionDetails();

                            if (resp.rag_settings) {
                              if (resp.rag_settings.provider)
                                setSelectedProvider(resp.rag_settings.provider);
                              if (resp.rag_settings.model)
                                setSelectedQueryModel(resp.rag_settings.model);
                              if (resp.rag_settings.embedding_provider)
                                setSelectedEmbeddingProvider(
                                  resp.rag_settings.embedding_provider
                                );
                              if (resp.rag_settings.embedding_model)
                                setSelectedEmbeddingModel(
                                  resp.rag_settings.embedding_model
                                );
                              if (
                                resp.rag_settings.use_reranker_service !==
                                undefined
                              )
                                setUseRerankerService(
                                  resp.rag_settings.use_reranker_service
                                );
                              if (resp.rag_settings.reranker_type)
                                setSelectedRerankerType(
                                  resp.rag_settings.reranker_type
                                );
                            }
                          } catch (e: any) {
                            setError(e.message || "Ayarlar kaydedilemedi");
                          } finally {
                            setSavingSettings(false);
                          }
                        }}
                        disabled={savingSettings}
                        size="lg"
                        className="gap-2"
                      >
                        {savingSettings ? (
                          <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            Kaydediliyor...
                          </>
                        ) : (
                          <>
                            <Upload className="w-4 h-4" />
                            RAG Ayarlarını Kaydet
                          </>
                        )}
                      </Button>
                      {savedSettingsInfo && (
                        <span className="text-sm text-green-600 dark:text-green-400 flex items-center gap-1">
                          <ChevronRight className="w-4 h-4" />
                          {savedSettingsInfo}
                        </span>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="chunks" className="mt-0">
              <div className="p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-base font-semibold text-foreground">
                      Döküman Parçaları
                    </h2>
                    <p className="text-sm text-muted-foreground mt-0.5">
                      {chunks.length} parça
                      {chunks.filter((c) => c.chunk_metadata?.llm_improved)
                        .length > 0 && (
                        <span className="ml-2 text-violet-600 font-medium">
                          (✨{" "}
                          {
                            chunks.filter((c) => c.chunk_metadata?.llm_improved)
                              .length
                          }{" "}
                          iyileştirildi)
                        </span>
                      )}
                    </p>
                  </div>
                  <button
                    onClick={fetchChunks}
                    disabled={loading}
                    className="py-2 px-3 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {loading ? "Yenileniyor..." : "Yenile"}
                  </button>
                </div>

                {/* Compact Document Filter */}
                {chunks.length > 0 &&
                  (() => {
                    const uniqueDocuments = Array.from(
                      new Set(chunks.map((c) => c.document_name))
                    ).sort();
                    const filteredChunks = selectedDocumentFilter
                      ? chunks.filter(
                          (c) => c.document_name === selectedDocumentFilter
                        )
                      : chunks;

                    // Filter documents by search term
                    const filteredDocuments = uniqueDocuments.filter((doc) =>
                      doc.toLowerCase().includes(searchTerm.toLowerCase())
                    );

                    const documentsToShow = showAllFiles
                      ? filteredDocuments
                      : filteredDocuments.slice(0, 3);

                    return (
                      <div className="bg-muted/30 border border-border rounded-lg p-3 space-y-3">
                        {/* Compact Header */}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <svg
                              className="w-4 h-4 text-blue-600"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                              />
                            </svg>
                            <span className="text-sm font-medium text-foreground">
                              Dosya Filtresi
                            </span>
                            <span className="text-xs text-muted-foreground">
                              (
                              {selectedDocumentFilter
                                ? filteredChunks.length
                                : chunks.length}{" "}
                              parça)
                            </span>
                          </div>
                          {selectedDocumentFilter && (
                            <button
                              onClick={() => setSelectedDocumentFilter(null)}
                              className="text-xs text-blue-600 hover:text-blue-700 font-medium flex items-center gap-1 px-2 py-1 rounded hover:bg-blue-50 transition-colors"
                            >
                              <svg
                                className="w-3 h-3"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M6 18L18 6M6 6l12 12"
                                />
                              </svg>
                              Temizle
                            </button>
                          )}
                        </div>

                        {/* Search Input */}
                        {uniqueDocuments.length > 4 && (
                          <div className="relative">
                            <svg
                              className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                              />
                            </svg>
                            <input
                              type="text"
                              placeholder="Dosya ara..."
                              value={searchTerm}
                              onChange={(e) => setSearchTerm(e.target.value)}
                              className="w-full pl-10 pr-3 py-2 text-sm border border-border rounded-lg bg-background focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                            />
                          </div>
                        )}

                        {/* File Selection - Horizontal Layout */}
                        <div className="space-y-2">
                          {/* All Files Chip */}
                          <div className="flex flex-wrap gap-2">
                            <button
                              onClick={() => setSelectedDocumentFilter(null)}
                              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                                selectedDocumentFilter === null
                                  ? "bg-blue-600 text-white shadow-md hover:bg-blue-700"
                                  : "bg-muted text-muted-foreground hover:bg-muted/80 border border-border"
                              }`}
                            >
                              <svg
                                className="w-3 h-3"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                                />
                              </svg>
                              <span>Tümü</span>
                              <span className="text-xs opacity-75">
                                ({chunks.length})
                              </span>
                            </button>

                            {/* Individual File Chips */}
                            {documentsToShow.map((docName) => {
                              const docChunkCount = chunks.filter(
                                (c) => c.document_name === docName
                              ).length;
                              const isActive =
                                selectedDocumentFilter === docName;
                              const displayName = docName
                                .replace(".md", "")
                                .replace(/^.*[\\\/]/, "");

                              return (
                                <button
                                  key={docName}
                                  onClick={() =>
                                    setSelectedDocumentFilter(docName)
                                  }
                                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all max-w-[200px] ${
                                    isActive
                                      ? "bg-blue-600 text-white shadow-md hover:bg-blue-700"
                                      : "bg-muted text-muted-foreground hover:bg-muted/80 border border-border hover:border-border/80"
                                  }`}
                                  title={docName}
                                >
                                  <svg
                                    className="w-3 h-3 flex-shrink-0"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth={2}
                                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                    />
                                  </svg>
                                  <span className="truncate">
                                    {displayName}
                                  </span>
                                  <span className="text-xs opacity-75 flex-shrink-0">
                                    ({docChunkCount})
                                  </span>
                                </button>
                              );
                            })}

                            {/* Show More Button */}
                            {filteredDocuments.length > 3 && (
                              <button
                                onClick={() => setShowAllFiles(!showAllFiles)}
                                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-medium text-primary hover:text-primary/80 bg-primary/10 hover:bg-primary/20 transition-all"
                              >
                                {showAllFiles ? (
                                  <>
                                    <svg
                                      className="w-3 h-3"
                                      fill="none"
                                      stroke="currentColor"
                                      viewBox="0 0 24 24"
                                    >
                                      <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M5 15l7-7 7 7"
                                      />
                                    </svg>
                                    <span>Daha Az</span>
                                  </>
                                ) : (
                                  <>
                                    <span>
                                      +{filteredDocuments.length - 3} Daha
                                    </span>
                                    <svg
                                      className="w-3 h-3"
                                      fill="none"
                                      stroke="currentColor"
                                      viewBox="0 0 24 24"
                                    >
                                      <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M19 9l-7 7-7-7"
                                      />
                                    </svg>
                                  </>
                                )}
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })()}

                {loading ? (
                  <div className="text-center py-16">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent mb-3"></div>
                    <p className="text-sm text-muted-foreground">
                      Yükleniyor...
                    </p>
                  </div>
                ) : chunks.length === 0 ? (
                  <div className="text-center py-16">
                    <p className="text-sm text-muted-foreground mb-1">
                      Henüz parça bulunmuyor
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Markdown yükleyerek başlayın
                    </p>
                  </div>
                ) : (
                  (() => {
                    // Apply document filter
                    const filteredChunks = selectedDocumentFilter
                      ? chunks.filter(
                          (c) => c.document_name === selectedDocumentFilter
                        )
                      : chunks;

                    return (
                      <>
                        {/* Table - Mobile Card View for Small Screens */}
                        <div className="block sm:hidden space-y-3 p-3">
                          {filteredChunks
                            .slice(
                              (chunkPage - 1) * CHUNKS_PER_PAGE,
                              chunkPage * CHUNKS_PER_PAGE
                            )
                            .map((chunk, idx) => {
                              return (
                                <div
                                  key={
                                    chunk.chunk_metadata?.chunk_id ||
                                    `${chunk.document_name}-${chunk.chunk_index}-${idx}`
                                  }
                                  className="rounded-lg p-3 space-y-2 bg-muted/30"
                                >
                                  <div className="flex justify-between items-center">
                                    <div className="flex items-center gap-2">
                                      <span className="text-sm font-medium">
                                        #{chunk.chunk_index}
                                      </span>
                                    </div>
                                    <span className="text-xs text-muted-foreground">
                                      {chunk.chunk_text.length} karakter
                                    </span>
                                  </div>
                                  <div className="text-sm text-foreground font-medium truncate">
                                    {chunk.document_name}
                                  </div>
                                  <button
                                    onClick={() => {
                                      setSelectedChunkForModal(chunk);
                                      setShowChunkModal(true);
                                    }}
                                    className="w-full text-sm text-primary cursor-pointer hover:underline text-left py-2"
                                  >
                                    İçeriği Göster
                                  </button>
                                </div>
                              );
                            })}
                        </div>

                        {/* Table - Desktop View */}
                        <div className="hidden sm:block overflow-x-auto">
                          <table className="w-full">
                            <thead className="bg-muted/50">
                              <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                                  #
                                </th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                                  Döküman
                                </th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                                  Karakter
                                </th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                                  İçerik
                                </th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-border">
                              {filteredChunks
                                .slice(
                                  (chunkPage - 1) * CHUNKS_PER_PAGE,
                                  chunkPage * CHUNKS_PER_PAGE
                                )
                                .map((chunk, idx) => {
                                  return (
                                    <tr
                                      key={
                                        chunk.chunk_metadata?.chunk_id ||
                                        `${chunk.document_name}-${chunk.chunk_index}-${idx}`
                                      }
                                      className="hover:bg-muted/30 transition-colors"
                                    >
                                      <td className="px-4 py-3 text-sm text-foreground font-medium">
                                        {chunk.chunk_index}
                                      </td>
                                      <td className="px-4 py-3 text-sm text-foreground">
                                        {chunk.document_name}
                                      </td>
                                      <td className="px-4 py-3 text-sm text-muted-foreground">
                                        {chunk.chunk_text.length}
                                      </td>
                                      <td className="px-4 py-3">
                                        <button
                                          onClick={() => {
                                            setSelectedChunkForModal(chunk);
                                            setShowChunkModal(true);
                                          }}
                                          className="text-sm text-primary cursor-pointer hover:underline"
                                        >
                                          Göster
                                        </button>
                                      </td>
                                    </tr>
                                  );
                                })}
                            </tbody>
                          </table>
                        </div>

                        {/* Pagination */}
                        {filteredChunks.length > CHUNKS_PER_PAGE && (
                          <div className="flex items-center justify-between px-3 sm:px-5 py-3 border-t border-border">
                            <button
                              onClick={() =>
                                setChunkPage((p) => Math.max(1, p - 1))
                              }
                              disabled={chunkPage === 1}
                              className="py-3 px-4 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors min-h-[44px]"
                            >
                              Önceki
                            </button>
                            <span className="text-sm text-muted-foreground">
                              <span className="hidden sm:inline">Sayfa </span>
                              {chunkPage} /{" "}
                              {Math.ceil(
                                filteredChunks.length / CHUNKS_PER_PAGE
                              )}
                            </span>
                            <button
                              onClick={() =>
                                setChunkPage((p) =>
                                  Math.min(
                                    Math.ceil(
                                      filteredChunks.length / CHUNKS_PER_PAGE
                                    ),
                                    p + 1
                                  )
                                )
                              }
                              disabled={
                                chunkPage >=
                                Math.ceil(
                                  filteredChunks.length / CHUNKS_PER_PAGE
                                )
                              }
                              className="py-3 px-4 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors min-h-[44px]"
                            >
                              Sonraki
                            </button>
                          </div>
                        )}
                      </>
                    );
                  })()
                )}
              </div>
            </TabsContent>

            {apragEnabled && (
              <>
                <TabsContent value="topics" className="mt-0">
                  <div className="p-6">
                    <TopicManagementPanel
                      sessionId={sessionId}
                      apragEnabled={apragEnabled}
                    />
                  </div>
                </TabsContent>

                <TabsContent value="interactions" className="mt-0">
                  <div className="p-6">
                    {interactionsLoading ? (
                      <div className="text-center py-12">
                        <div className="inline-block animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent mb-3"></div>
                        <p className="text-sm text-muted-foreground">
                          Yükleniyor...
                        </p>
                      </div>
                    ) : interactions.length === 0 ? (
                      <div className="text-center py-12">
                        <p className="text-sm text-muted-foreground">
                          Henüz soru sorulmamış
                        </p>
                      </div>
                    ) : (
                      <>
                        <div className="space-y-4">
                          {interactions.map((interaction, index) => {
                            // Get student name: prefer first_name + last_name, fallback to student_name, then "Bilinmeyen Öğrenci"
                            const studentName =
                              interaction.first_name && interaction.last_name
                                ? `${interaction.first_name} ${interaction.last_name}`
                                : interaction.first_name ||
                                  interaction.last_name
                                ? interaction.first_name ||
                                  interaction.last_name
                                : interaction.student_name &&
                                  !interaction.student_name.startsWith(
                                    "Öğrenci (ID:"
                                  )
                                ? interaction.student_name
                                : "Bilinmeyen Öğrenci";

                            return (
                              <Card key={interaction.interaction_id}>
                                <CardContent className="p-4">
                                  <div className="flex items-start gap-3 mb-3">
                                    <div className="flex-shrink-0 w-7 h-7 bg-primary/10 text-primary rounded flex items-center justify-center font-medium text-sm">
                                      {(interactionsPage - 1) *
                                        INTERACTIONS_PER_PAGE +
                                        index +
                                        1}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <div className="flex flex-col gap-2 mb-2">
                                        <div className="flex items-center gap-2 flex-wrap">
                                          <span className="text-sm font-medium text-foreground">
                                            👤 {studentName}
                                          </span>
                                          {interaction.topic_info && (
                                            <span className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded-full">
                                              📚 {interaction.topic_info.title}
                                              {interaction.topic_info
                                                .confidence && (
                                                <span className="ml-1 opacity-70">
                                                  (
                                                  {Math.round(
                                                    interaction.topic_info
                                                      .confidence * 100
                                                  )}
                                                  %)
                                                </span>
                                              )}
                                            </span>
                                          )}
                                        </div>
                                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                          <span>
                                            {new Date(
                                              interaction.timestamp
                                            ).toLocaleString("tr-TR")}
                                          </span>
                                          {interaction.processing_time_ms && (
                                            <span>
                                              • {interaction.processing_time_ms}
                                              ms
                                            </span>
                                          )}
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                  <div className="space-y-3">
                                    <div>
                                      <p className="text-sm font-bold text-blue-600 mb-1">
                                        Soru
                                      </p>
                                      <p className="text-sm text-foreground">
                                        {interaction.query}
                                      </p>
                                    </div>
                                    <div>
                                      <p className="text-sm font-bold text-green-600 mb-1">
                                        Cevap
                                      </p>
                                      <p className="text-sm text-foreground whitespace-pre-wrap">
                                        {interaction.personalized_response ||
                                          interaction.original_response}
                                      </p>
                                    </div>
                                    {interaction.sources &&
                                      interaction.sources.length > 0 && (
                                        <div>
                                          <p className="text-sm font-bold text-purple-600 mb-1.5">
                                            Kaynaklar
                                          </p>
                                          <div className="flex flex-wrap gap-1.5">
                                            {interaction.sources.map(
                                              (source, idx) => (
                                                <span
                                                  key={idx}
                                                  className="text-xs bg-muted text-muted-foreground px-2 py-1 rounded"
                                                >
                                                  {source.source}
                                                  {source.score !== undefined &&
                                                    ` (${(
                                                      source.score * 100
                                                    ).toFixed(1)}%)`}
                                                </span>
                                              )
                                            )}
                                          </div>
                                        </div>
                                      )}
                                  </div>
                                </CardContent>
                              </Card>
                            );
                          })}
                        </div>

                        {/* Pagination */}
                        {interactionsTotal > INTERACTIONS_PER_PAGE && (
                          <div className="flex items-center justify-between px-3 sm:px-5 py-3 border-t border-border mt-4">
                            <button
                              onClick={() => {
                                const newPage = Math.max(
                                  1,
                                  interactionsPage - 1
                                );
                                setInteractionsPage(newPage);
                              }}
                              disabled={
                                interactionsPage === 1 || interactionsLoading
                              }
                              className="py-3 px-4 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors min-h-[44px]"
                            >
                              Önceki
                            </button>
                            <span className="text-sm text-muted-foreground">
                              <span className="hidden sm:inline">Sayfa </span>
                              {interactionsPage} /{" "}
                              {Math.ceil(
                                interactionsTotal / INTERACTIONS_PER_PAGE
                              )}
                              <span className="hidden sm:inline">
                                {" "}
                                (Toplam {interactionsTotal} soru)
                              </span>
                            </span>
                            <button
                              onClick={() => {
                                const newPage = Math.min(
                                  Math.ceil(
                                    interactionsTotal / INTERACTIONS_PER_PAGE
                                  ),
                                  interactionsPage + 1
                                );
                                setInteractionsPage(newPage);
                              }}
                              disabled={
                                interactionsPage >=
                                  Math.ceil(
                                    interactionsTotal / INTERACTIONS_PER_PAGE
                                  ) || interactionsLoading
                              }
                              className="py-3 px-4 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors min-h-[44px]"
                            >
                              Sonraki
                            </button>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </TabsContent>
              </>
            )}

            <TabsContent value="session-settings" className="mt-0">
              <div className="p-6">
                <SessionSettingsPanel sessionId={sessionId} />
              </div>
            </TabsContent>
          </Tabs>
        </Card>

        {/* File Upload Modal */}
        <FileUploadModal
          isOpen={showModal}
          onClose={() => setShowModal(false)}
          sessionId={sessionId}
          onSuccess={handleModalSuccess}
          onError={handleModalError}
          isProcessing={processing}
          setIsProcessing={setProcessing}
          defaultEmbeddingModel={
            session?.rag_settings?.embedding_model || undefined
          }
        />

        {/* Chunk Content Modal */}
        {showChunkModal && selectedChunkForModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-card border border-border rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
              <div className="flex items-center justify-between p-4 sm:p-6 border-b border-border">
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg font-semibold text-foreground mb-1">
                    Chunk #{selectedChunkForModal.chunk_index}
                  </h3>
                  <p className="text-sm text-muted-foreground truncate">
                    {selectedChunkForModal.document_name}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {selectedChunkForModal.chunk_text.length} karakter
                  </p>
                </div>
                <button
                  onClick={() => {
                    setShowChunkModal(false);
                    setSelectedChunkForModal(null);
                  }}
                  className="ml-4 p-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded-md transition-colors"
                  aria-label="Kapat"
                >
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-4 sm:p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap text-sm text-foreground font-mono bg-muted/30 rounded-lg p-4 border border-border">
                    {selectedChunkForModal.chunk_text}
                  </pre>
                </div>
              </div>
              <div className="p-4 sm:p-6 border-t border-border">
                <button
                  onClick={() => {
                    setShowChunkModal(false);
                    setSelectedChunkForModal(null);
                  }}
                  className="w-full sm:w-auto px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
                >
                  Kapat
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </TeacherLayout>
  );
}
