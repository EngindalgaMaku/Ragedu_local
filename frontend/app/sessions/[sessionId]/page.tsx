"use client";
import React, { useState, useEffect } from "react";
import {
  getChunksForSession,
  listSessions,
  SessionMeta,
  Chunk,
  reprocessSessionDocuments,
  listAvailableEmbeddingModels,
  saveSessionRagSettings,
  getSessionInteractions,
  APRAGInteraction,
  getApiUrl,
} from "@/lib/api";
import TopicManagementPanel from "@/components/TopicManagementPanel";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import FileUploadModal from "@/components/FileUploadModal";
import DocumentUploadModal from "@/components/DocumentUploadModal";
import TeacherLayout from "@/app/components/TeacherLayout";

// Icons
const BackIcon = () => (
  <svg
    className="w-4 h-4"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M10 19l-7-7m0 0l7-7m-7 7h18"
    />
  </svg>
);

export default function SessionPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;

  // State management
  const [session, setSession] = useState<SessionMeta | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [chunkPage, setChunkPage] = useState(1);
  const [showModal, setShowModal] = useState(false);
  const [showReprocessModal, setShowReprocessModal] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [backgroundReprocessing, setBackgroundReprocessing] = useState(false);
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
  }>({ ollama: [], huggingface: [] });
  const [embeddingModelsLoading, setEmbeddingModelsLoading] = useState(false);
  const [interactions, setInteractions] = useState<APRAGInteraction[]>([]);
  const [interactionsLoading, setInteractionsLoading] = useState(false);
  const [showInteractions, setShowInteractions] = useState(false);
  const [apragEnabled, setApragEnabled] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'chunks' | 'topics' | 'interactions'>('chunks');
  const CHUNKS_PER_PAGE = 10;

  // Fetch session details
  const fetchSessionDetails = async () => {
    try {
      const sessions = await listSessions();
      const currentSession = sessions.find((s) => s.session_id === sessionId);
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
      setChunks(sessionChunks);
    } catch (e: any) {
      setError(e.message || "Parçalar yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  // Handle modal success
  const handleModalSuccess = async (result: any) => {
    // Defensive programming: ensure we have valid values to prevent undefined display
    const processedCount = result?.processed_count ?? 0;
    const totalChunks = result?.total_chunks_added ?? 0;

    setSuccess(
      `RAG işlemi tamamlandı! ${processedCount} dosya işlendi, ${totalChunks} parça oluşturuldu.`
    );
    setProcessing(false);
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
    } finally {
      setEmbeddingModelsLoading(false);
    }
  };

  // Fetch interactions for the session
  const fetchInteractions = async () => {
    try {
      setInteractionsLoading(true);
      const data = await getSessionInteractions(sessionId, 50, 0);
      setInteractions(data.interactions || []);
    } catch (e: any) {
      console.error("Failed to fetch interactions:", e);
      setInteractions([]);
    } finally {
      setInteractionsLoading(false);
    }
  };

  // Check APRAG status
  useEffect(() => {
    const checkApragStatus = async () => {
      try {
        const response = await fetch(
          `${getApiUrl()}/api/aprag/settings/status`
        );
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
    <TeacherLayout activeTab="sessions">
      <div className="space-y-6">
        {/* Minimal Header */}
        <div className="border-b border-border pb-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold text-foreground mb-1">
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
              className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <BackIcon />
              <span className="hidden sm:inline">Geri</span>
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

        {/* RAG Configuration Section - Minimal */}
        <div className="bg-card border border-border rounded-lg p-5">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-4">
            <div>
              <h2 className="text-base font-semibold text-foreground mb-1">
                Döküman Yönetimi
              </h2>
              <p className="text-sm text-muted-foreground">
                Markdown yükleyin veya mevcut dökümanları yeniden işleyin
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setShowModal(true)}
                disabled={processing || reprocessing}
                className="py-2 px-4 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Markdown Yükle
              </button>
              <button
                onClick={() => setShowReprocessModal(true)}
                disabled={processing || reprocessing || chunks.length === 0}
                className="py-2 px-4 bg-secondary text-secondary-foreground rounded-md text-sm font-medium hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                title={
                  chunks.length === 0
                    ? "Yeniden işlemek için önce döküman yüklemelisiniz"
                    : "Mevcut dökümanları yeni embedding modeli ile yeniden işle"
                }
              >
                Yeniden İşle
              </button>
            </div>
          </div>

          {/* Processing Status */}
          {processing && (
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-3">
              <div className="flex items-center gap-2 text-sm text-blue-800 dark:text-blue-200">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent"></div>
                <span>Markdown işlemi devam ediyor...</span>
              </div>
            </div>
          )}

          {/* Background Reprocessing Status */}
          {backgroundReprocessing && (
            <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-md p-3">
              <div className="flex items-center gap-2 text-sm text-purple-800 dark:text-purple-200">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-purple-600 border-t-transparent"></div>
                <span>Dökümanlar yeniden işleniyor...</span>
              </div>
            </div>
          )}
        </div>

        {/* Tab Navigation */}
        <div className="bg-card border border-border rounded-lg">
          <div className="border-b border-border">
            <div className="flex gap-1 p-2">
              <button
                onClick={() => setActiveTab('chunks')}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                  activeTab === 'chunks'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                }`}
              >
                Döküman Parçaları ({chunks.length})
              </button>
              {apragEnabled && (
                <>
                  <button
                    onClick={() => setActiveTab('topics')}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                      activeTab === 'topics'
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                    }`}
                  >
                    Konu Yönetimi
                  </button>
                  <button
                    onClick={() => {
                      setActiveTab('interactions');
                      if (interactions.length === 0) {
                        fetchInteractions();
                      }
                    }}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                      activeTab === 'interactions'
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                    }`}
                  >
                    Öğrenci Soruları ({interactions.length})
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Chunks Tab Content */}
          {activeTab === 'chunks' && (
            <>
              <div className="flex items-center justify-between p-5 border-b border-border">
                <div>
                  <h2 className="text-base font-semibold text-foreground">
                    Döküman Parçaları
                  </h2>
                  <p className="text-sm text-muted-foreground mt-0.5">
                    {chunks.length} parça
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

              {loading ? (
                <div className="text-center py-16">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent mb-3"></div>
                  <p className="text-sm text-muted-foreground">Yükleniyor...</p>
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
                <>
                  {/* Table */}
                  <div className="overflow-x-auto">
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
                        {chunks
                          .slice(
                            (chunkPage - 1) * CHUNKS_PER_PAGE,
                            chunkPage * CHUNKS_PER_PAGE
                          )
                          .map((chunk) => (
                            <tr
                              key={`${chunk.document_name}-${chunk.chunk_index}`}
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
                                <details className="group">
                                  <summary className="text-sm text-primary cursor-pointer hover:underline">
                                    Göster
                                  </summary>
                                  <div className="mt-2 text-xs text-muted-foreground bg-muted/50 rounded p-3 max-h-48 overflow-y-auto">
                                    <p className="whitespace-pre-wrap leading-relaxed">
                                      {chunk.chunk_text}
                                    </p>
                                  </div>
                                </details>
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Pagination */}
                  {chunks.length > CHUNKS_PER_PAGE && (
                    <div className="flex items-center justify-between px-5 py-3 border-t border-border">
                      <button
                        onClick={() => setChunkPage((p) => Math.max(1, p - 1))}
                        disabled={chunkPage === 1}
                        className="py-1.5 px-3 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        Önceki
                      </button>
                      <span className="text-sm text-muted-foreground">
                        Sayfa {chunkPage} /{" "}
                        {Math.ceil(chunks.length / CHUNKS_PER_PAGE)}
                      </span>
                      <button
                        onClick={() =>
                          setChunkPage((p) =>
                            Math.min(
                              Math.ceil(chunks.length / CHUNKS_PER_PAGE),
                              p + 1
                            )
                          )
                        }
                        disabled={
                          chunkPage >= Math.ceil(chunks.length / CHUNKS_PER_PAGE)
                        }
                        className="py-1.5 px-3 text-sm bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        Sonraki
                      </button>
                    </div>
                  )}
                </>
              )}
            </>
          )}

          {/* Topics Tab Content */}
          {activeTab === 'topics' && apragEnabled && (
            <div className="p-5">
              <TopicManagementPanel
                sessionId={sessionId}
                apragEnabled={apragEnabled}
              />
            </div>
          )}

          {/* Interactions Tab Content */}
          {activeTab === 'interactions' && apragEnabled && (
            <div className="p-5">
              {interactionsLoading ? (
                <div className="text-center py-12">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent mb-3"></div>
                  <p className="text-sm text-muted-foreground">Yükleniyor...</p>
                </div>
              ) : interactions.length === 0 ? (
                <div className="text-center py-12">
                  <p className="text-sm text-muted-foreground">
                    Henüz soru sorulmamış
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {interactions.map((interaction, index) => (
                    <div
                      key={interaction.interaction_id}
                      className="border border-border rounded-lg p-4"
                    >
                      <div className="flex items-start gap-3 mb-3">
                        <div className="flex-shrink-0 w-7 h-7 bg-primary/10 text-primary rounded flex items-center justify-center font-medium text-sm">
                          {index + 1}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-2 text-xs text-muted-foreground">
                            <span>
                              {new Date(interaction.timestamp).toLocaleString(
                                "tr-TR"
                              )}
                            </span>
                            {interaction.processing_time_ms && (
                              <span>• {interaction.processing_time_ms}ms</span>
                            )}
                            {interaction.chain_type && (
                              <span className="bg-primary/10 text-primary px-1.5 py-0.5 rounded">
                                {interaction.chain_type}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="space-y-3">
                        <div>
                          <p className="text-xs font-medium text-muted-foreground mb-1">
                            Soru
                          </p>
                          <p className="text-sm text-foreground">
                            {interaction.query}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs font-medium text-muted-foreground mb-1">
                            Cevap
                          </p>
                          <p className="text-sm text-foreground whitespace-pre-wrap">
                            {interaction.personalized_response ||
                              interaction.original_response}
                          </p>
                        </div>
                        {interaction.sources && interaction.sources.length > 0 && (
                          <div>
                            <p className="text-xs font-medium text-muted-foreground mb-1.5">
                              Kaynaklar
                            </p>
                            <div className="flex flex-wrap gap-1.5">
                              {interaction.sources.map((source, idx) => (
                                <span
                                  key={idx}
                                  className="text-xs bg-muted text-muted-foreground px-2 py-1 rounded"
                                >
                                  {source.source}
                                  {source.score !== undefined &&
                                    ` (${(source.score * 100).toFixed(1)}%)`}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

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

        {/* Reprocess Modal */}
        {showReprocessModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-card border border-border rounded-lg shadow-xl max-w-md w-full p-6">
              <h3 className="text-lg font-semibold text-foreground mb-3">
                Dökümanları Yeniden İşle
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                Mevcut dökümanları yeni bir embedding modeli ile yeniden
                işleyeceksiniz.
              </p>

              <div className="mb-4">
                <label className="block text-sm font-medium text-foreground mb-2">
                  Embedding Modeli
                </label>
                {embeddingModelsLoading ? (
                  <div className="w-full px-3 py-2 border border-border rounded-md bg-muted text-muted-foreground text-sm">
                    Modeller yükleniyor...
                  </div>
                ) : (
                  <select
                    value={selectedEmbeddingModel}
                    onChange={(e) => setSelectedEmbeddingModel(e.target.value)}
                    className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground text-sm focus:ring-2 focus:ring-primary focus:border-primary"
                  >
                    {/* Ollama Models */}
                    {availableEmbeddingModels.ollama.length > 0 && (
                      <>
                        {availableEmbeddingModels.ollama.map((model) => (
                          <option key={model} value={model}>
                            {model} (Ollama)
                          </option>
                        ))}
                      </>
                    )}
                    {/* HuggingFace Models */}
                    {availableEmbeddingModels.huggingface.length > 0 && (
                      <>
                        {availableEmbeddingModels.huggingface.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.name}{" "}
                            {model.description ? `- ${model.description}` : ""}{" "}
                            (HuggingFace)
                          </option>
                        ))}
                      </>
                    )}
                    {/* Fallback if no models loaded */}
                    {availableEmbeddingModels.ollama.length === 0 &&
                      availableEmbeddingModels.huggingface.length === 0 && (
                        <>
                          <option value="nomic-embed-text">
                            nomic-embed-text (Ollama)
                          </option>
                          <option value="sentence-transformers/all-MiniLM-L6-v2">
                            all-MiniLM-L6-v2 (HuggingFace)
                          </option>
                          <option value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2">
                            paraphrase-multilingual-MiniLM-L12-v2 (HuggingFace)
                          </option>
                          <option value="intfloat/multilingual-e5-base">
                            multilingual-e5-base (HuggingFace)
                          </option>
                        </>
                      )}
                  </select>
                )}
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => {
                    // Start background processing - don't await
                    setBackgroundReprocessing(true);
                    setError(null);
                    setSuccess(null);

                    // Close modal immediately
                    setShowReprocessModal(false);

                    // Show info message
                    setSuccess(
                      "İşlem arka planda başlatıldı. Tamamlandığında bildirim alacaksınız."
                    );

                    // Start reprocessing in background (fire and forget)
                    (async () => {
                      try {
                        const result = await reprocessSessionDocuments(
                          sessionId,
                          selectedEmbeddingModel
                        );

                        if (result.success) {
                          // IMPORTANT: Save embedding model to session rag_settings
                          // This ensures chat queries use the same embedding model
                          try {
                            await saveSessionRagSettings(sessionId, {
                              embedding_model: selectedEmbeddingModel,
                            });
                            console.log(
                              `Saved embedding model ${selectedEmbeddingModel} to session rag_settings`
                            );
                          } catch (e: any) {
                            console.error(
                              "Failed to save embedding model to session settings:",
                              e
                            );
                            // Don't fail the whole operation if settings save fails
                          }

                          // Show success notification
                          setSuccess(
                            `Yeniden işleme tamamlandı! ${result.chunks_processed} parça oluşturuldu.`
                          );

                          // Refresh data
                          await fetchChunks();
                          await fetchSessionDetails();
                        } else {
                          setError(
                            result.message || "Yeniden işleme başarısız"
                          );
                        }
                      } catch (e: any) {
                        setError(e.message || "Yeniden işleme başarısız");
                      } finally {
                        setBackgroundReprocessing(false);
                      }
                    })();
                  }}
                  disabled={backgroundReprocessing}
                  className="flex-1 py-2 px-4 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Yeniden İşle
                </button>
                <button
                  onClick={() => {
                    setShowReprocessModal(false);
                    setError(null);
                  }}
                  disabled={backgroundReprocessing}
                  className="flex-1 py-2 px-4 bg-secondary text-secondary-foreground rounded-md text-sm font-medium hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  İptal
                </button>
              </div>
            </div>
          </div>
        )}

      </div>
    </TeacherLayout>
  );
}
