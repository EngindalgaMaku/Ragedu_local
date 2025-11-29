"use client";
import React, { useState, useEffect, FormEvent } from "react";
import {
  configureAndProcess,
  listMarkdownFiles,
  getSession,
  getChunksForSession,
} from "@/lib/api";

interface FileUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  sessionId: string;
  onSuccess: (result: any) => void;
  onError: (error: string) => void;
  isProcessing: boolean;
  setIsProcessing: (processing: boolean) => void;
  defaultEmbeddingModel?: string;
}

const CloseIcon = () => (
  <svg
    className="w-6 h-6"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M6 18L18 6M6 6l12 12"
    />
  </svg>
);

const UploadIcon = ({ className = "w-8 h-8" }: { className?: string }) => (
  <svg
    className={className}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
    />
  </svg>
);

const ProcessingIcon = () => (
  <svg
    className="w-8 h-8 animate-spin"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
    />
  </svg>
);

export default function FileUploadModal({
  isOpen,
  onClose,
  sessionId,
  onSuccess,
  onError,
  isProcessing,
  setIsProcessing,
  defaultEmbeddingModel,
}: FileUploadModalProps) {
  const [markdownFiles, setMarkdownFiles] = useState<string[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [processedFiles, setProcessedFiles] = useState<Set<string>>(new Set());

  // Configuration states - Fixed values (read-only in UI)
  const chunkStrategy = "lightweight"; // Fixed: lightweight chunking
  const chunkSize = 800; // Fixed: display only
  const chunkOverlap = 100; // Fixed: display only
  const [embeddingModel, setEmbeddingModel] = useState(
    defaultEmbeddingModel || "nomic-embed-text"
  ); // Use session's embedding model

  // NEW: LLM Post-Processing option
  const [useLLMPostProcessing, setUseLLMPostProcessing] = useState(false);
  const [llmProvider, setLlmProvider] = useState<"ollama" | "grok">("grok"); // Default: Grok

  // Model name based on provider
  const llmModelName =
    llmProvider === "ollama" ? "llama3:8b" : "llama-3.1-8b-instant";

  // Processing status
  const [processingStep, setProcessingStep] = useState("");

  // Update embedding model when defaultEmbeddingModel prop changes (from session settings)
  useEffect(() => {
    if (defaultEmbeddingModel) {
      setEmbeddingModel(defaultEmbeddingModel);
    }
  }, [defaultEmbeddingModel]);

  // Note: chunkStrategies and embeddingModels removed - no longer needed in UI

  // Normalize filename for comparison (remove .md extension, convert to lowercase)
  const normalizeFilename = (filename: string): string => {
    return filename.toLowerCase().replace(/\.md$/i, "");
  };

  // Fetch processed files from session chunks (optimized)
  const fetchProcessedFiles = async () => {
    try {
      const chunks = await getChunksForSession(sessionId);
      console.log("🔍 Fetched chunks:", chunks.length, "chunks");

      // Extract unique document names from chunks (simplified)
      const processed = new Set<string>();
      chunks.forEach((chunk: any) => {
        const docName = chunk.document_name;
        if (docName && docName !== "Unknown" && typeof docName === "string") {
          const normalized = normalizeFilename(docName);
          processed.add(normalized);
        }
      });

      console.log("🔍 Processed files:", Array.from(processed));
      setProcessedFiles(processed);
      return processed;
    } catch (e: any) {
      console.warn("Could not fetch processed files:", e);
      return new Set<string>();
    }
  };

  // Fetch available markdown files and filter out processed ones
  const fetchMarkdownFiles = async () => {
    try {
      setLoading(true);

      // Fetch both markdown files and processed files in parallel
      const [files, processed] = await Promise.all([
        listMarkdownFiles(),
        fetchProcessedFiles(),
      ]);

      // Filter out already processed files (normalize for comparison)
      const availableFiles = files.filter((file: string) => {
        const normalized = normalizeFilename(file);
        return !processed.has(normalized);
      });
      setMarkdownFiles(availableFiles);
    } catch (e: any) {
      onError(e.message || "Markdown dosyaları yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  // Handle file selection toggle
  const handleFileToggle = (filename: string) => {
    setSelectedFiles((prev) =>
      prev.includes(filename)
        ? prev.filter((f) => f !== filename)
        : [...prev, filename]
    );
  };

  // Handle form submission
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (selectedFiles.length === 0) {
      onError("En az bir Markdown dosyası seçmelisiniz");
      return;
    }

    try {
      setIsProcessing(true);
      setProcessingStep("Konfigürasyon hazırlanıyor...");

      const result = await configureAndProcess({
        session_id: sessionId,
        markdown_files: selectedFiles,
        chunk_strategy: chunkStrategy,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        embedding_model: embeddingModel,
        use_llm_post_processing: useLLMPostProcessing,
        llm_model_name: llmModelName,
      });

      if (result.success) {
        setProcessingStep("İşlem tamamlandı!");
        onSuccess(result);
        setSelectedFiles([]);
        // Refresh the file list to exclude newly processed files
        await fetchMarkdownFiles();
        setTimeout(() => {
          onClose();
          setProcessingStep("");
        }, 2000);
      } else {
        onError(result.message || "İşlem başarısız");
        setIsProcessing(false);
        setProcessingStep("");
      }
    } catch (e: any) {
      onError(e.message || "RAG konfigürasyonu başarısız");
      setIsProcessing(false);
      setProcessingStep("");
    }
  };

  // Load files when modal opens or session changes
  useEffect(() => {
    if (isOpen && sessionId) {
      fetchMarkdownFiles();
    }
  }, [isOpen, sessionId]);

  // When modal opens or session default changes, preselect embedding from session settings
  useEffect(() => {
    if (isOpen && defaultEmbeddingModel) {
      setEmbeddingModel(defaultEmbeddingModel);
    }
  }, [isOpen, defaultEmbeddingModel]);

  // Guarantee sync with backend: fetch session rag_settings on open
  useEffect(() => {
    const load = async () => {
      try {
        if (!isOpen || !sessionId) return;
        const s = await getSession(sessionId);
        const m = s?.rag_settings?.embedding_model;
        if (m) setEmbeddingModel(m);
      } catch (_) {
        // ignore; fallback to current state
      }
    };
    load();
  }, [isOpen, sessionId]);

  // Update processing steps
  useEffect(() => {
    if (isProcessing && processingStep === "Konfigürasyon hazırlanıyor...") {
      const steps = [
        "Dosyalar okunuyor...",
        "Metin parçaları oluşturuluyor...",
        "Embedding vektörleri hesaplanıyor...",
        "Veritabanı güncelleniyor...",
      ];

      let stepIndex = 0;
      const interval = setInterval(() => {
        if (stepIndex < steps.length) {
          setProcessingStep(steps[stepIndex]);
          stepIndex++;
        }
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [isProcessing, processingStep]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="bg-card rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <h2 className="text-xl font-bold text-foreground">
            Belge Yükle & İşle
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-muted rounded-lg transition-colors"
          >
            <CloseIcon />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {isProcessing ? (
            // Processing State
            <div className="text-center py-8">
              <div className="flex flex-col items-center gap-4">
                <ProcessingIcon />
                <div>
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    Markdown İşlemi Devam Ediyor
                  </h3>
                  <p className="text-muted-foreground mb-4">{processingStep}</p>
                  <div className="bg-primary/10 text-primary p-3 rounded-lg text-sm">
                    💡 Modal'ı kapatabilirsiniz - İşlem arka planda devam edecek
                    ve bitince sonuçlar burada görünecektir
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // File Selection Form
            <form
              id="upload-form"
              onSubmit={handleSubmit}
              className="space-y-6"
            >
              {/* File Selection */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-3">
                  <div className="flex items-center gap-2">
                    <UploadIcon className="w-5 h-5" />
                    Markdown Dosyaları Seçin
                  </div>
                </label>

                {loading ? (
                  <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent mx-auto"></div>
                    <p className="text-muted-foreground text-sm mt-2">
                      Dosyalar yükleniyor...
                    </p>
                  </div>
                ) : (
                  <div className="max-h-48 overflow-y-auto border border-border rounded-lg bg-background">
                    {markdownFiles.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground">
                        <div className="text-sm">
                          Markdown dosyası bulunamadı
                        </div>
                      </div>
                    ) : (
                      markdownFiles.map((filename) => (
                        <div
                          key={filename}
                          className="flex items-start p-4 hover:bg-muted/50 border-b border-border last:border-b-0 transition-colors"
                        >
                          <input
                            type="checkbox"
                            checked={selectedFiles.includes(filename)}
                            onChange={() => handleFileToggle(filename)}
                            className="mt-0.5 mr-4 h-5 w-5 text-primary rounded border-border focus:ring-primary focus:ring-2 flex-shrink-0"
                          />
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium text-foreground truncate">
                              {filename.replace(".md", "")}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1 truncate">
                              {filename}
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}

                {selectedFiles.length > 0 && (
                  <div className="mt-3 px-3 py-2 bg-primary/10 rounded-lg">
                    <div className="text-sm text-primary font-medium">
                      {selectedFiles.length} dosya seçili
                    </div>
                  </div>
                )}
              </div>

              {/* LLM Post-Processing Option */}
              <div className="bg-gradient-to-r from-violet-500/10 to-purple-500/10 rounded-lg p-4 border border-violet-500/20">
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    id="use-llm-processing"
                    checked={useLLMPostProcessing}
                    onChange={(e) => setUseLLMPostProcessing(e.target.checked)}
                    className="mt-1 h-5 w-5 text-violet-600 rounded border-violet-300 focus:ring-violet-500 focus:ring-2"
                  />
                  <div className="flex-1">
                    <label
                      htmlFor="use-llm-processing"
                      className="block text-sm font-semibold text-foreground cursor-pointer"
                    >
                      🤖 LLM ile Gelişmiş İşleme (Batch Mode)
                    </label>
                    <p className="text-xs text-muted-foreground mt-1">
                      LLM kullanarak chunk kalitesini artırır.
                      <span className="font-medium text-violet-600">
                        {" "}
                        5 chunks/çağrı
                      </span>{" "}
                      -
                      <span className="font-medium text-amber-600">
                        {" "}
                        ~6-7 dakika
                      </span>{" "}
                      ekstra süre
                    </p>

                    {useLLMPostProcessing && (
                      <div className="mt-3 p-3 bg-violet-500/10 rounded-lg border border-violet-500/20 space-y-3">
                        {/* LLM Provider Selection */}
                        <div className="flex items-center gap-2 pb-2 border-b border-violet-500/20">
                          <span className="text-xs font-medium text-foreground">
                            Provider:
                          </span>
                          <button
                            type="button"
                            onClick={() => setLlmProvider("grok")}
                            className={`px-2 py-1 text-xs rounded transition-all ${
                              llmProvider === "grok"
                                ? "bg-violet-600 text-white"
                                : "bg-muted text-muted-foreground hover:bg-muted/80"
                            }`}
                          >
                            ⚡ Grok API
                          </button>
                          <button
                            type="button"
                            onClick={() => setLlmProvider("ollama")}
                            className={`px-2 py-1 text-xs rounded transition-all ${
                              llmProvider === "ollama"
                                ? "bg-violet-600 text-white"
                                : "bg-muted text-muted-foreground hover:bg-muted/80"
                            }`}
                          >
                            🏠 Ollama
                          </button>
                        </div>

                        <div className="text-xs space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-foreground">
                              Model:
                            </span>
                            <span className="text-muted-foreground">
                              {llmModelName}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-foreground">
                              Provider:
                            </span>
                            <span className="text-muted-foreground">
                              {llmProvider === "grok"
                                ? "Grok API (Hızlı, Internet gerekli)"
                                : "Ollama (Local, RAM kullanır)"}
                            </span>
                          </div>
                          {llmProvider === "ollama" && (
                            <div className="text-xs text-amber-600 font-medium mt-1 p-2 bg-amber-500/10 rounded border border-amber-500/20">
                              ⚠️ Ollama batch processing desteklemiyor. Upload
                              sonrası chunk listesinden tek tek
                              iyileştirebilirsiniz.
                            </div>
                          )}
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-foreground">
                              İşlem Modu:
                            </span>
                            <span className="text-muted-foreground">
                              Batch Processing (80% daha az API çağrısı)
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-foreground">
                              Tahmini Süre:
                            </span>
                            <span className="text-amber-600 font-medium">
                              {selectedFiles.length > 0
                                ? `~${Math.ceil(
                                    selectedFiles.length * 2.5
                                  )} dakika`
                                : "Dosya seçin"}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Configuration Parameters - Read-only Information */}
              <div className="bg-muted/30 rounded-lg p-4 border border-border">
                <h3 className="text-sm font-semibold text-foreground mb-3">
                  Temel İşleme Ayarları
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="block text-xs font-medium text-muted-foreground">
                      Parçalama Stratejisi
                    </label>
                    <div className="px-3 py-2 text-sm border border-border rounded-lg bg-muted/50 text-foreground">
                      Lightweight Turkish
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="block text-xs font-medium text-muted-foreground">
                      Parça Boyutu
                    </label>
                    <div className="px-3 py-2 text-sm border border-border rounded-lg bg-muted/50 text-foreground">
                      Anlamsal (~400-1200)
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="block text-xs font-medium text-muted-foreground">
                      Parça Çakışması
                    </label>
                    <div className="px-3 py-2 text-sm border border-border rounded-lg bg-muted/50 text-foreground">
                      Otomatik (cümle bazlı)
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="block text-xs font-medium text-muted-foreground">
                      Embedding Model
                    </label>
                    <div className="px-3 py-2 text-sm border border-border rounded-lg bg-muted/50 text-foreground">
                      {embeddingModel}
                    </div>
                  </div>
                </div>
              </div>
            </form>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-6 border-t border-border bg-muted/30">
          <div className="flex gap-3 justify-end">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              {isProcessing ? "Arka Planda Devam Et" : "İptal"}
            </button>
            {!isProcessing && (
              <button
                type="submit"
                form="upload-form"
                disabled={selectedFiles.length === 0}
                className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium text-sm hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                İşlemeyi Başlat
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
