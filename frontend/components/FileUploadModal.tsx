"use client";
import React, { useState, useEffect, FormEvent } from "react";
import {
  addMarkdownDocumentsToSession,
  getBatchProcessingStatus,
  listMarkdownFiles,
  listMarkdownFilesWithCategories,
  listMarkdownCategories,
  getSession,
  getChunksForSession,
  MarkdownFileWithCategory,
  MarkdownCategory,
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

const FilterIcon = () => (
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
      d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.707A1 1 0 013 7V4z"
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
  const [markdownFiles, setMarkdownFiles] = useState<
    MarkdownFileWithCategory[]
  >([]);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [processedFiles, setProcessedFiles] = useState<Set<string>>(new Set());

  // Category filtering states
  const [categories, setCategories] = useState<MarkdownCategory[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<number | null>(null);
  const [filteredFiles, setFilteredFiles] = useState<
    MarkdownFileWithCategory[]
  >([]);

  // Configuration states - Fixed values (read-only in UI)
  const chunkStrategy = "lightweight"; // Fixed: lightweight chunking
  const chunkSize = 800; // Fixed: display only
  const chunkOverlap = 100; // Fixed: display only
  const [embeddingModel, setEmbeddingModel] = useState(
    defaultEmbeddingModel || "nomic-embed-text"
  ); // Use session's embedding model

  // Processing status
  const [processingStep, setProcessingStep] = useState("");
  const [batchJobId, setBatchJobId] = useState<string | null>(null);
  const [batchProgress, setBatchProgress] = useState<{
    processed: number;
    total: number;
    current_file: string | null;
    total_chunks: number;
  }>({ processed: 0, total: 0, current_file: null, total_chunks: 0 });

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

  // Update filtered files when markdownFiles or selectedCategory change
  useEffect(() => {
    if (selectedCategory === null) {
      setFilteredFiles(markdownFiles);
    } else {
      setFilteredFiles(
        markdownFiles.filter((file) => file.category_id === selectedCategory)
      );
    }
  }, [markdownFiles, selectedCategory]);

  // Handle category selection
  const handleCategoryChange = (categoryId: number | null) => {
    setSelectedCategory(categoryId);
  };

  // Fetch processed files from session chunks - OPTIMIZED
  const fetchProcessedFiles = async () => {
    try {
      const chunks = await getChunksForSession(sessionId);

      // Only log count, not individual chunks to prevent spam
      console.log(
        `📊 Processing ${chunks.length} chunks for session ${sessionId}`
      );

      // Extract unique document names from chunks - OPTIMIZED
      const processed = new Set<string>();
      const seenDocNames = new Set<string>(); // Prevent duplicate processing

      chunks.forEach((chunk: any) => {
        // Try multiple possible locations for filename
        const docName =
          chunk.document_name ||
          chunk.chunk_metadata?.source_file ||
          chunk.chunk_metadata?.filename ||
          chunk.chunk_metadata?.document_name ||
          chunk.chunk_metadata?.source_files;

        // Skip if already processed this doc name
        if (!docName || docName === "Unknown" || seenDocNames.has(docName)) {
          return;
        }
        seenDocNames.add(docName);

        // Handle array case (source_files might be an array)
        let filesToProcess: string[] = [];
        if (Array.isArray(docName)) {
          filesToProcess = docName;
        } else if (typeof docName === "string") {
          // Try parsing as JSON if it's a JSON string
          try {
            const parsed = JSON.parse(docName);
            if (Array.isArray(parsed)) {
              filesToProcess = parsed;
            } else {
              filesToProcess = [docName];
            }
          } catch {
            filesToProcess = [docName];
          }
        }

        filesToProcess.forEach((file: string) => {
          if (file && file !== "Unknown") {
            const normalized = normalizeFilename(file);
            processed.add(normalized);
          }
        });
      });

      console.log(
        `✅ Found ${processed.size} processed files for session ${sessionId}`
      );
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

      // Fetch markdown files with categories, categories, and processed files in parallel
      const [filesWithCategories, categoriesList, processed] =
        await Promise.all([
          listMarkdownFilesWithCategories(),
          listMarkdownCategories(),
          fetchProcessedFiles(),
        ]);

      // Filter out already processed files (normalize for comparison)
      const availableFiles = filesWithCategories.filter(
        (file: MarkdownFileWithCategory) => {
          const normalized = normalizeFilename(file.filename);
          return !processed.has(normalized);
        }
      );

      setMarkdownFiles(availableFiles);
      setCategories(categoriesList);
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

  // Handle select all toggle
  const handleSelectAll = () => {
    const currentFilteredFilenames = filteredFiles.map((file) => file.filename);

    if (
      selectedFiles.length === currentFilteredFilenames.length &&
      currentFilteredFilenames.every((filename) =>
        selectedFiles.includes(filename)
      )
    ) {
      // If all filtered files are selected, deselect all
      setSelectedFiles([]);
    } else {
      // Otherwise, select all filtered files
      setSelectedFiles(currentFilteredFilenames);
    }
  };

  // Poll batch processing status (only while modal is open)
  useEffect(() => {
    if (!batchJobId || !isProcessing || !isOpen) return;

    const pollInterval = setInterval(async () => {
      try {
        const status = await getBatchProcessingStatus(batchJobId);
        const job = status.job;

        setBatchProgress({
          processed: job.processed_successfully,
          total: job.total_files,
          current_file: job.current_file,
          total_chunks: job.total_chunks,
        });

        if (job.current_file) {
          setProcessingStep(`İşleniyor: ${job.current_file} (${job.processed_successfully}/${job.total_files})`);
        }

        if (job.status === "completed" || job.status === "completed_with_errors" || job.status === "failed") {
          clearInterval(pollInterval);
          setProcessingStep("İşlem tamamlandı!");
          
          onSuccess({
            success: true,
            message: `Batch işlem tamamlandı: ${job.processed_successfully}/${job.total_files} dosya, ${job.total_chunks} chunk`,
            processed_count: job.processed_successfully,
            total_chunks_added: job.total_chunks,
            errors: job.errors,
            job_id: job.job_id,
          });

          setSelectedFiles([]);
          await fetchMarkdownFiles();

          setTimeout(() => {
            setIsProcessing(false);
            setBatchJobId(null);
            setProcessingStep("");
            onClose();
          }, 2000);
        }
      } catch (e: any) {
        console.error("Error polling batch status:", e);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, [batchJobId, isProcessing, isOpen]);

  // Handle form submission with batch processing
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (selectedFiles.length === 0) {
      onError("En az bir Markdown dosyası seçmelisiniz");
      return;
    }

    try {
      setIsProcessing(true);
      setProcessingStep("Batch işlem başlatılıyor...");

      console.log(`🚀 Starting batch processing for ${selectedFiles.length} files`);

      const result = await addMarkdownDocumentsToSession(
        sessionId,
        selectedFiles,
        embeddingModel
      );

      if (result.success && result.job_id) {
        // Batch processing started - close modal immediately
        setBatchJobId(result.job_id);
        setBatchProgress({
          processed: 0,
          total: result.total_files || selectedFiles.length,
          current_file: null,
          total_chunks: 0,
        });
        
        // Show success message
        onSuccess({
          success: true,
          message: `Batch işlem başlatıldı: ${result.total_files || selectedFiles.length} dosya arka planda işleniyor...`,
          job_id: result.job_id,
          background_processing: true,
        });
        
        // Close modal immediately
        setTimeout(() => {
          onClose();
        }, 500); // Small delay to show message
        
        // Processing will continue in background, status will be polled
        // Success callback will be called when job completes
      } else {
        throw new Error(result.message || "Batch işlem başlatılamadı");
      }
    } catch (e: any) {
      console.error("❌ Processing error:", e);
      onError(e.message || "RAG konfigürasyonu başarısız");
      setIsProcessing(false);
      setProcessingStep("");
      setBatchJobId(null);
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

  // Update processing steps with proper cleanup
  useEffect(() => {
    let stepInterval: NodeJS.Timeout | null = null;

    if (isProcessing && processingStep === "Konfigürasyon hazırlanıyor...") {
      const steps = [
        "Dosyalar okunuyor...",
        "Metin parçaları oluşturuluyor...",
        "Embedding vektörleri hesaplanıyor...",
        "Veritabanı güncelleniyor...",
        "İşlem tamamlanıyor...",
      ];

      let stepIndex = 0;
      stepInterval = setInterval(() => {
        if (stepIndex < steps.length) {
          setProcessingStep(steps[stepIndex]);
          stepIndex++;
        } else {
          // Clear interval when steps are done
          if (stepInterval) {
            clearInterval(stepInterval);
            stepInterval = null;
          }
        }
      }, 3000);
    }

    // Cleanup function
    return () => {
      if (stepInterval) {
        clearInterval(stepInterval);
      }
    };
  }, [isProcessing, processingStep]);

  // Global cleanup when component unmounts
  useEffect(() => {
    return () => {
      // Clear any remaining processing state
      setIsProcessing(false);
      setProcessingStep("");
    };
  }, []);

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
              {/* Category Filter */}
              {categories.length > 0 && (
                <div className="mb-6">
                  <label className="block text-sm font-medium text-foreground mb-3">
                    <div className="flex items-center gap-2">
                      <FilterIcon />
                      Kategori Filtresi
                    </div>
                  </label>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => handleCategoryChange(null)}
                      className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                        selectedCategory === null
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      Tümü
                      <span className="ml-2 text-xs opacity-70">
                        ({markdownFiles.length})
                      </span>
                    </button>
                    {categories.map((category) => (
                      <button
                        key={category.id}
                        type="button"
                        onClick={() => handleCategoryChange(category.id)}
                        className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                          selectedCategory === category.id
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {category.name}
                        <span className="ml-2 text-xs opacity-70">
                          (
                          {
                            markdownFiles.filter(
                              (file) => file.category_id === category.id
                            ).length
                          }
                          )
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* File Selection */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <label className="block text-sm font-medium text-foreground">
                    <div className="flex items-center gap-2">
                      <UploadIcon className="w-5 h-5" />
                      Markdown Dosyaları Seçin
                      {selectedCategory !== null && (
                        <span className="text-xs text-muted-foreground ml-2">
                          (Kategori:{" "}
                          {
                            categories.find((c) => c.id === selectedCategory)
                              ?.name
                          }
                          )
                        </span>
                      )}
                    </div>
                  </label>
                  {filteredFiles.length > 0 && (
                    <button
                      type="button"
                      onClick={handleSelectAll}
                      className="text-xs text-primary hover:text-primary/80 font-medium px-2 py-1 rounded hover:bg-primary/10 transition-colors"
                    >
                      {selectedFiles.length === filteredFiles.length &&
                      filteredFiles.every((file) =>
                        selectedFiles.includes(file.filename)
                      )
                        ? "Hiçbirini Seçme"
                        : "Tümünü Seç"}
                    </button>
                  )}
                </div>

                {loading ? (
                  <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent mx-auto"></div>
                    <p className="text-muted-foreground text-sm mt-2">
                      Dosyalar yükleniyor...
                    </p>
                  </div>
                ) : (
                  <div className="max-h-48 overflow-y-auto border border-border rounded-lg bg-background">
                    {filteredFiles.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground">
                        <div className="text-sm">
                          {selectedCategory === null
                            ? "Markdown dosyası bulunamadı"
                            : `${
                                categories.find(
                                  (c) => c.id === selectedCategory
                                )?.name
                              } kategorisinde dosya bulunamadı`}
                        </div>
                      </div>
                    ) : (
                      filteredFiles.map((file) => (
                        <div
                          key={file.filename}
                          className="flex items-start p-4 hover:bg-muted/50 border-b border-border last:border-b-0 transition-colors"
                        >
                          <input
                            type="checkbox"
                            checked={selectedFiles.includes(file.filename)}
                            onChange={() => handleFileToggle(file.filename)}
                            className="mt-0.5 mr-4 h-5 w-5 text-primary rounded border-border focus:ring-primary focus:ring-2 flex-shrink-0"
                          />
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium text-foreground truncate">
                              {file.filename.replace(".md", "")}
                            </div>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                              <span className="truncate">{file.filename}</span>
                              {file.category_name && (
                                <span className="px-2 py-1 bg-muted/50 rounded text-xs font-medium flex-shrink-0">
                                  {file.category_name}
                                </span>
                              )}
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
