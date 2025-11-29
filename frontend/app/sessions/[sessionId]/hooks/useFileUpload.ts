"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useExtendedChunksStore } from "../stores/chunksStore";
import { useSessionStore } from "../stores/sessionStore";
import { useChunksErrorHandler } from "./useErrorHandler";

// API functions from the original FileUploadModal
const getApiUrl = () => {
  if (typeof window === "undefined") return "http://localhost:8000";
  return window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "https://api-gateway:8000";
};

const listMarkdownFiles = async (): Promise<string[]> => {
  const response = await fetch(`${getApiUrl()}/api/markdown-files`);
  if (!response.ok) throw new Error("Failed to fetch markdown files");
  const data = await response.json();
  return data.files || [];
};

const configureAndProcess = async (config: any) => {
  const response = await fetch(`${getApiUrl()}/api/configure-and-process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error("Failed to process files");
  return response.json();
};

const getChunksForSession = async (sessionId: string) => {
  const response = await fetch(
    `${getApiUrl()}/api/sessions/${sessionId}/chunks`
  );
  if (!response.ok) throw new Error("Failed to fetch chunks");
  const data = await response.json();
  return data.chunks || [];
};

interface FileUploadConfig {
  chunk_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  use_llm_post_processing: boolean;
  llm_model_name: string;
  llm_provider: "ollama" | "grok";
}

interface FileUploadState {
  isProcessing: boolean;
  processingStep: string;
  selectedFiles: string[];
  uploadProgress: Record<string, number>;
  availableFiles: string[];
  processedFiles: Set<string>;
  config: FileUploadConfig;
  error: string | null;
}

interface FileUploadOperations {
  handleFileUpload: (
    files: string[],
    config?: Partial<FileUploadConfig>
  ) => Promise<void>;
  refreshFiles: () => Promise<void>;
  setSelectedFiles: (files: string[]) => void;
  toggleFileSelection: (filename: string) => void;
  updateConfig: (updates: Partial<FileUploadConfig>) => void;
  clearProcessingState: () => void;
  clearError: () => void;
}

const DEFAULT_CONFIG: FileUploadConfig = {
  chunk_strategy: "lightweight",
  chunk_size: 800,
  chunk_overlap: 100,
  embedding_model: "nomic-embed-text",
  use_llm_post_processing: false,
  llm_model_name: "llama-3.1-8b-instant",
  llm_provider: "grok",
};

export const useFileUpload = (
  sessionId: string
): FileUploadState & FileUploadOperations => {
  const [state, setState] = useState<FileUploadState>({
    isProcessing: false,
    processingStep: "",
    selectedFiles: [],
    uploadProgress: {},
    availableFiles: [],
    processedFiles: new Set(),
    config: DEFAULT_CONFIG,
    error: null,
  });

  const { handleChunksError } = useChunksErrorHandler();
  const chunksStore = useExtendedChunksStore();
  const sessionStore = useSessionStore();
  const processingTimeoutRef = useRef<NodeJS.Timeout>();

  // Normalize filename for comparison (remove .md extension, convert to lowercase)
  const normalizeFilename = useCallback((filename: string): string => {
    return filename.toLowerCase().replace(/\.md$/i, "");
  }, []);

  // Fetch processed files from session chunks
  const fetchProcessedFiles = useCallback(async (): Promise<Set<string>> => {
    try {
      const chunks = await getChunksForSession(sessionId);
      console.log("🔍 Fetched chunks:", chunks.length, "chunks");

      const processed = new Set<string>();
      chunks.forEach((chunk: any, index: number) => {
        const docName =
          chunk.document_name ||
          chunk.chunk_metadata?.source_file ||
          chunk.chunk_metadata?.filename ||
          chunk.chunk_metadata?.document_name ||
          chunk.chunk_metadata?.source_files;

        if (index < 3) {
          console.log(`🔍 Chunk ${index}:`, {
            document_name: chunk.document_name,
            metadata: chunk.chunk_metadata,
            extracted: docName,
          });
        }

        if (docName && docName !== "Unknown") {
          let filesToProcess: string[] = [];
          if (Array.isArray(docName)) {
            filesToProcess = docName;
          } else if (typeof docName === "string") {
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
              if (index < 3)
                console.log(
                  `✅ Added processed file: "${file}" -> "${normalized}"`
                );
            }
          });
        }
      });

      console.log("🔍 Processed files set:", Array.from(processed));
      return processed;
    } catch (e: any) {
      console.warn("Could not fetch processed files:", e);
      return new Set<string>();
    }
  }, [sessionId, normalizeFilename]);

  // Fetch available markdown files
  const refreshFiles = useCallback(async () => {
    try {
      setState((prev) => ({ ...prev, error: null }));

      const [files, processed] = await Promise.all([
        listMarkdownFiles(),
        fetchProcessedFiles(),
      ]);

      const availableFiles = files.filter((file: string) => {
        const normalized = normalizeFilename(file);
        return !processed.has(normalized);
      });

      setState((prev) => ({
        ...prev,
        availableFiles,
        processedFiles: processed,
      }));
    } catch (error: any) {
      const errorMessage = error.message || "Markdown dosyaları yüklenemedi";
      setState((prev) => ({ ...prev, error: errorMessage }));
      handleChunksError(error, "load_files");
    }
  }, [fetchProcessedFiles, normalizeFilename, handleChunksError]);

  // Handle file upload and processing
  const handleFileUpload = useCallback(
    async (files: string[], configOverrides?: Partial<FileUploadConfig>) => {
      if (files.length === 0) {
        setState((prev) => ({
          ...prev,
          error: "En az bir Markdown dosyası seçmelisiniz",
        }));
        return;
      }

      const finalConfig = { ...state.config, ...configOverrides };

      try {
        setState((prev) => ({
          ...prev,
          isProcessing: true,
          processingStep: "Konfigürasyon hazırlanıyor...",
          selectedFiles: files,
          error: null,
          uploadProgress: files.reduce(
            (acc, file) => ({ ...acc, [file]: 0 }),
            {}
          ),
        }));

        // Simulate progress updates
        const progressSteps = [
          "Dosyalar okunuyor...",
          "Metin parçaları oluşturuluyor...",
          "Embedding vektörleri hesaplanıyor...",
          "Veritabanı güncelleniyor...",
        ];

        let stepIndex = 0;
        processingTimeoutRef.current = setInterval(() => {
          if (stepIndex < progressSteps.length) {
            setState((prev) => ({
              ...prev,
              processingStep: progressSteps[stepIndex],
              uploadProgress: files.reduce(
                (acc, file) => ({
                  ...acc,
                  [file]: Math.min(
                    ((stepIndex + 1) / progressSteps.length) * 100,
                    90
                  ),
                }),
                {}
              ),
            }));
            stepIndex++;
          }
        }, 3000);

        const result = await configureAndProcess({
          session_id: sessionId,
          markdown_files: files,
          ...finalConfig,
        });

        if (processingTimeoutRef.current) {
          clearInterval(processingTimeoutRef.current);
        }

        if (result.success) {
          setState((prev) => ({
            ...prev,
            processingStep: "İşlem tamamlandı!",
            uploadProgress: files.reduce(
              (acc, file) => ({ ...acc, [file]: 100 }),
              {}
            ),
          }));

          // Refresh data
          await Promise.all([
            refreshFiles(),
            chunksStore.fetchChunks(sessionId),
          ]);

          // Clear processing state after delay
          setTimeout(() => {
            setState((prev) => ({
              ...prev,
              isProcessing: false,
              processingStep: "",
              selectedFiles: [],
              uploadProgress: {},
            }));
          }, 2000);
        } else {
          throw new Error(result.message || "İşlem başarısız");
        }
      } catch (error: any) {
        if (processingTimeoutRef.current) {
          clearInterval(processingTimeoutRef.current);
        }

        const errorMessage = error.message || "RAG konfigürasyonu başarısız";
        setState((prev) => ({
          ...prev,
          isProcessing: false,
          processingStep: "",
          uploadProgress: {},
          error: errorMessage,
        }));
        handleChunksError(error, "process_failed");
      }
    },
    [state.config, sessionId, refreshFiles, chunksStore, handleChunksError]
  );

  // Set selected files
  const setSelectedFiles = useCallback((files: string[]) => {
    setState((prev) => ({ ...prev, selectedFiles: files }));
  }, []);

  // Toggle file selection
  const toggleFileSelection = useCallback((filename: string) => {
    setState((prev) => ({
      ...prev,
      selectedFiles: prev.selectedFiles.includes(filename)
        ? prev.selectedFiles.filter((f) => f !== filename)
        : [...prev.selectedFiles, filename],
    }));
  }, []);

  // Update configuration
  const updateConfig = useCallback((updates: Partial<FileUploadConfig>) => {
    setState((prev) => ({
      ...prev,
      config: { ...prev.config, ...updates },
    }));
  }, []);

  // Clear processing state
  const clearProcessingState = useCallback(() => {
    if (processingTimeoutRef.current) {
      clearInterval(processingTimeoutRef.current);
    }
    setState((prev) => ({
      ...prev,
      isProcessing: false,
      processingStep: "",
      selectedFiles: [],
      uploadProgress: {},
    }));
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  // DISABLED: Load files on mount and when sessionId changes
  // This was causing automatic API calls and CORS errors
  // Original system requires manual user action to load files
  // useEffect(() => {
  //   if (sessionId) {
  //     refreshFiles();
  //   }
  // }, [sessionId, refreshFiles]);

  // Get embedding model from session settings - temporarily disabled
  useEffect(() => {
    // Note: SessionSettings doesn't have rag_settings, this would need to be handled differently
    // For now, keep default config
    // const sessionSettings = sessionStore.sessionSettings;
    // if (sessionSettings?.embedding_model) {
    //   updateConfig({ embedding_model: sessionSettings.embedding_model });
    // }
  }, [sessionStore.sessionSettings, updateConfig]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (processingTimeoutRef.current) {
        clearInterval(processingTimeoutRef.current);
      }
    };
  }, []);

  return {
    ...state,
    handleFileUpload,
    refreshFiles,
    setSelectedFiles,
    toggleFileSelection,
    updateConfig,
    clearProcessingState,
    clearError,
  };
};

export default useFileUpload;
