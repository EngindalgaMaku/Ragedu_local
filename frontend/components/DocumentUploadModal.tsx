"use client";
import React, { useState, FormEvent, useEffect, useRef } from "react";
import { convertPdfToMarkdown, convertMarker } from "@/lib/api";

interface DocumentUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (message: string) => void;
  onError: (error: string) => void;
  onMarkdownFilesUpdate: () => void;
  conversionMethod?: "nanonets" | "pdfplumber" | "marker"; // Pre-selected method
  selectedSessionId?: string; // Optional session ID for file naming
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

const DocumentIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
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

export default function DocumentUploadModal({
  isOpen,
  onClose,
  onSuccess,
  onError,
  onMarkdownFilesUpdate,
  conversionMethod, // Pre-selected: "nanonets" or "pdfplumber"
  selectedSessionId,
}: DocumentUploadModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [processingStep, setProcessingStep] = useState("");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const stepIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Use conversionMethod if provided, otherwise default to nanonets
  const useFallback = conversionMethod === "pdfplumber";
  const useMarker = conversionMethod === "marker";

  const handlePdfUpload = async (e?: FormEvent) => {
    if (e) e.preventDefault();
    if (!selectedFile) return;

    try {
      setIsConverting(true);
      setProcessingStep("Belge analiz ediliyor...");

      // Simulate processing steps with more realistic timing
      const steps = useMarker
        ? [
            "Belge okunuyor...",
            "Marker OCR çalışıyor...",
            "Düzen ve format analiz ediliyor...",
            "Karmaşık içerik işleniyor...",
            "Markdown formatına dönüştürülüyor...",
            "Dosya kaydediliyor...",
          ]
        : [
            "Belge okunuyor...",
            "İçerik ayıklanıyor...",
            "Büyük dosyalar için işlem devam ediyor...",
            "Markdown formatına dönüştürülüyor...",
            "Dosya kaydediliyor...",
          ];

      let stepIndex = 0;
      if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
      stepIntervalRef.current = setInterval(
        () => {
          if (stepIndex < steps.length) {
            setProcessingStep(steps[stepIndex]);
            stepIndex++;
          } else {
            // Loop back to show we're still processing
            setProcessingStep("İşlem devam ediyor... Lütfen bekleyin.");
          }
        },
        useMarker ? 8000 : 5000
      ); // Slower for Marker (8s vs 5s)

      // Marker için doğrudan Marker API'sini kullan, diğerlerinde mevcut PDF dönüştürme yolunu koru
      const result = useMarker
        ? await convertMarker(selectedFile)
        : await convertPdfToMarkdown(selectedFile, useFallback);

      // Clear interval immediately after request completes
      if (stepIntervalRef.current) {
        clearInterval(stepIntervalRef.current);
        stepIntervalRef.current = null;
      }

      if (result.success) {
        setProcessingStep("İşlem tamamlandı!");
        setErrorMessage(""); // Clear any errors
        setIsConverting(false); // Stop converting state
        onSuccess(
          `Belge başarıyla Markdown formatına dönüştürüldü: ${result.markdown_filename}`
        );
        setSelectedFile(null);

        setTimeout(() => {
          onMarkdownFilesUpdate();
          onClose();
          setProcessingStep("");
        }, 2000);
      } else {
        const errorMsg = result.message || "Dönüştürme işlemi başarısız";
        console.error("Conversion failed:", errorMsg, result);
        setIsConverting(false); // Stop converting state
        setProcessingStep(""); // Clear processing step
        setErrorMessage(errorMsg);
        onError(errorMsg);
      }
    } catch (e: any) {
      // Clear interval immediately on error
      if (stepIntervalRef.current) {
        clearInterval(stepIntervalRef.current);
        stepIntervalRef.current = null;
      }

      console.error("Document conversion error:", e);
      console.error("Error type:", typeof e);
      console.error("Error keys:", Object.keys(e || {}));

      let errorMsg = "Belge dönüştürme işlemi başarısız";

      // Try to extract error message from various sources
      if (e?.message) {
        errorMsg = e.message;
      } else if (e?.response?.data?.detail) {
        errorMsg = e.response.data.detail;
      } else if (e?.response?.data?.message) {
        errorMsg = e.response.data.message;
      } else if (typeof e === "string") {
        errorMsg = e;
      } else if (e?.toString && e.toString() !== "[object Object]") {
        errorMsg = e.toString();
      } else if (e?.detail) {
        errorMsg = e.detail;
      }

      // If it's a network error, provide more context
      if (
        errorMsg.includes("Failed to fetch") ||
        errorMsg.includes("NetworkError")
      ) {
        errorMsg =
          "Sunucuya bağlanılamadı. Lütfen internet bağlantınızı kontrol edin veya daha sonra tekrar deneyin.";
      }

      // If it's an Internal Server Error, provide helpful message
      if (
        errorMsg.includes("Internal Server Error") ||
        errorMsg.includes("500")
      ) {
        errorMsg =
          "Sunucu hatası oluştu. Büyük/karmaşık PDF'ler için 'Hızlı Dönüştür' yöntemini deneyin veya daha küçük bir dosya kullanın.";
      }

      console.error("Final extracted error message:", errorMsg);
      setIsConverting(false); // CRITICAL: Stop converting state
      setProcessingStep(""); // Clear processing step
      setErrorMessage(errorMsg);
      onError(errorMsg);
    }
  };

  // Reset modal state whenever it opens fresh
  useEffect(() => {
    if (isOpen) {
      // Fresh open: clear previous state
      setSelectedFile(null);
      setIsConverting(false);
      setProcessingStep("");
      setIsDragOver(false);
      setErrorMessage(""); // Clear errors when opening
    } else {
      // Closed: clear any running interval
      if (stepIntervalRef.current) {
        clearInterval(stepIntervalRef.current);
        stepIntervalRef.current = null;
      }
    }
    // Cleanup on unmount
    return () => {
      if (stepIntervalRef.current) {
        clearInterval(stepIntervalRef.current);
        stepIntervalRef.current = null;
      }
    };
  }, [isOpen]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      handleFileValidation(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (file) {
      handleFileValidation(file);
    }
  };

  const handleFileValidation = (file: File) => {
    const supportedTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // DOCX
      "application/vnd.openxmlformats-officedocument.presentationml.presentation", // PPTX
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // XLSX
    ];

    if (supportedTypes.includes(file.type)) {
      setSelectedFile(file);
      setErrorMessage(""); // Clear any previous errors
      onError(""); // Clear any previous errors
    } else {
      const errorMsg = "Lütfen sadece PDF, DOCX, PPTX veya XLSX dosyası seçin";
      setErrorMessage(errorMsg);
      onError(errorMsg);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-2 sm:p-4 z-50">
      <div className="bg-card rounded-xl shadow-2xl w-full max-w-sm sm:max-w-md md:max-w-lg lg:max-w-2xl max-h-[95vh] sm:max-h-[90vh] overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-3 sm:p-4 md:p-6 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 text-primary rounded-lg">
              <UploadIcon className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-foreground">
                {conversionMethod === "pdfplumber"
                  ? "⚡ Hızlı Dönüştür"
                  : conversionMethod === "marker"
                  ? "📚 Marker (En Kaliteli)"
                  : "🌐 Gelişmiş Dönüştür"}
              </h2>
              <p className="text-sm text-muted-foreground">
                {conversionMethod === "pdfplumber"
                  ? "Basit metin PDF'ler için hızlı dönüştürme (30 sn)"
                  : conversionMethod === "marker"
                  ? "PDF/PPT/DOC için en kaliteli OCR ve düzen korumalı dönüştürme (5-15 dk)"
                  : "Taranmış belgeler ve karmaşık PDF'ler için OCR (2-10 dk)"}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-muted rounded-lg transition-colors"
          >
            <CloseIcon />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-3 sm:p-4 md:p-6">
          {/* Error Message Display - Always visible when there's an error */}
          {errorMessage && (
            <div className="mb-4 p-4 bg-red-50 dark:bg-red-900/20 border-2 border-red-300 dark:border-red-700 rounded-lg shadow-lg">
              <div className="flex items-start gap-3">
                <svg
                  className="w-6 h-6 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <div className="flex-1 min-w-0">
                  <h4 className="text-base font-bold text-red-800 dark:text-red-300 mb-2">
                    ⚠️ Hata Oluştu
                  </h4>
                  <p className="text-sm text-red-700 dark:text-red-400 whitespace-pre-wrap break-words">
                    {errorMessage}
                  </p>
                </div>
                <button
                  onClick={() => {
                    setErrorMessage("");
                    onError(""); // Also clear parent error
                  }}
                  className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300 flex-shrink-0 p-1"
                  title="Kapat"
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
                      strokeWidth="2"
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            </div>
          )}

          {isConverting ? (
            // Processing State
            <div className="text-center py-4 sm:py-6 md:py-8">
              <div className="flex flex-col items-center gap-4">
                <ProcessingIcon />
                <div>
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    Belge Dönüştürülüyor
                  </h3>
                  <p className="text-muted-foreground mb-4">{processingStep}</p>
                  <div className="bg-primary/10 text-primary p-3 rounded-lg text-sm">
                    💡 Modal'ı kapatabilirsiniz - İşlem arka planda devam edecek
                    ve bitince sonuçlar görünecektir
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // Upload Form
            <div className="space-y-4 md:space-y-6">
              {/* Drag & Drop Zone */}
              <div
                className={`relative border-2 border-dashed rounded-xl p-4 sm:p-6 md:p-8 text-center transition-all ${
                  selectedFile ? "cursor-default" : "cursor-pointer"
                } ${
                  isDragOver
                    ? "border-primary bg-primary/5"
                    : selectedFile
                    ? "border-green-300 bg-green-50"
                    : "border-border hover:border-primary/50 hover:bg-muted/30"
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {selectedFile ? (
                  <div className="flex flex-col items-center gap-4">
                    <div className="p-4 rounded-full bg-green-100">
                      <DocumentIcon />
                    </div>
                    <div>
                      <p className="text-lg font-medium text-green-600">
                        ✅ Dosya Seçildi
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">
                        {selectedFile.name} -{" "}
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-4">
                    <div
                      className={`p-4 rounded-full ${
                        isDragOver ? "bg-primary/20" : "bg-muted/50"
                      }`}
                    >
                      <UploadIcon className="w-12 h-12" />
                    </div>
                    <div>
                      <p className="text-lg font-medium text-foreground">
                        Belgenizi buraya sürükleyin
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">
                        PDF, DOCX, PPTX, XLSX • Maks. 50MB
                      </p>
                    </div>
                  </div>
                )}
                <input
                  type="file"
                  id="file-upload"
                  onChange={handleFileSelect}
                  className="absolute inset-0 opacity-0 cursor-pointer"
                  accept=".pdf,.docx,.pptx,.xlsx"
                  disabled={isConverting}
                />
              </div>

              <input
                type="file"
                id="file-upload"
                onChange={handleFileSelect}
                className="hidden"
                accept=".pdf,.docx,.pptx,.xlsx"
                disabled={isConverting}
              />
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-3 sm:p-4 md:p-6 border-t border-border bg-muted/30">
          <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 justify-end">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors min-h-[44px] order-2 sm:order-1"
            >
              {isConverting ? "Arka Planda Devam Et" : "İptal"}
            </button>
            {!isConverting && (
              <button
                type="button"
                onClick={() => handlePdfUpload()}
                disabled={!selectedFile}
                className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium text-sm hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all min-h-[44px] order-1 sm:order-2"
              >
                Dönüştür ve Yükle
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
