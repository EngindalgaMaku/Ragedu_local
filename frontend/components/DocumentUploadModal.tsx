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
}: DocumentUploadModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [processingStep, setProcessingStep] = useState("");
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
      stepIntervalRef.current = setInterval(() => {
        if (stepIndex < steps.length) {
          setProcessingStep(steps[stepIndex]);
          stepIndex++;
        } else {
          // Loop back to show we're still processing
          setProcessingStep("İşlem devam ediyor... Lütfen bekleyin.");
        }
      }, useMarker ? 8000 : 5000); // Slower for Marker (8s vs 5s)

      const result = useMarker 
        ? await convertMarker(selectedFile)
        : await convertPdfToMarkdown(selectedFile, useFallback);

      if (stepIntervalRef.current) {
        clearInterval(stepIntervalRef.current);
        stepIntervalRef.current = null;
      }

      if (result.success) {
        setProcessingStep("İşlem tamamlandı!");
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
        onError(result.message || "Dönüştürme işlemi başarısız");
        setIsConverting(false);
        setProcessingStep("");
      }
    } catch (e: any) {
      onError(e.message || "Belge dönüştürme işlemi başarısız");
      setIsConverting(false);
      setProcessingStep("");
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
      onError(""); // Clear any previous errors
    } else {
      onError("Lütfen sadece PDF, DOCX, PPTX veya XLSX dosyası seçin");
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="bg-card rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
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
        <div className="p-6">
          {isConverting ? (
            // Processing State
            <div className="text-center py-8">
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
            <div className="space-y-6">
              {/* Drag & Drop Zone */}
              <div
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all ${
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
        <div className="p-6 border-t border-border bg-muted/30">
          <div className="flex gap-3 justify-end">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              {isConverting ? "Arka Planda Devam Et" : "İptal"}
            </button>
            {!isConverting && (
              <button
                type="button"
                onClick={() => handlePdfUpload()}
                disabled={!selectedFile}
                className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium text-sm hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
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
