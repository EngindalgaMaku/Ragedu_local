"use client";
import React, { useState, FormEvent } from "react";
import { convertPdfToMarkdown, uploadMarkdownFile } from "@/lib/api";

interface EnhancedDocumentUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (message: string) => void;
  onError: (error: string) => void;
  onMarkdownFilesUpdate: () => void;
  selectedSessionId?: string; // Optional session ID for file naming
}

const CloseIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

const UploadIcon = ({ className = "w-8 h-8" }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
  </svg>
);

const DocumentIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

const MarkdownIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
  </svg>
);

const ProcessingIcon = () => (
  <svg className="w-8 h-8 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
  </svg>
);

const CheckIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
  </svg>
);

type UploadMode = 'convert' | 'direct';

export default function EnhancedDocumentUploadModal({
  isOpen,
  onClose,
  onSuccess,
  onError,
  onMarkdownFilesUpdate,
  selectedSessionId,
}: EnhancedDocumentUploadModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [processingStep, setProcessingStep] = useState("");
  const [uploadMode, setUploadMode] = useState<UploadMode>('convert');

  const handleFileUpload = async (e?: FormEvent) => {
    if (e) e.preventDefault();
    if (!selectedFile) return;

    try {
      setIsProcessing(true);
      
      if (uploadMode === 'direct') {
        // Direct markdown upload
        setProcessingStep("Markdown dosyası yükleniyor...");
        const result = await uploadMarkdownFile(selectedFile);
        
        if (result.success) {
          setProcessingStep("İşlem tamamlandı!");
          onSuccess(`Markdown dosyası başarıyla yüklendi: ${result.markdown_filename}`);
          setSelectedFile(null);
          
          setTimeout(() => {
            onMarkdownFilesUpdate();
            setIsProcessing(false);
            setProcessingStep("");
          }, 1200);
        } else {
          onError(result.message || "Yükleme işlemi başarısız");
          setIsProcessing(false);
          setProcessingStep("");
        }
      } else {
        // Document conversion
        setProcessingStep("Belge analiz ediliyor...");
        
        const steps = [
          "Belge okunuyor...",
          "İçerik ayıklanıyor...",
          "Markdown formatına dönüştürülüyor...",
          "Dosya kaydediliyor...",
        ];

        let stepIndex = 0;
        const stepInterval = setInterval(() => {
          if (stepIndex < steps.length) {
            setProcessingStep(steps[stepIndex]);
            stepIndex++;
          }
        }, 2000);

        const result = await convertPdfToMarkdown(selectedFile, false);
        clearInterval(stepInterval);

        if (result.success) {
          setProcessingStep("İşlem tamamlandı!");
          onSuccess(`Belge başarıyla Markdown formatına dönüştürüldü: ${result.markdown_filename}`);
          setSelectedFile(null);

          setTimeout(() => {
            onMarkdownFilesUpdate();
            setIsProcessing(false);
            setProcessingStep("");
          }, 1200);
        } else {
          onError(result.message || "Dönüştürme işlemi başarısız");
          setIsProcessing(false);
          setProcessingStep("");
        }
      }
    } catch (e: any) {
      onError(e.message || "İşlem başarısız");
      setIsProcessing(false);
      setProcessingStep("");
    }
  };

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
    const isMarkdown = file.name.toLowerCase().endsWith('.md');
    const supportedConvertTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // DOCX
      "application/vnd.openxmlformats-officedocument.presentationml.presentation", // PPTX
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // XLSX
    ];

    if (isMarkdown) {
      setUploadMode('direct');
      setSelectedFile(file);
      onError(""); // Clear any previous errors
    } else if (supportedConvertTypes.includes(file.type)) {
      setUploadMode('convert');
      setSelectedFile(file);
      onError(""); // Clear any previous errors
    } else {
      onError("Lütfen PDF, DOCX, PPTX, XLSX veya MD dosyası seçin");
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
  };

  const getAcceptedFiles = () => {
    return ".pdf,.docx,.pptx,.xlsx,.md";
  };

  const getFileTypeDescription = () => {
    if (uploadMode === 'direct') {
      return "Markdown dosyası direkt yüklenecek";
    }
    return "Belge Markdown formatına dönüştürülecek";
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-2 sm:p-4 z-50">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm sm:max-w-md md:max-w-lg lg:max-w-2xl xl:max-w-3xl max-h-[95vh] sm:max-h-[90vh] overflow-hidden">
        {/* Modal Header */}
        <div className="relative bg-gradient-to-r from-blue-600 to-purple-600 text-white p-3 sm:p-4 md:p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
                <UploadIcon className="w-8 h-8" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">Belge Yönetimi</h2>
                <p className="text-blue-100 mt-1">
                  Belgelerinizi yükleyin ve RAG sistemi için hazırlayın
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors"
            >
              <CloseIcon />
            </button>
          </div>
        </div>

        {/* Modal Content */}
        <div className="p-3 sm:p-4 md:p-6 lg:p-8">
          {isProcessing ? (
            // Processing State
            <div className="text-center py-6 sm:py-8 md:py-12">
              <div className="flex flex-col items-center gap-6">
                <div className="relative">
                  <div className="w-20 h-20 border-4 border-blue-200 rounded-full"></div>
                  <div className="absolute inset-0 w-20 h-20 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-3">
                    {uploadMode === 'direct' ? 'Markdown Yükleniyor' : 'Belge Dönüştürülüyor'}
                  </h3>
                  <p className="text-gray-600 mb-6 text-lg">{processingStep}</p>
                  <div className="bg-blue-50 border border-blue-200 text-blue-800 p-4 rounded-xl text-sm max-w-md mx-auto">
                    💡 Modal'ı kapatabilirsiniz - İşlem arka planda devam edecek
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // Upload Form
            <div className="space-y-4 sm:space-y-6 lg:space-y-8">
              {/* Upload Mode Selector */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                <div className={`p-4 sm:p-5 lg:p-6 rounded-xl border-2 cursor-pointer transition-all ${
                  uploadMode === 'convert' 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`} onClick={() => setUploadMode('convert')}>
                  <div className="flex items-center gap-3 mb-3">
                    <DocumentIcon />
                    <h3 className="font-semibold text-gray-800">Belge Dönüştür</h3>
                  </div>
                  <p className="text-sm text-gray-600">
                    PDF, DOCX, PPTX, XLSX belgelerini Markdown'a dönüştür
                  </p>
                </div>
                
                <div className={`p-4 sm:p-5 lg:p-6 rounded-xl border-2 cursor-pointer transition-all ${
                  uploadMode === 'direct' 
                    ? 'border-purple-500 bg-purple-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`} onClick={() => setUploadMode('direct')}>
                  <div className="flex items-center gap-3 mb-3">
                    <MarkdownIcon />
                    <h3 className="font-semibold text-gray-800">Markdown Yükle</h3>
                  </div>
                  <p className="text-sm text-gray-600">
                    Hazır Markdown (.md) dosyalarını direkt yükle
                  </p>
                </div>
              </div>

              {/* Drag & Drop Zone */}
              <div
                className={`relative border-2 border-dashed rounded-2xl p-6 sm:p-8 lg:p-12 text-center transition-all ${
                  selectedFile ? "cursor-default" : "cursor-pointer"
                } ${
                  isDragOver
                    ? uploadMode === 'direct' 
                      ? "border-purple-400 bg-purple-50" 
                      : "border-blue-400 bg-blue-50"
                    : selectedFile
                    ? "border-green-400 bg-green-50"
                    : uploadMode === 'direct'
                    ? "border-purple-200 hover:border-purple-300 hover:bg-purple-25"
                    : "border-blue-200 hover:border-blue-300 hover:bg-blue-25"
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {selectedFile ? (
                  <div className="flex flex-col items-center gap-6">
                    <div className="p-6 rounded-full bg-green-100">
                      {uploadMode === 'direct' ? <MarkdownIcon /> : <DocumentIcon />}
                    </div>
                    <div>
                      <div className="flex items-center justify-center gap-2 mb-2">
                        <CheckIcon />
                        <p className="text-xl font-semibold text-green-600">
                          Dosya Seçildi
                        </p>
                      </div>
                      <p className="text-gray-600 mb-2">
                        {selectedFile.name}
                      </p>
                      <p className="text-sm text-gray-500">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • {getFileTypeDescription()}
                      </p>
                      <button
                        onClick={handleRemoveFile}
                        className="mt-4 px-4 py-2 text-sm text-red-600 hover:text-red-800 hover:bg-red-50 rounded-lg transition-colors"
                      >
                        Dosyayı Değiştir
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-4 sm:gap-6">
                    <div className={`p-4 sm:p-6 lg:p-8 rounded-full ${
                      isDragOver 
                        ? uploadMode === 'direct' ? "bg-purple-100" : "bg-blue-100"
                        : "bg-gray-100"
                    }`}>
                      <UploadIcon className="w-12 h-12 sm:w-14 sm:h-14 lg:w-16 lg:h-16 text-gray-400" />
                    </div>
                    <div>
                      <p className="text-lg sm:text-xl lg:text-2xl font-semibold text-gray-800 mb-2">
                        Dosyanızı buraya sürükleyin
                      </p>
                      <p className="text-gray-600 mb-4">
                        {uploadMode === 'direct' 
                          ? "Markdown (.md) • Maks. 50MB"
                          : "PDF, DOCX, PPTX, XLSX • Maks. 50MB"
                        }
                      </p>
                      <div className="text-sm text-gray-500">
                        veya dosya seçmek için tıklayın
                      </div>
                    </div>
                  </div>
                )}
                <input
                  type="file"
                  onChange={handleFileSelect}
                  className="absolute inset-0 opacity-0 cursor-pointer"
                  accept={getAcceptedFiles()}
                  disabled={isProcessing}
                />
              </div>

              {/* File Select Button */}
              {!selectedFile && (
                <div className="text-center">
                  <button
                    type="button"
                    onClick={() => (document.querySelector('input[type="file"]') as HTMLInputElement)?.click()}
                    className={`inline-flex items-center gap-3 px-6 sm:px-8 py-3 sm:py-4 rounded-xl font-semibold text-white transition-all transform hover:scale-105 min-h-[44px] ${
                      uploadMode === 'direct'
                        ? 'bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700'
                        : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700'
                    }`}
                  >
                    <UploadIcon className="w-6 h-6" />
                    Dosya Seç
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-3 sm:px-4 md:px-6 lg:px-8 py-4 sm:py-6 bg-gray-50 border-t border-gray-200">
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-end">
            <button
              onClick={onClose}
              className="px-6 py-3 text-gray-600 hover:text-gray-800 font-medium transition-colors min-h-[44px] order-2 sm:order-1"
            >
              {isProcessing ? "Arka Planda Devam Et" : "İptal"}
            </button>
            {!isProcessing && selectedFile && (
              <button
                type="button"
                onClick={() => handleFileUpload()}
                className={`px-6 sm:px-8 py-3 rounded-xl font-semibold text-white transition-all transform hover:scale-105 min-h-[44px] order-1 sm:order-2 ${
                  uploadMode === 'direct'
                    ? 'bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700'
                    : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700'
                }`}
              >
                {uploadMode === 'direct' ? 'Markdown Yükle' : 'Dönüştür ve Yükle'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
