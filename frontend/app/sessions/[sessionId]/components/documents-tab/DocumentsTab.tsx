"use client";

import React, { useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  RefreshCw,
  FileText,
  Upload,
  Settings,
  Info,
  Brain,
  Scissors,
  Layers,
  Database,
  HelpCircle,
} from "lucide-react";

import { useSessionData, useSessionChunks } from "../../hooks/useSessionData";
import { useChunksErrorHandler } from "../../hooks/useErrorHandler";
import { useExtendedChunksStore } from "../../stores/chunksStore";
import { useSessionStats } from "../../hooks/useSessionStats";

import DocumentUploadSection from "./DocumentUploadSection";
import ProcessingStatus from "./ProcessingStatus";
import SessionInfo from "./SessionInfo";
import { useFileUpload } from "../../hooks/useFileUpload";

interface DocumentsTabProps {
  sessionId: string;
}

export default function DocumentsTab({ sessionId }: DocumentsTabProps) {
  const {
    chunksLoading,
    chunksError,
    hasChunks,
    chunksCount,
    loadChunks,
    clearChunksError,
  } = useSessionData();

  const { chunks } = useSessionChunks();
  const { handleChunksError } = useChunksErrorHandler();
  const chunksStore = useExtendedChunksStore();

  // NEW: Use lightweight statistics that load immediately
  const {
    stats,
    loading: statsLoading,
    refreshStats,
  } = useSessionStats(sessionId);

  const {
    isProcessing,
    processingStep,
    selectedFiles,
    uploadProgress,
    availableFiles,
    processedFiles,
    refreshFiles,
    handleFileUpload,
    clearProcessingState,
  } = useFileUpload(sessionId);

  // Statistics: Use lightweight stats when available, fallback to chunks store for detailed view
  const documentInfo = chunksStore.getDocumentInfo();
  const hasDetailedChunks = documentInfo.length > 0;

  // Priority: detailed chunks data > lightweight stats > zeros
  const totalDocuments = hasDetailedChunks
    ? documentInfo.length
    : stats?.total_documents || 0;
  const totalChunks = hasDetailedChunks
    ? chunksCount
    : stats?.total_chunks || 0;
  const totalCharacters = hasDetailedChunks
    ? documentInfo.reduce((sum, doc) => sum + doc.total_characters, 0)
    : stats?.total_characters || 0;
  const llmImprovedCount = hasDetailedChunks
    ? documentInfo.reduce((sum, doc) => sum + doc.llm_improved_chunks, 0)
    : stats?.llm_improved || 0;

  // DEBUG: Add logging to validate statistics sources
  console.log(`🔍 [STATS DEBUG] DocumentsTab for session ${sessionId}:`);
  console.log(`📊 chunksStore.chunks.length: ${chunksStore.chunks.length}`);
  console.log(`📊 hasDetailedChunks: ${hasDetailedChunks}`);
  console.log(`📊 stats from API:`, stats);
  console.log(
    `📊 Final values: docs=${totalDocuments}, chunks=${totalChunks}, chars=${totalCharacters}, llm=${llmImprovedCount}`
  );
  console.log(
    `📊 statsLoading: ${statsLoading}, chunksLoading: ${chunksLoading}`
  );

  // DISABLED: Load chunks when component mounts
  // This was causing CORS errors and automatic API calls
  // Original system uses manual upload, not automatic chunk loading
  // useEffect(() => {
  //   if (sessionId && !chunksLoading && chunksCount === 0) {
  //     loadChunks();
  //   }
  // }, [sessionId, chunksLoading, chunksCount, loadChunks]);

  // Auto-refresh files when processing completes
  useEffect(() => {
    if (!isProcessing && selectedFiles.length > 0) {
      const timer = setTimeout(() => {
        refreshFiles();
        loadChunks();
        clearProcessingState();
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [
    isProcessing,
    selectedFiles.length,
    refreshFiles,
    loadChunks,
    clearProcessingState,
  ]);

  const handleRefresh = async () => {
    try {
      clearChunksError();
      await Promise.all([loadChunks(), refreshFiles(), refreshStats()]);
    } catch (error) {
      handleChunksError(error, "refresh");
    }
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h2 className="text-2xl font-bold text-foreground">Belgeler</h2>
          <p className="text-muted-foreground">
            Dosya yükleme, işleme durumu ve oturum bilgileri
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={chunksLoading || isProcessing}
            className="gap-2"
          >
            <RefreshCw
              className={`w-4 h-4 ${chunksLoading ? "animate-spin" : ""}`}
            />
            Yenile
          </Button>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Toplam Belge</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalDocuments}</div>
            <p className="text-xs text-muted-foreground">
              İşlenmiş dosya sayısı
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Toplam Parça</CardTitle>
            <Upload className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalChunks}</div>
            <p className="text-xs text-muted-foreground">
              Metin parçası sayısı
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Toplam Karakter
            </CardTitle>
            <Settings className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {totalCharacters.toLocaleString("tr-TR")}
            </div>
            <p className="text-xs text-muted-foreground">
              İşlenmiş içerik miktarı
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              LLM İyileştirilmiş
            </CardTitle>
            <Info className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{llmImprovedCount}</div>
            <p className="text-xs text-muted-foreground">
              {totalChunks > 0
                ? `%${Math.round((llmImprovedCount / totalChunks) * 100)}`
                : "%0"}{" "}
              iyileştirilmiş
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Processing Settings Information */}
      <Card className="bg-gradient-to-r from-blue-50/50 to-indigo-50/50 dark:from-blue-950/20 dark:to-indigo-950/20 border-blue-200/50 dark:border-blue-800/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-blue-800 dark:text-blue-200">
            <Settings className="w-5 h-5" />
            Sistem İşleme Ayarları
          </CardTitle>
          <p className="text-sm text-blue-600 dark:text-blue-300">
            Bu oturumdaki tüm belgeler aşağıdaki ayarlarla işlenmektedir
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Chunking Strategy */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Scissors className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <h3 className="font-semibold text-foreground">
                  Parçalama Stratejisi
                </h3>
                <HelpCircle className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="bg-blue-100/50 dark:bg-blue-900/20 rounded-lg p-3 border border-blue-200/30 dark:border-blue-700/30">
                <div className="font-medium text-blue-800 dark:text-blue-200 mb-1">
                  Lightweight Turkish
                </div>
                <p className="text-xs text-blue-600 dark:text-blue-300 leading-relaxed">
                  Türkçe dil yapısına optimize edilmiş akıllı parçalama. Cümle
                  sınırlarını korur, anlamsal bütünlüğü sağlar ve Türkçe dil
                  bilgisine uygun şekilde metin parçalar.
                </p>
              </div>
            </div>

            {/* Chunk Size */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-green-600 dark:text-green-400" />
                <h3 className="font-semibold text-foreground">Parça Boyutu</h3>
                <HelpCircle className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="bg-green-100/50 dark:bg-green-900/20 rounded-lg p-3 border border-green-200/30 dark:border-green-700/30">
                <div className="font-medium text-green-800 dark:text-green-200 mb-1">
                  Anlamsal (~400-1200 karakter)
                </div>
                <p className="text-xs text-green-600 dark:text-green-300 leading-relaxed">
                  İçerik yapısına göre dinamik boyutlandırma. Paragraf ve cümle
                  bütünlüğünü korurken optimal arama ve anlam çıkarma
                  performansı için ideal boyutta parçalar oluşturur.
                </p>
              </div>
            </div>

            {/* Chunk Overlap */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                <h3 className="font-semibold text-foreground">
                  Parça Çakışması
                </h3>
                <HelpCircle className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="bg-purple-100/50 dark:bg-purple-900/20 rounded-lg p-3 border border-purple-200/30 dark:border-purple-700/30">
                <div className="font-medium text-purple-800 dark:text-purple-200 mb-1">
                  Otomatik (cümle bazlı)
                </div>
                <p className="text-xs text-purple-600 dark:text-purple-300 leading-relaxed">
                  Parçalar arasında anlamsal süreklilik sağlar. Cümle
                  sınırlarında doğal çakışma oluşturarak bilgi kaybını önler ve
                  sorgu-cevap kalitesini artırır.
                </p>
              </div>
            </div>

            {/* Embedding Model */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                <h3 className="font-semibold text-foreground">
                  Embedding Modeli
                </h3>
                <HelpCircle className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="bg-amber-100/50 dark:bg-amber-900/20 rounded-lg p-3 border border-amber-200/30 dark:border-amber-700/30">
                <div className="font-medium text-amber-800 dark:text-amber-200 mb-1">
                  nomic-embed-text
                </div>
                <p className="text-xs text-amber-600 dark:text-amber-300 leading-relaxed">
                  Yüksek performanslı çok dilli embedding modeli. Türkçe
                  metinler için optimize edilmiş anlamsal vektör temsilciler
                  oluşturarak arama ve benzerlik hesaplama kalitesini artırır.
                </p>
              </div>
            </div>
          </div>

          {/* Additional Info */}
          <div className="mt-6 p-4 bg-gradient-to-r from-slate-100/50 to-gray-100/50 dark:from-slate-800/30 dark:to-gray-800/30 rounded-lg border border-slate-200/30 dark:border-slate-700/30">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-slate-600 dark:text-slate-400 mt-0.5 flex-shrink-0" />
              <div className="space-y-2">
                <h4 className="font-medium text-slate-800 dark:text-slate-200">
                  Önemli Bilgiler
                </h4>
                <ul className="text-xs text-slate-600 dark:text-slate-300 space-y-1 leading-relaxed">
                  <li>
                    • Bu ayarlar sistem genelinde sabitlenmiştir ve
                    değiştirilemez
                  </li>
                  <li>• Tüm belgeler aynı kalitede ve tutarlılıkta işlenir</li>
                  <li>
                    • Ayarlar Türkçe eğitim içerikleri için özel olarak optimize
                    edilmiştir
                  </li>
                  <li>
                    • İşleme süresi belge boyutuna ve karmaşıklığına göre
                    değişiklik gösterir
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error Display */}
      {chunksError && (
        <Alert variant="destructive">
          <AlertDescription className="flex items-center justify-between">
            <span>{chunksError}</span>
            <Button variant="outline" size="sm" onClick={clearChunksError}>
              Kapat
            </Button>
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - File Upload */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Dosya Yükleme
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <DocumentUploadSection
                sessionId={sessionId}
                availableFiles={availableFiles}
                processedFiles={processedFiles}
                selectedFiles={selectedFiles}
                onFileUpload={handleFileUpload}
                disabled={isProcessing}
                loading={chunksLoading}
              />
            </CardContent>
          </Card>

          {/* Processing Status */}
          {(isProcessing || processingStep) && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="w-5 h-5" />
                  İşlem Durumu
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ProcessingStatus
                  isProcessing={isProcessing}
                  processingStep={processingStep}
                  selectedFiles={selectedFiles}
                  uploadProgress={uploadProgress}
                />
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Column - Session Info */}
        <div>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="w-5 h-5" />
                Oturum Bilgileri
              </CardTitle>
            </CardHeader>
            <CardContent>
              <SessionInfo sessionId={sessionId} />
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Document List */}
      {hasChunks && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              İşlenmiş Belgeler
              <Badge variant="secondary" className="ml-2">
                {totalDocuments}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {documentInfo.map((doc, index) => (
                <div
                  key={doc.name}
                  className="flex items-center justify-between p-4 rounded-lg border bg-muted/30"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <FileText className="w-4 h-4 text-muted-foreground" />
                      <div>
                        <h3 className="font-medium text-foreground">
                          {doc.name.replace(".md", "")}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          {doc.chunk_count} parça •{" "}
                          {doc.total_characters.toLocaleString("tr-TR")}{" "}
                          karakter
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {doc.llm_improved_chunks > 0 && (
                      <Badge variant="outline" className="text-xs">
                        {doc.llm_improved_chunks} LLM
                      </Badge>
                    )}
                    <Badge
                      variant={
                        doc.status === "processed" ? "default" : "secondary"
                      }
                      className="text-xs"
                    >
                      {doc.status === "processed" ? "İşlenmiş" : "Bekliyor"}
                    </Badge>
                  </div>
                </div>
              ))}

              {documentInfo.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>Henüz işlenmiş belge bulunmuyor</p>
                  <p className="text-sm">
                    Yukarıdan dosya yükleyerek başlayabilirsiniz
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
