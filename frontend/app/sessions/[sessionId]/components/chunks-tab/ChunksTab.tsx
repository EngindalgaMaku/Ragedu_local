"use client";

import React, { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import {
  FileText,
  RefreshCw,
  Sparkles,
  Eye,
  EyeOff,
  BarChart3,
  TrendingUp,
  Layers,
  Zap,
  AlertCircle,
  CheckCircle,
} from "lucide-react";

// Import our custom components
import { useChunksData } from "../../hooks/useChunksData";
import ChunksFilter from "./ChunksFilter";
import ChunksTable from "./ChunksTable";
import ChunksMobileView from "./ChunksMobileView";
import ChunksPagination, { CompactChunksPagination } from "./ChunksPagination";
import ChunkModal from "./ChunkModal";
import { ChunksLoadingSkeleton } from "../../components/shared/LoadingSpinner";
import type { Chunk } from "../../types/chunks.types";

export interface ChunksTabProps {
  sessionId: string;
}

export default function ChunksTab({ sessionId }: ChunksTabProps) {
  // Hooks
  const chunksData = useChunksData({
    sessionId,
    initialPageSize: 10,
    enableAutoRefresh: false,
  });

  // Local state
  const [selectedChunks, setSelectedChunks] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<"table" | "cards">("table");
  const [sortBy, setSortBy] = useState<string>("chunk_index");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  // Handlers
  const handleChunkSelect = useCallback((chunkId: string) => {
    setSelectedChunks((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(chunkId)) {
        newSet.delete(chunkId);
      } else {
        newSet.add(chunkId);
      }
      return newSet;
    });
  }, []);

  const handleSelectAll = useCallback(
    (selected: boolean) => {
      if (selected) {
        const allIds = new Set(
          chunksData.paginatedChunks.map((chunk) => chunk.id)
        );
        setSelectedChunks(allIds);
      } else {
        setSelectedChunks(new Set());
      }
    },
    [chunksData.paginatedChunks]
  );

  const handleSort = useCallback(
    (column: string) => {
      if (sortBy === column) {
        setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
      } else {
        setSortBy(column);
        setSortDirection("asc");
      }
    },
    [sortBy]
  );

  const handleSaveChunk = useCallback(
    async (chunkId: string, improvedContent: string) => {
      // TODO: Implement actual save functionality
      console.log("Saving chunk:", chunkId, improvedContent);
    },
    []
  );

  const handleDeleteChunk = useCallback(async (chunk: Chunk) => {
    if (
      confirm(
        `"${chunk.document_name}" belgesindeki ${chunk.chunk_index}. parçayı silmek istediğinizden emin misiniz?`
      )
    ) {
      // TODO: Implement actual delete functionality
      console.log("Deleting chunk:", chunk.id);
    }
  }, []);

  const handleImproveChunk = useCallback(
    async (chunk: Chunk) => {
      chunksData.openChunkModal(chunk);
    },
    [chunksData]
  );

  const handleBulkAction = useCallback(
    (action: string) => {
      if (selectedChunks.size === 0) return;

      switch (action) {
        case "delete":
          if (
            confirm(
              `Seçili ${selectedChunks.size} parçayı silmek istediğinizden emin misiniz?`
            )
          ) {
            // TODO: Implement bulk delete
            console.log("Bulk deleting:", Array.from(selectedChunks));
            setSelectedChunks(new Set());
          }
          break;
        case "improve":
          // TODO: Implement bulk improvement
          console.log("Bulk improving:", Array.from(selectedChunks));
          break;
        default:
          break;
      }
    },
    [selectedChunks]
  );

  // Statistics calculations
  const stats = {
    totalChunks: chunksData.totalChunks,
    filteredChunks: chunksData.filteredCount,
    improvedChunks: chunksData.improvedChunksCount,
    improvementRate:
      chunksData.totalChunks > 0
        ? Math.round(
            (chunksData.improvedChunksCount / chunksData.totalChunks) * 100
          )
        : 0,
    avgLength:
      chunksData.chunks.length > 0
        ? Math.round(
            chunksData.chunks.reduce(
              (sum, chunk) => sum + chunk.character_count,
              0
            ) / chunksData.chunks.length
          )
        : 0,
    documentsCount: chunksData.availableDocuments.length,
  };

  if (chunksData.loading && chunksData.chunks.length === 0) {
    return <ChunksLoadingSkeleton />;
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header Section */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-2xl font-bold text-foreground">
            Döküman Parçaları
          </h2>
          <p className="text-muted-foreground">
            Ayrıştırılmış metin parçalarını görüntüleyin ve yönetin
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              setViewMode(viewMode === "table" ? "cards" : "table")
            }
            className="gap-2"
          >
            {viewMode === "table" ? (
              <>
                <Layers className="w-4 h-4" />
                Kart Görünümü
              </>
            ) : (
              <>
                <BarChart3 className="w-4 h-4" />
                Tablo Görünümü
              </>
            )}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={chunksData.refreshChunks}
            disabled={chunksData.loading}
            className="gap-2"
          >
            <RefreshCw
              className={cn("w-4 h-4", chunksData.loading && "animate-spin")}
            />
            Yenile
          </Button>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Toplam Parça</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalChunks}</div>
            <p className="text-xs text-muted-foreground">
              {stats.documentsCount} belgeden
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Filtrelenmiş</CardTitle>
            <Eye className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.filteredChunks}</div>
            <p className="text-xs text-muted-foreground">
              {stats.totalChunks > 0
                ? Math.round((stats.filteredChunks / stats.totalChunks) * 100)
                : 0}
              % görünür
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              LLM İyileştirilmiş
            </CardTitle>
            <Sparkles className="h-4 w-4 text-violet-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-violet-600">
              {stats.improvedChunks}
            </div>
            <p className="text-xs text-muted-foreground">
              %{stats.improvementRate} iyileştirme oranı
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Ortalama Uzunluk
            </CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats.avgLength.toLocaleString("tr-TR")}
            </div>
            <p className="text-xs text-muted-foreground">karakter/parça</p>
          </CardContent>
        </Card>
      </div>

      {/* Error Display */}
      {chunksData.error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="flex items-center justify-between">
            <span>{chunksData.error}</span>
            <Button
              variant="outline"
              size="sm"
              onClick={chunksData.clearMessages}
            >
              Kapat
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Success Display */}
      {chunksData.success && (
        <Alert className="border-green-200 bg-green-50 dark:bg-green-900/20">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="flex items-center justify-between text-green-800 dark:text-green-200">
            <span>{chunksData.success}</span>
            <Button
              variant="outline"
              size="sm"
              onClick={chunksData.clearMessages}
            >
              Kapat
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Filter Section */}
      <ChunksFilter
        searchTerm={chunksData.searchTerm}
        selectedDocumentFilter={chunksData.selectedDocumentFilter}
        statusFilter={chunksData.statusFilter}
        showAllFiles={chunksData.showAllFiles}
        availableDocuments={chunksData.availableDocuments}
        filteredCount={chunksData.filteredCount}
        totalCount={chunksData.totalChunks}
        onSearchChange={chunksData.setSearchTerm}
        onDocumentFilterChange={chunksData.setSelectedDocumentFilter}
        onStatusFilterChange={chunksData.setStatusFilter}
        onShowAllFilesChange={chunksData.setShowAllFiles}
        onClearFilters={chunksData.clearFilters}
        disabled={chunksData.loading}
      />

      {/* Bulk Actions */}
      {selectedChunks.size > 0 && (
        <Card className="border-primary/20 bg-primary/5">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Badge variant="secondary">
                  {selectedChunks.size} parça seçili
                </Badge>
                <span className="text-sm text-muted-foreground">
                  Toplu işlem yapabilirsiniz
                </span>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleBulkAction("improve")}
                  className="gap-1"
                >
                  <Sparkles className="w-3 h-3" />
                  İyileştir
                </Button>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleBulkAction("delete")}
                  className="gap-1 text-red-600 hover:text-red-700"
                >
                  <AlertCircle className="w-3 h-3" />
                  Sil
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedChunks(new Set())}
                >
                  İptal
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Content */}
      <Card className="overflow-hidden">
        {chunksData.chunks.length === 0 && !chunksData.loading ? (
          <CardContent className="py-12">
            <div className="text-center">
              <FileText className="w-16 h-16 mx-auto mb-4 text-muted-foreground/30" />
              <h3 className="text-lg font-medium text-muted-foreground mb-2">
                Henüz Parça Bulunmuyor
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                Belge yükleyerek parçalar oluşturabilirsiniz
              </p>
              <Button
                variant="outline"
                onClick={() => window.location.reload()}
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Sayfayı Yenile
              </Button>
            </div>
          </CardContent>
        ) : (
          <>
            {/* Desktop/Mobile View Toggle */}
            <div className="block lg:hidden">
              <ChunksMobileView
                chunks={chunksData.paginatedChunks}
                selectedChunks={selectedChunks}
                onChunkSelect={handleChunkSelect}
                onChunkView={chunksData.openChunkModal}
                onChunkImprove={handleImproveChunk}
                onChunkDelete={handleDeleteChunk}
                loading={chunksData.loading}
                disabled={chunksData.processing}
                showSelection={true}
              />
            </div>

            <div className="hidden lg:block">
              {viewMode === "table" ? (
                <ChunksTable
                  chunks={chunksData.paginatedChunks}
                  selectedChunks={selectedChunks}
                  onChunkSelect={handleChunkSelect}
                  onSelectAll={handleSelectAll}
                  onChunkView={chunksData.openChunkModal}
                  onChunkImprove={handleImproveChunk}
                  onChunkDelete={handleDeleteChunk}
                  sortBy={sortBy}
                  sortDirection={sortDirection}
                  onSort={handleSort}
                  loading={chunksData.loading}
                  disabled={chunksData.processing}
                />
              ) : (
                <div className="p-4">
                  <ChunksMobileView
                    chunks={chunksData.paginatedChunks}
                    selectedChunks={selectedChunks}
                    onChunkSelect={handleChunkSelect}
                    onChunkView={chunksData.openChunkModal}
                    onChunkImprove={handleImproveChunk}
                    onChunkDelete={handleDeleteChunk}
                    loading={chunksData.loading}
                    disabled={chunksData.processing}
                    showSelection={true}
                  />
                </div>
              )}
            </div>

            {/* Pagination */}
            {chunksData.totalPages > 1 && (
              <div className="block sm:hidden">
                <CompactChunksPagination
                  currentPage={chunksData.currentPage}
                  totalPages={chunksData.totalPages}
                  onPageChange={chunksData.setCurrentPage}
                  disabled={chunksData.loading || chunksData.processing}
                />
              </div>
            )}

            <div className="hidden sm:block">
              <ChunksPagination
                currentPage={chunksData.currentPage}
                totalPages={chunksData.totalPages}
                chunksPerPage={chunksData.chunksPerPage}
                totalItems={chunksData.totalChunks}
                filteredItems={chunksData.filteredCount}
                onPageChange={chunksData.setCurrentPage}
                onPageSizeChange={chunksData.setChunksPerPage}
                disabled={chunksData.loading || chunksData.processing}
              />
            </div>
          </>
        )}
      </Card>

      {/* Chunk Modal */}
      <ChunkModal
        chunk={chunksData.selectedChunk}
        isOpen={chunksData.isModalOpen}
        onClose={chunksData.closeChunkModal}
        onSave={handleSaveChunk}
        sessionId={sessionId}
        disabled={chunksData.processing}
      />
    </div>
  );
}
