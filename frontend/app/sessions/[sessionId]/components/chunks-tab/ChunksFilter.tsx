"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  Search,
  Filter,
  FileText,
  X,
  FolderOpen,
  Sparkles,
  ChevronDown,
} from "lucide-react";

export interface ChunksFilterProps {
  // Filter values
  searchTerm: string;
  selectedDocumentFilter: string | null;
  statusFilter: "all" | "improved" | "original";
  showAllFiles: boolean;

  // Available options
  availableDocuments: string[];
  filteredCount: number;
  totalCount: number;

  // Change handlers
  onSearchChange: (value: string) => void;
  onDocumentFilterChange: (document: string | null) => void;
  onStatusFilterChange: (status: "all" | "improved" | "original") => void;
  onShowAllFilesChange: (show: boolean) => void;
  onClearFilters: () => void;

  // Loading state
  disabled?: boolean;
}

export default function ChunksFilter({
  searchTerm,
  selectedDocumentFilter,
  statusFilter,
  showAllFiles,
  availableDocuments,
  filteredCount,
  totalCount,
  onSearchChange,
  onDocumentFilterChange,
  onStatusFilterChange,
  onShowAllFilesChange,
  onClearFilters,
  disabled = false,
}: ChunksFilterProps) {
  // Show limited documents or all based on showAllFiles state
  const documentsToShow = showAllFiles
    ? availableDocuments
    : availableDocuments.slice(0, 3);

  const hasActiveFilters =
    searchTerm || selectedDocumentFilter || statusFilter !== "all";

  const getDocumentDisplayName = (docName: string) => {
    return docName.replace(".md", "").replace(/^.*[\\\/]/, "");
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-primary" />
            <CardTitle className="text-base">Filtreleme</CardTitle>
            <Badge variant="secondary" className="text-xs">
              {filteredCount}/{totalCount}
            </Badge>
          </div>
          {hasActiveFilters && (
            <Button
              variant="outline"
              size="sm"
              onClick={onClearFilters}
              disabled={disabled}
              className="gap-1 text-xs"
            >
              <X className="w-3 h-3" />
              Temizle
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Search Input */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Parça içeriği, dosya adı ara..."
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            disabled={disabled}
            className="pl-10 pr-4"
          />
          {searchTerm && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onSearchChange("")}
              disabled={disabled}
              className="absolute right-1 top-1/2 transform -translate-y-1/2 h-7 w-7 p-0 hover:bg-muted"
            >
              <X className="w-3 h-3" />
            </Button>
          )}
        </div>

        {/* Status Filter */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-violet-600" />
            <span className="text-sm font-medium">Durum Filtresi</span>
          </div>
          <div className="flex gap-2 flex-wrap">
            <Button
              variant={statusFilter === "all" ? "default" : "outline"}
              size="sm"
              onClick={() => onStatusFilterChange("all")}
              disabled={disabled}
              className="gap-2 text-sm"
            >
              <div className="w-2 h-2 rounded-full bg-muted-foreground" />
              Tümü
            </Button>
            <Button
              variant={statusFilter === "improved" ? "default" : "outline"}
              size="sm"
              onClick={() => onStatusFilterChange("improved")}
              disabled={disabled}
              className="gap-2 text-sm"
            >
              <div className="w-2 h-2 rounded-full bg-violet-600" />
              İyileştirilmiş
            </Button>
            <Button
              variant={statusFilter === "original" ? "default" : "outline"}
              size="sm"
              onClick={() => onStatusFilterChange("original")}
              disabled={disabled}
              className="gap-2 text-sm"
            >
              <div className="w-2 h-2 rounded-full bg-blue-600" />
              Orijinal
            </Button>
          </div>
        </div>

        {/* Document Filter */}
        {availableDocuments.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <FolderOpen className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium">Dosya Filtresi</span>
              <Badge variant="outline" className="text-xs">
                {selectedDocumentFilter
                  ? `1/${availableDocuments.length}`
                  : `${availableDocuments.length} dosya`}
              </Badge>
            </div>

            {/* All Files Button */}
            <Button
              variant={selectedDocumentFilter === null ? "default" : "outline"}
              size="sm"
              onClick={() => onDocumentFilterChange(null)}
              disabled={disabled}
              className="w-full justify-start gap-2 text-sm"
            >
              <FileText className="w-3 h-3" />
              Tüm Dosyalar
              <Badge variant="secondary" className="ml-auto text-xs">
                {totalCount}
              </Badge>
            </Button>

            {/* Document List */}
            <div className="space-y-2">
              <div className="flex flex-wrap gap-2">
                {documentsToShow.map((docName) => {
                  const docChunkCount = availableDocuments.filter(
                    (name) => name === docName
                  ).length;
                  const isActive = selectedDocumentFilter === docName;
                  const displayName = getDocumentDisplayName(docName);

                  return (
                    <Button
                      key={docName}
                      variant={isActive ? "default" : "outline"}
                      size="sm"
                      onClick={() =>
                        onDocumentFilterChange(isActive ? null : docName)
                      }
                      disabled={disabled}
                      className={cn(
                        "gap-2 text-xs max-w-[200px] justify-start",
                        isActive && "shadow-md"
                      )}
                      title={docName}
                    >
                      <FileText className="w-3 h-3 flex-shrink-0" />
                      <span className="truncate">{displayName}</span>
                      <Badge
                        variant={isActive ? "secondary" : "outline"}
                        className="ml-auto text-xs"
                      >
                        {docChunkCount}
                      </Badge>
                    </Button>
                  );
                })}
              </div>

              {/* Show More/Less Toggle */}
              {availableDocuments.length > 3 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onShowAllFilesChange(!showAllFiles)}
                  disabled={disabled}
                  className="w-full gap-2 text-xs"
                >
                  {showAllFiles ? (
                    <>
                      Daha Az Göster
                      <ChevronDown className="w-3 h-3 rotate-180" />
                    </>
                  ) : (
                    <>
                      +{availableDocuments.length - 3} Dosya Daha
                      <ChevronDown className="w-3 h-3" />
                    </>
                  )}
                </Button>
              )}
            </div>
          </div>
        )}

        {/* Active Filters Summary */}
        {hasActiveFilters && (
          <div className="pt-2 border-t border-border">
            <div className="flex flex-wrap gap-1">
              {searchTerm && (
                <Badge variant="secondary" className="gap-1 text-xs">
                  Arama: "{searchTerm}"
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onSearchChange("")}
                    disabled={disabled}
                    className="h-auto p-0 hover:bg-transparent"
                  >
                    <X className="w-2 h-2" />
                  </Button>
                </Badge>
              )}
              {selectedDocumentFilter && (
                <Badge variant="secondary" className="gap-1 text-xs">
                  Dosya: {getDocumentDisplayName(selectedDocumentFilter)}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onDocumentFilterChange(null)}
                    disabled={disabled}
                    className="h-auto p-0 hover:bg-transparent"
                  >
                    <X className="w-2 h-2" />
                  </Button>
                </Badge>
              )}
              {statusFilter !== "all" && (
                <Badge variant="secondary" className="gap-1 text-xs">
                  Durum:{" "}
                  {statusFilter === "improved" ? "İyileştirilmiş" : "Orijinal"}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onStatusFilterChange("all")}
                    disabled={disabled}
                    className="h-auto p-0 hover:bg-transparent"
                  >
                    <X className="w-2 h-2" />
                  </Button>
                </Badge>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
