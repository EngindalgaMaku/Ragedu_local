"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import {
  FileText,
  Eye,
  Sparkles,
  Trash2,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  MoreHorizontal,
} from "lucide-react";
import type { Chunk } from "../../types/chunks.types";

export interface ChunksTableProps {
  chunks: Chunk[];
  selectedChunks: Set<string>;
  onChunkSelect: (chunkId: string) => void;
  onSelectAll: (selected: boolean) => void;
  onChunkView: (chunk: Chunk) => void;
  onChunkImprove?: (chunk: Chunk) => void;
  onChunkDelete?: (chunk: Chunk) => void;
  sortBy?: string;
  sortDirection?: "asc" | "desc";
  onSort?: (column: string) => void;
  loading?: boolean;
  disabled?: boolean;
}

interface SortableHeaderProps {
  column: string;
  children: React.ReactNode;
  sortBy?: string;
  sortDirection?: "asc" | "desc";
  onSort?: (column: string) => void;
  disabled?: boolean;
  className?: string;
}

function SortableHeader({
  column,
  children,
  sortBy,
  sortDirection,
  onSort,
  disabled,
  className,
}: SortableHeaderProps) {
  const handleClick = () => {
    if (!disabled && onSort) {
      onSort(column);
    }
  };

  const isActive = sortBy === column;
  const Icon =
    isActive && sortDirection === "asc"
      ? ArrowUp
      : isActive && sortDirection === "desc"
      ? ArrowDown
      : ArrowUpDown;

  return (
    <th
      className={cn(
        "px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider",
        onSort && !disabled && "cursor-pointer hover:bg-muted/50 select-none",
        className
      )}
      onClick={handleClick}
    >
      <div className="flex items-center gap-1">
        {children}
        {onSort && (
          <Icon
            className={cn(
              "w-3 h-3",
              isActive ? "text-primary" : "text-muted-foreground/50"
            )}
          />
        )}
      </div>
    </th>
  );
}

export default function ChunksTable({
  chunks,
  selectedChunks,
  onChunkSelect,
  onSelectAll,
  onChunkView,
  onChunkImprove,
  onChunkDelete,
  sortBy,
  sortDirection,
  onSort,
  loading = false,
  disabled = false,
}: ChunksTableProps) {
  const allSelected =
    chunks.length > 0 && chunks.every((chunk) => selectedChunks.has(chunk.id));
  const someSelected = chunks.some((chunk) => selectedChunks.has(chunk.id));

  const handleSelectAll = (checked: boolean) => {
    onSelectAll(checked);
  };

  const getDocumentDisplayName = (docName: string) => {
    return docName.replace(".md", "").replace(/^.*[\\\/]/, "");
  };

  const formatCharacterCount = (count: number) => {
    return count.toLocaleString("tr-TR");
  };

  if (chunks.length === 0) {
    return (
      <div className="text-center py-12 bg-muted/30 rounded-lg border border-dashed border-border">
        <FileText className="w-12 h-12 mx-auto mb-4 text-muted-foreground/50" />
        <h3 className="text-lg font-medium text-muted-foreground mb-2">
          Parça Bulunamadı
        </h3>
        <p className="text-sm text-muted-foreground">
          Filtrelerinizi değiştirerek daha fazla sonuç bulabilirsiniz
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <div className="overflow-x-auto">
        <table className="w-full divide-y divide-border">
          <thead className="bg-muted/50">
            <tr>
              <th className="px-4 py-3 w-12">
                <Checkbox
                  checked={allSelected}
                  ref={(el) => {
                    if (el) el.indeterminate = someSelected && !allSelected;
                  }}
                  onCheckedChange={handleSelectAll}
                  disabled={disabled || loading}
                  aria-label="Tümünü seç"
                />
              </th>

              <SortableHeader
                column="chunk_index"
                sortBy={sortBy}
                sortDirection={sortDirection}
                onSort={onSort}
                disabled={disabled || loading}
              >
                #
              </SortableHeader>

              <SortableHeader
                column="document_name"
                sortBy={sortBy}
                sortDirection={sortDirection}
                onSort={onSort}
                disabled={disabled || loading}
              >
                Döküman
              </SortableHeader>

              <SortableHeader
                column="character_count"
                sortBy={sortBy}
                sortDirection={sortDirection}
                onSort={onSort}
                disabled={disabled || loading}
              >
                Karakter
              </SortableHeader>

              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Durum
              </th>

              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                İçerik
              </th>

              <th className="px-4 py-3 text-right text-xs font-medium text-muted-foreground uppercase tracking-wider">
                İşlemler
              </th>
            </tr>
          </thead>

          <tbody className="bg-background divide-y divide-border">
            {chunks.map((chunk) => {
              const isSelected = selectedChunks.has(chunk.id);
              const isImproved = chunk.chunk_metadata?.llm_improved;

              return (
                <tr
                  key={chunk.id}
                  className={cn(
                    "hover:bg-muted/30 transition-colors",
                    isSelected && "bg-muted/20",
                    loading && "opacity-60"
                  )}
                >
                  <td className="px-4 py-3">
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={() => onChunkSelect(chunk.id)}
                      disabled={disabled || loading}
                    />
                  </td>

                  <td className="px-4 py-3 text-sm font-medium text-foreground">
                    #{chunk.chunk_index}
                  </td>

                  <td className="px-4 py-3 text-sm text-foreground max-w-[200px]">
                    <div className="truncate" title={chunk.document_name}>
                      {getDocumentDisplayName(chunk.document_name)}
                    </div>
                  </td>

                  <td className="px-4 py-3 text-sm text-muted-foreground">
                    {formatCharacterCount(chunk.character_count)}
                  </td>

                  <td className="px-4 py-3">
                    {isImproved ? (
                      <Badge variant="secondary" className="gap-1 text-xs">
                        <Sparkles className="w-3 h-3" />
                        İyileştirilmiş
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="text-xs">
                        Orijinal
                      </Badge>
                    )}
                  </td>

                  <td className="px-4 py-3 text-sm text-muted-foreground max-w-[300px]">
                    <div className="truncate" title={chunk.chunk_text}>
                      {chunk.chunk_text}
                    </div>
                  </td>

                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onChunkView(chunk)}
                        disabled={disabled || loading}
                        className="h-8 w-8 p-0"
                        title="İçeriği görüntüle"
                      >
                        <Eye className="w-4 h-4" />
                      </Button>

                      {onChunkImprove && !isImproved && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onChunkImprove(chunk)}
                          disabled={disabled || loading}
                          className="h-8 w-8 p-0 text-violet-600 hover:text-violet-700 hover:bg-violet-50"
                          title="LLM ile iyileştir"
                        >
                          <Sparkles className="w-4 h-4" />
                        </Button>
                      )}

                      {onChunkDelete && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onChunkDelete(chunk)}
                          disabled={disabled || loading}
                          className="h-8 w-8 p-0 text-red-600 hover:text-red-700 hover:bg-red-50"
                          title="Parçayı sil"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {loading && (
        <div className="absolute inset-0 bg-background/50 flex items-center justify-center">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary border-t-transparent" />
            Yükleniyor...
          </div>
        </div>
      )}
    </div>
  );
}

// Compact table variant for smaller screens
export function CompactChunksTable({
  chunks,
  onChunkView,
  onChunkImprove,
  loading = false,
  disabled = false,
}: {
  chunks: Chunk[];
  onChunkView: (chunk: Chunk) => void;
  onChunkImprove?: (chunk: Chunk) => void;
  loading?: boolean;
  disabled?: boolean;
}) {
  if (chunks.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">Parça bulunamadı</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {chunks.map((chunk) => {
        const isImproved = chunk.chunk_metadata?.llm_improved;

        return (
          <div
            key={chunk.id}
            className={cn(
              "p-3 rounded-lg border border-border bg-background hover:bg-muted/30 transition-colors",
              loading && "opacity-60"
            )}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant="outline" className="text-xs">
                    #{chunk.chunk_index}
                  </Badge>

                  {isImproved ? (
                    <Badge variant="secondary" className="gap-1 text-xs">
                      <Sparkles className="w-3 h-3" />
                      İyileştirilmiş
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="text-xs">
                      Orijinal
                    </Badge>
                  )}
                </div>

                <h4 className="text-sm font-medium text-foreground truncate mb-1">
                  {chunk.document_name.replace(".md", "")}
                </h4>

                <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
                  {chunk.chunk_text}
                </p>

                <div className="text-xs text-muted-foreground">
                  {chunk.character_count.toLocaleString("tr-TR")} karakter
                </div>
              </div>

              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onChunkView(chunk)}
                  disabled={disabled || loading}
                  className="h-8 w-8 p-0"
                >
                  <Eye className="w-4 h-4" />
                </Button>

                {onChunkImprove && !isImproved && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onChunkImprove(chunk)}
                    disabled={disabled || loading}
                    className="h-8 w-8 p-0 text-violet-600 hover:text-violet-700"
                  >
                    <Sparkles className="w-4 h-4" />
                  </Button>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
