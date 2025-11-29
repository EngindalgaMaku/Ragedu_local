"use client";

import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import {
  FileText,
  Eye,
  Sparkles,
  Trash2,
  MoreVertical,
  Calendar,
  Hash,
} from "lucide-react";
import type { Chunk } from "../../types/chunks.types";

export interface ChunksMobileViewProps {
  chunks: Chunk[];
  selectedChunks: Set<string>;
  onChunkSelect: (chunkId: string) => void;
  onChunkView: (chunk: Chunk) => void;
  onChunkImprove?: (chunk: Chunk) => void;
  onChunkDelete?: (chunk: Chunk) => void;
  loading?: boolean;
  disabled?: boolean;
  showSelection?: boolean;
}

export default function ChunksMobileView({
  chunks,
  selectedChunks,
  onChunkSelect,
  onChunkView,
  onChunkImprove,
  onChunkDelete,
  loading = false,
  disabled = false,
  showSelection = true,
}: ChunksMobileViewProps) {
  const getDocumentDisplayName = (docName: string) => {
    return docName.replace(".md", "").replace(/^.*[\\\/]/, "");
  };

  const formatCharacterCount = (count: number) => {
    if (count < 1000) return `${count}`;
    if (count < 1000000) return `${(count / 1000).toFixed(1)}k`;
    return `${(count / 1000000).toFixed(1)}M`;
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString("tr-TR", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
      });
    } catch {
      return "";
    }
  };

  if (chunks.length === 0) {
    return (
      <div className="text-center py-12 px-4">
        <div className="max-w-sm mx-auto">
          <FileText className="w-16 h-16 mx-auto mb-4 text-muted-foreground/30" />
          <h3 className="text-lg font-medium text-muted-foreground mb-2">
            Parça Bulunamadı
          </h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Arama kriterlerinizi değiştirerek daha fazla sonuç bulabilirsiniz
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3 p-3">
      {chunks.map((chunk) => {
        const isSelected = selectedChunks.has(chunk.id);
        const isImproved = chunk.chunk_metadata?.llm_improved;

        return (
          <Card
            key={chunk.id}
            className={cn(
              "transition-all duration-200 hover:shadow-md",
              isSelected && "ring-2 ring-primary ring-offset-2 bg-primary/5",
              loading && "opacity-60 pointer-events-none"
            )}
          >
            <CardContent className="p-4">
              {/* Header Row */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  {showSelection && (
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={() => onChunkSelect(chunk.id)}
                      disabled={disabled || loading}
                      className="mt-0.5"
                    />
                  )}

                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <Badge variant="outline" className="text-xs gap-1">
                      <Hash className="w-3 h-3" />
                      {chunk.chunk_index}
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
                </div>

                {/* Actions Button */}
                <div className="flex items-center gap-1 ml-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onChunkView(chunk)}
                    disabled={disabled || loading}
                    className="h-8 w-8 p-0"
                  >
                    <Eye className="w-4 h-4" />
                  </Button>

                  {(onChunkImprove || onChunkDelete) && (
                    <div className="flex items-center">
                      {onChunkImprove && !isImproved && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onChunkImprove(chunk)}
                          disabled={disabled || loading}
                          className="h-8 w-8 p-0 text-violet-600 hover:text-violet-700 hover:bg-violet-50"
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
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Document Info */}
              <div className="flex items-center gap-2 mb-2">
                <FileText className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                <span className="text-sm font-medium text-foreground truncate">
                  {getDocumentDisplayName(chunk.document_name)}
                </span>
              </div>

              {/* Content Preview */}
              <div className="mb-3">
                <p className="text-sm text-muted-foreground line-clamp-3 leading-relaxed">
                  {chunk.chunk_text}
                </p>
              </div>

              {/* Footer Info */}
              <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t border-border">
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-blue-500" />
                    {formatCharacterCount(chunk.character_count)} karakter
                  </span>

                  {chunk.created_at && (
                    <span className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" />
                      {formatDate(chunk.created_at)}
                    </span>
                  )}
                </div>

                {/* Quality Score or Language */}
                {chunk.quality_metrics?.coherence_score && (
                  <div className="text-right">
                    <span className="text-xs text-muted-foreground">
                      Kalite:{" "}
                      {Math.round(chunk.quality_metrics.coherence_score * 100)}%
                    </span>
                  </div>
                )}
              </div>

              {/* Improvement Info */}
              {isImproved && chunk.chunk_metadata && (
                <div className="mt-2 pt-2 border-t border-violet-200 bg-violet-50 -mx-4 -mb-4 px-4 pb-4 rounded-b-lg">
                  <div className="flex items-center gap-2 text-xs text-violet-700">
                    <Sparkles className="w-3 h-3" />
                    <span>
                      {chunk.chunk_metadata.improvement_model && (
                        <>Model: {chunk.chunk_metadata.improvement_model}</>
                      )}
                      {chunk.chunk_metadata.improvement_timestamp && (
                        <>
                          {" "}
                          •{" "}
                          {formatDate(
                            chunk.chunk_metadata.improvement_timestamp
                          )}
                        </>
                      )}
                    </span>
                  </div>

                  {chunk.chunk_metadata.original_length &&
                    chunk.chunk_metadata.improved_length && (
                      <div className="text-xs text-violet-600 mt-1">
                        {chunk.chunk_metadata.original_length} →{" "}
                        {chunk.chunk_metadata.improved_length} karakter
                        {chunk.chunk_metadata.improved_length >
                          chunk.chunk_metadata.original_length && (
                          <span className="ml-1 text-green-600">
                            (+
                            {(
                              (chunk.chunk_metadata.improved_length /
                                chunk.chunk_metadata.original_length -
                                1) *
                              100
                            ).toFixed(1)}
                            %)
                          </span>
                        )}
                      </div>
                    )}
                </div>
              )}
            </CardContent>
          </Card>
        );
      })}

      {/* Loading Overlay */}
      {loading && (
        <div className="fixed inset-0 bg-background/50 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-background p-4 rounded-lg shadow-lg border border-border">
            <div className="flex items-center gap-3">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-primary border-t-transparent" />
              <span className="text-sm text-muted-foreground">
                Yükleniyor...
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Compact mobile view for very small screens
export function CompactChunksMobileView({
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
      <div className="text-center py-8 px-4">
        <FileText className="w-12 h-12 mx-auto mb-3 text-muted-foreground/30" />
        <p className="text-sm text-muted-foreground">Parça bulunamadı</p>
      </div>
    );
  }

  return (
    <div className="space-y-2 p-2">
      {chunks.map((chunk) => {
        const isImproved = chunk.chunk_metadata?.llm_improved;
        const docName = chunk.document_name
          .replace(".md", "")
          .replace(/^.*[\\\/]/, "");

        return (
          <div
            key={chunk.id}
            className={cn(
              "p-3 rounded-lg border border-border bg-background",
              "hover:bg-muted/30 transition-colors",
              loading && "opacity-60"
            )}
          >
            <div className="flex items-start gap-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant="outline" className="text-xs">
                    #{chunk.chunk_index}
                  </Badge>

                  {isImproved && (
                    <Badge variant="secondary" className="gap-1 text-xs">
                      <Sparkles className="w-3 h-3" />
                      LLM
                    </Badge>
                  )}
                </div>

                <h4 className="text-sm font-medium text-foreground truncate mb-1">
                  {docName}
                </h4>

                <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
                  {chunk.chunk_text}
                </p>

                <div className="text-xs text-muted-foreground">
                  {chunk.character_count.toLocaleString("tr-TR")} karakter
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onChunkView(chunk)}
                  disabled={disabled || loading}
                  className="h-8 w-8 p-0"
                >
                  <Eye className="w-3 h-3" />
                </Button>

                {onChunkImprove && !isImproved && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onChunkImprove(chunk)}
                    disabled={disabled || loading}
                    className="h-8 w-8 p-0 text-violet-600 hover:text-violet-700"
                  >
                    <Sparkles className="w-3 h-3" />
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
