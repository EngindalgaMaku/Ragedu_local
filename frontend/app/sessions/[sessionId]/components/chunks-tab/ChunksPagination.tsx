"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
} from "lucide-react";

export interface ChunksPaginationProps {
  currentPage: number;
  totalPages: number;
  chunksPerPage: number;
  totalItems: number;
  filteredItems: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (pageSize: number) => void;
  disabled?: boolean;
}

const PAGE_SIZE_OPTIONS = [10, 25, 50, 100] as const;

export default function ChunksPagination({
  currentPage,
  totalPages,
  chunksPerPage,
  totalItems,
  filteredItems,
  onPageChange,
  onPageSizeChange,
  disabled = false,
}: ChunksPaginationProps) {
  // Calculate display range
  const startItem = totalPages > 0 ? (currentPage - 1) * chunksPerPage + 1 : 0;
  const endItem = Math.min(currentPage * chunksPerPage, filteredItems);

  // Generate page numbers to show
  const getVisiblePages = () => {
    const pages: number[] = [];
    const maxVisible = 5;

    if (totalPages <= maxVisible) {
      // Show all pages if total is small
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Show smart pagination with ellipsis
      const start = Math.max(1, currentPage - 2);
      const end = Math.min(totalPages, currentPage + 2);

      // Always show first page
      if (start > 1) {
        pages.push(1);
        if (start > 2) {
          pages.push(-1); // Ellipsis marker
        }
      }

      // Show pages around current
      for (let i = start; i <= end; i++) {
        pages.push(i);
      }

      // Always show last page
      if (end < totalPages) {
        if (end < totalPages - 1) {
          pages.push(-1); // Ellipsis marker
        }
        pages.push(totalPages);
      }
    }

    return pages;
  };

  const visiblePages = getVisiblePages();

  if (totalPages <= 1 && totalItems <= chunksPerPage) {
    return null; // Don't show pagination if not needed
  }

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 px-4 py-3 border-t border-border bg-muted/30">
      {/* Items Info */}
      <div className="flex items-center gap-4 text-sm text-muted-foreground">
        <div className="flex items-center gap-2">
          <span>
            {startItem}-{endItem} / {filteredItems}
          </span>
          {filteredItems !== totalItems && (
            <Badge variant="secondary" className="text-xs">
              {totalItems} toplam
            </Badge>
          )}
        </div>

        {/* Page Size Selector */}
        <div className="hidden sm:flex items-center gap-2">
          <span className="text-xs">Sayfa başına:</span>
          <div className="flex gap-1">
            {PAGE_SIZE_OPTIONS.map((size) => (
              <Button
                key={size}
                variant={chunksPerPage === size ? "default" : "ghost"}
                size="sm"
                onClick={() => onPageSizeChange(size)}
                disabled={disabled}
                className={`h-6 px-2 text-xs ${
                  chunksPerPage === size ? "shadow-sm" : ""
                }`}
              >
                {size}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Navigation Controls */}
      {totalPages > 1 && (
        <div className="flex items-center gap-1">
          {/* First Page */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onPageChange(1)}
            disabled={disabled || currentPage === 1}
            className="h-8 w-8 p-0"
            title="İlk sayfa"
          >
            <ChevronsLeft className="w-4 h-4" />
          </Button>

          {/* Previous Page */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onPageChange(Math.max(1, currentPage - 1))}
            disabled={disabled || currentPage === 1}
            className="h-8 w-8 p-0"
            title="Önceki sayfa"
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>

          {/* Page Numbers */}
          <div className="flex items-center gap-1 mx-2">
            {visiblePages.map((page, index) => (
              <React.Fragment key={index}>
                {page === -1 ? (
                  <span className="px-2 py-1 text-sm text-muted-foreground">
                    ...
                  </span>
                ) : (
                  <Button
                    variant={currentPage === page ? "default" : "ghost"}
                    size="sm"
                    onClick={() => onPageChange(page)}
                    disabled={disabled}
                    className={`h-8 w-8 p-0 text-sm ${
                      currentPage === page ? "shadow-sm" : ""
                    }`}
                  >
                    {page}
                  </Button>
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Next Page */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
            disabled={disabled || currentPage === totalPages}
            className="h-8 w-8 p-0"
            title="Sonraki sayfa"
          >
            <ChevronRight className="w-4 h-4" />
          </Button>

          {/* Last Page */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onPageChange(totalPages)}
            disabled={disabled || currentPage === totalPages}
            className="h-8 w-8 p-0"
            title="Son sayfa"
          >
            <ChevronsRight className="w-4 h-4" />
          </Button>
        </div>
      )}

      {/* Mobile Page Size Selector */}
      <div className="flex sm:hidden items-center gap-2 text-sm">
        <span className="text-xs text-muted-foreground">Sayfa başına:</span>
        <div className="flex gap-1">
          {PAGE_SIZE_OPTIONS.slice(0, 3).map((size) => (
            <Button
              key={size}
              variant={chunksPerPage === size ? "default" : "outline"}
              size="sm"
              onClick={() => onPageSizeChange(size)}
              disabled={disabled}
              className="h-6 px-2 text-xs"
            >
              {size}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}

// Compact pagination for mobile or tight spaces
export function CompactChunksPagination({
  currentPage,
  totalPages,
  onPageChange,
  disabled = false,
}: {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  disabled?: boolean;
}) {
  if (totalPages <= 1) {
    return null;
  }

  return (
    <div className="flex items-center justify-between gap-2 px-3 py-2 border-t border-border">
      <Button
        variant="outline"
        size="sm"
        onClick={() => onPageChange(Math.max(1, currentPage - 1))}
        disabled={disabled || currentPage === 1}
        className="gap-1 text-xs"
      >
        <ChevronLeft className="w-3 h-3" />
        Önceki
      </Button>

      <Badge variant="secondary" className="text-xs">
        {currentPage} / {totalPages}
      </Badge>

      <Button
        variant="outline"
        size="sm"
        onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
        disabled={disabled || currentPage === totalPages}
        className="gap-1 text-xs"
      >
        Sonraki
        <ChevronRight className="w-3 h-3" />
      </Button>
    </div>
  );
}
