/**
 * InteractionsPagination Component
 * Specialized pagination for interactions with date-based navigation
 */

import React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

interface InteractionsPaginationProps {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  itemsPerPage: number;
  loading?: boolean;
  onPageChange: (page: number) => void;
  onItemsPerPageChange?: (itemsPerPage: number) => void;
  showItemsPerPageSelector?: boolean;
}

export function InteractionsPagination({
  currentPage,
  totalPages,
  totalItems,
  itemsPerPage,
  loading = false,
  onPageChange,
  onItemsPerPageChange,
  showItemsPerPageSelector = true,
}: InteractionsPaginationProps) {
  // Don't render if there's only one page or no items
  if (totalPages <= 1) {
    return null;
  }

  const handlePrevious = () => {
    if (currentPage > 1) {
      onPageChange(currentPage - 1);
    }
  };

  const handleNext = () => {
    if (currentPage < totalPages) {
      onPageChange(currentPage + 1);
    }
  };

  const startItem = (currentPage - 1) * itemsPerPage + 1;
  const endItem = Math.min(currentPage * itemsPerPage, totalItems);

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 px-3 sm:px-5 py-3 border-t border-border bg-muted/30">
      {/* Items per page selector */}
      {showItemsPerPageSelector && onItemsPerPageChange && (
        <div className="flex items-center gap-2 order-last sm:order-first">
          <span className="text-sm text-muted-foreground whitespace-nowrap">
            Sayfa başına:
          </span>
          <select
            value={itemsPerPage}
            onChange={(e) => onItemsPerPageChange(parseInt(e.target.value))}
            disabled={loading}
            className="text-sm border border-border rounded px-2 py-1 bg-background"
          >
            <option value={5}>5</option>
            <option value={10}>10</option>
            <option value={20}>20</option>
            <option value={50}>50</option>
          </select>
        </div>
      )}

      {/* Navigation controls */}
      <div className="flex items-center gap-2 mx-auto sm:mx-0">
        <Button
          variant="outline"
          size="sm"
          onClick={handlePrevious}
          disabled={currentPage === 1 || loading}
          className="flex items-center gap-1 min-h-[40px]"
        >
          <ChevronLeft className="h-4 w-4" />
          <span className="hidden sm:inline">Önceki</span>
        </Button>

        {/* Page info */}
        <div className="flex items-center gap-2 px-4">
          <span className="text-sm text-muted-foreground whitespace-nowrap">
            <span className="hidden sm:inline">Sayfa </span>
            <span className="font-medium text-foreground">{currentPage}</span>
            <span className="mx-1">/</span>
            <span className="font-medium text-foreground">{totalPages}</span>
          </span>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={handleNext}
          disabled={currentPage >= totalPages || loading}
          className="flex items-center gap-1 min-h-[40px]"
        >
          <span className="hidden sm:inline">Sonraki</span>
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      {/* Items count */}
      <div className="text-sm text-muted-foreground">
        <span className="hidden sm:inline">
          {startItem}-{endItem} / {totalItems} etkileşim
        </span>
        <span className="sm:hidden">{totalItems} etkileşim</span>
      </div>
    </div>
  );
}

// Enhanced pagination with jump-to-page functionality
export function InteractionsPaginationAdvanced({
  currentPage,
  totalPages,
  totalItems,
  itemsPerPage,
  loading = false,
  onPageChange,
  onItemsPerPageChange,
  showItemsPerPageSelector = true,
}: InteractionsPaginationProps) {
  if (totalPages <= 1) {
    return null;
  }

  const handleJumpToPage = (page: number) => {
    if (page >= 1 && page <= totalPages) {
      onPageChange(page);
    }
  };

  // Calculate which pages to show
  const getVisiblePages = () => {
    const delta = 2; // Number of pages to show on each side of current page
    const range: number[] = [];

    for (
      let i = Math.max(2, currentPage - delta);
      i <= Math.min(totalPages - 1, currentPage + delta);
      i++
    ) {
      range.push(i);
    }

    if (currentPage - delta > 2) {
      range.unshift(-1); // Placeholder for "..."
    }
    if (currentPage + delta < totalPages - 1) {
      range.push(-1); // Placeholder for "..."
    }

    range.unshift(1);
    if (totalPages > 1) {
      range.push(totalPages);
    }

    return range;
  };

  const visiblePages = getVisiblePages();
  const startItem = (currentPage - 1) * itemsPerPage + 1;
  const endItem = Math.min(currentPage * itemsPerPage, totalItems);

  return (
    <div className="flex flex-col gap-4 px-3 sm:px-5 py-4 border-t border-border bg-muted/30">
      {/* Top row: Items count and per-page selector */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          <span>
            {startItem}-{endItem} /{" "}
          </span>
          <span className="font-medium text-foreground">{totalItems}</span>
          <span> etkileşim gösteriliyor</span>
        </div>

        {showItemsPerPageSelector && onItemsPerPageChange && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Sayfa başına:</span>
            <select
              value={itemsPerPage}
              onChange={(e) => onItemsPerPageChange(parseInt(e.target.value))}
              disabled={loading}
              className="text-sm border border-border rounded px-2 py-1 bg-background"
            >
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
            </select>
          </div>
        )}
      </div>

      {/* Bottom row: Page navigation */}
      <div className="flex items-center justify-center gap-1">
        <Button
          variant="outline"
          size="sm"
          onClick={() => handleJumpToPage(currentPage - 1)}
          disabled={currentPage === 1 || loading}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>

        {visiblePages.map((page, index) => {
          if (page === -1) {
            return (
              <span
                key={index}
                className="px-2 py-1 text-sm text-muted-foreground"
              >
                ...
              </span>
            );
          }

          return (
            <Button
              key={page}
              variant={page === currentPage ? "default" : "outline"}
              size="sm"
              onClick={() => handleJumpToPage(page)}
              disabled={loading}
              className="min-w-[40px]"
            >
              {page}
            </Button>
          );
        })}

        <Button
          variant="outline"
          size="sm"
          onClick={() => handleJumpToPage(currentPage + 1)}
          disabled={currentPage >= totalPages || loading}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

export default InteractionsPagination;
