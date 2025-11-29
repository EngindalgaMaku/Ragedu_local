"use client";
import React from "react";
import { MarkdownCategory } from "@/lib/api";

interface FilterBarProps {
  categories: MarkdownCategory[];
  selectedCategoryId: number | null;
  onCategoryChange: (categoryId: number | null) => void;
  selectedFilesCount: number;
  onRefresh: () => void;
}

export function FilterBar({
  categories,
  selectedCategoryId,
  onCategoryChange,
  selectedFilesCount,
  onRefresh,
}: FilterBarProps) {
  return (
    <div className="flex items-center justify-between mb-3">
      <div className="flex items-center gap-3">
        <label className="label m-0">Mevcut Markdown Dosyaları</label>

        {/* Category Filter Dropdown */}
        <div className="relative group">
          <select
            value={selectedCategoryId || ""}
            onChange={(e) =>
              onCategoryChange(e.target.value ? Number(e.target.value) : null)
            }
            className="appearance-none bg-white border border-gray-300 rounded-md pl-3 pr-8 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">Tüm Kategoriler</option>
            {Array.isArray(categories) &&
              categories.map((cat) =>
                cat && cat.id && cat.name ? (
                  <option key={cat.id} value={cat.id}>
                    {cat.name}
                  </option>
                ) : null
              )}
          </select>
          <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
            <svg
              className="fill-current h-4 w-4"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
            >
              <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
            </svg>
          </div>
        </div>

        {selectedFilesCount > 0 && (
          <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
            {selectedFilesCount} dosya seçili
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onRefresh}
          className="text-sm text-primary-600 hover:text-primary-800 flex items-center gap-1 px-3 py-1.5 rounded-md hover:bg-gray-100 transition-colors"
        >
          <svg
            className="w-4 h-4"
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
          Yenile
        </button>
      </div>
    </div>
  );
}
