"use client";
import React from "react";
import { MarkdownCategory } from "@/lib/api";

interface CategoryManagerProps {
  isOpen: boolean;
  onClose: () => void;
  categories: MarkdownCategory[];
  newCategoryName: string;
  setNewCategoryName: (name: string) => void;
  onCreateCategory: () => void;
  onDeleteCategory: (id: number) => void;
  isLoading: boolean;
  selectedMarkdownFiles: string[];
  onAssignCategory: (categoryId: number | null) => void;
}

export function CategoryManager({
  isOpen,
  onClose,
  categories,
  newCategoryName,
  setNewCategoryName,
  onCreateCategory,
  onDeleteCategory,
  isLoading,
}: CategoryManagerProps) {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-xl shadow-2xl w-full max-w-md max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-6 pb-0">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-900">
              Markdown Kategorileri
            </h3>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700"
            >
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
            </button>
          </div>

          {/* Add New Category */}
          <div className="flex gap-2 mb-6">
            <input
              type="text"
              value={newCategoryName}
              onChange={(e) => setNewCategoryName(e.target.value)}
              placeholder="Yeni kategori adı"
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              onKeyDown={(e) => e.key === "Enter" && onCreateCategory()}
            />
            <button
              onClick={onCreateCategory}
              disabled={!newCategoryName.trim() || isLoading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "Ekleniyor..." : "Ekle"}
            </button>
          </div>

          {/* Categories List */}
          <div className="max-h-[50vh] overflow-y-auto">
            {categories.length === 0 ? (
              <p className="text-gray-500 text-center py-4">
                Henüz kategori bulunmuyor
              </p>
            ) : (
              <ul className="divide-y divide-gray-200">
                {categories.map((category) => (
                  <li
                    key={category.id}
                    className="py-3 px-1 flex items-center justify-between hover:bg-gray-50"
                  >
                    <span className="text-gray-800">{category.name}</span>
                    <button
                      onClick={() => onDeleteCategory(category.id)}
                      disabled={isLoading}
                      className="text-red-500 hover:text-red-700 disabled:opacity-50"
                      title="Kategoriyi sil"
                    >
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
                          d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                        />
                      </svg>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-gray-50 rounded-b-xl flex justify-end border-t border-gray-200">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Kapat
          </button>
        </div>
      </div>
    </div>
  );
}
