"use client";
import React from "react";
import { MarkdownFileWithCategory } from "@/lib/api";

interface FileListProps {
  files: MarkdownFileWithCategory[];
  selectedFiles: string[];
  onFileToggle: (filename: string) => void;
  loading: boolean;
  onViewFile?: (filename: string) => void;
  onDownloadFile?: (filename: string) => void;
  onDeleteFile?: (filename: string) => void;
  markdownPage?: number;
  setMarkdownPage?: (page: number) => void;
  MARKDOWN_FILES_PER_PAGE?: number;
}

export function FileList({
  files,
  selectedFiles,
  onFileToggle,
  loading,
  onViewFile,
  onDownloadFile,
  onDeleteFile,
  markdownPage = 1,
  setMarkdownPage,
  MARKDOWN_FILES_PER_PAGE = 20,
}: FileListProps) {
  if (loading) {
    return (
      <div className="text-center py-12 animate-fade-in">
        <div className="relative mx-auto w-16 h-16 mb-4">
          <div className="absolute inset-0 rounded-full border-4 border-muted"></div>
          <div className="absolute inset-0 rounded-full border-4 border-primary border-t-transparent animate-spin"></div>
        </div>
        <p className="text-muted-foreground animate-pulse-soft">
          Markdown dosyaları yükleniyor...
        </p>
        <div className="flex justify-center mt-3 space-x-1">
          <div
            className="w-1 h-1 bg-primary rounded-full animate-bounce"
            style={{ animationDelay: "0ms" }}
          ></div>
          <div
            className="w-1 h-1 bg-primary rounded-full animate-bounce"
            style={{ animationDelay: "150ms" }}
          ></div>
          <div
            className="w-1 h-1 bg-primary rounded-full animate-bounce"
            style={{ animationDelay: "300ms" }}
          ></div>
        </div>
      </div>
    );
  }

  if (files.length === 0) {
    return (
      <div className="text-center py-6 text-gray-500 border border-gray-200 rounded-md">
        Henüz markdown dosyası bulunmuyor.
      </div>
    );
  }

  const paginatedFiles = files.slice(
    (markdownPage - 1) * MARKDOWN_FILES_PER_PAGE,
    markdownPage * MARKDOWN_FILES_PER_PAGE
  );

  return (
    <>
      {/* Markdown Dosya Bilgisi */}
      <div className="mb-4 p-3 bg-gray-50 border border-gray-200 rounded-lg">
        <div className="text-sm text-gray-700">
          <span className="font-medium">{files.length} dosya bulundu</span>
          {selectedFiles.length > 0 && (
            <span className="text-gray-500">
              {" "}
              ({selectedFiles.length} seçili)
            </span>
          )}
        </div>
      </div>

      <div className="max-h-[600px] overflow-y-auto border border-gray-200 rounded-md bg-white">
        {paginatedFiles.map((file) => (
          <div
            key={file.filename}
            className="flex items-center p-3 hover:bg-gray-50 border-b border-gray-100 last:border-b-0 group"
          >
            <input
              type="checkbox"
              checked={selectedFiles.includes(file.filename)}
              onChange={() => onFileToggle(file.filename)}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />

            <div className="flex-1 ml-3">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-gray-900">
                  {file.filename.replace(".md", "")}
                </span>
                {/* Category Badge */}
                {file.category_name && (
                  <span className="px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                    {file.category_name}
                  </span>
                )}
              </div>
              <div className="text-xs text-gray-500">{file.filename}</div>
            </div>

            <div className="flex items-center gap-2">
              {onViewFile && (
                <button
                  onClick={() => onViewFile(file.filename)}
                  className="px-3 py-1 text-xs bg-blue-100 text-blue-700 hover:bg-blue-200 rounded-md transition-colors flex items-center gap-1"
                  title="Dosyayı görüntüle"
                >
                  <svg
                    className="w-3 h-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                    />
                  </svg>
                  Görüntüle
                </button>
              )}

              {onDownloadFile && (
                <button
                  onClick={() => onDownloadFile(file.filename)}
                  className="px-3 py-1 text-xs bg-green-100 text-green-700 hover:bg-green-200 rounded-md transition-colors flex items-center gap-1"
                  title="Dosyayı indir"
                >
                  <svg
                    className="w-3 h-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  İndir
                </button>
              )}

              {onDeleteFile && (
                <button
                  onClick={() => onDeleteFile(file.filename)}
                  className="px-3 py-1 text-xs bg-red-100 text-red-700 hover:bg-red-200 rounded-md transition-colors"
                  title="Dosyayı sil"
                >
                  Sil
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Pagination */}
      {files.length > MARKDOWN_FILES_PER_PAGE && setMarkdownPage && (
        <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-200">
          <button
            onClick={() => setMarkdownPage(Math.max(1, markdownPage - 1))}
            disabled={markdownPage === 1}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Önceki
          </button>
          <span className="text-sm text-gray-700">
            Sayfa {markdownPage} /{" "}
            {Math.ceil(files.length / MARKDOWN_FILES_PER_PAGE)}{" "}
            <span className="text-gray-500">(Toplam {files.length} dosya)</span>
          </span>
          <button
            onClick={() =>
              setMarkdownPage(
                Math.min(
                  Math.ceil(files.length / MARKDOWN_FILES_PER_PAGE),
                  markdownPage + 1
                )
              )
            }
            disabled={
              markdownPage >= Math.ceil(files.length / MARKDOWN_FILES_PER_PAGE)
            }
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Sonraki
          </button>
        </div>
      )}
    </>
  );
}
