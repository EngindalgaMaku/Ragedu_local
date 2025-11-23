"use client";

import React from "react";
import { X, FileText, Hash } from "lucide-react";
import { RAGSource } from "@/lib/api";

interface SourceModalProps {
  source: RAGSource | null;
  isOpen: boolean;
  onClose: () => void;
}

export default function SourceModal({ source, isOpen, onClose }: SourceModalProps) {
  if (!isOpen || !source) return null;

  const rawType =
    (source.metadata as any)?.source_type ||
    (source.metadata as any)?.source ||
    "unknown";

  const sourceTypeLabel =
    rawType === "direct_qa" || rawType === "qa_pair" || rawType === "question_answer"
      ? "Soru Bankası (Otomatik QA)"
      : rawType === "structured_kb"
      ? "Bilgi Tabanı Özeti"
      : rawType === "vector_search"
      ? "Döküman Parçası (RAG)"
      : "Döküman";

  const displayContent = (() => {
    const base = (source.content || "").toString().trim();
    if (base.length > 0) return base;

    // QA kaynağıysa ve soru/cevap metadata'da varsa, bunlardan içerik oluştur
    const meta: any = source.metadata || {};
    if (
      (rawType === "direct_qa" || rawType === "qa_pair" || rawType === "question_answer") &&
      (meta.qa_question || meta.qa_answer)
    ) {
      return `Soru: ${meta.qa_question || "-"}\n\nCevap: ${meta.qa_answer || "-"}`;
    }

    return "İçerik bulunamadı";
  })();

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
      onClick={handleBackdropClick}
    >
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6" />
            <div>
              <h3 className="text-lg font-semibold">Kaynak Detayları</h3>
              <p className="text-sm text-blue-100">
                {sourceTypeLabel}
                {source.metadata?.filename || source.metadata?.source_file
                  ? ` • ${source.metadata.filename || source.metadata.source_file}`
                  : ""}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/20 rounded-lg transition-colors"
            title="Kapat"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {/* Metadata Tags */}
          <div className="flex flex-wrap gap-2">
            <span className="inline-flex items-center gap-1 px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
              <Hash className="w-3 h-3" />
              Skor: {(source.score * 100).toFixed(1)}%
            </span>
            
            {source.metadata?.chunk_index !== undefined && (
              <span className="inline-flex items-center gap-1 px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">
                Parça {source.metadata.chunk_index + 1}
                {source.metadata?.total_chunks && ` / ${source.metadata.total_chunks}`}
              </span>
            )}
            
            {source.metadata?.page_number && (
              <span className="inline-flex items-center px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
                Sayfa {source.metadata.page_number}
              </span>
            )}
            
            {source.metadata?.section && (
              <span className="inline-flex items-center px-3 py-1 bg-yellow-100 text-yellow-700 rounded-full text-sm font-medium">
                {source.metadata.section}
              </span>
            )}
          </div>

          {/* Chunk Title */}
          {source.metadata?.chunk_title && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-gray-700 mb-1">Bölüm Başlığı</h4>
              <p className="text-gray-900 font-medium">{source.metadata.chunk_title}</p>
            </div>
          )}

          {/* Source Content */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h4 className="text-sm font-semibold text-gray-700 mb-3">İçerik</h4>
            <div className="prose prose-sm max-w-none">
              <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
                {displayContent}
              </p>
            </div>
          </div>

          {/* Additional Metadata */}
          {source.metadata && Object.keys(source.metadata).length > 0 && (
            <details className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <summary className="text-sm font-semibold text-gray-700 cursor-pointer hover:text-gray-900">
                Ek Bilgiler
              </summary>
              <div className="mt-3 space-y-2 text-sm">
                {Object.entries(source.metadata).map(([key, value]) => {
                  if (
                    [
                      "filename",
                      "source_file",
                      "chunk_title",
                      "chunk_index",
                      "total_chunks",
                      "page_number",
                      "section",
                      "qa_question",
                      "qa_answer",
                    ].includes(key)
                  ) {
                    return null; // Already displayed above
                  }
                  // Türkçe anahtar ismi
                  const label =
                    key === "source_type"
                      ? "kaynak_türü"
                      : key === "qa_similarity"
                      ? "benzerlik"
                      : key;
                  return (
                    <div key={key} className="flex items-start gap-2">
                      <span className="font-medium text-gray-600 min-w-[120px]">
                        {label}:
                      </span>
                      <span className="text-gray-800 flex-1">
                        {typeof value === "object"
                          ? JSON.stringify(value, null, 2)
                          : String(value)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </details>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-50 border-t border-gray-200 px-6 py-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
          >
            Kapat
          </button>
        </div>
      </div>
    </div>
  );
}

