"use client";

import React from "react";

interface RAGSource {
  content?: string;
  text?: string;
  metadata?: any;
  score: number;
  crag_score?: number;
}

interface SourcesViewerProps {
  isOpen: boolean;
  onClose: () => void;
  selectedSource: RAGSource | null;
}

export default function SourcesViewer({
  isOpen,
  onClose,
  selectedSource,
}: SourcesViewerProps) {
  if (!isOpen || !selectedSource) {
    return null;
  }

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      onClose();
    }
  };

  React.useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.removeEventListener("keydown", handleEscape);
      document.body.style.overflow = "unset";
    };
  }, [isOpen, onClose]);

  const filename =
    selectedSource.metadata?.source_file ||
    selectedSource.metadata?.filename ||
    "Belge";

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={handleOverlayClick}
      onKeyDown={handleKeyDown}
      tabIndex={-1}
    >
      <div
        className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 px-6 py-4 border-b border-gray-200 rounded-t-2xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <svg
                  className="w-5 h-5 text-blue-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-800">
                  Kaynak Detayları
                </h2>
                <p className="text-sm text-gray-600">
                  {filename.replace(".md", "")}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
              aria-label="Kapat"
            >
              <svg
                className="w-6 h-6 text-gray-500"
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
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Source Info Header */}
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-bold text-blue-800 flex items-center gap-2">
                <span>📄</span>
                <span>{filename}</span>
              </h3>
              <div className="flex items-center gap-3">
                {/* CRAG Score */}
                {selectedSource.crag_score != null && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-600 font-medium">
                      DYSK:
                    </span>
                    <div className="flex items-center gap-1">
                      <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-indigo-400 to-purple-500 rounded-full"
                          style={{
                            width: `${selectedSource.crag_score * 100}%`,
                          }}
                        ></div>
                      </div>
                      <span className="text-sm font-bold text-indigo-700">
                        {Math.round(selectedSource.crag_score * 100)}%
                      </span>
                    </div>
                  </div>
                )}

                {/* Similarity Score */}
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-600 font-medium">
                    Benzerlik:
                  </span>
                  <div className="flex items-center gap-1">
                    <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-green-400 to-blue-500 rounded-full"
                        style={{ width: `${selectedSource.score * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-bold text-green-600">
                      {Math.round(selectedSource.score * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Metadata Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              {selectedSource.metadata?.chunk_index && (
                <div className="bg-white p-3 rounded border">
                  <div className="text-gray-600 font-medium">Bölüm</div>
                  <div className="text-gray-800 font-bold">
                    {selectedSource.metadata.chunk_index} /{" "}
                    {selectedSource.metadata?.total_chunks || "?"}
                  </div>
                </div>
              )}

              {selectedSource.metadata?.chunk_title && (
                <div className="bg-white p-3 rounded border">
                  <div className="text-gray-600 font-medium">Başlık</div>
                  <div className="text-gray-800 font-bold text-xs">
                    {selectedSource.metadata.chunk_title}
                  </div>
                </div>
              )}

              <div className="bg-white p-3 rounded border">
                <div className="text-gray-600 font-medium">İçerik Uzunluğu</div>
                <div className="text-gray-800 font-bold">
                  {(selectedSource.content || selectedSource.text || "").length}{" "}
                  karakter
                </div>
              </div>
            </div>
          </div>

          {/* Content Display */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <h4 className="text-lg font-semibold text-gray-700">📝 İçerik</h4>
              {selectedSource.metadata?.chunk_title && (
                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full">
                  {selectedSource.metadata.chunk_title}
                </span>
              )}
            </div>

            <div className="bg-gray-50 border border-gray-200 rounded-lg max-h-96 overflow-y-auto">
              <div className="p-4">
                {selectedSource.content || selectedSource.text ? (
                  <div className="prose prose-sm max-w-none text-gray-700 leading-relaxed whitespace-pre-wrap">
                    {selectedSource.content || selectedSource.text}
                  </div>
                ) : (
                  <div className="text-gray-500 italic text-center py-8">
                    <svg
                      className="w-12 h-12 mx-auto mb-3 text-gray-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="1"
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                    <p>İçerik bulunamadı</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Additional Metadata */}
          {(selectedSource.metadata?.page_number ||
            selectedSource.metadata?.section ||
            selectedSource.metadata?.category) && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <h4 className="text-md font-semibold text-gray-700 mb-3 flex items-center gap-2">
                <span>📋</span>
                <span>Ek Bilgiler</span>
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                {selectedSource.metadata?.page_number && (
                  <div className="flex justify-between">
                    <span className="text-gray-600 font-medium">Sayfa:</span>
                    <span className="text-gray-800 font-semibold">
                      {selectedSource.metadata.page_number}
                    </span>
                  </div>
                )}

                {selectedSource.metadata?.section && (
                  <div className="flex justify-between">
                    <span className="text-gray-600 font-medium">Bölüm:</span>
                    <span className="text-gray-800 font-semibold">
                      {selectedSource.metadata.section}
                    </span>
                  </div>
                )}

                {selectedSource.metadata?.category && (
                  <div className="flex justify-between">
                    <span className="text-gray-600 font-medium">Kategori:</span>
                    <span className="text-gray-800 font-semibold">
                      {selectedSource.metadata.category}
                    </span>
                  </div>
                )}

                {selectedSource.metadata?.created_at && (
                  <div className="flex justify-between">
                    <span className="text-gray-600 font-medium">
                      Oluşturma:
                    </span>
                    <span className="text-gray-800 font-semibold text-xs">
                      {new Date(
                        selectedSource.metadata.created_at
                      ).toLocaleDateString("tr-TR")}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Score Analysis */}
          <div className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-4">
            <h4 className="text-md font-semibold text-gray-700 mb-3 flex items-center gap-2">
              <span>📊</span>
              <span>Skor Analizi</span>
            </h4>
            <div className="space-y-3">
              {/* Similarity Score Explanation */}
              <div className="bg-white p-3 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-green-700">
                    Benzerlik Skoru
                  </span>
                  <span className="text-2xl font-bold text-green-600">
                    {Math.round(selectedSource.score * 100)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-green-400 to-green-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${selectedSource.score * 100}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-600 mt-2">
                  Bu kaynak, sorunuzla ne kadar benzer içeriğe sahip olduğunu
                  gösterir.
                </p>
              </div>

              {/* CRAG Score Explanation */}
              {selectedSource.crag_score != null && (
                <div className="bg-white p-3 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-indigo-700">
                      DYSK Skoru
                    </span>
                    <span className="text-2xl font-bold text-indigo-600">
                      {Math.round(selectedSource.crag_score * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-indigo-400 to-purple-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${selectedSource.crag_score * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-gray-600 mt-2">
                    Doğrulayıcı Yeniden Sıralama Katmanı (DYSK) - gelişmiş
                    alakalılık skoru.
                  </p>
                </div>
              )}

              {/* Quality Indicator */}
              <div className="text-center">
                <div
                  className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium ${
                    selectedSource.score > 0.8
                      ? "bg-green-100 text-green-800 border border-green-300"
                      : selectedSource.score > 0.6
                      ? "bg-yellow-100 text-yellow-800 border border-yellow-300"
                      : "bg-red-100 text-red-800 border border-red-300"
                  }`}
                >
                  {selectedSource.score > 0.8 ? (
                    <>
                      <span>✅</span>
                      <span>Yüksek Kalite Kaynak</span>
                    </>
                  ) : selectedSource.score > 0.6 ? (
                    <>
                      <span>⚠️</span>
                      <span>Orta Kalite Kaynak</span>
                    </>
                  ) : (
                    <>
                      <span>❌</span>
                      <span>Düşük Kalite Kaynak</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 px-6 py-4 bg-gray-50 rounded-b-2xl">
          <div className="flex items-center justify-between">
            <div className="text-xs text-gray-500 flex items-center gap-2">
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
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>Bu kaynak otomatik olarak seçilmiş ve işlenmiştir</span>
            </div>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors font-medium text-sm"
            >
              Kapat
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
