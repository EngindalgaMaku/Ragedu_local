"use client";
import React, { useEffect, useMemo, useState } from "react";
import DocumentUploadModal from "@/components/DocumentUploadModal";
import {
  listMarkdownFiles,
  uploadMarkdownFile,
  getMarkdownFileContent,
  deleteMarkdownFile,
} from "@/lib/api";
import ModernAdminLayout from "../components/ModernAdminLayout";

export default function DocumentCenterPage() {
  const [markdownFiles, setMarkdownFiles] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Modals
  const [showNanonets, setShowNanonets] = useState(false);
  const [showPdfplumber, setShowPdfplumber] = useState(false);
  const [showMarker, setShowMarker] = useState(false);
  
  // Dropdown state
  const [markerDropdownOpen, setMarkerDropdownOpen] = useState(false);

  // Markdown upload (raw .md)
  const [uploadingMd, setUploadingMd] = useState(false);
  // Modal states
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState<string>("");
  const [selectedFileContent, setSelectedFileContent] = useState<string>("");
  const [contentLoading, setContentLoading] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);
  // Pagination
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 20;

  const refreshFiles = async (options?: {
    desiredPage?: number;
    preservePage?: boolean;
  }) => {
    try {
      setLoading(true);
      const files = await listMarkdownFiles();
      // Sort alphabetically (case-insensitive)
      const sorted = [...files].sort((a, b) =>
        a.localeCompare(b, "tr", { sensitivity: "accent" })
      );
      setMarkdownFiles(sorted);
      // Keep user on the same page unless explicitly overridden
      const newTotalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
      setPage((prev) => {
        const desired = options?.desiredPage ?? prev;
        const keep = options?.preservePage !== false; // default: keep page
        const target = keep ? desired : 1;
        return Math.min(Math.max(1, target), newTotalPages);
      });
    } catch (e: any) {
      setError(e.message || "Markdown dosyaları yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshFiles();
  }, []);

  const handleUploadMd = async (file?: File | null) => {
    try {
      if (!file) return;
      if (!file.name.toLowerCase().endsWith(".md")) {
        setError("Lütfen sadece .md dosyası yükleyin");
        return;
      }
      setUploadingMd(true);
      const res = await uploadMarkdownFile(file);
      if (res?.success) {
        setSuccess(`Yüklendi: ${res.markdown_filename}`);
        await refreshFiles();
      } else {
        setError(res?.message || "Yükleme başarısız");
      }
    } catch (e: any) {
      setError(e.message || "Yükleme başarısız");
    } finally {
      setUploadingMd(false);
    }
  };

  const pagedFiles = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return markdownFiles.slice(start, start + PAGE_SIZE);
  }, [markdownFiles, page]);

  const totalPages = useMemo(
    () => Math.max(1, Math.ceil(markdownFiles.length / PAGE_SIZE)),
    [markdownFiles.length]
  );

  const openViewer = async (filename: string) => {
    try {
      setIsModalOpen(true);
      setSelectedFileName(filename);
      setSelectedFileContent("");
      setModalError(null);
      setContentLoading(true);
      const res = await getMarkdownFileContent(filename);
      setSelectedFileContent(res.content || "");
    } catch (e: any) {
      setModalError(e.message || "Dosya içeriği yüklenemedi");
    } finally {
      setContentLoading(false);
    }
  };

  const closeViewer = () => {
    setIsModalOpen(false);
    setSelectedFileName("");
    setSelectedFileContent("");
    setModalError(null);
  };

  const handleDelete = async (filename: string) => {
    if (!confirm(`'${filename}' dosyasını silmek istiyor musunuz?`)) return;
    try {
      await deleteMarkdownFile(filename);
      setSuccess(`Silindi: ${filename}`);
      await refreshFiles();
      if (isModalOpen) closeViewer();
    } catch (e: any) {
      setError(e.message || "Silme işlemi başarısız");
    }
  };

  const handleDownload = async (filename: string) => {
    try {
      const res = await getMarkdownFileContent(filename);
      const blob = new Blob([res.content || ""], { type: "text/markdown" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e.message || "İndirme başarısız");
    }
  };

  return (
    <ModernAdminLayout
      title="Belge Merkezi"
      description="PDF/DOC/PPT dosyalarını Markdown'a dönüştür ve mevcut Markdown'ları yönet"
    >
      <div className="space-y-6 md:space-y-8">
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-100 dark:border-gray-700 p-6 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="text-4xl">📚</div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Belge Merkezi
                </h1>
                <p className="text-gray-600 text-sm">
                  PDF/DOC/PPT dosyalarını Markdown'a dönüştür ve mevcut
                  Markdown'ları yönet
                </p>
              </div>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
            <div className="text-red-800 dark:text-red-200">{error}</div>
          </div>
        )}
        {success && (
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md p-4">
            <div className="text-green-800 dark:text-green-200">{success}</div>
          </div>
        )}

        <div className="bg-white dark:bg-gray-800 p-4 md:p-6 lg:p-8 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6 gap-4">
            <div className="flex items-start sm:items-center gap-3 flex-1 min-w-0">
              <div className="p-3 bg-primary/10 text-primary rounded-xl flex-shrink-0">
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
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <h2 className="text-lg md:text-xl font-bold text-foreground">
                  Belgeleri Dönüştür
                </h2>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  PDF/DOC/PPT dosyalarını Markdown'a dönüştür
                </p>
              </div>
            </div>
            <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
              {/* Marker Dropdown - Ana Seçenek */}
              <div className="relative">
                <button
                  onClick={() => setMarkerDropdownOpen(!markerDropdownOpen)}
                  className="w-full sm:w-auto py-3 px-6 bg-orange-600 text-white rounded-lg font-medium text-sm hover:bg-orange-700 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2 transition-all flex items-center justify-center gap-2"
                >
                  <span>📚 Marker</span>
                  <span className="text-xs bg-orange-500 px-2 py-0.5 rounded">Önerilen</span>
                  <svg
                    className={`w-4 h-4 transition-transform ${markerDropdownOpen ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </button>
                
                {/* Dropdown Menu */}
                {markerDropdownOpen && (
                  <>
                    <div
                      className="fixed inset-0 z-40"
                      onClick={() => setMarkerDropdownOpen(false)}
                    />
                    <div className="absolute top-full left-0 mt-2 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 z-50 overflow-hidden">
                      {/* Marker - Ana Seçenek */}
                      <button
                        onClick={() => {
                          setShowMarker(true);
                          setMarkerDropdownOpen(false);
                        }}
                        className="w-full text-left px-4 py-3 bg-orange-50 dark:bg-orange-900/20 hover:bg-orange-100 dark:hover:bg-orange-900/30 transition-colors flex items-center gap-3 border-b border-gray-200 dark:border-gray-700"
                      >
                        <span className="text-xl">📚</span>
                        <div className="flex-1">
                          <div className="font-semibold text-gray-900 dark:text-white">
                            Marker
                          </div>
                          <div className="text-xs text-gray-600 dark:text-gray-400">
                            Önerilen dönüştürme yöntemi
                          </div>
                        </div>
                      </button>
                      
                      {/* Nanonets - Alt Seçenek */}
                      <button
                        onClick={() => {
                          setShowNanonets(true);
                          setMarkerDropdownOpen(false);
                        }}
                        className="w-full text-left px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors flex items-center gap-3 border-b border-gray-200 dark:border-gray-700"
                      >
                        <span className="text-xl">🌐</span>
                        <div className="flex-1">
                          <div className="font-medium text-gray-900 dark:text-white">
                            Nanonets
                          </div>
                          <div className="text-xs text-gray-600 dark:text-gray-400">
                            Gelişmiş OCR dönüştürme
                          </div>
                        </div>
                      </button>
                      
                      {/* Diğer Seçenekler */}
                      <div className="border-t border-gray-200 dark:border-gray-700 pt-2 pb-2">
                        <label className="w-full cursor-pointer px-4 py-2 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors flex items-center gap-3">
                          <input
                            type="file"
                            accept=".md"
                            className="hidden"
                            onChange={(e) => {
                              handleUploadMd(e.target.files?.[0] || null);
                              setMarkerDropdownOpen(false);
                            }}
                            disabled={uploadingMd}
                          />
                          <span className="text-xl">📄</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            Markdown Yükle
                          </span>
                        </label>
                        
                        <button
                          onClick={() => {
                            setShowPdfplumber(true);
                            setMarkerDropdownOpen(false);
                          }}
                          className="w-full text-left px-4 py-2 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors flex items-center gap-3"
                        >
                          <span className="text-xl">⚡</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            Hızlı Dönüştür
                          </span>
                        </button>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
          <div className="mt-4">
            {loading ? (
              <div className="text-sm text-muted-foreground">
                Liste yükleniyor...
              </div>
            ) : markdownFiles.length === 0 ? (
              <div className="text-sm text-muted-foreground">
                Henüz markdown dosyası yok.
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between mb-2 text-sm text-muted-foreground">
                  <span>{markdownFiles.length} dosya</span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      disabled={page === 1}
                      className="px-3 py-1 rounded-md border text-foreground disabled:opacity-50"
                    >
                      Önceki
                    </button>
                    <span>
                      Sayfa {page} / {totalPages}
                    </span>
                    <button
                      onClick={() =>
                        setPage((p) => Math.min(totalPages, p + 1))
                      }
                      disabled={page >= totalPages}
                      className="px-3 py-1 rounded-md border text-foreground disabled:opacity-50"
                    >
                      Sonraki
                    </button>
                  </div>
                </div>
                <ul className="divide-y divide-gray-200 dark:divide-gray-700 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
                  {pagedFiles.map((f) => (
                    <li
                      key={f}
                      className="px-4 py-3 text-sm text-foreground flex items-center justify-between hover:bg-muted/40 cursor-pointer"
                      onClick={() => openViewer(f)}
                    >
                      <span className="truncate max-w-[70%]" title={f}>
                        {f}
                      </span>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDownload(f);
                          }}
                          className="px-3 py-1 rounded-md border hover:bg-muted"
                          title="İndir"
                        >
                          ⬇️
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(f);
                          }}
                          className="px-3 py-1 rounded-md border text-red-600 hover:bg-red-50"
                          title="Sil"
                        >
                          🗑️
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </div>

        {/* Modals */}
        <DocumentUploadModal
          isOpen={showNanonets}
          conversionMethod="nanonets"
          onClose={() => setShowNanonets(false)}
          onSuccess={(message) => {
            setSuccess(message);
            setShowNanonets(false);
          }}
          onError={(msg) => setError(msg)}
          onMarkdownFilesUpdate={refreshFiles}
        />
        <DocumentUploadModal
          isOpen={showPdfplumber}
          conversionMethod="pdfplumber"
          onClose={() => setShowPdfplumber(false)}
          onSuccess={(message) => {
            setSuccess(message);
            setShowPdfplumber(false);
          }}
          onError={(msg) => setError(msg)}
          onMarkdownFilesUpdate={refreshFiles}
        />
        <DocumentUploadModal
          isOpen={showMarker}
          conversionMethod="marker"
          onClose={() => setShowMarker(false)}
          onSuccess={(message) => {
            setSuccess(message);
            setShowMarker(false);
          }}
          onError={(msg) => setError(msg)}
          onMarkdownFilesUpdate={refreshFiles}
        />

        {/* Markdown Viewer Modal */}
        {isModalOpen && (
          <div
            className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4"
            onClick={closeViewer}
          >
            <div
              className="bg-white dark:bg-gray-800 w-full max-w-4xl rounded-xl shadow-2xl overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between px-5 py-3 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-3">
                  <span className="font-semibold text-gray-900 dark:text-white truncate max-w-[50vw]">
                    {selectedFileName}
                  </span>
                  {contentLoading && (
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      Yükleniyor...
                    </span>
                  )}
                  {modalError && (
                    <span className="text-xs text-red-600 dark:text-red-400">
                      {modalError}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleDownload(selectedFileName)}
                    className="px-3 py-1 rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                    title="İndir"
                    disabled={contentLoading}
                  >
                    İndir
                  </button>
                  <button
                    onClick={() => handleDelete(selectedFileName)}
                    className="px-3 py-1 rounded-md border border-red-300 dark:border-red-700 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20"
                    title="Sil"
                    disabled={contentLoading}
                  >
                    Sil
                  </button>
                  <button
                    onClick={closeViewer}
                    className="px-3 py-1 rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                  >
                    Kapat
                  </button>
                </div>
              </div>
              <div className="max-h-[70vh] overflow-auto p-5 bg-gray-50 dark:bg-gray-900/50">
                <pre className="whitespace-pre-wrap text-sm leading-relaxed font-mono text-gray-900 dark:text-gray-100">
                  {selectedFileContent || (contentLoading ? "" : "İçerik boş")}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </ModernAdminLayout>
  );
}
