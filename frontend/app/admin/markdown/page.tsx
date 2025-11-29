"use client";
import React, { useEffect, useMemo, useState } from "react";
import DocumentUploadModal from "@/components/DocumentUploadModal";
import {
  listMarkdownFilesWithCategories,
  listMarkdownCategories,
  createMarkdownCategory,
  deleteMarkdownCategory,
  assignMarkdownCategory,
  type MarkdownCategory,
  type MarkdownFileWithCategory,
  uploadMarkdownFile,
  getMarkdownFileContent,
  deleteMarkdownFile,
} from "@/lib/api";
import ModernAdminLayout from "../components/ModernAdminLayout";

export default function DocumentCenterPage() {
  const [markdownFiles, setMarkdownFiles] = useState<
    MarkdownFileWithCategory[]
  >([]);
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

  // Category state
  const [categories, setCategories] = useState<MarkdownCategory[]>([]);
  const [selectedCategoryId, setSelectedCategoryId] = useState<number | null>(
    null
  );
  const [categoryModalOpen, setCategoryModalOpen] = useState(false);
  const [newCategoryName, setNewCategoryName] = useState("");
  const [newCategoryDescription, setNewCategoryDescription] = useState("");

  // Selection for assigning categories
  const [selectedFilenames, setSelectedFilenames] = useState<Set<string>>(
    new Set()
  );

  const refreshFiles = async (options?: {
    desiredPage?: number;
    preservePage?: boolean;
  }) => {
    try {
      setLoading(true);
      const [files, cats] = await Promise.all([
        listMarkdownFilesWithCategories(),
        listMarkdownCategories(),
      ]);
      // Sort alphabetically (case-insensitive) by filename
      const sorted = [...files].sort((a, b) =>
        a.filename.localeCompare(b.filename, "tr", { sensitivity: "accent" })
      );
      setMarkdownFiles(sorted);
      setCategories(cats);
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

  const toggleFileSelection = (filename: string) => {
    setSelectedFilenames((prev) => {
      const next = new Set(prev);
      if (next.has(filename)) next.delete(filename);
      else next.add(filename);
      return next;
    });
  };

  const clearSelection = () => setSelectedFilenames(new Set());

  const handleAssignCategory = async (categoryId: number | null) => {
    try {
      if (selectedFilenames.size === 0) return;
      await assignMarkdownCategory(Array.from(selectedFilenames), categoryId);
      setSuccess("Kategori ataması güncellendi");
      clearSelection();
      await refreshFiles({ preservePage: true });
    } catch (e: any) {
      setError(e.message || "Kategori ataması başarısız");
    }
  };

  const handleCreateCategory = async () => {
    try {
      if (!newCategoryName.trim()) return;
      await createMarkdownCategory({
        name: newCategoryName.trim(),
        description: newCategoryDescription.trim() || undefined,
      });
      setNewCategoryName("");
      setNewCategoryDescription("");
      const cats = await listMarkdownCategories();
      setCategories(cats);
    } catch (e: any) {
      setError(e.message || "Kategori oluşturma başarısız");
    }
  };

  const handleDeleteCategory = async (id: number) => {
    if (
      !confirm(
        "Kategoriyi silmek istiyor musunuz? İlişkili belgelerin kategorisi boşalır."
      )
    )
      return;
    try {
      await deleteMarkdownCategory(id);
      const cats = await listMarkdownCategories();
      setCategories(cats);
      if (selectedCategoryId === id) setSelectedCategoryId(null);
      await refreshFiles({ preservePage: true });
    } catch (e: any) {
      setError(e.message || "Kategori silme başarısız");
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

  const filteredFiles = useMemo(() => {
    if (selectedCategoryId === null) return markdownFiles;
    return markdownFiles.filter((f) => f.category_id === selectedCategoryId);
  }, [markdownFiles, selectedCategoryId]);

  const pagedFiles = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return filteredFiles.slice(start, start + PAGE_SIZE);
  }, [filteredFiles, page]);

  const totalPages = useMemo(
    () => Math.max(1, Math.ceil(filteredFiles.length / PAGE_SIZE)),
    [filteredFiles.length]
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

          {/* Category filter & actions */}
          <div className="flex flex-col md:flex-row gap-4 items-start md:items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Kategori:</span>
              <select
                className="border rounded-md px-2 py-1 text-sm bg-white dark:bg-gray-900"
                value={selectedCategoryId ?? ""}
                onChange={(e) => {
                  const v = e.target.value;
                  setSelectedCategoryId(v === "" ? null : Number(v));
                  setPage(1);
                }}
              >
                <option value="">Tümü</option>
                {categories.map((cat) => (
                  <option key={cat.id} value={cat.id}>
                    {cat.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setCategoryModalOpen(true)}
                className="px-3 py-1.5 text-xs rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800"
              >
                Kategorileri Yönet
              </button>
              {selectedFilenames.size > 0 && (
                <>
                  <select
                    className="border rounded-md px-2 py-1 text-xs bg-white dark:bg-gray-900"
                    onChange={(e) => {
                      const v = e.target.value;
                      if (v === "__none__") {
                        handleAssignCategory(null);
                      } else if (v) {
                        handleAssignCategory(Number(v));
                      }
                      e.target.value = "";
                    }}
                    defaultValue=""
                  >
                    <option value="" disabled>
                      Seçili belgeleri kategoriye ata
                    </option>
                    <option value="__none__">Kategori yok</option>
                    {categories.map((cat) => (
                      <option key={cat.id} value={cat.id}>
                        {cat.name}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={clearSelection}
                    className="px-2 py-1 text-xs rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800"
                  >
                    Seçimi Temizle
                  </button>
                </>
              )}
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
                  <span className="text-xs bg-orange-500 px-2 py-0.5 rounded">
                    Önerilen
                  </span>
                  <svg
                    className={`w-4 h-4 transition-transform ${
                      markerDropdownOpen ? "rotate-180" : ""
                    }`}
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

          {/* Existing markdown files */}
          <div className="mt-8">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">
                Mevcut Markdown Dosyaları ({filteredFiles.length})
              </h3>
            </div>

            {loading ? (
              <div className="text-center py-4">
                <span className="text-gray-600">Yükleniyor...</span>
              </div>
            ) : pagedFiles.length > 0 ? (
              <div className="grid gap-3">
                {pagedFiles.map((file) => (
                  <div
                    key={file.filename}
                    className="flex items-center gap-4 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-700"
                  >
                    <input
                      type="checkbox"
                      checked={selectedFilenames.has(file.filename)}
                      onChange={() => toggleFileSelection(file.filename)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-gray-900 dark:text-white truncate">
                        {file.filename}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {file.category_name && (
                          <span className="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-2 py-0.5 rounded text-xs mr-2">
                            {file.category_name}
                          </span>
                        )}
                        {file.created_at &&
                          `Eklendi: ${new Date(
                            file.created_at
                          ).toLocaleDateString("tr-TR")}`}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => openViewer(file.filename)}
                        className="px-3 py-1 text-sm rounded-md border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700"
                      >
                        Görüntüle
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                {selectedCategoryId
                  ? "Bu kategoride dosya bulunamadı"
                  : "Henüz markdown dosyası yok"}
              </div>
            )}

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex justify-center gap-2 mt-6">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md disabled:opacity-50 hover:bg-gray-50 dark:hover:bg-gray-800"
                >
                  Önceki
                </button>
                <span className="px-3 py-1 text-sm">
                  {page} / {totalPages}
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                  className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md disabled:opacity-50 hover:bg-gray-50 dark:hover:bg-gray-800"
                >
                  Sonraki
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Document Upload Modals */}
        {showNanonets && (
          <DocumentUploadModal
            isOpen={showNanonets}
            onClose={() => setShowNanonets(false)}
            conversionMethod="nanonets"
            onSuccess={() => {
              setSuccess("Dosya başarıyla yüklendi ve işleme alındı");
              refreshFiles();
            }}
            onError={(err) => setError(err)}
            onMarkdownFilesUpdate={refreshFiles}
          />
        )}

        {showPdfplumber && (
          <DocumentUploadModal
            isOpen={showPdfplumber}
            onClose={() => setShowPdfplumber(false)}
            conversionMethod="pdfplumber"
            onSuccess={() => {
              setSuccess("Dosya başarıyla yüklendi ve işleme alındı");
              refreshFiles();
            }}
            onError={(err) => setError(err)}
            onMarkdownFilesUpdate={refreshFiles}
          />
        )}

        {showMarker && (
          <DocumentUploadModal
            isOpen={showMarker}
            onClose={() => setShowMarker(false)}
            conversionMethod="marker"
            onSuccess={() => {
              setSuccess("Dosya başarıyla yüklendi ve işleme alındı");
              refreshFiles();
            }}
            onError={(err) => setError(err)}
            onMarkdownFilesUpdate={refreshFiles}
          />
        )}

        {/* Category Management Modal */}
        {categoryModalOpen && (
          <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-md">
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                <h2 className="text-lg font-semibold">Kategorileri Yönet</h2>
              </div>
              <div className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    Kategori Adı
                  </label>
                  <input
                    type="text"
                    value={newCategoryName}
                    onChange={(e) => setNewCategoryName(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-900"
                    placeholder="Kategori adı girin"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">
                    Açıklama (isteğe bağlı)
                  </label>
                  <textarea
                    value={newCategoryDescription}
                    onChange={(e) => setNewCategoryDescription(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-900"
                    placeholder="Kategori açıklaması"
                    rows={3}
                  />
                </div>
                <button
                  onClick={handleCreateCategory}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                  disabled={!newCategoryName.trim()}
                >
                  Kategori Oluştur
                </button>

                <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                  <h3 className="text-sm font-medium mb-2">
                    Mevcut Kategoriler
                  </h3>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {categories.map((cat) => (
                      <div
                        key={cat.id}
                        className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-900/50 rounded"
                      >
                        <div>
                          <div className="font-medium">{cat.name}</div>
                          {cat.description && (
                            <div className="text-xs text-gray-600 dark:text-gray-400">
                              {cat.description}
                            </div>
                          )}
                        </div>
                        <button
                          onClick={() => handleDeleteCategory(cat.id)}
                          className="text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 px-2 py-1 rounded text-xs"
                        >
                          Sil
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 flex justify-end">
                <button
                  onClick={() => setCategoryModalOpen(false)}
                  className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-800"
                >
                  Kapat
                </button>
              </div>
            </div>
          </div>
        )}

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
