"use client";
import { useState, useEffect } from "react";
import {
  listMarkdownFilesWithCategories,
  listMarkdownCategories,
  createMarkdownCategory,
  deleteMarkdownCategory,
  assignMarkdownCategory,
  getMarkdownFileContent,
  deleteMarkdownFile,
  deleteAllMarkdownFiles,
  MarkdownFileWithCategory,
  MarkdownCategory,
} from "@/lib/api";

export function useDocumentCenter() {
  // Markdown files
  const [markdownFiles, setMarkdownFiles] = useState<
    MarkdownFileWithCategory[]
  >([]);
  const [filteredMarkdownFiles, setFilteredMarkdownFiles] = useState<
    MarkdownFileWithCategory[]
  >([]);
  const [selectedMarkdownFiles, setSelectedMarkdownFiles] = useState<string[]>(
    []
  );

  // Categories
  const [categories, setCategories] = useState<MarkdownCategory[]>([]);
  const [selectedCategoryId, setSelectedCategoryId] = useState<number | null>(
    null
  );

  // Loading states
  const [markdownLoading, setMarkdownLoading] = useState(false);
  const [isCategoryLoading, setIsCategoryLoading] = useState(false);

  // Messages
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Pagination
  const [markdownPage, setMarkdownPage] = useState(1);
  const MARKDOWN_FILES_PER_PAGE = 20;

  // Modal states
  const [isCategoryModalOpen, setIsCategoryModalOpen] = useState(false);
  const [newCategoryName, setNewCategoryName] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedFileContent, setSelectedFileContent] = useState<string>("");
  const [selectedFileName, setSelectedFileName] = useState<string>("");
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);
  const [uploadStats, setUploadStats] = useState<any>(null);

  // Auto-clear success/error messages
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 8000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Markdown files
  const fetchMarkdownFiles = async () => {
    try {
      setMarkdownLoading(true);
      setError(null);

      const [files, cats] = await Promise.all([
        listMarkdownFilesWithCategories(selectedCategoryId || undefined),
        listMarkdownCategories(),
      ]);

      setMarkdownFiles(files);
      setFilteredMarkdownFiles(files);
      setCategories(cats);
    } catch (e: any) {
      setError(e.message || "Markdown dosyaları yüklenemedi");
    } finally {
      setMarkdownLoading(false);
    }
  };

  const handleCategoryFilter = (categoryId: number | null) => {
    setSelectedCategoryId(categoryId);
    setMarkdownPage(1);

    if (!categoryId) {
      setFilteredMarkdownFiles(markdownFiles);
    } else {
      setFilteredMarkdownFiles(
        markdownFiles.filter((file) => file.category_id === categoryId)
      );
    }
  };

  const handleCreateCategory = async () => {
    if (!newCategoryName.trim()) return;

    try {
      setIsCategoryLoading(true);
      const category = await createMarkdownCategory({
        name: newCategoryName.trim(),
        description: null,
      });

      setCategories([...categories, category]);
      setNewCategoryName("");
      setSuccess(`"${category.name}" kategorisi oluşturuldu`);
    } catch (e: any) {
      setError(e.message || "Kategori oluşturulurken hata oluştu");
    } finally {
      setIsCategoryLoading(false);
    }
  };

  const handleDeleteCategory = async (id: number) => {
    if (
      !confirm(
        "Bu kategoriyi silmek istediğinize emin misiniz? Bu işlem geri alınamaz."
      )
    ) {
      return;
    }

    try {
      setIsCategoryLoading(true);
      await deleteMarkdownCategory(id);
      setCategories(categories.filter((cat) => cat.id !== id));
      setSuccess("Kategori silindi");

      if (selectedCategoryId === id) {
        setSelectedCategoryId(null);
        setFilteredMarkdownFiles(markdownFiles);
      }
    } catch (e: any) {
      setError(e.message || "Kategori silinirken hata oluştu");
    } finally {
      setIsCategoryLoading(false);
    }
  };

  const handleAssignCategory = async (categoryId: number | null) => {
    if (selectedMarkdownFiles.length === 0) return;

    try {
      setMarkdownLoading(true);
      await assignMarkdownCategory(selectedMarkdownFiles, categoryId);

      const updatedFiles = markdownFiles.map((file) =>
        selectedMarkdownFiles.includes(file.filename)
          ? {
              ...file,
              category_id: categoryId,
              category_name: categoryId
                ? categories.find((c) => c.id === categoryId)?.name || null
                : null,
            }
          : file
      );

      setMarkdownFiles(updatedFiles);
      setFilteredMarkdownFiles(
        updatedFiles.filter(
          (file) =>
            !selectedCategoryId || file.category_id === selectedCategoryId
        )
      );

      setSuccess(
        `Seçili dosyalar ${
          categoryId ? "kategoriye eklendi" : "kategorisiz olarak işaretlendi"
        }`
      );
      setSelectedMarkdownFiles([]);
    } catch (e: any) {
      setError(e.message || "Kategori atanırken hata oluştu");
    } finally {
      setMarkdownLoading(false);
    }
  };

  // File operations
  const handleFileToggle = (filename: string) => {
    setSelectedMarkdownFiles((prev) =>
      prev.includes(filename)
        ? prev.filter((f) => f !== filename)
        : [...prev, filename]
    );
  };

  const handleViewMarkdownFile = async (filename: string) => {
    try {
      setIsLoadingContent(true);
      setIsModalOpen(true);
      setSelectedFileName(filename);
      setModalError(null);
      setSelectedFileContent("");

      const result = await getMarkdownFileContent(filename);
      setSelectedFileContent(result.content);
    } catch (e: any) {
      setModalError(e.message || "Dosya içeriği yüklenemedi");
    } finally {
      setIsLoadingContent(false);
    }
  };

  const handleDownloadMarkdownFile = async (filename: string) => {
    try {
      const result = await getMarkdownFileContent(filename);

      const blob = new Blob([result.content], { type: "text/markdown" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;

      document.body.appendChild(link);
      link.click();

      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e.message || "Dosya indirilemedi");
    }
  };

  const handleDeleteFile = async (filename: string) => {
    if (!confirm(`'${filename}' dosyasını silmek istiyor musunuz?`)) return;

    try {
      await deleteMarkdownFile(filename);
      setSelectedMarkdownFiles((prev) => prev.filter((f) => f !== filename));
      await fetchMarkdownFiles();
    } catch (e: any) {
      setError(e.message || "Silinemedi");
    }
  };

  const handleDeleteAllFiles = async () => {
    if (!confirm("Tüm markdown dosyalarını silmek istiyor musunuz?")) return;

    try {
      await deleteAllMarkdownFiles();
      setSelectedMarkdownFiles([]);
      await fetchMarkdownFiles();
    } catch (e: any) {
      setError(e.message || "Tüm dosyalar silinemedi");
    }
  };

  const handleDeleteSelectedFiles = async () => {
    if (selectedMarkdownFiles.length === 0) return;
    if (
      !confirm(
        `${selectedMarkdownFiles.length} dosyayı silmek istiyor musunuz?`
      )
    )
      return;

    try {
      for (const filename of selectedMarkdownFiles) {
        await deleteMarkdownFile(filename);
      }
      setSelectedMarkdownFiles([]);
      await fetchMarkdownFiles();
    } catch (e: any) {
      setError(e.message || "Seçili dosyalar silinemedi");
    }
  };

  // Modal handlers
  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedFileContent("");
    setSelectedFileName("");
    setModalError(null);
    setIsLoadingContent(false);
  };

  // Refresh all data
  const refreshData = async () => {
    await fetchMarkdownFiles();
  };

  useEffect(() => {
    fetchMarkdownFiles();
  }, [selectedCategoryId]);

  return {
    // States
    markdownFiles,
    filteredMarkdownFiles,
    selectedMarkdownFiles,
    categories,
    selectedCategoryId,
    markdownLoading,
    isCategoryLoading,
    error,
    success,
    markdownPage,
    setMarkdownPage,
    MARKDOWN_FILES_PER_PAGE,

    // Modal states
    isCategoryModalOpen,
    setIsCategoryModalOpen,
    newCategoryName,
    setNewCategoryName,
    isModalOpen,
    selectedFileContent,
    selectedFileName,
    isLoadingContent,
    modalError,
    uploadStats,
    setUploadStats,

    // Actions
    fetchMarkdownFiles,
    handleCategoryFilter,
    handleCreateCategory,
    handleDeleteCategory,
    handleAssignCategory,
    handleFileToggle,
    handleViewMarkdownFile,
    handleDownloadMarkdownFile,
    handleDeleteFile,
    handleDeleteAllFiles,
    handleDeleteSelectedFiles,
    handleCloseModal,
    refreshData,
  };
}

export type { MarkdownFileWithCategory, MarkdownCategory };
