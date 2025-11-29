"use client";
import React, { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useDocumentCenter } from "@/hooks/useDocumentCenter";
import { FilterBar } from "@/components/DocumentCenter/FilterBar";
import { FileList } from "@/components/DocumentCenter/FileList";
import { UploadSection } from "@/components/DocumentCenter/UploadSection";
import { CategoryManager } from "@/components/DocumentCenter/CategoryManager";
import Modal from "@/components/Modal";
import MarkdownViewer from "@/components/MarkdownViewer";
import EnhancedDocumentUploadModal from "@/components/EnhancedDocumentUploadModal";
import TeacherLayout from "@/app/components/TeacherLayout";

// Types
interface Category {
  id: number;
  name: string;
}

export default function DocumentCenterPage() {
  const router = useRouter();
  const { user } = useAuth();

  // Upload modal state
  const [isDocumentUploadModalOpen, setIsDocumentUploadModalOpen] =
    useState(false);

  // Custom hook ile tüm state management
  const {
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
  } = useDocumentCenter();

  // Sadece öğretmenler erişebilir
  const userRole = user?.role_name?.toLowerCase() || "teacher";
  if (userRole === "student") {
    router.push("/student");
    return null;
  }

  const handleDocumentUploadSuccess = (message: string) => {
    setUploadStats({ message });
    fetchMarkdownFiles();
  };

  const handleDocumentUploadError = (error: string) => {
    console.error("Upload error:", error);
  };

  return (
    <TeacherLayout activeTab="upload">
      <div className="space-y-8">
        {/* Header */}
        <div className="bg-gradient-to-br from-blue-50 via-white to-purple-50 rounded-2xl p-8 border border-blue-100">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="p-4 bg-gradient-to-br from-blue-500 to-purple-600 text-white rounded-2xl shadow-lg">
                <svg
                  className="w-8 h-8"
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
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Belge Merkezi
                </h1>
                <p className="text-gray-600 mt-1">
                  PDF dosyalarını Markdown'a dönüştürün, kategorize edin ve
                  yönetin
                </p>
              </div>
            </div>
          </div>

          {/* Success/Error Messages */}
          {success && (
            <div className="mb-6 bg-green-50 border border-green-200 text-green-800 p-4 rounded-xl flex items-center gap-3">
              <svg
                className="w-6 h-6 text-green-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              {success}
            </div>
          )}

          {error && (
            <div className="mb-6 bg-red-50 border border-red-200 text-red-800 p-4 rounded-xl flex items-center gap-3">
              <svg
                className="w-6 h-6 text-red-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              {error}
            </div>
          )}
        </div>

        {/* Upload Section */}
        <UploadSection
          onOpenUploadModal={() => setIsDocumentUploadModalOpen(true)}
        />

        {/* Main Content */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
          <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-8 py-6 border-b border-gray-200">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">
              Markdown Belge Yönetimi
            </h2>
            <p className="text-gray-600">
              Markdown dosyalarınızı kategorize edin, görüntüleyin ve yönetin
            </p>
          </div>

          <div className="p-8">
            {/* Filter Bar with Additional Actions */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-3">
                <FilterBar
                  categories={categories}
                  selectedCategoryId={selectedCategoryId}
                  onCategoryChange={handleCategoryFilter}
                  selectedFilesCount={selectedMarkdownFiles.length}
                  onRefresh={refreshData}
                />

                <div className="flex items-center gap-2">
                  {/* Assign Category Dropdown */}
                  <div className="relative group">
                    <select
                      value=""
                      onChange={(e) =>
                        handleAssignCategory(
                          e.target.value ? Number(e.target.value) : null
                        )
                      }
                      disabled={selectedMarkdownFiles.length === 0}
                      className={`appearance-none bg-white border border-gray-300 rounded-md pl-3 pr-8 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                        selectedMarkdownFiles.length === 0
                          ? "opacity-50 cursor-not-allowed"
                          : "cursor-pointer"
                      }`}
                    >
                      <option value="">
                        Kategori Ata ({selectedMarkdownFiles.length})
                      </option>
                      <option value="">Kategorisiz</option>
                      {categories.map((cat: Category) => (
                        <option key={cat.id} value={cat.id}>
                          {cat.name}
                        </option>
                      ))}
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

                  <button
                    onClick={() => setIsCategoryModalOpen(true)}
                    className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md transition-colors flex items-center gap-1.5"
                  >
                    <span>📂</span>
                    <span>Kategorileri Yönet</span>
                  </button>

                  <button
                    onClick={handleDeleteSelectedFiles}
                    className="text-sm text-red-600 hover:text-red-800 px-3 py-1.5 rounded-md hover:bg-red-50 transition-colors"
                    disabled={selectedMarkdownFiles.length === 0}
                    title="Seçili dosyaları sil"
                  >
                    🗑️ Seçilileri Sil
                  </button>

                  <button
                    onClick={handleDeleteAllFiles}
                    className="text-sm text-red-600 hover:text-red-800 px-3 py-1.5 rounded-md hover:bg-red-50 transition-colors"
                    title="Tümünü sil"
                  >
                    🧹 Tümünü Sil
                  </button>
                </div>
              </div>
            </div>

            {/* File List */}
            <FileList
              files={filteredMarkdownFiles}
              selectedFiles={selectedMarkdownFiles}
              onFileToggle={handleFileToggle}
              loading={markdownLoading}
              onViewFile={handleViewMarkdownFile}
              onDownloadFile={handleDownloadMarkdownFile}
              onDeleteFile={handleDeleteFile}
              markdownPage={markdownPage}
              setMarkdownPage={setMarkdownPage}
              MARKDOWN_FILES_PER_PAGE={MARKDOWN_FILES_PER_PAGE}
            />

            {/* Upload Stats */}
            {uploadStats && (
              <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-md">
                <h3 className="text-green-800 font-medium">
                  {uploadStats.message ||
                    "PDF'ler Başarıyla Markdown'a Dönüştürüldü!"}
                </h3>
                {uploadStats.processed_count && (
                  <div className="mt-2 text-sm text-green-700">
                    <p>
                      Dönüştürülen dosya sayısı: {uploadStats.processed_count}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Modals */}

        {/* Enhanced Document Upload Modal */}
        <EnhancedDocumentUploadModal
          isOpen={isDocumentUploadModalOpen}
          onClose={() => setIsDocumentUploadModalOpen(false)}
          onSuccess={handleDocumentUploadSuccess}
          onError={handleDocumentUploadError}
          onMarkdownFilesUpdate={fetchMarkdownFiles}
        />

        {/* Markdown Viewer Modal */}
        <Modal
          isOpen={isModalOpen}
          onClose={handleCloseModal}
          title={
            selectedFileName
              ? `${selectedFileName.replace(".md", "")}`
              : "Markdown Dosyası"
          }
          size="2xl"
        >
          <MarkdownViewer
            content={selectedFileContent}
            isLoading={isLoadingContent}
            error={modalError}
          />
        </Modal>

        {/* Category Management Modal */}
        <CategoryManager
          isOpen={isCategoryModalOpen}
          onClose={() => setIsCategoryModalOpen(false)}
          categories={categories}
          newCategoryName={newCategoryName}
          setNewCategoryName={setNewCategoryName}
          onCreateCategory={handleCreateCategory}
          onDeleteCategory={handleDeleteCategory}
          isLoading={isCategoryLoading}
          selectedMarkdownFiles={selectedMarkdownFiles}
          onAssignCategory={handleAssignCategory}
        />
      </div>
    </TeacherLayout>
  );
}
