"use client";

import React, { useState, useEffect } from "react";
import { LearningSession, AdminUser } from "@/lib/admin-api";
import { getRecentInteractions, getApiUrl } from "@/lib/api";

interface SessionsModalProps {
  session: LearningSession;
  users: AdminUser[];
  onClose: () => void;
}

interface Interaction {
  interaction_id: number;
  user_id: string;
  session_id: string;
  timestamp: string;
  query: string;
  response: string;
  processing_time_ms?: number;
  success?: boolean;
  error_message?: string;
  chain_type?: string;
  sources?: Array<{
    content?: string;
    text?: string;
    metadata?: any;
    score?: number;
    crag_score?: number;
  }>;
}

export default function SessionsModal({
  session,
  users,
  onClose,
}: SessionsModalProps) {
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [loadingInteractions, setLoadingInteractions] = useState(false);
  const [interactionsPage, setInteractionsPage] = useState(1);
  const [interactionsTotal, setInteractionsTotal] = useState(0);
  const [apragEnabled, setApragEnabled] = useState<boolean>(false);

  // Get teacher name
  const getTeacherName = (createdBy: string): string => {
    const userById = users.find((u) => u.id.toString() === createdBy);
    if (userById) {
      return `${userById.first_name} ${userById.last_name}`;
    }
    const userByUsername = users.find((u) => u.username === createdBy);
    if (userByUsername) {
      return `${userByUsername.first_name} ${userByUsername.last_name}`;
    }
    return createdBy;
  };

  // Check APRAG status
  useEffect(() => {
    const checkApragStatus = async () => {
      try {
        const response = await fetch(
          `${getApiUrl()}/aprag/settings/status`
        );
        if (response.ok) {
          const data = await response.json();
          setApragEnabled(data.global_enabled || false);
        }
      } catch (err) {
        console.error("Failed to check APRAG status:", err);
        setApragEnabled(false);
      }
    };
    checkApragStatus();
  }, []);

  // Fetch interactions (only if APRAG is enabled)
  useEffect(() => {
    if (session?.session_id && apragEnabled) {
      fetchInteractions();
    }
  }, [session?.session_id, interactionsPage, apragEnabled]);

  const fetchInteractions = async () => {
    setLoadingInteractions(true);
    try {
      const result = await getRecentInteractions({
        session_id: session.session_id,
        page: interactionsPage,
        limit: 10,
      });
      setInteractions(result.items || []);
      setInteractionsTotal(result.count || 0);
    } catch (error) {
      console.error("Failed to fetch interactions:", error);
      setInteractions([]);
    } finally {
      setLoadingInteractions(false);
    }
  };
  const formatDate = (dateString: string | null) => {
    if (!dateString) return "Bilinmiyor";
    return new Date(dateString).toLocaleDateString("tr-TR", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getStatusBadge = (status: string) => {
    const isActive = status === "active";
    return (
      <span
        className={`inline-flex px-3 py-1 text-sm font-semibold rounded-full ${
          isActive
            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
            : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
        }`}
      >
        {isActive ? "Aktif" : "İnaktif"}
      </span>
    );
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        {/* Background overlay */}
        <div
          className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity backdrop-blur-sm"
          onClick={onClose}
        ></div>

        {/* Modal */}
        <div className="inline-block align-bottom bg-white dark:bg-gray-800 rounded-lg px-6 pt-6 pb-6 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl sm:w-full">
          {/* Header */}
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg
                    className="w-6 h-6 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
                    />
                  </svg>
                </div>
              </div>
              <div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                  {session.name}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  Ders Oturumu Detayları
                </p>
              </div>
            </div>
            <button
              type="button"
              className="bg-white dark:bg-gray-800 rounded-md text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors"
              onClick={onClose}
            >
              <span className="sr-only">Kapat</span>
              <svg
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
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

          {/* Content */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Basic Information */}
            <div className="space-y-6">
              <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Genel Bilgiler
                </h4>
                <div className="space-y-3">
                  <div>
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Oturum ID
                    </label>
                    <p className="text-sm text-gray-900 dark:text-white font-mono bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded mt-1">
                      {session.session_id}
                    </p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Açıklama
                    </label>
                    <p className="text-sm text-gray-900 dark:text-white mt-1">
                      {session.description || "Açıklama bulunmuyor"}
                    </p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Durum
                    </label>
                    <div className="mt-1">{getStatusBadge(session.status)}</div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Kategori
                      </label>
                      <p className="text-sm text-gray-900 dark:text-white mt-1">
                        {session.category}
                      </p>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Seviye
                      </label>
                      <p className="text-sm text-gray-900 dark:text-white mt-1">
                        {session.grade_level}
                      </p>
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Konu Alanı
                    </label>
                    <p className="text-sm text-gray-900 dark:text-white mt-1">
                      {session.subject_area}
                    </p>
                  </div>
                </div>
              </div>

              {/* Teacher Information */}
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                  <svg
                    className="w-5 h-5 text-blue-600 dark:text-blue-400 mr-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                    />
                  </svg>
                  Öğretmen Bilgileri
                </h4>
                <div className="space-y-3">
                  <div>
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Oluşturan Öğretmen
                    </label>
                    <p className="text-sm text-gray-900 dark:text-white mt-1 font-medium">
                      {getTeacherName(session.created_by)}
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Oluşturma Tarihi
                      </label>
                      <p className="text-sm text-gray-900 dark:text-white mt-1">
                        {formatDate(session.created_at)}
                      </p>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Son Güncelleme
                      </label>
                      <p className="text-sm text-gray-900 dark:text-white mt-1">
                        {formatDate(session.updated_at)}
                      </p>
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Son Erişim
                    </label>
                    <p className="text-sm text-gray-900 dark:text-white mt-1">
                      {formatDate(session.last_accessed)}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Statistics and Learning Objectives */}
            <div className="space-y-6">
              {/* Statistics */}
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                  <svg
                    className="w-5 h-5 text-green-600 dark:text-green-400 mr-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                  İstatistikler
                </h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                      {session.document_count}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Belge Sayısı
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                      {session.total_chunks}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Toplam Chunk
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                      {session.student_entry_count}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Öğrenci Sayısı
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                      {session.query_count}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Soru Sayısı
                    </div>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-green-200 dark:border-green-800">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      Kullanıcı Değerlendirmesi
                    </span>
                    <div className="flex items-center">
                      <div className="flex text-yellow-400">
                        {[...Array(5)].map((_, i) => (
                          <svg
                            key={i}
                            className={`w-4 h-4 ${
                              i < Math.floor(session.user_rating)
                                ? "text-yellow-400"
                                : "text-gray-300"
                            }`}
                            fill="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                          </svg>
                        ))}
                      </div>
                      <span className="ml-2 text-sm text-gray-900 dark:text-white">
                        {session.user_rating.toFixed(1)}
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center mt-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      Görünürlük
                    </span>
                    <span
                      className={`text-sm font-medium ${
                        session.is_public
                          ? "text-green-600 dark:text-green-400"
                          : "text-red-600 dark:text-red-400"
                      }`}
                    >
                      {session.is_public ? "Genel" : "Özel"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Learning Objectives */}
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                  <svg
                    className="w-5 h-5 text-purple-600 dark:text-purple-400 mr-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z"
                    />
                  </svg>
                  Öğrenme Hedefleri
                </h4>
                <div className="space-y-2">
                  {session.learning_objectives &&
                  session.learning_objectives.length > 0 ? (
                    session.learning_objectives.map((objective, index) => (
                      <div key={index} className="flex items-start space-x-2">
                        <div className="flex-shrink-0 w-2 h-2 bg-purple-600 dark:bg-purple-400 rounded-full mt-2"></div>
                        <p className="text-sm text-gray-900 dark:text-white">
                          {objective}
                        </p>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500 dark:text-gray-400 italic">
                      Öğrenme hedefleri belirtilmemiş
                    </p>
                  )}
                </div>
              </div>

              {/* Tags */}
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                  <svg
                    className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mr-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
                    />
                  </svg>
                  Etiketler
                </h4>
                <div className="flex flex-wrap gap-2">
                  {session.tags && session.tags.length > 0 ? (
                    session.tags.map((tag, index) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200"
                      >
                        {tag}
                      </span>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500 dark:text-gray-400 italic">
                      Etiket bulunmuyor
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Student Questions and Answers - Only when APRAG is enabled */}
          {apragEnabled && (
            <div className="mt-6 bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <svg
                  className="w-5 h-5 text-orange-600 dark:text-orange-400 mr-2"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                  />
                </svg>
                Öğrenci Soruları ve Cevapları ({interactionsTotal})
              </h4>
              {loadingInteractions ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-600 mx-auto"></div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                    Yükleniyor...
                  </p>
                </div>
              ) : interactions.length === 0 ? (
                <p className="text-sm text-gray-500 dark:text-gray-400 italic text-center py-4">
                  Henüz soru sorulmamış
                </p>
              ) : (
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {interactions.map((interaction) => (
                    <div
                      key={interaction.interaction_id}
                      className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-orange-200 dark:border-orange-800"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                              {new Date(interaction.timestamp).toLocaleString(
                                "tr-TR"
                              )}
                            </span>
                            {interaction.processing_time_ms && (
                              <span className="text-xs text-gray-400 dark:text-gray-500">
                                • {interaction.processing_time_ms}ms
                              </span>
                            )}
                            {interaction.chain_type && (
                              <span className="text-xs px-2 py-0.5 bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-200 rounded">
                                {interaction.chain_type}
                              </span>
                            )}
                          </div>
                          <div className="mb-2">
                            <div className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-1">
                              Soru:
                            </div>
                            <div className="text-sm text-gray-900 dark:text-white bg-gray-50 dark:bg-gray-900/50 p-2 rounded">
                              {interaction.query}
                            </div>
                          </div>
                          <div className="mb-2">
                            <div className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-1">
                              Cevap:
                            </div>
                            <div className="text-sm text-gray-900 dark:text-white bg-green-50 dark:bg-green-900/20 p-2 rounded">
                              {interaction.response || "Cevap bulunamadı"}
                            </div>
                          </div>
                          {interaction.sources &&
                            interaction.sources.length > 0 && (
                              <div className="mt-2">
                                <div className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-1">
                                  Kaynaklar ({interaction.sources.length}):
                                </div>
                                <div className="space-y-1">
                                  {interaction.sources
                                    .slice(0, 3)
                                    .map((source, idx) => (
                                      <div
                                        key={idx}
                                        className="text-xs bg-purple-50 dark:bg-purple-900/20 p-2 rounded flex items-center justify-between"
                                      >
                                        <span className="text-gray-700 dark:text-gray-300 truncate flex-1">
                                          {source.content?.substring(0, 100) ||
                                            source.text?.substring(0, 100) ||
                                            "Kaynak"}
                                        </span>
                                        <div className="flex items-center gap-2 ml-2">
                                          {source.score !== undefined && (
                                            <span className="text-purple-600 dark:text-purple-400 font-medium">
                                              Skor:{" "}
                                              {(source.score * 100).toFixed(1)}%
                                            </span>
                                          )}
                                          {source.crag_score !== undefined && (
                                            <span className="text-orange-600 dark:text-orange-400 font-medium">
                                              DYSK:{" "}
                                              {source.crag_score.toFixed(1)}%
                                            </span>
                                          )}
                                        </div>
                                      </div>
                                    ))}
                                </div>
                              </div>
                            )}
                          {interaction.success === false && (
                            <div className="mt-2 text-xs text-red-600 dark:text-red-400">
                              Hata:{" "}
                              {interaction.error_message || "Bilinmeyen hata"}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {interactionsTotal > 10 && (
                <div className="mt-4 flex items-center justify-between">
                  <button
                    onClick={() =>
                      setInteractionsPage((p) => Math.max(1, p - 1))
                    }
                    disabled={interactionsPage === 1}
                    className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded disabled:opacity-50"
                  >
                    Önceki
                  </button>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    Sayfa {interactionsPage} /{" "}
                    {Math.ceil(interactionsTotal / 10)}
                  </span>
                  <button
                    onClick={() =>
                      setInteractionsPage((p) =>
                        Math.min(Math.ceil(interactionsTotal / 10), p + 1)
                      )
                    }
                    disabled={
                      interactionsPage >= Math.ceil(interactionsTotal / 10)
                    }
                    className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded disabled:opacity-50"
                  >
                    Sonraki
                  </button>
                </div>
              )}
            </div>
          )}

          {/* RAG Settings */}
          {session.rag_settings && (
            <div className="mt-6 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-4">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <svg
                  className="w-5 h-5 text-indigo-600 dark:text-indigo-400 mr-2"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                </svg>
                RAG Ayarları
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Model
                  </label>
                  <p className="text-sm text-gray-900 dark:text-white mt-1">
                    {session.rag_settings.model || "Belirtilmemiş"}
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Chain Türü
                  </label>
                  <p className="text-sm text-gray-900 dark:text-white mt-1">
                    {session.rag_settings.chain_type || "Belirtilmemiş"}
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Top K
                  </label>
                  <p className="text-sm text-gray-900 dark:text-white mt-1">
                    {session.rag_settings.top_k || "Belirtilmemiş"}
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Min Skor
                  </label>
                  <p className="text-sm text-gray-900 dark:text-white mt-1">
                    {session.rag_settings.min_score || "Belirtilmemiş"}
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Rerank Kullan
                  </label>
                  <p className="text-sm text-gray-900 dark:text-white mt-1">
                    {session.rag_settings.use_rerank ? "Evet" : "Hayır"}
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Doğrudan LLM
                  </label>
                  <p className="text-sm text-gray-900 dark:text-white mt-1">
                    {session.rag_settings.use_direct_llm ? "Evet" : "Hayır"}
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Embedding Model
                  </label>
                  <p className="text-sm text-gray-900 dark:text-white mt-1">
                    {session.rag_settings.embedding_model || "Belirtilmemiş"}
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Max Context
                  </label>
                  <p className="text-sm text-gray-900 dark:text-white mt-1">
                    {session.rag_settings.max_context_chars || "Belirtilmemiş"}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="mt-6 flex justify-end">
            <button
              type="button"
              onClick={onClose}
              className="px-6 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors"
            >
              Kapat
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
