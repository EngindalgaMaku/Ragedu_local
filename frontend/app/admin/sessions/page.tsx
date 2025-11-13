"use client";

import React, { useState, useEffect } from "react";
import ModernAdminLayout from "../components/ModernAdminLayout";
import SessionsTable from "./components/SessionsTable";
import SessionsModal from "./components/SessionsModal";
import { adminApiClient, LearningSession, AdminUser } from "@/lib/admin-api";

export default function SessionsPage() {
  const [sessions, setSessions] = useState<LearningSession[]>([]);
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [selectedSession, setSelectedSession] =
    useState<LearningSession | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSessions();
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const data = await adminApiClient.getUsers();
      setUsers(data);
    } catch (error) {
      console.error("Failed to fetch users:", error);
    }
  };

  const fetchSessions = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await adminApiClient.getLearningSessions();
      setSessions(data);
    } catch (error) {
      console.error("Failed to fetch learning sessions:", error);

      // Enhanced error handling with specific error messages
      if (error instanceof Error) {
        if (error.message.includes("ERR_CONNECTION_REFUSED")) {
          setError(
            "Backend servisi bağlantı hatası. Servisin çalışır durumda olduğunu kontrol edin."
          );
        } else if (error.message.includes("404")) {
          setError(
            "Sessions endpoint bulunamadı. Backend API kontrol edilmeli."
          );
        } else if (error.message.includes("500")) {
          setError("Server hatası. Backend loglarını kontrol edin.");
        } else if (error.message.includes("timeout")) {
          setError("Zaman aşımı hatası. Backend servisi yavaş yanıtlıyor.");
        } else {
          setError(`Ders oturumları yüklenemedi: ${error.message}`);
        }
      } else {
        setError("Ders oturumları yüklenemedi. Bilinmeyen hata.");
      }

      setSessions([]);
    } finally {
      setLoading(false);
    }
  };

  const handleViewSession = (session: LearningSession) => {
    setSelectedSession(session);
    setShowModal(true);
  };

  const handleDeleteSession = async (sessionId: string) => {
    if (confirm("Bu ders oturumunu silmek istediğinizden emin misiniz?")) {
      try {
        await adminApiClient.deleteLearningSession(sessionId);
        setSessions(
          sessions.filter((session) => session.session_id !== sessionId)
        );
        setError(null);
      } catch (error) {
        console.error("Failed to delete session:", error);

        if (error instanceof Error) {
          if (error.message.includes("ERR_CONNECTION_REFUSED")) {
            setError(
              "Backend servisi bağlantı hatası. Ders oturumu silinemedi."
            );
          } else if (error.message.includes("404")) {
            setError("Ders oturumu bulunamadı veya zaten silinmiş.");
          } else if (error.message.includes("403")) {
            setError("Bu ders oturumunu silme yetkiniz yok.");
          } else {
            setError(`Ders oturumu silinemedi: ${error.message}`);
          }
        } else {
          setError("Ders oturumu silinemedi. Bilinmeyen hata.");
        }
      }
    }
  };

  const handleToggleStatus = async (
    sessionId: string,
    currentStatus: string
  ) => {
    const newStatus = currentStatus === "active" ? "inactive" : "active";

    try {
      const updatedSession = await adminApiClient.updateLearningSessionStatus(
        sessionId,
        newStatus as "active" | "inactive"
      );
      setSessions(
        sessions.map((session) =>
          session.session_id === sessionId ? updatedSession : session
        )
      );
      setError(null);
    } catch (error) {
      console.error("Failed to update session status:", error);

      if (error instanceof Error) {
        if (error.message.includes("ERR_CONNECTION_REFUSED")) {
          setError("Backend servisi bağlantı hatası. Durum güncellenemedi.");
        } else if (error.message.includes("404")) {
          setError("Ders oturumu bulunamadı.");
        } else if (error.message.includes("403")) {
          setError("Bu ders oturumunu düzenleme yetkiniz yok.");
        } else {
          setError(`Ders oturumu durumu güncellenemedi: ${error.message}`);
        }
      } else {
        setError("Ders oturumu durumu güncellenemedi. Bilinmeyen hata.");
      }
    }
  };

  const handleBulkAction = async (action: string, selectedIds: string[]) => {
    if (selectedIds.length === 0) {
      setError("Hiçbir oturum seçilmedi");
      return;
    }

    try {
      if (action === "activate" || action === "deactivate") {
        const targetStatus = action === "activate" ? "active" : "inactive";

        // Update each session individually
        for (const sessionId of selectedIds) {
          await adminApiClient.updateLearningSessionStatus(
            sessionId,
            targetStatus as "active" | "inactive"
          );
        }

        // Refresh sessions list
        await fetchSessions();
        setError(null);
      } else if (action === "delete") {
        if (
          confirm(
            `Seçili ${selectedIds.length} ders oturumunu silmek istediğinizden emin misiniz?`
          )
        ) {
          for (const sessionId of selectedIds) {
            await adminApiClient.deleteLearningSession(sessionId);
          }

          // Refresh sessions list
          await fetchSessions();
          setError(null);
        }
      }
    } catch (error) {
      console.error("Failed to perform bulk action:", error);

      if (error instanceof Error) {
        if (error.message.includes("ERR_CONNECTION_REFUSED")) {
          setError(
            "Backend servisi bağlantı hatası. Toplu işlem başarısız oldu."
          );
        } else if (error.message.includes("403")) {
          setError("Toplu işlem için yetkiniz yok.");
        } else {
          setError(`Toplu işlem başarısız oldu: ${error.message}`);
        }
      } else {
        setError("Toplu işlem başarısız oldu. Bilinmeyen hata.");
      }
    }
  };

  return (
    <ModernAdminLayout
      title="Ders Oturumları Yönetimi"
      description="Sistem ders oturumlarını ve detaylarını yönetin"
    >
      <div className="space-y-3">
        {/* Error Message */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg
                  className="h-5 w-5 text-red-400"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-red-800 dark:text-red-200">
                  {error}
                </p>
              </div>
              <div className="ml-auto pl-3">
                <button
                  onClick={() => setError(null)}
                  className="inline-flex rounded-md bg-red-50 dark:bg-red-900/20 p-1.5 text-red-500 hover:bg-red-100 dark:hover:bg-red-900/40"
                >
                  <svg
                    className="h-5 w-5"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                      clipRule="evenodd"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Action Bar */}
        <div className="flex justify-between items-center bg-white dark:bg-gray-800 p-3 rounded-lg shadow-sm">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Ders Oturumları ({sessions.length})
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Aktif: {sessions.filter((s) => s.status === "active").length} |
              Pasif: {sessions.filter((s) => s.status === "inactive").length}
            </p>
          </div>
          <button
            onClick={fetchSessions}
            disabled={loading}
            className="inline-flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white text-sm font-medium rounded-md transition-colors"
          >
            <svg
              className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
            {loading ? "Yenileniyor..." : "Yenile"}
          </button>
        </div>

        {/* Sessions Table */}
        <SessionsTable
          sessions={sessions}
          users={users}
          loading={loading}
          onView={handleViewSession}
          onDelete={handleDeleteSession}
          onToggleStatus={handleToggleStatus}
          onBulkAction={handleBulkAction}
        />

        {/* Session Modal */}
        {showModal && selectedSession && (
          <SessionsModal
            session={selectedSession}
            users={users}
            onClose={() => {
              setShowModal(false);
              setSelectedSession(null);
            }}
          />
        )}
      </div>
    </ModernAdminLayout>
  );
}
