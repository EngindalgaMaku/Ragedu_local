"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import { studentStorage, StudentSession } from "@/lib/student-storage";
import { StudentChatMessage, RAGSource } from "@/lib/api";
import {
  BookOpen,
  Clock,
  MessageCircle,
  ArrowLeft,
  Search,
  History,
} from "lucide-react";

const MESSAGES_PER_PAGE = 8;

function getSourceTypes(sources?: RAGSource[]) {
  const types = new Set<string>();
  (sources || []).forEach((s) => {
    const t = (s.metadata?.source_type || s.metadata?.source || "").toString();
    if (t) types.add(t);
  });
  return types;
}

export default function StudentHistoryPage() {
  const { user } = useAuth();
  const router = useRouter();

  const [sessions, setSessions] = useState<StudentSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(
    null
  );
  const [messages, setMessages] = useState<StudentChatMessage[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);

  // Redirect if not logged in
  useEffect(() => {
    if (!user) {
      router.push("/login");
    }
  }, [user, router]);

  // Load student's active sessions with activity info
  useEffect(() => {
    const load = async () => {
      try {
        setLoadingSessions(true);
        const list = await studentStorage.getStudentSessions();
        setSessions(list);
        if (list.length > 0 && !selectedSessionId) {
          setSelectedSessionId(list[0].sessionId);
        }
      } finally {
        setLoadingSessions(false);
      }
    };
    if (user) {
      load();
    }
  }, [user, selectedSessionId]);

  // Load chat history when session changes
  useEffect(() => {
    const loadHistory = async () => {
      if (!selectedSessionId) return;
      try {
        setLoadingMessages(true);
        const history = await studentStorage.getChatHistory(selectedSessionId);
        setMessages(history || []);
        setCurrentPage(1);
      } finally {
        setLoadingMessages(false);
      }
    };
    loadHistory();
  }, [selectedSessionId]);

  if (!user) {
    return null;
  }

  const selectedSession = sessions.find(
    (s) => s.sessionId === selectedSessionId
  );

  // Filter and paginate messages (only those with a real question)
  const filteredMessages = messages.filter(
    (m) =>
      m.user &&
      m.user !== "..." &&
      (!searchTerm.trim() ||
        m.user.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (m.bot &&
          m.bot
            .toString()
            .toLowerCase()
            .includes(searchTerm.toLowerCase())))
  );

  const totalPages = Math.max(
    1,
    Math.ceil(filteredMessages.length / MESSAGES_PER_PAGE)
  );
  const startIdx = (currentPage - 1) * MESSAGES_PER_PAGE;
  const endIdx = startIdx + MESSAGES_PER_PAGE;
  const paginated = filteredMessages.slice(startIdx, endIdx);

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return "";
    try {
      const date = new Date(timestamp);
      return date.toLocaleString("tr-TR", {
        day: "2-digit",
        month: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "";
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <History className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Geçmiş Sorularım
            </h1>
            <p className="text-sm text-gray-600">
              Daha önce sorduğun soruları, yanıtları ve kullanılan kaynakları
              inceleyebilirsin.
            </p>
          </div>
        </div>
        <button
          onClick={() => router.push("/student/chat")}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-gray-200 bg-white text-sm text-gray-800 hover:bg-gray-50 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Canlı Sohbete Dön
        </button>
      </div>

      {/* Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Session list */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 lg:col-span-1">
          <div className="flex items-center gap-2 mb-3">
            <BookOpen className="w-4 h-4 text-blue-500" />
            <h2 className="text-sm font-semibold text-gray-900">
              Ders Oturumlarım
            </h2>
          </div>

          {loadingSessions ? (
            <div className="py-8 text-center text-sm text-gray-500">
              Oturumlar yükleniyor...
            </div>
          ) : sessions.length === 0 ? (
            <div className="py-8 text-center text-sm text-gray-500">
              Henüz soru sorulan bir oturum yok.
            </div>
          ) : (
            <div className="space-y-2 max-h-[380px] overflow-y-auto">
              {sessions.map((s) => (
                <button
                  key={s.sessionId}
                  onClick={() => setSelectedSessionId(s.sessionId)}
                  className={`w-full text-left px-3 py-2 rounded-lg border text-sm transition-colors ${
                    s.sessionId === selectedSessionId
                      ? "border-blue-500 bg-blue-50"
                      : "border-gray-200 bg-white hover:bg-gray-50"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-gray-900 line-clamp-1">
                      {s.sessionName}
                    </span>
                    <span className="text-xs text-gray-500 flex items-center gap-1">
                      <MessageCircle className="w-3 h-3" />
                      {s.messageCount}
                    </span>
                  </div>
                  <p className="text-[11px] text-gray-500">
                    Son aktivite: {formatTimestamp(s.lastAccessed)}
                  </p>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* History list */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 lg:col-span-2 flex flex-col">
          {/* Toolbar */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
            <div>
              <h2 className="text-sm font-semibold text-gray-900">
                {selectedSession?.sessionName || "Seçili Oturum Yok"}
              </h2>
              <p className="text-xs text-gray-500">
                Toplam {filteredMessages.length} soru
                {searchTerm && " (filtrelenmiş)"}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="w-4 h-4 text-gray-400 absolute left-2 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Sorularda ara..."
                  value={searchTerm}
                  onChange={(e) => {
                    setSearchTerm(e.target.value);
                    setCurrentPage(1);
                  }}
                  className="pl-8 pr-3 py-1.5 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-52"
                />
              </div>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto space-y-3 pr-1">
            {loadingMessages ? (
              <div className="py-10 text-center text-sm text-gray-500">
                Geçmiş sorular yükleniyor...
              </div>
            ) : paginated.length === 0 ? (
              <div className="py-10 text-center text-sm text-gray-500">
                Bu oturumda henüz kayıtlı soru yok.
              </div>
            ) : (
              paginated.map((m, idx) => {
                const types = getSourceTypes(m.sources);
                const hasKB = types.has("knowledge_base");
                const hasQA = types.has("qa_pair");
                const hasChunks = types.has("chunk") || types.size === 0;

                return (
                  <div
                    key={`${m.timestamp}-${idx}`}
                    className="border border-gray-200 rounded-lg p-3 bg-gray-50"
                  >
                    <div className="flex justify-between items-start gap-2 mb-1.5">
                      <p className="text-sm font-medium text-gray-900">
                        {m.user}
                      </p>
                      <span className="text-[11px] text-gray-500 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatTimestamp(m.timestamp)}
                      </span>
                    </div>

                    {m.bot && (
                      <p className="text-sm text-gray-800 bg-white rounded-md border border-gray-200 px-3 py-2">
                        {m.bot.length > 260
                          ? `${m.bot.slice(0, 260)}...`
                          : m.bot}
                      </p>
                    )}

                    {/* KB / QA usage badges */}
                    <div className="mt-2 flex flex-wrap gap-1.5 text-[11px]">
                      {hasKB && (
                        <span className="px-2 py-0.5 rounded-full bg-purple-50 border border-purple-200 text-purple-700">
                          📚 Bilgi Tabanı
                        </span>
                      )}
                      {hasQA && (
                        <span className="px-2 py-0.5 rounded-full bg-green-50 border border-green-200 text-green-700">
                          ❓ Soru Bankası
                        </span>
                      )}
                      {hasChunks && (
                        <span className="px-2 py-0.5 rounded-full bg-blue-50 border border-blue-200 text-blue-700">
                          📄 Döküman Parçaları
                        </span>
                      )}
                      {m.durationMs && (
                        <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
                          ⚡ {(m.durationMs / 1000).toFixed(1)} sn
                        </span>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Pagination */}
          {filteredMessages.length > 0 && (
            <div className="mt-4 pt-3 border-t border-gray-200 flex items-center justify-between text-xs text-gray-600">
              <span>
                Sayfa {currentPage} / {totalPages}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() =>
                    setCurrentPage((p) => Math.max(1, p - 1))
                  }
                  disabled={currentPage === 1}
                  className="px-2 py-1 rounded border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Önceki
                </button>
                <button
                  onClick={() =>
                    setCurrentPage((p) => Math.min(totalPages, p + 1))
                  }
                  disabled={currentPage >= totalPages}
                  className="px-2 py-1 rounded border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Sonraki
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}






