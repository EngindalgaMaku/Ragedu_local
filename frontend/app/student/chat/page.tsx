"use client";

import React, { useState, useEffect, useRef } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useStudentChat } from "@/hooks/useStudentChat";
import { listSessions, SessionMeta, RAGSource, getPedagogicalState, PedagogicalState, resetPedagogicalState } from "@/lib/api";
import { useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import {
  Send,
  Loader2,
  Trash2,
  BookOpen,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { QuickEmojiFeedback } from "@/components/EmojiFeedback";
import SourceModal from "@/components/SourceModal";
import KBRAGPersonalizationDebugPanel from "@/components/KBRAGPersonalizationDebugPanel";
import EBARSStatusPanel from "@/components/EBARSStatusPanel";

const MESSAGES_PER_PAGE = 10;

// Student Chat Page with Enhanced AI Loading Animation
export default function StudentChatPage() {
  const { user } = useAuth();
  const router = useRouter();
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedSource, setSelectedSource] = useState<RAGSource | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [sessionRagSettings, setSessionRagSettings] = useState<any>(null);
  const [pedagogicalState, setPedagogicalState] = useState<PedagogicalState | null>(null);
  const [pedagogicalStateLoading, setPedagogicalStateLoading] = useState(false);
  const [ebarsEnabled, setEbarsEnabled] = useState(false);
  const [ebarsRefreshTrigger, setEbarsRefreshTrigger] = useState(0);

  const {
    messages,
    isLoading: chatLoading,
    isQuerying,
    error: chatError,
    sendMessage,
    clearHistory,
    debugData,
    asyncTaskProgress,
  } = useStudentChat({
    sessionId: selectedSession || "",
    autoSave: true,
  });

  // Load student's current pedagogical state (ZPD, Bloom, Cognitive Load)
  const loadPedagogicalState = async () => {
    if (!selectedSession || !user?.id) {
      setPedagogicalState(null);
      return;
    }

    try {
      setPedagogicalStateLoading(true);
      const state = await getPedagogicalState(String(user.id), selectedSession);
      setPedagogicalState(state);
    } catch (err) {
      console.error("Failed to load pedagogical state:", err);
      setPedagogicalState(null);
    } finally {
      setPedagogicalStateLoading(false);
    }
  };

  useEffect(() => {
    loadPedagogicalState();
  }, [selectedSession, user?.id]);

  // Handle reset
  const handleResetProfile = async () => {
    if (!selectedSession || !user?.id) return;

    try {
      await resetPedagogicalState(String(user.id), selectedSession);
      // Reload the state after reset
      await loadPedagogicalState();
      alert("Öğrenme parametreleri başarıyla sıfırlandı!");
    } catch (err) {
      console.error("Failed to reset profile:", err);
      alert("Sıfırlama işlemi başarısız oldu. Lütfen tekrar deneyin.");
    }
  };

  // Load session RAG settings when session changes
  useEffect(() => {
    const loadSessionSettings = async () => {
      if (!selectedSession) {
        setSessionRagSettings(null);
        setEbarsEnabled(false);
        return;
      }
      try {
        const sessionData = sessions.find(
          (s) => s.session_id === selectedSession
        );
        if (sessionData?.rag_settings) {
          setSessionRagSettings(sessionData.rag_settings);
        } else {
          setSessionRagSettings(null);
        }
        
        // Check if EBARS is enabled for this session
        try {
          const response = await fetch(`/api/aprag/session-settings/${selectedSession}`);
          if (response.ok) {
            const data = await response.json();
            const enabled = data?.settings?.enable_ebars || false;
            setEbarsEnabled(enabled);
            console.log("✅ EBARS enabled:", enabled, "for session:", selectedSession, "data:", data);
          } else {
            const errorText = await response.text();
            console.warn("❌ Failed to load session settings, status:", response.status, "error:", errorText);
            setEbarsEnabled(false);
          }
        } catch (err) {
          console.error("❌ Failed to load session settings:", err);
          setEbarsEnabled(false);
        }
      } catch (err) {
        console.error("Failed to load session RAG settings:", err);
        setSessionRagSettings(null);
        setEbarsEnabled(false);
      }
    };
    loadSessionSettings();
  }, [selectedSession, sessions]);

  // Pagination
  const totalPages = Math.ceil(messages.length / MESSAGES_PER_PAGE);
  const startIdx = (currentPage - 1) * MESSAGES_PER_PAGE;
  const endIdx = startIdx + MESSAGES_PER_PAGE;
  const paginatedMessages = messages.slice(startIdx, endIdx);

  // Auto-scroll to bottom when new messages arrive on current page
  useEffect(() => {
    if (currentPage === totalPages) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, currentPage, totalPages]);

  // Load sessions
  useEffect(() => {
    const loadSessions = async () => {
      if (!user) {
        router.push("/login");
        return;
      }
      try {
        const sessionsList = await listSessions();
        setSessions(sessionsList);

        if (sessionsList.length > 0 && !selectedSession) {
          setSelectedSession(sessionsList[0].session_id);
        }
      } catch (err) {
        console.error("Failed to load sessions:", err);
      } finally {
        setLoading(false);
      }
    };
    loadSessions();
  }, [user, selectedSession, router]);

  // Reset to last page when new message arrives
  useEffect(() => {
    if (messages.length > 0) {
      setCurrentPage(Math.ceil(messages.length / MESSAGES_PER_PAGE));
    }
  }, [messages.length]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !selectedSession || isQuerying) return;

    const currentQuery = query;
    setQuery(""); // Clear input immediately for better UX
    await sendMessage(currentQuery, sessionRagSettings);
    
    // Trigger EBARS refresh after query
    if (ebarsEnabled) {
      setTimeout(() => setEbarsRefreshTrigger(prev => prev + 1), 1000);
    }
  };

  const handleSessionChange = (newSessionId: string) => {
    setSelectedSession(newSessionId);
    clearHistory();
    setCurrentPage(1);
  };

  const handleClearHistory = () => {
    if (confirm("Tüm sohbet geçmişini silmek istediğinize emin misiniz?")) {
      clearHistory();
      setCurrentPage(1);
    }
  };

  const handleSourceClick = (source: RAGSource) => {
    setSelectedSource(source);
    setIsModalOpen(true);
  };

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return "";
    try {
      const date = new Date(timestamp);
      const today = new Date();
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);

      const timeStr = date.toLocaleTimeString("tr-TR", {
        hour: "2-digit",
        minute: "2-digit",
      });

      if (date.toDateString() === today.toDateString()) {
        return `Bugün ${timeStr}`;
      } else if (date.toDateString() === yesterday.toDateString()) {
        return `Dün ${timeStr}`;
      } else {
        return date.toLocaleString("tr-TR", {
          day: "numeric",
          month: "short",
          hour: "2-digit",
          minute: "2-digit",
        });
      }
    } catch {
      return "";
    }
  };

  // Get unique sources (group by file, show all chunks)
  const getUniqueSources = (sources: RAGSource[]) => {
    if (!sources || sources.length === 0) return [];

    const sourceMap = new Map<string, RAGSource[]>();

    sources.forEach((source) => {
      const filename =
        source.metadata?.filename || source.metadata?.source_file || "unknown";
      if (!sourceMap.has(filename)) {
        sourceMap.set(filename, []);
      }
      sourceMap.get(filename)!.push(source);
    });

    return Array.from(sourceMap.entries()).map(([filename, chunks]) => ({
      filename,
      chunks: chunks.sort(
        (a, b) =>
          (a.metadata?.chunk_index ?? 0) - (b.metadata?.chunk_index ?? 0)
      ),
    }));
  };

  // Get high-level source types (chunk / knowledge_base / qa_pair)
  const getSourceTypes = (sources?: RAGSource[]): Set<string> => {
    const types = new Set<string>();
    if (sources) {
      sources.forEach((s) => {
        const t = (
          s.metadata?.source_type ||
          s.metadata?.source ||
          ""
        ).toString();
        if (t) types.add(t);
      });
    }
    return types;
  };

  // Early returns
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[calc(100vh-200px)]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Yükleniyor...</p>
        </div>
      </div>
    );
  }

  if (!user || sessions.length === 0) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
          <AlertCircle className="w-12 h-12 text-yellow-500 mx-auto mb-3" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Aktif Oturum Bulunamadı
          </h3>
          <p className="text-gray-600 mb-4">
            Soru sorabilmek için önce öğretmeninizin oluşturduğu bir oturuma
            dahil olmanız gerekmektedir.
          </p>
          <button
            onClick={() => router.push("/student")}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Dashboard'a Dön
          </button>
        </div>
      </div>
    );
  }

  const selectedSessionData = sessions.find(
    (s) => s.session_id === selectedSession
  );

  return (
    <div className="max-w-6xl mx-auto p-4 pb-6" style={{ minHeight: "calc(100vh - 100px)" }}>
      {/* Header Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Soru & Cevap</h2>
              <p className="text-sm text-gray-500">
                Seçili oturum hakkında sorularınızı sorun
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            <select
              value={selectedSession || ""}
              onChange={(e) => handleSessionChange(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white text-gray-900 font-medium"
            >
              {sessions.map((session) => (
                <option key={session.session_id} value={session.session_id}>
                  📚 {session.name}
                </option>
              ))}
            </select>

            {messages.length > 0 && (
              <button
                onClick={handleClearHistory}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors flex items-center gap-2"
                title="Sohbet geçmişini temizle"
              >
                <Trash2 className="w-4 h-4" />
                <span className="hidden sm:inline">Temizle</span>
              </button>
            )}
          </div>
        </div>

        {/* Session Info */}
        {selectedSessionData && (
          <div className="mt-4 pt-4 border-t border-gray-100">
            <div className="flex items-center gap-4 text-sm text-gray-600">
              <span>📖 {selectedSessionData.document_count || 0} Döküman</span>
              {selectedSessionData.rag_settings?.embedding_model && (
                <span>
                  🔮 {selectedSessionData.rag_settings.embedding_model}
                </span>
              )}
              {selectedSessionData.rag_settings?.chunk_strategy && (
                <span>
                  ✂️ {selectedSessionData.rag_settings.chunk_strategy}
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* EBARS Status Panel - Shows comprehension score and difficulty level when EBARS is enabled */}
      {user && selectedSession && (
        <div className="mb-4">
          {ebarsEnabled ? (
            <EBARSStatusPanel
              userId={user.id.toString()}
              sessionId={selectedSession}
              refreshTrigger={ebarsRefreshTrigger}
              onFeedbackSubmitted={() => {
                // Refresh EBARS state after feedback
                setEbarsRefreshTrigger(prev => prev + 1);
              }}
            />
          ) : (
            // Debug: Show why panel is not visible
            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-800">
              <div className="flex items-center gap-2">
                <span>⚠️</span>
                <div>
                  <strong>EBARS Panel Görünmüyor</strong>
                  <div className="text-xs mt-1">
                    EBARS aktif: {String(ebarsEnabled)} | 
                    User: {user ? 'var' : 'yok'} | 
                    Session: {selectedSession || 'yok'}
                  </div>
                  <div className="text-xs mt-1 text-yellow-700">
                    EBARS'ı aktif etmek için oturum ayarlarından "EBARS (Adaptif Zorluk Ayarlama)" toggle'ını açın.
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Chat Container */}
      <div
        className="bg-white rounded-lg shadow-sm border border-gray-200 flex flex-col"
        style={{ height: "calc(100vh - 280px)", minHeight: "600px" }}
      >
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-md">
                <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Send className="w-10 h-10 text-blue-500" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  Soru Sormaya Başla
                </h3>
                <p className="text-gray-600 mb-4">
                  Seçili oturumdaki dökümanlar hakkında istediğin soruyu
                  sorabilirsin. Yapay zeka asistanı sana yardımcı olacak! 🤖
                </p>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-left text-sm">
                  <p className="font-medium text-blue-900 mb-2">💡 İpucu:</p>
                  <ul className="text-blue-800 space-y-1 list-disc list-inside">
                    <li>Sorularını net ve açık sor</li>
                    <li>Cevapları emoji ile değerlendir</li>
                    <li>Anlamadığın yerleri tekrar sor</li>
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <>
              {paginatedMessages.map((message, index) => (
                <div key={index} className="space-y-4">
                  {/* User Question */}
                  {message.user && message.user !== "..." && (
                    <div className="flex justify-end">
                      <div className="max-w-[75%]">
                        <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-2xl rounded-tr-sm px-5 py-3 shadow-md">
                          <p className="text-sm leading-relaxed whitespace-pre-wrap">
                            {message.user}
                          </p>
                        </div>
                        {message.timestamp && (
                          <p className="text-xs text-gray-500 mt-1 text-right">
                            {formatTimestamp(message.timestamp)}
                          </p>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Bot Answer */}
                  {message.bot && message.bot !== "..." && (
                    <div className="flex justify-start">
                      <div className="max-w-[85%]">
                        <div className="bg-gray-50 border border-gray-200 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm group">
                          <div className="prose prose-sm max-w-none text-gray-800">
                            <ReactMarkdown
                              components={{
                                p: ({ children }) => (
                                  <p className="mb-2 last:mb-0">{children}</p>
                                ),
                                ul: ({ children }) => (
                                  <ul className="ml-4 mb-2 list-disc">
                                    {children}
                                  </ul>
                                ),
                                ol: ({ children }) => (
                                  <ol className="ml-4 mb-2 list-decimal">
                                    {children}
                                  </ol>
                                ),
                                li: ({ children }) => (
                                  <li className="mb-1">{children}</li>
                                ),
                                strong: ({ children }) => (
                                  <strong className="font-semibold text-gray-900">
                                    {children}
                                  </strong>
                                ),
                                code: ({ children }) => (
                                  <code className="bg-gray-200 px-1 py-0.5 rounded text-sm">
                                    {children}
                                  </code>
                                ),
                              }}
                            >
                              {message.bot}
                            </ReactMarkdown>
                          </div>

                          {/* High-level KB / QA usage summary */}
                          {message.sources && message.sources.length > 0 && (
                            <div className="mt-3 flex flex-wrap gap-1.5 text-[11px]">
                              {(() => {
                                const types = getSourceTypes(message.sources);
                                const hasKB = types.has("knowledge_base");
                                const hasQA = types.has("qa_pair");
                                const hasChunks =
                                  types.has("chunk") || types.size === 0;
                                return (
                                  <>
                                    {hasKB && (
                                      <span className="px-2 py-0.5 rounded-full bg-purple-50 border border-purple-200 text-purple-700">
                                        📚 Bilgi Tabanı Kullanıldı
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
                                  </>
                                );
                              })()}
                            </div>
                          )}

                          {/* Correction Notice */}
                          {message.correction &&
                            message.correction.was_corrected && (
                              <div className="mt-3 mb-2 p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm">
                                <div className="flex items-start gap-2">
                                  <div className="mt-0.5 text-amber-600">
                                    <AlertCircle className="w-4 h-4" />
                                  </div>
                                  <div>
                                    <p className="font-medium text-amber-800 mb-1">
                                      Otomatik Düzeltme Uygulandı
                                    </p>
                                    <p className="text-amber-700 text-xs mb-2">
                                      İlk analizde tespit edilen tutarsızlıklar
                                      nedeniyle cevap güncellendi:
                                    </p>
                                    <ul className="list-disc list-inside text-amber-700 text-xs space-y-1">
                                      {message.correction.issues.map(
                                        (issue, idx) => (
                                          <li key={idx}>{issue}</li>
                                        )
                                      )}
                                    </ul>
                                  </div>
                                </div>
                              </div>
                            )}

                          {/* Response Time & Emoji Feedback */}
                          <div className="mt-3 pt-3 border-t border-gray-200 flex items-center justify-between">
                            <span className="text-xs text-gray-500">
                              {message.durationMs &&
                                `⚡ ${(message.durationMs / 1000).toFixed(1)}s`}
                              {message.timestamp &&
                                !message.durationMs &&
                                formatTimestamp(message.timestamp)}
                            </span>

            {/* Emoji Feedback */}
            {message.aprag_interaction_id &&
              user &&
              selectedSession && (
                <QuickEmojiFeedback
                  interactionId={message.aprag_interaction_id}
                  userId={user.id.toString()}
                  sessionId={selectedSession}
                  onFeedbackSubmitted={() => {
                    // Trigger EBARS refresh after emoji feedback
                    if (ebarsEnabled) {
                      setTimeout(() => setEbarsRefreshTrigger(prev => prev + 1), 500);
                    }
                  }}
                />
              )}
                          </div>

                          {/* İlgili Sorular (Suggestions) */}
                          {Array.isArray(message.suggestions) &&
                            message.suggestions.length > 0 && (
                              <div className="mt-6 pt-4 border-t border-gray-200">
                                <div className="flex items-center gap-2 mb-3">
                                  <svg
                                    className="w-5 h-5 text-indigo-600"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth={2}
                                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                                    />
                                  </svg>
                                  <span className="text-sm font-semibold text-gray-700">
                                    İlgili Sorular
                                  </span>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                  {message.suggestions.map((suggestion, i) => (
                                    <button
                                      key={i}
                                      onClick={async () => {
                                        setQuery(suggestion);
                                        setTimeout(async () => {
                                          await sendMessage(suggestion, sessionRagSettings);
                                        }, 100);
                                      }}
                                      className="px-4 py-2 text-sm bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-full hover:shadow-md transition-all duration-200 transform hover:scale-105 min-h-[40px] flex items-center gap-2"
                                      title="Bu soruyu sor"
                                    >
                                      <span>💡</span>
                                      <span>{suggestion}</span>
                                    </button>
                                  ))}
                                </div>
                              </div>
                            )}

                          {/* Sources with Chunks */}
                          {message.sources && message.sources.length > 0 && (
                            <div className="mt-2 ml-4">
                              <details className="text-xs">
                                <summary className="cursor-pointer hover:text-gray-700 font-medium text-gray-600">
                                  📚 {message.sources.length} kaynak kullanıldı
                                </summary>
                                <div className="mt-2 space-y-2">
                                  {getUniqueSources(message.sources).map(
                                    (sourceGroup, idx) => {
                                      const filename = sourceGroup.filename || "unknown";
                                      const isKnowledgeBase = filename.toLowerCase() === "unknown" || filename.trim() === "";
                                      const displayName = isKnowledgeBase ? "Bilgi Tabanı" : filename;
                                      
                                      return (
                                        <div
                                          key={idx}
                                          className={`rounded-lg p-2 ${
                                            isKnowledgeBase 
                                              ? "bg-purple-50 border border-purple-200" 
                                              : "bg-gray-50"
                                          }`}
                                        >
                                          <div className={`font-medium mb-1 flex items-center gap-2 ${
                                            isKnowledgeBase 
                                              ? "text-purple-800" 
                                              : "text-gray-700"
                                          }`}>
                                            {isKnowledgeBase ? (
                                              <>
                                                <span className="text-purple-600">📚</span>
                                                <span>{displayName}</span>
                                              </>
                                            ) : (
                                              <>
                                                <span>📄</span>
                                                <span>{displayName}</span>
                                              </>
                                            )}
                                          </div>
                                          <div className="flex flex-wrap gap-1">
                                            {sourceGroup.chunks.map(
                                              (chunk, chunkIdx) => {
                                                const chunkNumber = chunkIdx + 1;
                                                return (
                                                  <button
                                                    key={chunkIdx}
                                                    onClick={() =>
                                                      handleSourceClick(chunk)
                                                    }
                                                    className="px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors text-xs"
                                                    title={`Parça ${chunkNumber} - Skor: ${(
                                                      chunk.score * 100
                                                    ).toFixed(0)}%`}
                                                  >
                                                    #{chunkNumber}
                                                    {chunk.metadata?.page_number &&
                                                      ` (s.${chunk.metadata.page_number})`}
                                                  </button>
                                                );
                                              }
                                            )}
                                          </div>
                                        </div>
                                      );
                                    }
                                  )}
                                </div>
                              </details>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {/* Loading Indicator - AI Teacher Thinking with Async Progress */}
              {isQuerying && messages.length > 0 && messages[messages.length - 1]?.bot === "..." && (
                <div className="flex justify-start">
                  <div className="relative bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 border-2 border-blue-300 rounded-2xl rounded-tl-sm px-6 py-5 shadow-2xl max-w-lg">
                    {/* Animated background effect */}
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-400/10 via-purple-400/10 to-pink-400/10 rounded-2xl animate-pulse"></div>

                    <div className="relative z-10">
                      {/* Header with AI Icon */}
                      <div className="flex items-center gap-3 mb-4">
                        <div className="relative">
                          {/* Main robot icon with glow */}
                          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 rounded-full flex items-center justify-center shadow-lg animate-ai-thinking">
                            <span className="text-2xl">🤖</span>
                          </div>
                          {/* Pulsing ring effect */}
                          <div className="absolute inset-0 rounded-full border-2 border-blue-400 animate-ping opacity-75"></div>
                          {/* Status indicator */}
                          <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 rounded-full border-2 border-white shadow-lg animate-glow">
                            <div className="absolute inset-0 bg-green-400 rounded-full animate-ping opacity-60"></div>
                          </div>
                        </div>
                        <div className="flex-1">
                          <p className="text-base font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
                            {asyncTaskProgress
                              ? `🚀 ${asyncTaskProgress.currentStep}`
                              : "🧠 AI Asistanı Cevap Hazırlıyor..."}
                          </p>
                          {/* Time estimate or animated dots */}
                          {asyncTaskProgress ? (
                            <p className="text-xs text-gray-600 mt-1">
                              ⏱️ Tahmini süre:{" "}
                              {Math.ceil(asyncTaskProgress.estimatedRemaining)}s
                            </p>
                          ) : (
                            <div className="flex items-center gap-1.5 mt-1.5">
                              <span
                                className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-dots-bounce shadow-md"
                                style={{ animationDelay: "0ms" }}
                              ></span>
                              <span
                                className="w-2.5 h-2.5 bg-purple-500 rounded-full animate-dots-bounce shadow-md"
                                style={{ animationDelay: "0.2s" }}
                              ></span>
                              <span
                                className="w-2.5 h-2.5 bg-pink-500 rounded-full animate-dots-bounce shadow-md"
                                style={{ animationDelay: "0.4s" }}
                              ></span>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Processing steps with dynamic progress or static steps */}
                      <div className="space-y-3 text-sm">
                        {asyncTaskProgress ? (
                          // Dynamic async progress steps
                          <div className="space-y-2">
                            <div className="flex items-center gap-3 p-2 bg-white/50 rounded-lg backdrop-blur-sm">
                              <div className="w-4 h-4 relative">
                                <div className="absolute inset-0 bg-blue-500 rounded-full animate-pulse"></div>
                                <div className="relative w-4 h-4 bg-blue-600 rounded-full"></div>
                              </div>
                              <span className="text-gray-700 font-medium text-xs">
                                {asyncTaskProgress.currentStep}
                              </span>
                            </div>
                            {asyncTaskProgress.progress > 0 && (
                              <div className="text-center">
                                <span className="text-xs text-gray-600 font-medium">
                                  %{Math.round(asyncTaskProgress.progress)}{" "}
                                  tamamlandı
                                </span>
                              </div>
                            )}
                          </div>
                        ) : (
                          // Static loading steps (for sync RAG)
                          <>
                            <div className="flex items-center gap-3 p-2 bg-white/50 rounded-lg backdrop-blur-sm">
                              <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                              <span className="text-gray-700 font-medium">
                                📚 Dökümanlar analiz ediliyor...
                              </span>
                            </div>
                            <div className="flex items-center gap-3 p-2 bg-white/50 rounded-lg backdrop-blur-sm">
                              <div className="w-4 h-4 relative">
                                <div className="absolute inset-0 bg-purple-500 rounded-full animate-ping opacity-75"></div>
                                <div className="relative w-4 h-4 bg-purple-500 rounded-full animate-pulse"></div>
                              </div>
                              <span className="text-gray-700 font-medium">
                                🎯 En uygun bilgiler seçiliyor...
                              </span>
                            </div>
                            <div className="flex items-center gap-3 p-2 bg-white/50 rounded-lg backdrop-blur-sm">
                              <div className="w-4 h-4 relative">
                                <div className="absolute inset-0 bg-pink-500 rounded-full animate-ping opacity-75"></div>
                                <div className="relative w-4 h-4 bg-pink-500 rounded-full animate-pulse"></div>
                              </div>
                              <span className="text-gray-700 font-medium">
                                ✨ Senin için özel cevap oluşturuluyor...
                              </span>
                            </div>
                          </>
                        )}
                      </div>

                      {/* Progress bar effect */}
                      <div className="mt-4 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-full"
                          style={{
                            width: asyncTaskProgress
                              ? `${Math.min(
                                  95,
                                  Math.max(10, asyncTaskProgress.progress)
                                )}%`
                              : "85%",
                          }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Error Message */}
              {chatError && (
                <div className="flex justify-center">
                  <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 max-w-md">
                    <div className="flex items-center gap-2 text-red-800">
                      <AlertCircle className="w-5 h-5" />
                      <span className="text-sm font-medium">
                        Hata: {chatError}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="border-t border-gray-200 bg-gray-50 px-4 py-2 flex items-center justify-between">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
            >
              <ChevronLeft className="w-4 h-4" />
              Önceki
            </button>
            <span className="text-sm text-gray-600">
              Sayfa {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
            >
              Sonraki
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}


        {/* Input Area - Fixed at Bottom */}
        <div className="border-t border-gray-200 bg-gray-50 p-4">
          <form onSubmit={handleSendMessage} className="flex gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Sorunuzu yazın..."
              disabled={isQuerying || !selectedSession}
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed text-gray-900 placeholder-gray-500"
            />
            <button
              type="submit"
              disabled={isQuerying || !query.trim() || !selectedSession}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2 font-medium shadow-md hover:shadow-lg"
            >
              {isQuerying ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>
                    {asyncTaskProgress ? "Cevap Üretiliyor" : "Gönderiliyor"}
                  </span>
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  <span>Gönder</span>
                </>
              )}
            </button>
          </form>

          {/* Help Text */}
          <p className="text-xs text-gray-500 mt-2 text-center">
            💡 Cevapları emojilerle değerlendirerek öğrenme deneyimini
            iyileştirebilirsin
          </p>
        </div>
      </div>

      {/* Source Modal */}
      <SourceModal
        source={selectedSource}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />

      {/* KBRAG & Personalization Debug Panel */}
      <KBRAGPersonalizationDebugPanel debugData={debugData} />
    </div>
  );
}
