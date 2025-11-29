"use client";

import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface RAGSource {
  content?: string;
  text?: string;
  metadata?: any;
  score: number;
  crag_score?: number;
}

interface CorrectionDetails {
  was_corrected: boolean;
  issues: string[];
}

interface ChatHistoryItem {
  user: string;
  bot: string;
  sources?: RAGSource[];
  durationMs?: number;
  suggestions?: string[];
  timestamp?: string;
  interactionId?: number;
  correction?: CorrectionDetails;
  aprag_interaction_id?: number;
}

interface StudentMessage {
  id?: number;
  user: string;
  bot: string;
  sources?: RAGSource[];
  durationMs?: number;
  suggestions?: string[];
  timestamp?: string;
  aprag_interaction_id?: number;
}

interface ChatHistoryProps {
  // Chat History (for teachers)
  chatHistory: ChatHistoryItem[];
  isQuerying: boolean;

  // Student Messages (for students)
  studentMessages: StudentMessage[];
  studentChatLoading: boolean;

  // User info
  isStudent: boolean;
  user?: any;
  selectedSessionId?: string;

  // Handlers
  onSuggestionClick: (suggestion: string) => void;
  onOpenSourceModal: (source: RAGSource) => void;
  clearChatHistory?: () => void;

  // Utilities
  formatTimestamp: (timestamp: string) => string;
}

export default function ChatHistory({
  chatHistory,
  isQuerying,
  studentMessages,
  studentChatLoading,
  isStudent,
  user,
  selectedSessionId,
  onSuggestionClick,
  onOpenSourceModal,
  clearChatHistory,
  formatTimestamp,
}: ChatHistoryProps) {
  // Determine which messages to show
  const messages = isStudent ? studentMessages : chatHistory;
  const isLoading = isStudent ? studentChatLoading : isQuerying;

  // Handle clear chat history with confirmation
  const handleClearHistory = () => {
    if (!clearChatHistory || messages.length === 0) return;

    const confirmed = window.confirm(
      "Sohbet geçmişi silinecek, emin misiniz?\n\nBu işlem geri alınamaz ve tüm sohbet geçmişiniz kalıcı olarak silinecektir."
    );

    if (confirmed) {
      clearChatHistory();
    }
  };

  const renderLoadingState = () => (
    <div className="py-8">
      <div className="flex items-center gap-4 mb-6">
        <div className="relative">
          <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center animate-pulse">
            <span className="text-2xl">🤖</span>
          </div>
          <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white animate-ping"></div>
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-3">
            <div className="h-2 w-2 bg-blue-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-gray-700">
              Sanal Asistanınız cevabı hazırlıyor...
            </span>
          </div>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-400 to-indigo-500 rounded-full animate-pulse"
                  style={{ width: "70%" }}
                ></div>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-xs">📚</span>
                <span className="text-xs text-gray-500">
                  Kaynaklar oluşturuluyor
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-purple-400 to-pink-500 rounded-full animate-pulse"
                  style={{ width: "45%", animationDelay: "150ms" }}
                ></div>
              </div>
              <span className="text-xs text-gray-500">Cevap üretiliyor</span>
            </div>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2 pl-16">
        <div className="flex gap-1">
          <div
            className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"
            style={{ animationDelay: "0ms" }}
          ></div>
          <div
            className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce"
            style={{ animationDelay: "150ms" }}
          ></div>
          <div
            className="w-2 h-2 bg-indigo-600 rounded-full animate-bounce"
            style={{ animationDelay: "300ms" }}
          ></div>
        </div>
        <span className="text-xs text-gray-400 italic">
          AI değerlendirme yapıyor...
        </span>
      </div>
    </div>
  );

  const renderSources = (sources: RAGSource[]) => {
    if (!sources || sources.length === 0) return null;

    return (
      <div className="mt-4 pt-4 border-t border-gray-100">
        <div className="flex items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <span className="text-sm">📚</span>
            <span className="text-sm font-medium text-gray-600">
              Güvenilir Kaynaklar ({sources.length})
            </span>
          </div>
        </div>

        <div className="bg-gray-50 rounded-lg border border-gray-200 overflow-hidden">
          <div className="flex flex-wrap border-b border-gray-200 bg-white">
            {sources.map((source, idx) => (
              <button
                key={idx}
                className="px-3 py-2 text-xs font-medium text-gray-700 hover:text-blue-600 hover:bg-blue-50 border-r border-gray-200 last:border-r-0 transition-colors flex items-center gap-2"
                onClick={() => onOpenSourceModal(source)}
              >
                <span className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-full text-xs font-bold">
                  {idx + 1}
                </span>
                <span className="truncate max-w-20 cursor-pointer hover:text-blue-700 underline">
                  {source.metadata?.source_file?.replace(".md", "") ||
                    source.metadata?.filename?.replace(".md", "") ||
                    `Kaynak ${idx + 1}`}
                </span>
                <span className="flex items-center gap-1">
                  {source.crag_score != null ? (
                    <>
                      <span className="text-xs text-gray-500">DYSK:</span>
                      <span className="text-indigo-700 font-bold">
                        {Math.round(source.crag_score * 100)}%
                      </span>
                    </>
                  ) : (
                    <span className="text-green-600 font-bold">
                      {Math.round(source.score * 100)}%
                    </span>
                  )}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="mt-2 text-xs text-gray-500 text-center">
          💡 Kaynak detaylarını görmek için üzerine tıklayın
        </div>
      </div>
    );
  };

  const renderCorrection = (correction?: CorrectionDetails) => {
    if (!correction?.was_corrected) return null;

    return (
      <div className="mt-4 mb-4 p-4 bg-amber-50 border-l-4 border-amber-500 rounded-r-lg shadow-sm">
        <div className="flex items-start gap-3">
          <div className="text-amber-600 mt-0.5">
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
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <div className="flex-1">
            <h4 className="text-sm font-bold text-amber-800 mb-1">
              ⚠️ Otomatik Doğrulama ve Düzeltme
            </h4>
            <p className="text-xs text-amber-700 mb-2">
              Yapay zeka, ilk cevabında tutarsızlıklar tespit etti ve aşağıdaki
              nedenlerle cevabı güncelledi:
            </p>
            <ul className="list-disc list-inside text-xs text-amber-800 space-y-1 bg-amber-100/50 p-2 rounded">
              {correction.issues.map((issue: string, idx: number) => (
                <li key={idx}>{issue}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    );
  };

  const renderSuggestions = (suggestions?: string[]) => {
    if (!Array.isArray(suggestions) || suggestions.length === 0) return null;

    return (
      <div className="mt-6 pt-4 border-t border-gray-100">
        <div className="flex items-center gap-2 mb-3">
          <svg
            className="w-4 h-4 text-indigo-600"
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
          {suggestions.map((suggestion, i) => (
            <button
              key={i}
              onClick={() => onSuggestionClick(suggestion)}
              className="group px-4 py-3 text-sm bg-gradient-to-r from-indigo-50 to-purple-50 text-indigo-700 border border-indigo-200 rounded-lg hover:from-indigo-100 hover:to-purple-100 hover:border-indigo-300 hover:shadow-md transition-all duration-200 flex items-center gap-2"
              title="Bu soruyu sor"
            >
              <svg
                className="w-4 h-4 text-indigo-500 group-hover:text-indigo-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>{suggestion}</span>
            </button>
          ))}
        </div>
      </div>
    );
  };

  const renderMessage = (message: any, index: number) => (
    <div key={message.id || index} className="space-y-4">
      {/* User Question - FULL WIDTH */}
      <div className="w-full">
        <div className="w-full bg-gradient-to-r from-gray-100 to-gray-200 rounded-2xl p-4 border-l-4 border-gray-500 shadow-sm">
          <div className="flex items-start gap-3">
            <span className="text-xl">{isStudent ? "👨‍🎓" : "👨‍🏫"}</span>
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <p className="font-medium text-sm text-gray-700">Soru</p>
                {message.timestamp && (
                  <p className="text-xs text-gray-500">
                    {formatTimestamp(message.timestamp)}
                  </p>
                )}
              </div>
              <p className="text-gray-800">{message.user}</p>
            </div>
          </div>
        </div>
      </div>

      {/* AI Assistant Response - FULL WIDTH */}
      <div className="w-full">
        <div className="w-full bg-gradient-to-br from-blue-50 to-indigo-100 border-2 border-blue-200 rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow">
          {message.bot === "..." ? (
            renderLoadingState()
          ) : (
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-white text-xl shadow-lg">
                🎓
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-4 pb-3 border-b border-blue-200">
                  <p className="font-bold text-lg text-blue-800">Cevap</p>
                  <div className="flex items-center gap-2">
                    {message.timestamp && (
                      <span className="text-xs text-blue-600">
                        {formatTimestamp(message.timestamp)}
                      </span>
                    )}
                    {(message.sources?.length ?? 0) > 0 && (
                      <span className="text-xs font-medium text-blue-700 bg-blue-50 px-2.5 py-1 rounded-full border border-blue-300">
                        {message.sources?.length} kaynak
                      </span>
                    )}
                    {message.durationMs != null && (
                      <span className="text-xs text-blue-600 bg-blue-100 px-2.5 py-1 rounded-full font-medium">
                        {message.durationMs} ms
                      </span>
                    )}
                  </div>
                </div>

                <div className="prose prose-base max-w-none text-gray-800 leading-relaxed markdown-content">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ node, ...props }) => (
                        <p
                          className="mb-4 text-gray-800 leading-7"
                          {...props}
                        />
                      ),
                      h1: ({ node, ...props }) => (
                        <h1
                          className="text-2xl font-bold mb-3 mt-6 text-gray-900 border-b border-gray-200 pb-2"
                          {...props}
                        />
                      ),
                      h2: ({ node, ...props }) => (
                        <h2
                          className="text-xl font-bold mb-3 mt-5 text-gray-900"
                          {...props}
                        />
                      ),
                      h3: ({ node, ...props }) => (
                        <h3
                          className="text-lg font-semibold mb-2 mt-4 text-gray-900"
                          {...props}
                        />
                      ),
                      ul: ({ node, ...props }) => (
                        <ul
                          className="list-disc list-inside mb-4 space-y-2 text-gray-800"
                          {...props}
                        />
                      ),
                      ol: ({ node, ...props }) => (
                        <ol
                          className="list-decimal list-inside mb-4 space-y-2 text-gray-800"
                          {...props}
                        />
                      ),
                      li: ({ node, ...props }) => (
                        <li className="ml-4" {...props} />
                      ),
                      code: ({ node, inline, ...props }: any) =>
                        inline ? (
                          <code
                            className="bg-gray-100 text-indigo-700 px-1.5 py-0.5 rounded text-sm font-mono"
                            {...props}
                          />
                        ) : (
                          <code
                            className="block bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm font-mono mb-4"
                            {...props}
                          />
                        ),
                      pre: ({ node, ...props }) => (
                        <pre
                          className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4"
                          {...props}
                        />
                      ),
                      blockquote: ({ node, ...props }) => (
                        <blockquote
                          className="border-l-4 border-indigo-500 pl-4 italic text-gray-700 my-4 bg-indigo-50 py-2 rounded-r"
                          {...props}
                        />
                      ),
                      strong: ({ node, ...props }) => (
                        <strong
                          className="font-bold text-gray-900"
                          {...props}
                        />
                      ),
                      em: ({ node, ...props }) => (
                        <em className="italic text-gray-700" {...props} />
                      ),
                      a: ({ node, ...props }) => (
                        <a
                          className="text-indigo-600 hover:text-indigo-800 underline"
                          {...props}
                        />
                      ),
                    }}
                  >
                    {message.bot}
                  </ReactMarkdown>
                </div>

                {/* Correction Notice */}
                {renderCorrection(message.correction)}

                {/* Sources Display */}
                {renderSources(message.sources)}

                {/* Suggestions */}
                {renderSuggestions(message.suggestions)}

                {/* Debug: Show feedback component status for students with APRAG */}
                {isStudent &&
                  message.aprag_interaction_id &&
                  user?.id &&
                  selectedSessionId && (
                    <div className="mt-4 text-xs text-gray-500 p-3 bg-yellow-50 border border-yellow-200 rounded">
                      <div className="font-semibold mb-1">
                        🔍 Feedback Debug Info:
                      </div>
                      <div>
                        aprag_interaction_id: ✅ {message.aprag_interaction_id}
                      </div>
                      <div>user_id: ✅ {user.id}</div>
                      <div>session_id: ✅ {selectedSessionId}</div>
                      <div>Will render EmojiFeedback: ✅ YES</div>
                    </div>
                  )}

                {/* Emoji Feedback Component for Students */}
                {isStudent &&
                  message.aprag_interaction_id &&
                  user?.id &&
                  selectedSessionId && (
                    <div className="mt-6 pt-4 border-t border-gray-100">
                      {/* Placeholder for EmojiFeedback - would be imported and used */}
                      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                        <div className="text-sm font-medium text-blue-800 mb-2">
                          📝 Geri Bildirim
                        </div>
                        <div className="text-xs text-blue-600">
                          EmojiFeedback component buraya gelecek (interactionId:{" "}
                          {message.aprag_interaction_id})
                        </div>
                      </div>
                    </div>
                  )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  // Chat container ref for auto-scroll
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [isUserScrolling, setIsUserScrolling] = useState(false);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatContainerRef.current && !isUserScrolling) {
      const container = chatContainerRef.current;
      container.scrollTo({
        top: container.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages.length, isLoading, isUserScrolling]);

  // Handle scroll events to show/hide scroll-to-bottom button
  const handleScroll = () => {
    if (!chatContainerRef.current) return;

    const container = chatContainerRef.current;
    const isNearBottom =
      container.scrollTop + container.clientHeight >=
      container.scrollHeight - 100;

    setShowScrollButton(!isNearBottom && messages.length > 0);
    setIsUserScrolling(true);

    // Reset user scrolling flag after a delay
    clearTimeout((window as any).scrollTimeout);
    (window as any).scrollTimeout = setTimeout(() => {
      setIsUserScrolling(false);
    }, 1000);
  };

  // Scroll to bottom function
  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: "smooth",
      });
      setShowScrollButton(false);
      setIsUserScrolling(false);
    }
  };

  return (
    <div className="relative flex flex-col h-full">
      {/* Header with Clear Button */}
      {messages.length > 0 && clearChatHistory && (
        <div className="flex items-center justify-between mb-4 px-2">
          <h3 className="text-lg font-semibold text-gray-700">
            Sohbet Geçmişi ({messages.length} mesaj)
          </h3>
          <button
            onClick={handleClearHistory}
            className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-red-600 bg-red-50 border border-red-200 rounded-lg hover:bg-red-100 hover:border-red-300 transition-all duration-200 group"
            title="Sohbet geçmişini temizle"
          >
            <svg
              className="w-4 h-4 group-hover:scale-110 transition-transform"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
            <span>Sohbeti Temizle</span>
          </button>
        </div>
      )}

      <div
        ref={chatContainerRef}
        id="chat-history-container"
        className="flex-1 min-h-[50vh] max-h-[70vh] overflow-y-auto bg-gradient-to-b from-indigo-50/40 to-white rounded-xl border border-gray-200 p-4 scroll-smooth"
        onScroll={handleScroll}
        style={{ scrollBehavior: "smooth" }}
      >
        {/* Empty State */}
        {messages.length === 0 && !isLoading && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center py-16">
              <div className="text-6xl mb-4">🎓</div>
              <h3 className="text-xl font-semibold text-gray-700 mb-2">
                {isStudent
                  ? "Merhaba! Size nasıl yardımcı olabilirim?"
                  : "Eğitim Asistanınıza Hoş Geldiniz!"}
              </h3>
              <p className="text-gray-500 max-w-md mx-auto">
                {isStudent
                  ? "Ders hakkında sorularınızı bekliyorum. Detaylı cevaplar verebilirim!"
                  : "Ders materyalleriniz hakkında soru sorarak öğrenme sürecinizi destekleyin."}
              </p>
            </div>
          </div>
        )}

        {/* Messages - Gemini-style vertical layout (oldest to newest) */}
        <div className="flex flex-col space-y-6">
          {messages.map((message, index) => renderMessage(message, index))}
        </div>

        {/* Loading indicator if currently querying */}
        {isLoading && <div className="mt-4">{renderLoadingState()}</div>}
      </div>

      {/* Scroll to Bottom Button */}
      {showScrollButton && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-4 right-4 p-3 bg-indigo-600 text-white rounded-full shadow-lg hover:bg-indigo-700 transition-all duration-200 z-10 flex items-center gap-2"
          title="En alta kaydır"
        >
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
              d="M19 14l-7 7m0 0l-7-7m7 7V3"
            />
          </svg>
          <span className="text-sm font-medium">Aşağı</span>
        </button>
      )}
    </div>
  );
}
