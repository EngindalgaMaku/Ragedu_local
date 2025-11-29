"use client";

import React, { FormEvent } from "react";

interface QueryFormProps {
  query: string;
  setQuery: (query: string) => void;
  onSubmit: (e: FormEvent, suggestionText?: string) => Promise<void>;
  isQuerying: boolean;
  studentChatLoading?: boolean;
  selectedQueryModel?: string;
  sessionRagSettings?: any;
  answerLength: "short" | "normal" | "detailed";
  setAnswerLength: (length: "short" | "normal" | "detailed") => void;
  isStudent: boolean;
  chatHistoryLength: number;
  onSuggestionClick: (suggestion: string) => void;
}

export default function QueryForm({
  query,
  setQuery,
  onSubmit,
  isQuerying,
  studentChatLoading = false,
  selectedQueryModel,
  sessionRagSettings,
  answerLength,
  setAnswerLength,
  isStudent,
  chatHistoryLength,
  onSuggestionClick,
}: QueryFormProps) {
  const isLoading = isQuerying || studentChatLoading;
  const hasModel = sessionRagSettings?.model || selectedQueryModel;

  // Default suggestions for when no chat history exists
  const defaultSuggestions = [
    "Kısa özet çıkar",
    "Bu başlığın ana fikirleri neler?",
    "Öğrenciye 3 soru hazırla",
    "Kaynaklardan alıntıyla cevapla",
  ];

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-lg p-4 mb-6">
      {/* Answer Length Selector - Only for teachers */}
      {!isStudent && (
        <div className="mb-3 flex items-center gap-2">
          <span className="text-xs font-medium text-gray-600">
            Cevap Uzunluğu:
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setAnswerLength("short")}
              className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                answerLength === "short"
                  ? "bg-blue-500 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              🔸 Kısa
            </button>
            <button
              type="button"
              onClick={() => setAnswerLength("normal")}
              className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                answerLength === "normal"
                  ? "bg-blue-500 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              🔹 Normal
            </button>
            <button
              type="button"
              onClick={() => setAnswerLength("detailed")}
              className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                answerLength === "detailed"
                  ? "bg-blue-500 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              🔶 Detaylı
            </button>
          </div>
        </div>
      )}

      {/* Query Form */}
      <form onSubmit={onSubmit}>
        <div className="flex items-center gap-3">
          <div className="flex-1 relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                isStudent
                  ? "Ders hakkında soru sorabilirsiniz..."
                  : "Ders materyalleri hakkında sorunuzu yazın..."
              }
              className="w-full p-4 pr-12 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-gray-800 placeholder-gray-400"
              disabled={isLoading || (!hasModel && !isStudent)}
            />
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400">
              💭
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading || !query.trim() || (!hasModel && !isStudent)}
            className="px-6 py-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg font-medium flex items-center gap-2"
          >
            {isLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Düşünüyor...</span>
              </>
            ) : (
              <>
                <span className="text-lg">🚀</span>
                <span>Sor</span>
              </>
            )}
          </button>
        </div>

        {/* Tips */}
        <div className="mt-2 text-xs text-gray-500 flex items-center gap-1">
          <span>💡</span>
          <span>İpucu: Spesifik sorular daha iyi sonuçlar verir</span>
        </div>
      </form>

      {/* Default Suggestions - Show when no chat history */}
      {chatHistoryLength === 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          <div className="w-full mb-2">
            <span className="text-xs font-medium text-gray-600 flex items-center gap-1">
              <span>💡</span>
              <span>Örnek sorular:</span>
            </span>
          </div>
          {defaultSuggestions.map((suggestion, i) => (
            <button
              type="button"
              key={i}
              onClick={() => onSuggestionClick(suggestion)}
              className="px-3 py-2 text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-lg hover:bg-indigo-100 transition-colors shadow-sm"
              title="Öneriyi sor"
              disabled={isLoading}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
