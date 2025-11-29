"use client";

import React from "react";
import Link from "next/link";

interface SessionMeta {
  session_id: string;
  name: string;
  description?: string;
  status: string;
  document_count: number;
  total_chunks: number;
  query_count: number;
  rag_settings?: any;
  updated_at: string;
}

interface SessionSelectionProps {
  sessions: SessionMeta[];
  selectedSessionId: string;
  onSessionChange: (sessionId: string) => void;
  sessionRagSettings?: any;
  loading: boolean;
  useDirectLLM: boolean;
  isStudent: boolean;
}

export default function SessionSelection({
  sessions,
  selectedSessionId,
  onSessionChange,
  sessionRagSettings,
  loading,
  useDirectLLM,
  isStudent,
}: SessionSelectionProps) {
  // Filter active sessions
  const activeSessions = sessions.filter((s) => s.status === "active");

  // Find selected session details
  const selectedSession = sessions.find(
    (s) => s.session_id === selectedSessionId
  );

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
      {/* Session Selection - Horizontal */}
      <div className="flex-1 min-w-0">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          📚 Ders Oturumu
        </label>
        <select
          value={selectedSessionId}
          onChange={(e) => onSessionChange(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all bg-white shadow-sm"
          disabled={loading || isStudent}
        >
          <option value="">
            {useDirectLLM
              ? "Direkt LLM (Oturum Gerekmez)"
              : "Ders Oturumu Seçin"}
          </option>
          {activeSessions.map((session) => (
            <option key={session.session_id} value={session.session_id}>
              {session.name} ({session.document_count} doküman)
            </option>
          ))}
        </select>
        {!useDirectLLM && (
          <p className="text-xs text-gray-500 mt-1 flex items-center gap-1">
            <span>💡</span>
            <span>RAG ayarlarını düzenlemek için oturum sayfasına gidin</span>
          </p>
        )}
      </div>

      {/* Selected Session Info - Horizontal Compact */}
      {selectedSession && !useDirectLLM && (
        <div className="flex items-center gap-4 px-4 py-2 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
          <div className="text-center">
            <div className="text-lg font-bold text-blue-600">
              {selectedSession.document_count}
            </div>
            <div className="text-xs text-gray-600">Doküman</div>
          </div>
          <div className="w-px h-8 bg-blue-300"></div>
          <div className="text-center">
            <div className="text-lg font-bold text-green-600">
              {selectedSession.total_chunks}
            </div>
            <div className="text-xs text-gray-600">İçerik</div>
          </div>
          <div className="w-px h-8 bg-blue-300"></div>
          <div className="text-center">
            <div className="text-lg font-bold text-purple-600">
              {selectedSession.query_count}
            </div>
            <div className="text-xs text-gray-600">Toplam</div>
          </div>

        </div>
      )}

      {/* Direct LLM Mode Info - Compact */}
      {useDirectLLM && (
        <div className="px-3 py-2 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg border border-amber-200 text-sm">
          <span className="text-amber-800 font-medium">🤖 Direkt LLM Modu Aktif</span>
        </div>
      )}

      {/* No Sessions Warning - Compact */}
      {activeSessions.length === 0 && !useDirectLLM && (
        <div className="px-3 py-2 bg-red-50 border border-red-200 rounded-lg text-sm">
          <span className="text-red-800 font-medium">⚠️ Aktif Oturum Bulunamadı</span>
        </div>
      )}
    </div>
  );
}
