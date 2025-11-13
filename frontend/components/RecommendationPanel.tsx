"use client";
import React, { useState, useEffect } from "react";
import {
  getRecommendations,
  acceptRecommendation,
  dismissRecommendation,
  APRAGRecommendation,
} from "@/lib/api";

interface RecommendationPanelProps {
  userId: string;
  sessionId?: string;
  onQuestionClick?: (question: string) => void;
}

export default function RecommendationPanel({
  userId,
  sessionId,
  onQuestionClick,
}: RecommendationPanelProps) {
  const [recommendations, setRecommendations] = useState<APRAGRecommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(true);

  const fetchRecommendations = async () => {
    if (!userId) return;
    
    try {
      setLoading(true);
      setError(null);
      const data = await getRecommendations(userId, sessionId, 10);
      setRecommendations(data.recommendations || []);
    } catch (e: any) {
      console.error("Failed to fetch recommendations:", e);
      setError("Öneriler yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendations();
    // Refresh recommendations every 30 seconds
    const interval = setInterval(fetchRecommendations, 30000);
    return () => clearInterval(interval);
  }, [userId, sessionId]);

  const handleAccept = async (recommendationId: number) => {
    try {
      await acceptRecommendation(recommendationId);
      setRecommendations((prev) =>
        prev.filter((r) => r.recommendation_id !== recommendationId)
      );
    } catch (e: any) {
      console.error("Failed to accept recommendation:", e);
    }
  };

  const handleDismiss = async (recommendationId: number) => {
    try {
      await dismissRecommendation(recommendationId);
      setRecommendations((prev) =>
        prev.filter((r) => r.recommendation_id !== recommendationId)
      );
    } catch (e: any) {
      console.error("Failed to dismiss recommendation:", e);
    }
  };

  const handleQuestionClick = (question: string) => {
    if (onQuestionClick) {
      onQuestionClick(question);
    }
  };

  if (!expanded && recommendations.length === 0) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="fixed bottom-4 right-4 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg hover:bg-blue-700 transition-colors z-40"
      >
        💡 Öneriler
      </button>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
      <div
        className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white p-4 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl">💡</span>
            <h3 className="font-semibold">Kişiselleştirilmiş Öneriler</h3>
            {recommendations.length > 0 && (
              <span className="bg-white/20 px-2 py-0.5 rounded-full text-xs">
                {recommendations.length}
              </span>
            )}
          </div>
          <svg
            className={`w-5 h-5 transition-transform ${expanded ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </div>

      {expanded && (
        <div className="p-4 max-h-96 overflow-y-auto">
          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-blue-500 border-t-transparent mx-auto"></div>
              <p className="text-sm text-gray-500 mt-2">Öneriler yükleniyor...</p>
            </div>
          ) : error ? (
            <div className="text-center py-8">
              <p className="text-sm text-red-600">{error}</p>
              <button
                onClick={fetchRecommendations}
                className="mt-2 text-sm text-blue-600 hover:underline"
              >
                Tekrar Dene
              </button>
            </div>
          ) : recommendations.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-sm text-gray-500">
                Henüz öneri bulunmuyor. Daha fazla soru sordukça öneriler oluşturulacak.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {recommendations.map((rec, index) => (
                <div
                  key={rec.recommendation_id || index}
                  className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 bg-gray-50 dark:bg-gray-900"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <h4 className="font-semibold text-sm text-gray-800 dark:text-gray-200">
                        {rec.title}
                      </h4>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        {rec.description}
                      </p>
                    </div>
                    <div className="flex gap-1 ml-2">
                      {rec.recommendation_id && (
                        <>
                          <button
                            onClick={() => handleAccept(rec.recommendation_id!)}
                            className="text-green-600 hover:text-green-700 text-xs"
                            title="Kabul Et"
                          >
                            ✓
                          </button>
                          <button
                            onClick={() => handleDismiss(rec.recommendation_id!)}
                            className="text-red-600 hover:text-red-700 text-xs"
                            title="Reddet"
                          >
                            ×
                          </button>
                        </>
                      )}
                    </div>
                  </div>

                  {rec.recommendation_type === "question" &&
                    rec.content.suggested_questions && (
                      <div className="mt-2 space-y-1">
                        {rec.content.suggested_questions.map((question, qIdx) => (
                          <button
                            key={qIdx}
                            onClick={() => handleQuestionClick(question)}
                            className="block w-full text-left text-xs text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 hover:underline p-2 rounded bg-white dark:bg-gray-800 border border-blue-200 dark:border-blue-800 transition-colors"
                          >
                            💬 {question}
                          </button>
                        ))}
                      </div>
                    )}

                  {rec.recommendation_type === "topic" && (
                    <div className="mt-2">
                      <span className="text-xs text-gray-600 dark:text-gray-400">
                        Konu: {rec.content.topic || "Genel"}
                      </span>
                    </div>
                  )}

                  <div className="mt-2 flex items-center gap-2">
                    <span className="text-xs text-gray-500">
                      Öncelik: {rec.priority}/10
                    </span>
                    <span className="text-xs text-gray-500">
                      • İlgililik: {Math.round(rec.relevance_score * 100)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

