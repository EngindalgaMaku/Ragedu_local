"use client";

import React, { useEffect, useState } from "react";
import {
  getMultiDimensionalStats,
  getEmojiStats,
  MultiDimensionalStats,
} from "@/lib/api";

interface EmojiStatsResponse {
  user_id: string;
  session_id: string;
  total_feedback_count: number;
  emoji_distribution: Record<string, number>;
  avg_score: number;
  most_common_emoji: string | null;
  recent_trend: string;
}

interface MultiFeedbackAnalyticsProps {
  userId: string;
  sessionId: string;
  className?: string;
}

interface DimensionInsight {
  dimension: string;
  title: string;
  score: number;
  status: "excellent" | "good" | "needs_improvement" | "poor";
  color: string;
  description: string;
}

const DIMENSION_CONFIG = {
  understanding: {
    title: "Anlama Düzeyi",
    icon: "🧠",
    description: "Açıklamaları ne kadar iyi anlıyor",
  },
  relevance: {
    title: "İlgili Olma",
    icon: "🎯",
    description: "Cevaplar sorularla ne kadar uyumlu",
  },
  clarity: {
    title: "Açıklık",
    icon: "💡",
    description: "Açıklamalar ne kadar net ve anlaşılır",
  },
};

const getScoreStatus = (
  score: number
): {
  status: "excellent" | "good" | "needs_improvement" | "poor";
  color: string;
  description: string;
} => {
  if (score >= 4.5)
    return {
      status: "excellent",
      color: "text-green-600 bg-green-50 border-green-200",
      description: "Mükemmel performans",
    };
  if (score >= 3.5)
    return {
      status: "good",
      color: "text-blue-600 bg-blue-50 border-blue-200",
      description: "İyi performans",
    };
  if (score >= 2.5)
    return {
      status: "needs_improvement",
      color: "text-yellow-600 bg-yellow-50 border-yellow-200",
      description: "Geliştirilmeli",
    };
  return {
    status: "poor",
    color: "text-red-600 bg-red-50 border-red-200",
    description: "Dikkat gerekli",
  };
};

const getProgressWidth = (score: number) => `${(score / 5) * 100}%`;

export default function MultiFeedbackAnalytics({
  userId,
  sessionId,
  className = "",
}: MultiFeedbackAnalyticsProps) {
  const [multiStats, setMultiStats] = useState<MultiDimensionalStats | null>(
    null
  );
  const [emojiStats, setEmojiStats] = useState<EmojiStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        setLoading(true);
        setError(null);

        const [multiData, emojiData] = await Promise.all([
          getMultiDimensionalStats(userId, sessionId),
          getEmojiStats(userId, sessionId).catch(() => null), // Emoji stats might not exist
        ]);

        setMultiStats(multiData);
        setEmojiStats(emojiData);
      } catch (err: any) {
        console.error("Failed to fetch feedback analytics:", err);
        setError("Analitik veriler yüklenemedi");
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, [userId, sessionId]);

  if (loading) {
    return (
      <div className={`p-6 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded mb-4 w-1/3"></div>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`p-6 ${className}`}>
        <div className="text-center py-8">
          <div className="text-red-500 text-4xl mb-2">⚠️</div>
          <p className="text-red-600 font-medium">{error}</p>
        </div>
      </div>
    );
  }

  const hasMultiData = multiStats && multiStats.dimension_feedback_count > 0;
  const hasEmojiData = emojiStats && emojiStats.total_feedback_count > 0;

  if (!hasMultiData && !hasEmojiData) {
    return (
      <div className={`p-6 ${className}`}>
        <div className="text-center py-8">
          <div className="text-gray-400 text-4xl mb-2">📊</div>
          <p className="text-gray-600 font-medium">
            Henüz geri bildirim verisi yok
          </p>
          <p className="text-gray-500 text-sm mt-1">
            Cevaplara geri bildirim verdikçe analitikler burada görünecek
          </p>
        </div>
      </div>
    );
  }

  // Prepare dimension insights
  const dimensionInsights: DimensionInsight[] = [];

  if (hasMultiData) {
    if (multiStats!.avg_understanding) {
      const statusInfo = getScoreStatus(multiStats!.avg_understanding);
      dimensionInsights.push({
        dimension: "understanding",
        title: DIMENSION_CONFIG.understanding.title,
        score: multiStats!.avg_understanding,
        ...statusInfo,
      });
    }

    if (multiStats!.avg_relevance) {
      const statusInfo = getScoreStatus(multiStats!.avg_relevance);
      dimensionInsights.push({
        dimension: "relevance",
        title: DIMENSION_CONFIG.relevance.title,
        score: multiStats!.avg_relevance,
        ...statusInfo,
      });
    }

    if (multiStats!.avg_clarity) {
      const statusInfo = getScoreStatus(multiStats!.avg_clarity);
      dimensionInsights.push({
        dimension: "clarity",
        title: DIMENSION_CONFIG.clarity.title,
        score: multiStats!.avg_clarity,
        ...statusInfo,
      });
    }
  }

  return (
    <div
      className={`bg-white rounded-lg border border-gray-200 shadow-sm ${className}`}
    >
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-800">
            Geri Bildirim Analitikleri
          </h3>
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            {hasMultiData && (
              <div className="flex items-center space-x-1">
                <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                <span>{multiStats!.dimension_feedback_count} detaylı</span>
              </div>
            )}
            {hasEmojiData && (
              <div className="flex items-center space-x-1">
                <span className="w-2 h-2 bg-gray-400 rounded-full"></span>
                <span>{emojiStats!.total_feedback_count} emoji</span>
              </div>
            )}
          </div>
        </div>

        {/* Multi-Dimensional Analysis */}
        {hasMultiData && dimensionInsights.length > 0 && (
          <div className="mb-8">
            <h4 className="text-md font-medium text-gray-800 mb-4 flex items-center">
              <span className="mr-2">📏</span>
              Boyutsal Analiz
            </h4>

            <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-3">
              {dimensionInsights.map((insight) => {
                const config =
                  DIMENSION_CONFIG[
                    insight.dimension as keyof typeof DIMENSION_CONFIG
                  ];
                return (
                  <div
                    key={insight.dimension}
                    className={`p-4 rounded-lg border-2 ${insight.color}`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-xl">{config.icon}</span>
                        <h5 className="font-medium">{insight.title}</h5>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold">
                          {insight.score.toFixed(1)}
                        </div>
                        <div className="text-xs opacity-75">/5.0</div>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                      <div
                        className={`h-2 rounded-full ${
                          insight.status === "excellent"
                            ? "bg-green-500"
                            : insight.status === "good"
                            ? "bg-blue-500"
                            : insight.status === "needs_improvement"
                            ? "bg-yellow-500"
                            : "bg-red-500"
                        }`}
                        style={{ width: getProgressWidth(insight.score) }}
                      ></div>
                    </div>

                    <p className="text-xs">{config.description}</p>
                    <p className="text-xs font-medium mt-1">
                      {insight.description}
                    </p>
                  </div>
                );
              })}
            </div>

            {/* Overall Score */}
            {multiStats!.avg_overall && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="text-xl">⭐</span>
                    <span className="font-medium text-gray-800">
                      Genel Ortalama
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-bold text-blue-600">
                      {multiStats!.avg_overall.toFixed(1)}
                    </div>
                    <div className="text-sm text-gray-500">/5.0</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Emoji Overview */}
        {hasEmojiData && (
          <div className="mb-6">
            <h4 className="text-md font-medium text-gray-800 mb-4 flex items-center">
              <span className="mr-2">😊</span>
              Emoji Dağılımı
            </h4>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(emojiStats!.emoji_distribution).map(
                ([emoji, count]) => (
                  <div
                    key={emoji}
                    className="text-center p-3 bg-gray-50 rounded-lg"
                  >
                    <div className="text-2xl mb-1">{emoji}</div>
                    <div className="text-lg font-semibold text-gray-800">
                      {count as number}
                    </div>
                    <div className="text-xs text-gray-500">kez</div>
                  </div>
                )
              )}
            </div>

            <div className="mt-3 flex items-center justify-between text-sm">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-1">
                  <span className="text-gray-600">Ortalama:</span>
                  <span className="font-medium">
                    {(emojiStats!.avg_score * 5).toFixed(1)}/5.0
                  </span>
                </div>
                {emojiStats!.most_common_emoji && (
                  <div className="flex items-center space-x-1">
                    <span className="text-gray-600">En çok:</span>
                    <span className="text-lg">
                      {emojiStats!.most_common_emoji}
                    </span>
                  </div>
                )}
              </div>
              <div
                className={`px-2 py-1 rounded text-xs font-medium ${
                  emojiStats!.recent_trend === "positive"
                    ? "bg-green-100 text-green-700"
                    : emojiStats!.recent_trend === "negative"
                    ? "bg-red-100 text-red-700"
                    : "bg-gray-100 text-gray-700"
                }`}
              >
                {emojiStats!.recent_trend === "positive"
                  ? "📈 Pozitif"
                  : emojiStats!.recent_trend === "negative"
                  ? "📉 Negatif"
                  : "➡️ Kararlı"}
              </div>
            </div>
          </div>
        )}

        {/* Improvement Insights */}
        {hasMultiData && multiStats!.weak_dimensions.length > 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <h5 className="font-medium text-yellow-800 mb-2 flex items-center">
              <span className="mr-2">💡</span>
              İyileştirme Önerileri
            </h5>
            <div className="text-sm text-yellow-700 space-y-1">
              {multiStats!.weak_dimensions.map((dim) => (
                <div key={dim}>
                  •{" "}
                  <strong>
                    {
                      DIMENSION_CONFIG[dim as keyof typeof DIMENSION_CONFIG]
                        ?.title
                    }
                  </strong>{" "}
                  boyutunda gelişim fırsatı var
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Strong Points */}
        {hasMultiData && multiStats!.strong_dimensions.length > 0 && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 mt-4">
            <h5 className="font-medium text-green-800 mb-2 flex items-center">
              <span className="mr-2">🌟</span>
              Güçlü Yönler
            </h5>
            <div className="text-sm text-green-700 space-y-1">
              {multiStats!.strong_dimensions.map((dim) => (
                <div key={dim}>
                  •{" "}
                  <strong>
                    {
                      DIMENSION_CONFIG[dim as keyof typeof DIMENSION_CONFIG]
                        ?.title
                    }
                  </strong>{" "}
                  boyutunda başarılı
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Trend Information */}
        {hasMultiData &&
          multiStats!.improvement_trend !== "insufficient_data" && (
            <div className="mt-4 text-center">
              <div
                className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                  multiStats!.improvement_trend === "improving"
                    ? "bg-green-100 text-green-700"
                    : multiStats!.improvement_trend === "declining"
                    ? "bg-red-100 text-red-700"
                    : "bg-gray-100 text-gray-700"
                }`}
              >
                <span className="mr-1">
                  {multiStats!.improvement_trend === "improving"
                    ? "📈"
                    : multiStats!.improvement_trend === "declining"
                    ? "📉"
                    : "➡️"}
                </span>
                {multiStats!.improvement_trend === "improving"
                  ? "Gelişim gösteriyor"
                  : multiStats!.improvement_trend === "declining"
                  ? "Düşüş eğilimi"
                  : "Kararlı performans"}
              </div>
            </div>
          )}
      </div>
    </div>
  );
}
