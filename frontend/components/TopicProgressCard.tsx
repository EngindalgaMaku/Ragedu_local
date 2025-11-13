"use client";

import React, { useState, useEffect } from "react";
import {
  getStudentProgress,
  TopicProgress,
  StudentProgressResponse,
} from "@/lib/api";

interface TopicProgressCardProps {
  userId: string;
  sessionId: string;
  apragEnabled: boolean;
}

const TopicProgressCard: React.FC<TopicProgressCardProps> = ({
  userId,
  sessionId,
  apragEnabled,
}) => {
  const [progress, setProgress] = useState<StudentProgressResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchProgress = async () => {
    if (!userId || !sessionId || !apragEnabled) return;

    try {
      setLoading(true);
      setError(null);
      const data = await getStudentProgress(userId, sessionId);
      setProgress(data);
    } catch (e: any) {
      console.error("Failed to fetch progress:", e);
      setError("İlerleme bilgisi yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProgress();
    // Refresh every 30 seconds
    const interval = setInterval(fetchProgress, 30000);
    return () => clearInterval(interval);
  }, [userId, sessionId, apragEnabled]);

  if (!apragEnabled) {
    return null;
  }

  if (loading && !progress) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (error && !progress) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
      </div>
    );
  }

  if (!progress || !progress.success || progress.progress.length === 0) {
    return null;
  }

  const currentTopic = progress.current_topic;
  const nextTopic = progress.next_recommended_topic;

  const getMasteryColor = (masteryLevel: string | null) => {
    switch (masteryLevel) {
      case "mastered":
        return "bg-green-500";
      case "learning":
        return "bg-yellow-500";
      case "needs_review":
        return "bg-red-500";
      default:
        return "bg-gray-300";
    }
  };

  const getMasteryText = (masteryLevel: string | null) => {
    switch (masteryLevel) {
      case "mastered":
        return "Tamamlandı";
      case "learning":
        return "Öğreniyor";
      case "needs_review":
        return "Tekrar Gerekli";
      default:
        return "Başlanmadı";
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 space-y-4">
      {/* Current Topic */}
      {currentTopic && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
              Mevcut Konu
            </h3>
            <span
              className={`text-xs px-2 py-1 rounded ${
                currentTopic.mastery_level === "mastered"
                  ? "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300"
                  : currentTopic.mastery_level === "learning"
                  ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-300"
                  : "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300"
              }`}
            >
              {getMasteryText(currentTopic.mastery_level)}
            </span>
          </div>
          <p className="text-sm font-medium text-gray-800 dark:text-gray-200 mb-2">
            {currentTopic.topic_title}
          </p>
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
              <span>İlerleme</span>
              <span>
                {currentTopic.questions_asked} soru soruldu
                {currentTopic.mastery_score !== null &&
                  ` • ${(currentTopic.mastery_score * 100).toFixed(0)}%`}
              </span>
            </div>
            {currentTopic.mastery_score !== null && (
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all ${getMasteryColor(
                    currentTopic.mastery_level
                  )}`}
                  style={{
                    width: `${currentTopic.mastery_score * 100}%`,
                  }}
                ></div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Next Recommended Topic */}
      {nextTopic && (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <div className="flex items-center gap-2 mb-2">
            <svg
              className="w-4 h-4 text-blue-600 dark:text-blue-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 7l5 5m0 0l-5 5m5-5H6"
              />
            </svg>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
              Sonraki Önerilen Konu
            </h3>
          </div>
          <p className="text-sm font-medium text-gray-800 dark:text-gray-200 mb-1">
            {nextTopic.topic_title}
          </p>
          {nextTopic.readiness_score !== null && (
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Hazır olma durumu:{" "}
              {(nextTopic.readiness_score * 100).toFixed(0)}%
            </p>
          )}
        </div>
      )}

      {/* Progress Summary */}
      {progress.progress.length > 0 && (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
              Genel İlerleme
            </h3>
            <span className="text-xs text-gray-600 dark:text-gray-400">
              {progress.progress.filter((p) => p.mastery_level === "mastered")
                .length}{" "}
              / {progress.progress.length} konu tamamlandı
            </span>
          </div>
          <div className="space-y-1">
            {progress.progress
              .slice(0, 5)
              .map((topicProgress) => (
                <div
                  key={topicProgress.topic_id}
                  className="flex items-center justify-between text-xs"
                >
                  <span className="text-gray-700 dark:text-gray-300 truncate flex-1">
                    {topicProgress.topic_title}
                  </span>
                  <div className="flex items-center gap-2 ml-2">
                    <span className="text-gray-500 dark:text-gray-400">
                      {topicProgress.questions_asked} soru
                    </span>
                    <div
                      className={`w-12 h-1.5 rounded-full ${
                        topicProgress.mastery_level === "mastered"
                          ? "bg-green-500"
                          : topicProgress.mastery_level === "learning"
                          ? "bg-yellow-500"
                          : "bg-gray-300"
                      }`}
                    ></div>
                  </div>
                </div>
              ))}
            {progress.progress.length > 5 && (
              <p className="text-xs text-gray-500 dark:text-gray-400 text-center pt-1">
                +{progress.progress.length - 5} konu daha
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default TopicProgressCard;



