"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import {
  getAnalyticsSummary,
  getStudentProgress,
  getSessionTopics,
  listSessions,
  getAPRAGSettings,
  SessionMeta,
  APRAGAnalyticsSummary,
  StudentProgressResponse,
  Topic,
  APRAGSettings,
} from "@/lib/api";
import {
  BookOpen,
  TrendingUp,
  Target,
  Award,
  CheckCircle,
  Clock,
  ArrowRight,
  BarChart3,
  Brain,
} from "lucide-react";

interface StudentDashboardProps {
  userId: string;
}

export default function StudentDashboard({ userId }: StudentDashboardProps) {
  const { user } = useAuth();
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<APRAGAnalyticsSummary | null>(null);
  const [progress, setProgress] = useState<StudentProgressResponse | null>(null);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [apragSettings, setApragSettings] = useState<APRAGSettings | null>(null);

  // Load sessions
  useEffect(() => {
    const loadSessions = async () => {
      try {
        const sessionsList = await listSessions();
        setSessions(sessionsList);
        
        // Auto-select first session
        if (sessionsList.length > 0 && !selectedSession) {
          setSelectedSession(sessionsList[0].session_id);
        }
      } catch (err) {
        console.error("Failed to load sessions:", err);
        setError("Oturumlar yüklenemedi");
      }
    };

    loadSessions();
  }, []);

  // Load analytics and progress when session changes
  useEffect(() => {
    const loadData = async () => {
      if (!selectedSession || !userId) return;

      setLoading(true);
      setError(null);

      try {
        // Check APRAG settings first
        const settings = await getAPRAGSettings(selectedSession);
        setApragSettings(settings);

        // Only load APRAG data if enabled
        if (settings.enabled) {
          // Load analytics, progress, and topics in parallel
          const [analyticsData, progressData, topicsData] = await Promise.all([
            getAnalyticsSummary(userId, selectedSession),
            getStudentProgress(userId, selectedSession),
            getSessionTopics(selectedSession),
          ]);

          setAnalytics(analyticsData);
          setProgress(progressData);
          setTopics(topicsData.topics || []);
        } else {
          // APRAG disabled, clear data
          setAnalytics(null);
          setProgress(null);
          setTopics([]);
        }
      } catch (err) {
        console.error("Failed to load student data:", err);
        setError("Veriler yüklenirken bir hata oluştu");
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedSession, userId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Yükleniyor...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
        <p className="text-red-600">{error}</p>
      </div>
    );
  }

  // APRAG disabled message
  if (!loading && apragSettings && !apragSettings.enabled) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-yellow-50 border-2 border-yellow-200 rounded-xl p-8 text-center">
          <div className="w-20 h-20 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-10 h-10 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            APRAG Sistemi Devre Dışı
          </h2>
          <p className="text-gray-600 mb-4">
            Öğrenme analitikleri ve ilerleme takibi şu anda kullanılamıyor.
          </p>
          <p className="text-sm text-gray-500">
            Bu özellikler admin panelinden etkinleştirilebilir.
          </p>
        </div>
      </div>
    );
  }

  // Calculate mastery percentage
  const calculateMastery = () => {
    if (!progress?.progress || progress.progress.length === 0) return 0;
    
    const masteredCount = progress.progress.filter(
      (p) => p.mastery_level === "mastered"
    ).length;
    
    return Math.round((masteredCount / progress.progress.length) * 100);
  };

  // Get trend icon and color
  const getTrendInfo = (trend: string) => {
    switch (trend) {
      case "improving":
        return { icon: TrendingUp, color: "text-green-500", label: "Gelişiyor" };
      case "stable":
        return { icon: Target, color: "text-blue-500", label: "Sabit" };
      case "declining":
        return { icon: TrendingUp, color: "text-red-500 rotate-180", label: "Azalıyor" };
      default:
        return { icon: Target, color: "text-gray-500", label: "Yetersiz Veri" };
    }
  };

  const trendInfo = analytics ? getTrendInfo(analytics.improvement_trend) : null;
  const TrendIcon = trendInfo?.icon;

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Hoş Geldin, {user?.first_name || user?.username}! 👋
            </h1>
            <p className="text-gray-600 mt-1">Öğrenme yolculuğuna devam et</p>
          </div>
          
          {/* Session Selector */}
          {sessions.length > 0 && (
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium text-gray-700">Oturum:</label>
              <select
                value={selectedSession || ""}
                onChange={(e) => setSelectedSession(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {sessions.map((session) => (
                  <option key={session.session_id} value={session.session_id}>
                    {session.name}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Questions */}
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-lg p-6 text-white">
          <div className="flex items-center justify-between mb-3">
            <BookOpen className="w-8 h-8 opacity-80" />
            <span className="text-3xl font-bold">
              {analytics?.total_interactions || 0}
            </span>
          </div>
          <h3 className="text-sm font-medium opacity-90">Toplam Soru</h3>
          <p className="text-xs opacity-75 mt-1">Sorduğun soru sayısı</p>
        </div>

        {/* Average Understanding */}
        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg shadow-lg p-6 text-white">
          <div className="flex items-center justify-between mb-3">
            <Brain className="w-8 h-8 opacity-80" />
            <span className="text-3xl font-bold">
              {analytics?.average_understanding
                ? analytics.average_understanding.toFixed(1)
                : "-"}
            </span>
          </div>
          <h3 className="text-sm font-medium opacity-90">Anlama Düzeyi</h3>
          <p className="text-xs opacity-75 mt-1">5 üzerinden ortalama</p>
        </div>

        {/* Mastery Progress */}
        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg shadow-lg p-6 text-white">
          <div className="flex items-center justify-between mb-3">
            <Award className="w-8 h-8 opacity-80" />
            <span className="text-3xl font-bold">{calculateMastery()}%</span>
          </div>
          <h3 className="text-sm font-medium opacity-90">Hakimiyet</h3>
          <p className="text-xs opacity-75 mt-1">
            {progress?.progress.filter((p) => p.mastery_level === "mastered").length || 0} /{" "}
            {progress?.progress.length || 0} konu tamamlandı
          </p>
        </div>

        {/* Engagement Level */}
        <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg shadow-lg p-6 text-white">
          <div className="flex items-center justify-between mb-3">
            {TrendIcon && <TrendIcon className={`w-8 h-8 opacity-80 ${trendInfo?.color}`} />}
            <span className="text-3xl font-bold capitalize">
              {analytics?.engagement_level || "-"}
            </span>
          </div>
          <h3 className="text-sm font-medium opacity-90">Katılım Seviyesi</h3>
          <p className="text-xs opacity-75 mt-1">{trendInfo?.label}</p>
        </div>
      </div>

      {/* Current Topic & Recommendations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Current Topic */}
        {progress?.current_topic && (
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-5 h-5 text-blue-500" />
              <h2 className="text-xl font-bold text-gray-900">Şu Anki Konum</h2>
            </div>
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-100">
              <h3 className="font-semibold text-lg text-gray-900 mb-2">
                {progress.current_topic.topic_title}
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Sorular:</span>
                  <span className="font-medium">{progress.current_topic.questions_asked}</span>
                </div>
                {progress.current_topic.average_understanding && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Anlama:</span>
                    <span className="font-medium">
                      {progress.current_topic.average_understanding.toFixed(1)} / 5.0
                    </span>
                  </div>
                )}
                {progress.current_topic.mastery_level && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Seviye:</span>
                    <span className="font-medium capitalize">
                      {progress.current_topic.mastery_level}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Next Recommended Topic */}
        {progress?.next_recommended_topic && (
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <div className="flex items-center gap-2 mb-4">
              <ArrowRight className="w-5 h-5 text-green-500" />
              <h2 className="text-xl font-bold text-gray-900">Sıradaki Konu</h2>
            </div>
            <div className="bg-green-50 rounded-lg p-4 border border-green-100">
              <h3 className="font-semibold text-lg text-gray-900 mb-2">
                {progress.next_recommended_topic.topic_title}
              </h3>
              <p className="text-sm text-gray-600 mb-3">
                Bu konuya geçmeye hazırsın! 🎯
              </p>
              <button className="w-full bg-green-500 hover:bg-green-600 text-white font-medium py-2 px-4 rounded-lg transition-colors">
                Başla
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Learning Patterns */}
      {analytics?.key_patterns && analytics.key_patterns.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-5 h-5 text-purple-500" />
            <h2 className="text-xl font-bold text-gray-900">Öğrenme Desenleri</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {analytics.key_patterns.map((pattern, index) => (
              <div
                key={index}
                className="bg-gray-50 rounded-lg p-4 border border-gray-200"
              >
                <div className="flex items-start gap-2 mb-2">
                  <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-1">
                      {pattern.description}
                    </h4>
                    <p className="text-sm text-gray-600">{pattern.recommendation}</p>
                    <span
                      className={`inline-block mt-2 px-2 py-1 text-xs rounded-full ${
                        pattern.strength === "high"
                          ? "bg-green-100 text-green-700"
                          : "bg-yellow-100 text-yellow-700"
                      }`}
                    >
                      {pattern.strength === "high" ? "Yüksek" : "Orta"}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Topic Progress List */}
      {progress?.progress && progress.progress.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <Target className="w-5 h-5 text-blue-500" />
            <h2 className="text-xl font-bold text-gray-900">Konu İlerlemesi</h2>
          </div>
          <div className="space-y-3">
            {progress.progress.map((topicProgress) => (
              <div
                key={topicProgress.topic_id}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border border-gray-200 hover:border-blue-300 transition-colors"
              >
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <h3 className="font-semibold text-gray-900">
                      {topicProgress.topic_title}
                    </h3>
                    {topicProgress.mastery_level === "mastered" && (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    )}
                  </div>
                  <div className="flex gap-4 mt-2 text-sm text-gray-600">
                    <span>{topicProgress.questions_asked} soru</span>
                    {topicProgress.average_understanding && (
                      <span>
                        Anlama: {topicProgress.average_understanding.toFixed(1)}/5.0
                      </span>
                    )}
                    {topicProgress.time_spent_minutes && (
                      <span>{topicProgress.time_spent_minutes} dk</span>
                    )}
                  </div>
                </div>
                
                {/* Progress Bar */}
                {topicProgress.mastery_score !== null && (
                  <div className="w-32">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all"
                        style={{
                          width: `${Math.min(topicProgress.mastery_score * 100, 100)}%`,
                        }}
                      ></div>
                    </div>
                    <p className="text-xs text-center text-gray-600 mt-1">
                      {Math.round((topicProgress.mastery_score || 0) * 100)}%
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No Data State */}
      {(!progress || progress.progress.length === 0) && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-8 text-center">
          <BookOpen className="w-16 h-16 text-blue-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Henüz soru sormadın
          </h3>
          <p className="text-gray-600 mb-4">
            Öğrenme yolculuğuna başlamak için bir soru sor!
          </p>
          <a
            href={`/sessions/${selectedSession}`}
            className="inline-flex items-center gap-2 bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-6 rounded-lg transition-colors"
          >
            Oturuma Git
            <ArrowRight className="w-4 h-4" />
          </a>
        </div>
      )}
    </div>
  );
}

