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
  TopicProgress,
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
  Eye,
  Info,
} from "lucide-react";
import TopicDetailModal from "@/components/TopicDetailModal";

interface StudentDashboardProps {
  userId: string;
}

export default function StudentDashboard({ userId }: StudentDashboardProps) {
  const { user } = useAuth();
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<APRAGAnalyticsSummary | null>(
    null
  );
  const [progress, setProgress] = useState<StudentProgressResponse | null>(
    null
  );
  const [topics, setTopics] = useState<Topic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [apragSettings, setApragSettings] = useState<APRAGSettings | null>(
    null
  );
  const [topicPage, setTopicPage] = useState(1);
  const TOPICS_PER_PAGE = 8;

  // Topic detail modal states
  const [selectedTopicForDetail, setSelectedTopicForDetail] =
    useState<Topic | null>(null);
  const [selectedTopicProgress, setSelectedTopicProgress] =
    useState<TopicProgress | null>(null);
  const [isTopicDetailModalOpen, setIsTopicDetailModalOpen] = useState(false);

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

  // Reset topic page when session changes
  useEffect(() => {
    setTopicPage(1);
  }, [selectedSession]);

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
            <svg
              className="w-10 h-10 text-yellow-600"
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
        return {
          icon: TrendingUp,
          color: "text-green-500",
          label: "Gelişiyor",
        };
      case "stable":
        return { icon: Target, color: "text-blue-500", label: "Sabit" };
      case "declining":
        return {
          icon: TrendingUp,
          color: "text-red-500 rotate-180",
          label: "Azalıyor",
        };
      default:
        return { icon: Target, color: "text-gray-500", label: "Yetersiz Veri" };
    }
  };

  const trendInfo = analytics
    ? getTrendInfo(analytics.improvement_trend)
    : null;
  const TrendIcon = trendInfo?.icon;

  // Build topic hierarchy for progress (ana konu + alt konular)
  const topicProgressList = progress?.progress || [];
  const progressMap = new Map<number, TopicProgress>();
  topicProgressList.forEach((tp) => {
    progressMap.set(tp.topic_id, tp);
  });

  const mainTopicsMeta = topics.filter((t) => !t.parent_topic_id);
  const subtopicsMap = new Map<number, Topic[]>();
  topics
    .filter((t) => t.parent_topic_id)
    .forEach((t) => {
      const parentId = t.parent_topic_id as number;
      if (!subtopicsMap.has(parentId)) {
        subtopicsMap.set(parentId, []);
      }
      subtopicsMap.get(parentId)!.push(t);
    });

  const mainTopicsWithProgress = mainTopicsMeta.map((mt) => ({
    mainTopic: mt,
    mainProgress: progressMap.get(mt.topic_id) || null,
    subtopics:
      subtopicsMap.get(mt.topic_id)?.map((st) => ({
        topic: st,
        progress: progressMap.get(st.topic_id) || null,
      })) || [],
  }));

  const totalTopicPages = Math.max(
    1,
    Math.ceil(mainTopicsWithProgress.length / TOPICS_PER_PAGE)
  );
  const paginatedMainTopics = mainTopicsWithProgress.slice(
    (topicPage - 1) * TOPICS_PER_PAGE,
    topicPage * TOPICS_PER_PAGE
  );

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

          <div className="flex flex-col items-end gap-2">
            {/* Session Selector */}
            {sessions.length > 0 && (
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-gray-700">
                  Oturum:
                </label>
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
            <div className="flex flex-wrap gap-2">
              <a
                href="/student/chat"
                className="inline-flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors"
              >
                Soru Sor
              </a>
              <a
                href="/student/history"
                className="inline-flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-md border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors"
              >
                Geçmiş Sorularım
              </a>
            </div>
          </div>
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
            {progress?.progress.filter((p) => p.mastery_level === "mastered")
              .length || 0}{" "}
            / {progress?.progress.length || 0} konu tamamlandı
          </p>
        </div>

        {/* Engagement Level */}
        <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg shadow-lg p-6 text-white">
          <div className="flex items-center justify-between mb-3">
            {TrendIcon && (
              <TrendIcon className={`w-8 h-8 opacity-80 ${trendInfo?.color}`} />
            )}
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
        {progress?.current_topic ? (
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
                  <span className="font-medium">
                    {progress.current_topic.questions_asked || 0}
                  </span>
                </div>
                {progress.current_topic.average_understanding && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Anlama:</span>
                    <span className="font-medium">
                      {progress.current_topic.average_understanding.toFixed(1)}{" "}
                      / 5.0
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
        ) : progress?.progress && progress.progress.length > 0 ? (
          // Show first topic if no current topic is set
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-5 h-5 text-blue-500" />
              <h2 className="text-xl font-bold text-gray-900">Şu Anki Konum</h2>
            </div>
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-100">
              <h3 className="font-semibold text-lg text-gray-900 mb-2">
                {progress.progress[0].topic_title}
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Sorular:</span>
                  <span className="font-medium">
                    {progress.progress[0].questions_asked || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Seviye:</span>
                  <span className="font-medium capitalize">
                    {progress.progress[0].mastery_level || "not_started"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ) : null}

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
            <h2 className="text-xl font-bold text-gray-900">
              Öğrenme Desenleri
            </h2>
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
                    <p className="text-sm text-gray-600">
                      {pattern.recommendation}
                    </p>
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

      {/* Topic Progress List (Ana konu + alt konular hiyerarşik) */}
      {mainTopicsWithProgress.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <div className="flex items-center justify-between gap-2 mb-4">
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5 text-blue-500" />
              <h2 className="text-xl font-bold text-gray-900">
                Konu İlerlemesi
              </h2>
            </div>
            {totalTopicPages > 1 && (
              <div className="flex items-center gap-2 text-xs text-gray-600">
                <span>
                  Sayfa {topicPage} / {totalTopicPages}
                </span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setTopicPage((p) => Math.max(1, p - 1))}
                    disabled={topicPage === 1}
                    className="px-2 py-1 rounded border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Önceki
                  </button>
                  <button
                    onClick={() =>
                      setTopicPage((p) => Math.min(totalTopicPages, p + 1))
                    }
                    disabled={topicPage >= totalTopicPages}
                    className="px-2 py-1 rounded border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Sonraki
                  </button>
                </div>
              </div>
            )}
          </div>
          <div className="space-y-3">
            {paginatedMainTopics.map(
              ({ mainTopic, mainProgress, subtopics }) => (
                <div
                  key={mainTopic.topic_id}
                  className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border border-gray-200 hover:border-blue-300 hover:shadow-md transition-all cursor-pointer group"
                  onClick={() => {
                    setSelectedTopicForDetail(mainTopic);
                    setSelectedTopicProgress(mainProgress || null);
                    setIsTopicDetailModalOpen(true);
                  }}
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <h3 className="font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
                        {mainTopic.topic_title}
                      </h3>
                      {mainProgress?.mastery_level === "mastered" && (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      )}
                      <Eye className="w-4 h-4 text-gray-400 group-hover:text-blue-500 transition-colors" />
                    </div>

                    {/* Short description or keywords preview */}
                    {(mainTopic.description ||
                      (mainTopic.keywords &&
                        mainTopic.keywords.length > 0)) && (
                      <div className="mt-1 text-xs text-gray-500">
                        {mainTopic.description
                          ? mainTopic.description.length > 80
                            ? `${mainTopic.description.substring(0, 80)}...`
                            : mainTopic.description
                          : mainTopic.keywords?.slice(0, 3).join(", ")}
                      </div>
                    )}

                    <div className="flex gap-4 mt-2 text-sm text-gray-600">
                      <span className="flex items-center gap-1">
                        <BookOpen className="w-3 h-3" />
                        {mainProgress?.questions_asked ?? 0} soru
                      </span>
                      {mainProgress?.average_understanding && (
                        <span className="flex items-center gap-1">
                          <Brain className="w-3 h-3" />
                          Anlama:{" "}
                          {mainProgress.average_understanding.toFixed(1)}/5.0
                        </span>
                      )}
                      {mainProgress?.time_spent_minutes && (
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {mainProgress.time_spent_minutes} dk
                        </span>
                      )}
                      {mainTopic.keywords && mainTopic.keywords.length > 0 && (
                        <span className="flex items-center gap-1 text-purple-600">
                          <span className="w-2 h-2 bg-purple-400 rounded-full"></span>
                          {mainTopic.keywords.length} keyword
                        </span>
                      )}
                    </div>

                    {/* Alt konular */}
                    {subtopics.length > 0 && (
                      <div className="mt-3 pl-3 border-l border-gray-300 space-y-1">
                        <div className="text-xs text-gray-500 mb-1 font-medium">
                          Alt Konular:
                        </div>
                        {subtopics.slice(0, 3).map(({ topic, progress }) => (
                          <div
                            key={topic.topic_id}
                            className="flex items-center justify-between text-xs text-gray-700"
                          >
                            <span className="truncate flex-1">
                              • {topic.topic_title}
                            </span>
                            <div className="flex items-center gap-2 ml-2">
                              <span className="text-gray-500">
                                {progress?.questions_asked ?? 0} soru
                              </span>
                              {progress?.mastery_score !== null &&
                                progress?.mastery_score !== undefined && (
                                  <div className="w-12">
                                    <div className="w-full bg-gray-200 rounded-full h-1">
                                      <div
                                        className="bg-blue-400 h-1 rounded-full"
                                        style={{
                                          width: `${Math.min(
                                            progress.mastery_score * 100,
                                            100
                                          )}%`,
                                        }}
                                      ></div>
                                    </div>
                                  </div>
                                )}
                            </div>
                          </div>
                        ))}
                        {subtopics.length > 3 && (
                          <div className="text-xs text-gray-500 italic">
                            +{subtopics.length - 3} alt konu daha...
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Progress Bar and Detail Button */}
                  <div className="flex items-center gap-3">
                    {mainProgress?.mastery_score !== null &&
                      mainProgress?.mastery_score !== undefined && (
                        <div className="w-24 text-right">
                          <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
                            <div
                              className="bg-blue-500 h-2 rounded-full transition-all"
                              style={{
                                width: `${Math.min(
                                  mainProgress.mastery_score * 100,
                                  100
                                )}%`,
                              }}
                            ></div>
                          </div>
                          <p className="text-xs text-gray-600">
                            {Math.round(
                              (mainProgress.mastery_score || 0) * 100
                            )}
                            % tamamlandı
                          </p>
                        </div>
                      )}

                    <button className="p-2 bg-white rounded-full shadow-sm border border-gray-200 hover:bg-blue-50 hover:border-blue-300 transition-all group-hover:scale-105">
                      <Info className="w-4 h-4 text-gray-500 group-hover:text-blue-500" />
                    </button>
                  </div>
                </div>
              )
            )}
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
            href="/student/chat"
            className="inline-flex items-center gap-2 bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-6 rounded-lg transition-colors"
          >
            Soru Sor
            <ArrowRight className="w-4 h-4" />
          </a>
        </div>
      )}

      {/* Topic Detail Modal */}
      <TopicDetailModal
        isOpen={isTopicDetailModalOpen}
        onClose={() => {
          setIsTopicDetailModalOpen(false);
          setSelectedTopicForDetail(null);
          setSelectedTopicProgress(null);
        }}
        topic={selectedTopicForDetail}
        progress={selectedTopicProgress}
        sessionId={selectedSession || ""}
        userId={userId}
      />
    </div>
  );
}
