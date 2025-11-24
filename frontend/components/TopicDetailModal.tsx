"use client";

import React, { useState, useEffect } from "react";
import {
  Topic,
  TopicProgress,
  getSessionTopics,
  getStudentProgress,
} from "@/lib/api";
import {
  X,
  BookOpen,
  Tag,
  Clock,
  Target,
  CheckCircle,
  Star,
  Users,
  BarChart3,
  FileText,
  Brain,
  TrendingUp,
  AlertCircle,
  Award,
  Database,
  Eye,
  List,
  Lightbulb,
} from "lucide-react";

interface TopicDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  topic: Topic | null;
  progress: TopicProgress | null;
  sessionId: string;
  userId: string;
}

export default function TopicDetailModal({
  isOpen,
  onClose,
  topic,
  progress,
  sessionId,
  userId,
}: TopicDetailModalProps) {
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<
    "overview" | "summary" | "content" | "progress"
  >("overview");
  const [knowledgeStats, setKnowledgeStats] = useState<{
    qa_pairs: number;
    chunks_linked: number;
    difficulty_breakdown: Record<string, number>;
  } | null>(null);
  const [topicSummary, setTopicSummary] = useState<string | null>(null);
  const [sampleContent, setSampleContent] = useState<{
    chunks: Array<{ content: string; score: number; source?: string }>;
    qa_pairs: Array<{ question: string; answer: string }>;
  } | null>(null);

  useEffect(() => {
    const loadTopicData = async () => {
      if (!topic || !isOpen) return;

      setLoading(true);
      try {
        // Bu endpoint henüz yok, simulated data kullanacağız
        // Gerçek implementasyonda APRAG service'den knowledge stats alınacak
        const mockStats = {
          qa_pairs: Math.floor(Math.random() * 50) + 10,
          chunks_linked: Math.floor(Math.random() * 25) + 5,
          difficulty_breakdown: {
            basic: Math.floor(Math.random() * 15) + 5,
            intermediate: Math.floor(Math.random() * 20) + 8,
            advanced: Math.floor(Math.random() * 10) + 2,
          },
        };
        setKnowledgeStats(mockStats);

        // Mock topic summary
        const mockSummary = `Bu konu ${topic.topic_title.toLowerCase()} ile ilgili temel kavramları kapsar. Öğrenciler bu konuda ${
          topic.keywords?.join(", ") || "temel kavramları"
        } öğrenecekler. Konu, ${
          topic.estimated_difficulty || "orta"
        } seviyede olup, öğrencilerin bu alandaki temel bilgilerini geliştirmelerine yardımcı olur.

Bu konuyu öğrenmek için öğrencilerin önceki konulardaki temel bilgilere sahip olmaları önerilir. Konu boyunca praktik örnekler ve interaktif sorularla öğrenme desteklenir.

Öğrenim süreci boyunca öğrenciler:
• Temel kavramları anlayacak
• Pratik uygulamalar yapacak
• Diğer konularla bağlantı kuracak
• Kendi öğrenme hızında ilerleyecek`;
        setTopicSummary(mockSummary);

        // Mock content samples
        const mockContent = {
          chunks: [
            {
              content: `${topic.topic_title} konusunda temel bilgiler: Bu bölümde konunun temelleri açıklanmaktadır. Öğrenciler bu bilgileri kullanarak daha ileri seviye konuları öğrenebilirler. Konu, sistemli bir yaklaşımla ele alınmış olup, öğrencilerin kolayca anlayabileceği şekilde sunulmuştur.`,
              score: 0.95,
              source: "Ders Kitabı Bölüm 1",
            },
            {
              content: `${topic.topic_title} ile ilgili önemli noktalar ve kavramlar burada detaylandırılmıştır. Bu bilgiler sınav ve ödevlerde kullanılacaktır. Konunun pratik uygulamaları da dahil olmak üzere gerçek hayat örnekleri ile zenginleştirilmiştir.`,
              score: 0.88,
              source: "Ders Notları",
            },
            {
              content: `Praktik uygulamalar ve örnekler ile ${topic.topic_title} konusunun pekiştirilmesi için çeşitli alıştırmalar sunulmaktadır. Bu alıştırmalar öğrencilerin konuyu daha iyi anlamalarına yardımcı olur.`,
              score: 0.82,
              source: "Alıştırma Kitabı",
            },
          ],
          qa_pairs: [
            {
              question: `${topic.topic_title} nedir ve neden önemlidir?`,
              answer: `${topic.topic_title}, ${
                topic.description ||
                "önemli bir konudur ve öğrencilerin bu alandaki temel bilgilerini geliştirmelerine yardımcı olur."
              } Bu konu, öğrencilerin akademik gelişimi açısından kritik önem taşır.`,
            },
            {
              question: `${topic.topic_title} konusunun temel özellikleri nelerdir?`,
              answer:
                "Bu konu, öğrencilerin temel bilgilerini geliştirmek ve daha ileri seviye konulara hazırlık yapmak için kritik öneme sahiptir. Sistematik bir yaklaşımla öğretilir ve pratik uygulamalarla desteklenir.",
            },
            {
              question: `${topic.topic_title} konusunu öğrenmek için hangi ön bilgiler gereklidir?`,
              answer:
                "Bu konuyu etkili şekilde öğrenebilmek için temel matematik ve fen bilgisi yeterlidir. Önceki konulardaki temel kavramların bilinmesi öğrenme sürecini hızlandırır.",
            },
          ],
        };
        setSampleContent(mockContent);
      } catch (error) {
        console.error("Failed to load topic data:", error);
      } finally {
        setLoading(false);
      }
    };

    loadTopicData();
  }, [topic, isOpen]);

  // Reset active tab when modal opens
  useEffect(() => {
    if (isOpen) {
      setActiveTab("overview");
    }
  }, [isOpen]);

  if (!isOpen || !topic) return null;

  const getDifficultyColor = (difficulty: string | null) => {
    switch (difficulty?.toLowerCase()) {
      case "basic":
      case "kolay":
        return "bg-green-100 text-green-700 border-green-200";
      case "intermediate":
      case "orta":
        return "bg-yellow-100 text-yellow-700 border-yellow-200";
      case "advanced":
      case "zor":
        return "bg-red-100 text-red-700 border-red-200";
      default:
        return "bg-gray-100 text-gray-700 border-gray-200";
    }
  };

  const getMasteryColor = (masteryLevel: string | null) => {
    switch (masteryLevel?.toLowerCase()) {
      case "mastered":
        return "bg-green-500";
      case "learning":
        return "bg-yellow-500";
      case "introduced":
        return "bg-blue-500";
      default:
        return "bg-gray-400";
    }
  };

  const getMasteryLabel = (masteryLevel: string | null) => {
    switch (masteryLevel?.toLowerCase()) {
      case "mastered":
        return "Uzmanlaştı";
      case "learning":
        return "Öğreniyor";
      case "introduced":
        return "Tanıtıldı";
      default:
        return "Henüz başlanmadı";
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-xl max-w-5xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-6 relative">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-1 hover:bg-white/20 rounded-full transition-colors"
          >
            <X className="w-6 h-6" />
          </button>

          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center">
              <BookOpen className="w-6 h-6" />
            </div>

            <div>
              <h2 className="text-2xl font-bold mb-2">{topic.topic_title}</h2>
              <div className="flex items-center gap-4 text-sm opacity-90">
                <span className="flex items-center gap-1">
                  <Target className="w-4 h-4" />
                  Sıra: {topic.topic_order}
                </span>
                {topic.extraction_confidence && (
                  <span className="flex items-center gap-1">
                    <Star className="w-4 h-4" />
                    Güven: {(topic.extraction_confidence * 100).toFixed(0)}%
                  </span>
                )}
                <span
                  className={`px-2 py-1 rounded-full text-xs border ${getDifficultyColor(
                    topic.estimated_difficulty
                  )}`}
                >
                  {topic.estimated_difficulty || "Belirlenmemiş"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="border-b border-gray-200 px-6">
          <nav className="flex space-x-8">
            {[
              { key: "overview", label: "Genel Bakış", icon: Eye },
              { key: "summary", label: "Konu Özeti", icon: FileText },
              { key: "content", label: "KB İçerikleri", icon: Database },
              { key: "progress", label: "İlerleme Detayı", icon: BarChart3 },
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key as any)}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === key
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                <div className="flex items-center gap-2">
                  <Icon className="w-4 h-4" />
                  {label}
                </div>
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {/* Overview Tab */}
          {activeTab === "overview" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Column - Topic Details */}
              <div className="space-y-6">
                {/* Description */}
                {topic.description && (
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h3 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
                      <FileText className="w-4 h-4 text-blue-500" />
                      Açıklama
                    </h3>
                    <p className="text-gray-700 text-sm leading-relaxed">
                      {topic.description}
                    </p>
                  </div>
                )}

                {/* Keywords */}
                {topic.keywords && topic.keywords.length > 0 && (
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <Tag className="w-4 h-4 text-purple-500" />
                      Anahtar Kelimeler
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {topic.keywords.map((keyword, index) => (
                        <span
                          key={index}
                          className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium border border-purple-200"
                        >
                          {keyword}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Prerequisites */}
                {topic.prerequisites && topic.prerequisites.length > 0 && (
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-orange-500" />
                      Ön Koşullar
                    </h3>
                    <div className="space-y-2">
                      {topic.prerequisites.map((prereq, index) => (
                        <div
                          key={index}
                          className="flex items-center gap-2 p-2 bg-orange-50 rounded-lg border border-orange-200"
                        >
                          <CheckCircle className="w-4 h-4 text-orange-500" />
                          <span className="text-sm text-orange-800">
                            Konu ID: {prereq}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Knowledge Base Stats */}
                {knowledgeStats && (
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <Brain className="w-4 h-4 text-indigo-500" />
                      Bilgi Tabanı İstatistikleri
                    </h3>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-3 bg-indigo-50 rounded-lg border border-indigo-200">
                        <div className="flex items-center gap-2">
                          <Users className="w-4 h-4 text-indigo-500" />
                          <span className="text-sm font-medium text-indigo-800">
                            Soru-Cevap Çiftleri
                          </span>
                        </div>
                        <span className="text-lg font-bold text-indigo-600">
                          {knowledgeStats.qa_pairs}
                        </span>
                      </div>

                      <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-blue-500" />
                          <span className="text-sm font-medium text-blue-800">
                            Bağlantılı İçerikler
                          </span>
                        </div>
                        <span className="text-lg font-bold text-blue-600">
                          {knowledgeStats.chunks_linked}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Right Column - Progress Overview */}
              <div className="space-y-6">
                {/* Quick Progress */}
                {progress && (
                  <div className="bg-gradient-to-br from-green-50 to-blue-50 rounded-lg p-4 border border-green-200">
                    <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-green-500" />
                      Hızlı Bakış
                    </h3>

                    <div className="grid grid-cols-2 gap-3">
                      <div className="text-center p-3 bg-white rounded-lg border">
                        <div className="text-2xl font-bold text-blue-600 mb-1">
                          {progress.questions_asked}
                        </div>
                        <div className="text-xs text-gray-600">Sorular</div>
                      </div>

                      <div className="text-center p-3 bg-white rounded-lg border">
                        <div className="text-2xl font-bold text-purple-600 mb-1">
                          {progress.average_understanding?.toFixed(1) || "0.0"}
                        </div>
                        <div className="text-xs text-gray-600">Anlama /5</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Topic Status */}
                <div className="bg-white rounded-lg border p-4">
                  <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <Award className="w-4 h-4 text-amber-500" />
                    Konu Durumu
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Aktif Durum</span>
                      <span
                        className={`px-2 py-1 rounded-full text-xs font-medium ${
                          topic.is_active
                            ? "bg-green-100 text-green-700"
                            : "bg-red-100 text-red-700"
                        }`}
                      >
                        {topic.is_active ? "Aktif" : "Pasif"}
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Konu Sırası</span>
                      <span className="font-medium text-gray-900">
                        #{topic.topic_order}
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">
                        Çıkarılma Güveni
                      </span>
                      <span className="font-medium text-gray-900">
                        {topic.extraction_confidence
                          ? `${(topic.extraction_confidence * 100).toFixed(1)}%`
                          : "N/A"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Summary Tab */}
          {activeTab === "summary" && (
            <div className="max-w-4xl">
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 mb-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                  <Lightbulb className="w-6 h-6 text-yellow-500" />
                  Konu Özeti: {topic?.topic_title}
                </h3>

                {loading ? (
                  <div className="animate-pulse space-y-3">
                    <div className="h-4 bg-gray-200 rounded w-full"></div>
                    <div className="h-4 bg-gray-200 rounded w-5/6"></div>
                    <div className="h-4 bg-gray-200 rounded w-4/6"></div>
                  </div>
                ) : (
                  <div className="prose prose-sm max-w-none">
                    <p className="text-gray-700 leading-relaxed whitespace-pre-line">
                      {topicSummary}
                    </p>
                  </div>
                )}
              </div>

              {/* Learning Objectives */}
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h4 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <Target className="w-5 h-5 text-green-500" />
                  Öğrenme Hedefleri
                </h4>
                <ul className="space-y-2">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-gray-700">
                      {topic?.topic_title} konusunun temel kavramlarını anlama
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-gray-700">
                      Konuyla ilgili pratik uygulamaları yapabilme
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-gray-700">
                      İlgili konularla bağlantı kurabilme
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-gray-700">
                      Konuyu günlük hayata uygulayabilme
                    </span>
                  </li>
                </ul>
              </div>
            </div>
          )}

          {/* Content Tab */}
          {activeTab === "content" && (
            <div className="space-y-6">
              {/* Sample Chunks */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <FileText className="w-5 h-5 text-blue-500" />
                  Örnek İçerikler
                </h3>

                {loading ? (
                  <div className="space-y-4">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="animate-pulse">
                        <div className="h-20 bg-gray-200 rounded-lg"></div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="space-y-4">
                    {sampleContent?.chunks.map((chunk, index) => (
                      <div
                        key={index}
                        className="bg-gray-50 border border-gray-200 rounded-lg p-4"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs text-gray-500 font-medium">
                            {chunk.source}
                          </span>
                          <span className="text-xs text-green-600 font-medium">
                            Relevans: {(chunk.score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-sm text-gray-700 leading-relaxed">
                          {chunk.content}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Sample Q&A Pairs */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <Users className="w-5 h-5 text-purple-500" />
                  Örnek Soru-Cevaplar
                </h3>

                {loading ? (
                  <div className="space-y-4">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="animate-pulse">
                        <div className="h-24 bg-gray-200 rounded-lg"></div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="space-y-4">
                    {sampleContent?.qa_pairs.map((qa, index) => (
                      <div
                        key={index}
                        className="bg-purple-50 border border-purple-200 rounded-lg p-4"
                      >
                        <div className="mb-3">
                          <span className="text-xs font-medium text-purple-600 uppercase tracking-wide">
                            Soru
                          </span>
                          <p className="text-sm font-medium text-gray-900 mt-1">
                            {qa.question}
                          </p>
                        </div>
                        <div>
                          <span className="text-xs font-medium text-purple-600 uppercase tracking-wide">
                            Cevap
                          </span>
                          <p className="text-sm text-gray-700 mt-1 leading-relaxed">
                            {qa.answer}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Progress Tab */}
          {activeTab === "progress" && (
            <div className="space-y-6">
              {progress ? (
                <>
                  {/* Detailed Progress Stats */}
                  <div className="bg-gradient-to-br from-green-50 to-blue-50 rounded-lg p-6">
                    <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                      <BarChart3 className="w-6 h-6 text-green-500" />
                      Detaylı İlerleme Analizi
                    </h3>

                    {/* Progress Grid */}
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                      <div className="text-center p-4 bg-white rounded-lg border">
                        <div className="text-3xl font-bold text-blue-600 mb-2">
                          {progress.questions_asked}
                        </div>
                        <div className="text-sm text-gray-600">Soru Sayısı</div>
                      </div>

                      <div className="text-center p-4 bg-white rounded-lg border">
                        <div className="text-3xl font-bold text-purple-600 mb-2">
                          {progress.average_understanding?.toFixed(1) || "0.0"}
                        </div>
                        <div className="text-sm text-gray-600">Anlama /5</div>
                      </div>

                      <div className="text-center p-4 bg-white rounded-lg border">
                        <div className="text-3xl font-bold text-green-600 mb-2">
                          {progress.time_spent_minutes || 0}
                        </div>
                        <div className="text-sm text-gray-600">Dakika</div>
                      </div>

                      <div className="text-center p-4 bg-white rounded-lg border">
                        <div className="text-3xl font-bold text-orange-600 mb-2">
                          {progress.readiness_score
                            ? (progress.readiness_score * 100).toFixed(0)
                            : 0}
                          %
                        </div>
                        <div className="text-sm text-gray-600">Hazırlık</div>
                      </div>
                    </div>

                    {/* Mastery Progress */}
                    <div className="mb-6">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-semibold text-gray-900">
                          Uzmanlaşma İlerlemesi
                        </h4>
                        <span
                          className={`px-3 py-1 rounded-full text-sm font-medium ${getMasteryColor(
                            progress.mastery_level
                          )} text-white`}
                        >
                          {getMasteryLabel(progress.mastery_level)}
                        </span>
                      </div>

                      {progress.mastery_score !== null && (
                        <div>
                          <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
                            <div
                              className={`h-3 rounded-full transition-all ${getMasteryColor(
                                progress.mastery_level
                              )}`}
                              style={{
                                width: `${Math.min(
                                  progress.mastery_score * 100,
                                  100
                                )}%`,
                              }}
                            />
                          </div>
                          <div className="text-right text-sm text-gray-600">
                            {Math.min(
                              progress.mastery_score * 100,
                              100
                            ).toFixed(1)}
                            % tamamlandı
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Learning Timeline */}
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">
                        Öğrenme Süreci
                      </h4>
                      <div className="space-y-3">
                        <div className="flex items-center gap-3">
                          <div
                            className={`w-3 h-3 rounded-full ${
                              progress.questions_asked > 0
                                ? "bg-green-500"
                                : "bg-gray-300"
                            }`}
                          ></div>
                          <span className="text-sm text-gray-700">
                            Konuyu keşfetmeye başladı
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div
                            className={`w-3 h-3 rounded-full ${
                              progress.questions_asked >= 3
                                ? "bg-green-500"
                                : "bg-gray-300"
                            }`}
                          ></div>
                          <span className="text-sm text-gray-700">
                            Aktif öğrenme aşamasında
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div
                            className={`w-3 h-3 rounded-full ${
                              progress.mastery_level === "mastered"
                                ? "bg-green-500"
                                : "bg-gray-300"
                            }`}
                          ></div>
                          <span className="text-sm text-gray-700">
                            Konuda uzmanlaştı
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="bg-gray-50 rounded-lg p-8 text-center">
                  <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <h4 className="font-medium text-gray-900 mb-2">
                    Henüz veri yok
                  </h4>
                  <p className="text-sm text-gray-600 mb-4">
                    Bu konuyla ilgili henüz herhangi bir aktivite
                    gerçekleştirmedin
                  </p>
                  <button className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                    İlk Sorununu Sor
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 border-t flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-100 transition-colors"
          >
            Kapat
          </button>
          <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
            Bu Konuyla İlgili Soru Sor
          </button>
        </div>
      </div>
    </div>
  );
}
