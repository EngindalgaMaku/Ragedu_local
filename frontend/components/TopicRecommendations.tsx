"use client";

import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Brain,
  AlertTriangle,
  CheckCircle,
  Target,
  TrendingUp,
  Lightbulb,
  BookOpen,
  Users,
  ArrowRight,
  Star,
  Zap,
  AlertCircle,
} from "lucide-react";

interface RecommendationInsight {
  topic_id: number;
  topic_title: string;
  estimated_difficulty: string;
  topic_strength_level: "critical_weakness" | "moderate_weakness" | "strength";
  engagement_level:
    | "no_engagement"
    | "low_engagement"
    | "medium_engagement"
    | "high_engagement";
  recommended_intervention: string;
  learning_path_suggestion: string;
  student_count: number;
  avg_understanding: number;
  avg_satisfaction: number;
  qa_pairs_count: number;
  kb_quality: number;
}

interface ActionItem {
  immediate_attention: string[];
  monitor_closely: string[];
  leverage_strengths: string[];
}

interface TopicRecommendationsProps {
  sessionId: string;
  data?: any;
}

const TopicRecommendations: React.FC<TopicRecommendationsProps> = ({
  sessionId,
  data,
}) => {
  const [insights, setInsights] = useState<RecommendationInsight[]>([]);
  const [actionItems, setActionItems] = useState<ActionItem | null>(null);
  const [interventions, setInterventions] = useState<{ [key: string]: any[] }>(
    {}
  );
  const [learningPaths, setLearningPaths] = useState<{ [key: string]: any[] }>(
    {}
  );
  const [sessionHealth, setSessionHealth] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<
    "insights" | "interventions" | "paths"
  >("insights");

  // Use provided data or fetch it
  useEffect(() => {
    if (data) {
      setInsights(data.insights || []);
      setActionItems(data.action_items || null);
      setInterventions(data.intervention_strategies || {});
      setLearningPaths(data.learning_path_recommendations || {});
      setSessionHealth(data.session_health || null);
    } else if (sessionId) {
      fetchRecommendations();
    }
  }, [sessionId, data]);

  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `/api/aprag/analytics/topics/recommendation-insights?session_id=${sessionId}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        setInsights(result.insights || []);
        setActionItems(result.action_items || null);
        setInterventions(result.intervention_strategies || {});
        setLearningPaths(result.learning_path_recommendations || {});
        setSessionHealth(result.session_health || null);
      } else {
        throw new Error("Failed to load recommendations");
      }
    } catch (e: any) {
      setError(e.message || "Failed to load recommendations");
      console.error("Recommendations error:", e);
    } finally {
      setLoading(false);
    }
  };

  // Get strength level color and icon
  const getStrengthInfo = (level: string) => {
    switch (level) {
      case "critical_weakness":
        return {
          color: "text-red-600",
          bgColor: "bg-red-50 border-red-200",
          icon: AlertTriangle,
          label: "Kritik Zayıflık",
        };
      case "moderate_weakness":
        return {
          color: "text-amber-600",
          bgColor: "bg-amber-50 border-amber-200",
          icon: AlertCircle,
          label: "Orta Düzey Sorun",
        };
      case "strength":
        return {
          color: "text-green-600",
          bgColor: "bg-green-50 border-green-200",
          icon: CheckCircle,
          label: "Güçlü Alan",
        };
      default:
        return {
          color: "text-gray-600",
          bgColor: "bg-gray-50 border-gray-200",
          icon: Target,
          label: "Belirsiz",
        };
    }
  };

  const getEngagementColor = (level: string) => {
    switch (level) {
      case "high_engagement":
        return "bg-green-100 text-green-800 border-green-200";
      case "medium_engagement":
        return "bg-blue-100 text-blue-800 border-blue-200";
      case "low_engagement":
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "no_engagement":
        return "bg-red-100 text-red-800 border-red-200";
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getInterventionIcon = (intervention: string) => {
    if (intervention.includes("practice")) return BookOpen;
    if (intervention.includes("knowledge")) return Brain;
    if (intervention.includes("accessible")) return Users;
    if (intervention.includes("approach")) return Target;
    if (intervention.includes("advanced")) return Star;
    return Lightbulb;
  };

  const getInterventionPriority = (topics: any[]) => {
    const criticalCount = topics.filter(
      (t) => t.priority === "critical_weakness"
    ).length;
    const moderateCount = topics.filter(
      (t) => t.priority === "moderate_weakness"
    ).length;

    if (criticalCount > 0) return "high";
    if (moderateCount > 0) return "medium";
    return "low";
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>AI Destekli Öneriler</CardTitle>
          <CardDescription>
            Konu bazlı müdahale stratejileri ve öğrenme yolu önerileri
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent"></div>
            <span className="ml-3 text-sm text-muted-foreground">
              Öneriler analiz ediliyor...
            </span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>AI Destekli Öneriler</CardTitle>
          <CardDescription>
            Konu bazlı müdahale stratejileri ve öğrenme yolu önerileri
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Brain className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-sm text-muted-foreground mb-4">{error}</p>
            <Button onClick={fetchRecommendations} variant="outline" size="sm">
              Yeniden Dene
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Session Health & Quick Actions */}
      {sessionHealth && actionItems && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Session Health */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Session Sağlık Durumu
              </CardTitle>
              <CardDescription>Genel durum değerlendirmesi</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-primary mb-2">
                    {sessionHealth.overall_score?.toFixed(1)}/3.0
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Genel Sağlık Skoru
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-lg font-semibold text-red-600">
                      {sessionHealth.critical_topics || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Kritik</div>
                  </div>
                  <div>
                    <div className="text-lg font-semibold text-amber-600">
                      {sessionHealth.moderate_issues || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Orta</div>
                  </div>
                  <div>
                    <div className="text-lg font-semibold text-green-600">
                      {sessionHealth.strong_topics || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Güçlü</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Acil Eylem Planı
              </CardTitle>
              <CardDescription>
                Öncelikli müdahale gerektiren konular
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Immediate Attention */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="h-4 w-4 text-red-600" />
                    <span className="font-medium text-sm">Acil Müdahale</span>
                  </div>
                  <div className="space-y-1">
                    {actionItems.immediate_attention.length > 0 ? (
                      actionItems.immediate_attention
                        .slice(0, 3)
                        .map((topic, index) => (
                          <div
                            key={index}
                            className="text-sm px-2 py-1 bg-red-50 border-l-2 border-red-500 text-red-800"
                          >
                            {topic}
                          </div>
                        ))
                    ) : (
                      <p className="text-sm text-muted-foreground">
                        Acil müdahale gerektiren konu yok
                      </p>
                    )}
                  </div>
                </div>

                {/* Monitor Closely */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <AlertCircle className="h-4 w-4 text-amber-600" />
                    <span className="font-medium text-sm">Yakın Takip</span>
                  </div>
                  <div className="space-y-1">
                    {actionItems.monitor_closely.length > 0 ? (
                      actionItems.monitor_closely
                        .slice(0, 2)
                        .map((topic, index) => (
                          <div
                            key={index}
                            className="text-sm px-2 py-1 bg-amber-50 border-l-2 border-amber-500 text-amber-800"
                          >
                            {topic}
                          </div>
                        ))
                    ) : (
                      <p className="text-sm text-muted-foreground">
                        Takip gerektiren konu yok
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Recommendations */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Detaylı Öneriler ve Stratejiler
              </CardTitle>
              <CardDescription>
                AI destekli analiz sonuçları ve eylem planları
              </CardDescription>
            </div>

            {/* Tab Controls */}
            <div className="flex gap-2">
              <Button
                variant={activeTab === "insights" ? "default" : "outline"}
                size="sm"
                onClick={() => setActiveTab("insights")}
                className="text-xs"
              >
                <Target className="h-3 w-3 mr-1" />
                İçgörüler
              </Button>
              <Button
                variant={activeTab === "interventions" ? "default" : "outline"}
                size="sm"
                onClick={() => setActiveTab("interventions")}
                className="text-xs"
              >
                <Lightbulb className="h-3 w-3 mr-1" />
                Müdahaleler
              </Button>
              <Button
                variant={activeTab === "paths" ? "default" : "outline"}
                size="sm"
                onClick={() => setActiveTab("paths")}
                className="text-xs"
              >
                <ArrowRight className="h-3 w-3 mr-1" />
                Yollar
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {activeTab === "insights" && (
            <div className="space-y-4">
              {insights.length === 0 ? (
                <div className="text-center py-8">
                  <Brain className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Henüz analiz verisi bulunmamaktadır
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {insights.map((insight) => {
                    const strengthInfo = getStrengthInfo(
                      insight.topic_strength_level
                    );
                    const StrengthIcon = strengthInfo.icon;

                    return (
                      <div
                        key={insight.topic_id}
                        className={`p-4 rounded-lg border-2 ${strengthInfo.bgColor}`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <StrengthIcon
                                className={`h-5 w-5 ${strengthInfo.color}`}
                              />
                              <h4 className="font-semibold text-foreground">
                                {insight.topic_title}
                              </h4>
                              <Badge
                                variant="outline"
                                className={getEngagementColor(
                                  insight.engagement_level
                                )}
                              >
                                {insight.engagement_level === "high_engagement"
                                  ? "Yüksek Katılım"
                                  : insight.engagement_level ===
                                    "medium_engagement"
                                  ? "Orta Katılım"
                                  : insight.engagement_level ===
                                    "low_engagement"
                                  ? "Düşük Katılım"
                                  : "Katılım Yok"}
                              </Badge>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                              <div>
                                <span className="font-medium text-muted-foreground">
                                  Durum:{" "}
                                </span>
                                <span className={strengthInfo.color}>
                                  {strengthInfo.label}
                                </span>
                              </div>
                              <div>
                                <span className="font-medium text-muted-foreground">
                                  Öğrenci:{" "}
                                </span>
                                <span>{insight.student_count}</span>
                              </div>
                              <div>
                                <span className="font-medium text-muted-foreground">
                                  Anlama:{" "}
                                </span>
                                <span>
                                  {insight.avg_understanding.toFixed(2)}/5.0
                                </span>
                              </div>
                            </div>

                            {/* Recommendations */}
                            <div className="mt-3 space-y-2">
                              <div className="text-sm">
                                <span className="font-medium text-primary">
                                  Önerilen Müdahale:{" "}
                                </span>
                                <span className="text-foreground">
                                  {insight.recommended_intervention ===
                                  "add_more_practice_questions"
                                    ? "Daha fazla pratik sorusu ekle"
                                    : insight.recommended_intervention ===
                                      "improve_knowledge_base"
                                    ? "Bilgi tabanını geliştir"
                                    : insight.recommended_intervention ===
                                      "make_more_accessible"
                                    ? "Daha erişilebilir hale getir"
                                    : insight.recommended_intervention ===
                                      "review_teaching_approach"
                                    ? "Öğretim yaklaşımını gözden geçir"
                                    : insight.recommended_intervention ===
                                      "add_advanced_content"
                                    ? "İleri düzey içerik ekle"
                                    : "Mevcut yaklaşımı sürdür"}
                                </span>
                              </div>
                              <div className="text-sm">
                                <span className="font-medium text-secondary-foreground">
                                  Öğrenme Yolu:{" "}
                                </span>
                                <span className="text-foreground">
                                  {insight.learning_path_suggestion ===
                                  "ready_for_intermediate"
                                    ? "Orta seviyeye hazır"
                                    : insight.learning_path_suggestion ===
                                      "ready_for_advanced"
                                    ? "İleri seviyeye hazır"
                                    : insight.learning_path_suggestion ===
                                      "needs_prerequisite_review"
                                    ? "Ön koşul gözden geçirmesi gerekli"
                                    : "Mevcut seviyeyi sürdür"}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {activeTab === "interventions" && (
            <div className="space-y-4">
              {Object.keys(interventions).length === 0 ? (
                <div className="text-center py-8">
                  <Lightbulb className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Müdahale önerisi bulunmamaktadır
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {Object.entries(interventions).map(
                    ([intervention, topics]) => {
                      const InterventionIcon =
                        getInterventionIcon(intervention);
                      const priority = getInterventionPriority(topics);
                      const priorityColor =
                        priority === "high"
                          ? "text-red-600"
                          : priority === "medium"
                          ? "text-amber-600"
                          : "text-green-600";

                      return (
                        <Card
                          key={intervention}
                          className="border-l-4 border-l-primary"
                        >
                          <CardHeader className="pb-3">
                            <CardTitle className="text-base flex items-center gap-2">
                              <InterventionIcon className="h-4 w-4" />
                              {intervention === "add_more_practice_questions"
                                ? "Daha Fazla Pratik Sorusu Ekle"
                                : intervention === "improve_knowledge_base"
                                ? "Bilgi Tabanını Geliştir"
                                : intervention === "make_more_accessible"
                                ? "Daha Erişilebilir Hale Getir"
                                : intervention === "review_teaching_approach"
                                ? "Öğretim Yaklaşımını Gözden Geçir"
                                : intervention === "add_advanced_content"
                                ? "İleri Düzey İçerik Ekle"
                                : "Diğer Müdahale"}
                              <Badge
                                variant="outline"
                                className={`ml-auto ${priorityColor}`}
                              >
                                {priority === "high"
                                  ? "Yüksek Öncelik"
                                  : priority === "medium"
                                  ? "Orta Öncelik"
                                  : "Düşük Öncelik"}
                              </Badge>
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-2">
                              {topics.map((topic, index) => (
                                <div
                                  key={index}
                                  className="flex items-center justify-between p-2 bg-muted/30 rounded"
                                >
                                  <span className="font-medium text-sm">
                                    {topic.topic_title}
                                  </span>
                                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                    <span>{topic.student_count} öğrenci</span>
                                    <span>•</span>
                                    <span>
                                      {topic.understanding_score.toFixed(1)}{" "}
                                      anlama
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      );
                    }
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === "paths" && (
            <div className="space-y-4">
              {Object.keys(learningPaths).length === 0 ? (
                <div className="text-center py-8">
                  <ArrowRight className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Öğrenme yolu önerisi bulunmamaktadır
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {Object.entries(learningPaths).map(([path, topics]) => (
                    <Card key={path} className="border-l-4 border-l-secondary">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-base flex items-center gap-2">
                          <TrendingUp className="h-4 w-4" />
                          {path === "ready_for_intermediate"
                            ? "Orta Seviyeye Hazır"
                            : path === "ready_for_advanced"
                            ? "İleri Seviyeye Hazır"
                            : path === "needs_prerequisite_review"
                            ? "Ön Koşul Gözden Geçirmesi"
                            : "Mevcut Seviyeyi Sürdür"}
                          <Badge variant="secondary" className="ml-auto">
                            {topics.length} konu
                          </Badge>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          {topics.map((topic, index) => (
                            <div
                              key={index}
                              className="flex items-center justify-between p-2 bg-muted/30 rounded"
                            >
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-sm">
                                  {topic.topic_title}
                                </span>
                                <Badge variant="outline">
                                  {topic.current_difficulty === "beginner"
                                    ? "Başlangıç"
                                    : topic.current_difficulty ===
                                      "intermediate"
                                    ? "Orta"
                                    : "İleri"}
                                </Badge>
                              </div>
                              <div className="text-xs text-muted-foreground">
                                {topic.understanding_score.toFixed(1)} anlama
                                skoru
                              </div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default TopicRecommendations;
