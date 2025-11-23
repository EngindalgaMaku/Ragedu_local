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
  BarChart3,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Target,
  Clock,
  Users,
  Brain,
  Star,
  ArrowUp,
  ArrowDown,
  Minus,
} from "lucide-react";

interface DifficultyData {
  topic_id: number;
  topic_title: string;
  estimated_difficulty: "beginner" | "intermediate" | "advanced";
  total_questions: number;
  avg_question_complexity: number;
  students_attempted: number;
  avg_understanding_score: number;
  avg_satisfaction_score: number;
  success_rate: number;
  avg_time_spent: number;
  kb_quality_score: number;
  available_qa_pairs: number;
  difficulty_recommendation: string;
}

interface TopicDifficultyAnalysisProps {
  sessionId: string;
  data?: any;
  compact?: boolean;
}

const TopicDifficultyAnalysis: React.FC<TopicDifficultyAnalysisProps> = ({
  sessionId,
  data,
  compact = false,
}) => {
  const [difficultyData, setDifficultyData] = useState<DifficultyData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<
    "understanding" | "success_rate" | "time"
  >("understanding");
  const [filterBy, setFilterBy] = useState<
    "all" | "needs_adjustment" | "appropriate"
  >("all");

  // Use provided data or fetch it
  useEffect(() => {
    if (data?.difficulty_data) {
      setDifficultyData(data.difficulty_data);
    } else if (sessionId) {
      fetchDifficultyData();
    }
  }, [sessionId, data]);

  const fetchDifficultyData = async () => {
    try {
      setLoading(true);
      setError(null);

      const url = sessionId
        ? `/api/aprag/analytics/topics/difficulty-analysis?session_id=${sessionId}`
        : `/api/aprag/analytics/topics/difficulty-analysis`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        setDifficultyData(result.difficulty_data || []);
      } else {
        throw new Error("Failed to load difficulty analysis");
      }
    } catch (e: any) {
      setError(e.message || "Failed to load difficulty analysis");
      console.error("Difficulty analysis error:", e);
    } finally {
      setLoading(false);
    }
  };

  // Get difficulty color and icon
  const getDifficultyInfo = (difficulty: string) => {
    switch (difficulty) {
      case "beginner":
        return {
          color: "text-green-600",
          bgColor: "bg-green-50 border-green-200",
          icon: Target,
          label: "Başlangıç",
        };
      case "intermediate":
        return {
          color: "text-yellow-600",
          bgColor: "bg-yellow-50 border-yellow-200",
          icon: BarChart3,
          label: "Orta",
        };
      case "advanced":
        return {
          color: "text-red-600",
          bgColor: "bg-red-50 border-red-200",
          icon: Brain,
          label: "İleri",
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

  const getRecommendationInfo = (recommendation: string) => {
    switch (recommendation) {
      case "needs_simplification":
        return {
          color: "text-red-600",
          icon: ArrowDown,
          label: "Basitleştir",
          priority: "high",
        };
      case "can_increase_difficulty":
        return {
          color: "text-blue-600",
          icon: ArrowUp,
          label: "Zorlaştır",
          priority: "medium",
        };
      case "needs_support_materials":
        return {
          color: "text-amber-600",
          icon: Star,
          label: "Destek Materyali",
          priority: "medium",
        };
      case "insufficient_data":
        return {
          color: "text-gray-600",
          icon: Clock,
          label: "Yetersiz Veri",
          priority: "low",
        };
      case "appropriate_level":
        return {
          color: "text-green-600",
          icon: CheckCircle,
          label: "Uygun Seviye",
          priority: "low",
        };
      default:
        return {
          color: "text-gray-600",
          icon: Minus,
          label: "Belirsiz",
          priority: "low",
        };
    }
  };

  // Filter and sort data
  const filteredAndSortedData = difficultyData
    .filter((topic) => {
      if (filterBy === "needs_adjustment") {
        return (
          topic.difficulty_recommendation !== "appropriate_level" &&
          topic.difficulty_recommendation !== "insufficient_data"
        );
      }
      if (filterBy === "appropriate") {
        return topic.difficulty_recommendation === "appropriate_level";
      }
      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case "understanding":
          return a.avg_understanding_score - b.avg_understanding_score;
        case "success_rate":
          return a.success_rate - b.success_rate;
        case "time":
          return b.avg_time_spent - a.avg_time_spent;
        default:
          return 0;
      }
    });

  // Calculate statistics
  const stats = {
    needsAdjustment: difficultyData.filter(
      (t) =>
        t.difficulty_recommendation !== "appropriate_level" &&
        t.difficulty_recommendation !== "insufficient_data"
    ).length,
    appropriate: difficultyData.filter(
      (t) => t.difficulty_recommendation === "appropriate_level"
    ).length,
    avgSuccessRate:
      difficultyData.length > 0
        ? (difficultyData.reduce((sum, t) => sum + t.success_rate, 0) /
            difficultyData.length) *
          100
        : 0,
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Konu Zorluk Analizi</CardTitle>
          <CardDescription>
            Zorluk seviyesi vs öğrenci performansı korelasyonu
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent"></div>
            <span className="ml-3 text-sm text-muted-foreground">
              Analiz ediliyor...
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
          <CardTitle>Konu Zorluk Analizi</CardTitle>
          <CardDescription>
            Zorluk seviyesi vs öğrenci performansı korelasyonu
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-sm text-muted-foreground mb-4">{error}</p>
            <Button onClick={fetchDifficultyData} variant="outline" size="sm">
              Yeniden Dene
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (compact) {
    // Compact view for dashboard overview
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Zorluk Analizi Özeti</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-lg font-semibold text-red-600">
                {stats.needsAdjustment}
              </div>
              <div className="text-xs text-muted-foreground">Ayar Gerekli</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-green-600">
                {stats.appropriate}
              </div>
              <div className="text-xs text-muted-foreground">Uygun Seviye</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-blue-600">
                {stats.avgSuccessRate.toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">
                Ortalama Başarı
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Toplam Konu</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{difficultyData.length}</div>
            <p className="text-xs text-muted-foreground">Analiz edilen</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Ayar Gerekli</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">
              {stats.needsAdjustment}
            </div>
            <p className="text-xs text-muted-foreground">Zorluk ayarı</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Uygun Seviye</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {stats.appropriate}
            </div>
            <p className="text-xs text-muted-foreground">Doğru zorluk</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">
              Ortalama Başarı
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">
              {stats.avgSuccessRate.toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">Başarı oranı</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Analysis */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Detaylı Zorluk Analizi
              </CardTitle>
              <CardDescription>
                Konu zorluk seviyeleri ve öğrenci performans korelasyonu
              </CardDescription>
            </div>

            {/* Controls */}
            <div className="flex flex-wrap gap-2">
              <div className="flex gap-1">
                <Button
                  variant={sortBy === "understanding" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSortBy("understanding")}
                  className="text-xs"
                >
                  Anlama
                </Button>
                <Button
                  variant={sortBy === "success_rate" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSortBy("success_rate")}
                  className="text-xs"
                >
                  Başarı
                </Button>
                <Button
                  variant={sortBy === "time" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSortBy("time")}
                  className="text-xs"
                >
                  Süre
                </Button>
              </div>

              <div className="flex gap-1">
                <Button
                  variant={filterBy === "all" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setFilterBy("all")}
                  className="text-xs"
                >
                  Tümü
                </Button>
                <Button
                  variant={
                    filterBy === "needs_adjustment" ? "default" : "outline"
                  }
                  size="sm"
                  onClick={() => setFilterBy("needs_adjustment")}
                  className="text-xs"
                >
                  Ayar Gerekli
                </Button>
                <Button
                  variant={filterBy === "appropriate" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setFilterBy("appropriate")}
                  className="text-xs"
                >
                  Uygun
                </Button>
              </div>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {filteredAndSortedData.length === 0 ? (
            <div className="text-center py-8">
              <Target className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-sm text-muted-foreground">
                Filtrelenen kriterlere uygun konu bulunmamaktadır
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredAndSortedData.map((topic) => {
                const difficultyInfo = getDifficultyInfo(
                  topic.estimated_difficulty
                );
                const recommendationInfo = getRecommendationInfo(
                  topic.difficulty_recommendation
                );
                const DifficultyIcon = difficultyInfo.icon;
                const RecommendationIcon = recommendationInfo.icon;

                return (
                  <div
                    key={topic.topic_id}
                    className="p-4 border rounded-lg hover:bg-muted/30 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h4 className="font-semibold text-foreground">
                            {topic.topic_title}
                          </h4>
                          <Badge
                            variant="outline"
                            className={difficultyInfo.bgColor}
                          >
                            <DifficultyIcon
                              className={`h-3 w-3 mr-1 ${difficultyInfo.color}`}
                            />
                            {difficultyInfo.label}
                          </Badge>
                        </div>

                        {/* Recommendation */}
                        <div className="flex items-center gap-2 mb-3">
                          <RecommendationIcon
                            className={`h-4 w-4 ${recommendationInfo.color}`}
                          />
                          <span
                            className={`text-sm font-medium ${recommendationInfo.color}`}
                          >
                            {recommendationInfo.label}
                          </span>
                          {recommendationInfo.priority === "high" && (
                            <Badge variant="destructive" className="text-xs">
                              Yüksek Öncelik
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Metrics Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-6 gap-4 text-center">
                      <div>
                        <div className="text-lg font-semibold text-blue-600">
                          {topic.avg_understanding_score.toFixed(1)}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Anlama Skoru
                        </div>
                      </div>

                      <div>
                        <div className="text-lg font-semibold text-green-600">
                          {(topic.success_rate * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Başarı Oranı
                        </div>
                      </div>

                      <div>
                        <div className="text-lg font-semibold text-purple-600">
                          {Math.round(topic.avg_time_spent)}m
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Ortalama Süre
                        </div>
                      </div>

                      <div>
                        <div className="text-lg font-semibold text-orange-600">
                          {topic.students_attempted}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          <Users className="h-3 w-3 inline mr-1" />
                          Öğrenci
                        </div>
                      </div>

                      <div>
                        <div className="text-lg font-semibold text-indigo-600">
                          {topic.total_questions}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Toplam Soru
                        </div>
                      </div>

                      <div>
                        <div className="text-lg font-semibold text-pink-600">
                          {topic.avg_satisfaction_score.toFixed(1)}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Memnuniyet
                        </div>
                      </div>
                    </div>

                    {/* Additional Info */}
                    <div className="mt-3 pt-3 border-t border-gray-100 flex justify-between text-xs text-muted-foreground">
                      <span>
                        KB Kalitesi: {topic.kb_quality_score.toFixed(2)}
                      </span>
                      <span>
                        Soru Karmaşıklığı:{" "}
                        {topic.avg_question_complexity.toFixed(1)}
                      </span>
                      <span>
                        Mevcut Soru Havuzu: {topic.available_qa_pairs}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default TopicDifficultyAnalysis;
