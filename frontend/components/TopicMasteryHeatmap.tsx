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
  Target,
  TrendingUp,
  TrendingDown,
  Users,
  MessageSquare,
  Brain,
  AlertCircle,
  CheckCircle,
  Clock,
} from "lucide-react";

interface HeatmapData {
  topic_id: number;
  topic_title: string;
  difficulty: string;
  students_attempted: number;
  mastery_score: number;
  understanding_score: number;
  satisfaction_score: number;
  performance_level: string;
  status: string;
}

interface TopicMasteryHeatmapProps {
  sessionId: string;
  data?: any;
}

const TopicMasteryHeatmap: React.FC<TopicMasteryHeatmapProps> = ({
  sessionId,
  data,
}) => {
  const [heatmapData, setHeatmapData] = useState<HeatmapData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<
    "mastery" | "understanding" | "students"
  >("understanding");
  const [filterBy, setFilterBy] = useState<"all" | "needs_attention" | "good">(
    "all"
  );

  // Use provided data or fetch it
  useEffect(() => {
    if (data?.heatmap_data) {
      setHeatmapData(data.heatmap_data);
    } else if (sessionId) {
      fetchHeatmapData();
    }
  }, [sessionId, data]);

  const fetchHeatmapData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `/api/aprag/analytics/topics/mastery-overview/${sessionId}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success && result.heatmap_data) {
        setHeatmapData(result.heatmap_data);
      } else {
        throw new Error("Failed to load heatmap data");
      }
    } catch (e: any) {
      setError(e.message || "Failed to load heatmap data");
      console.error("Heatmap data error:", e);
    } finally {
      setLoading(false);
    }
  };

  // Get color based on score
  const getScoreColor = (
    score: number,
    type: "mastery" | "understanding" | "satisfaction"
  ) => {
    if (score >= 4.0) return "bg-green-500";
    if (score >= 3.5) return "bg-green-400";
    if (score >= 3.0) return "bg-yellow-400";
    if (score >= 2.0) return "bg-orange-400";
    if (score >= 1.0) return "bg-red-400";
    return "bg-red-500";
  };

  const getScoreTextColor = (score: number) => {
    return score >= 3.0 ? "text-white" : "text-white";
  };

  const getDifficultyBadge = (difficulty: string) => {
    const variants = {
      beginner: "bg-green-100 text-green-800 border-green-200",
      intermediate: "bg-yellow-100 text-yellow-800 border-yellow-200",
      advanced: "bg-red-100 text-red-800 border-red-200",
    };
    return (
      variants[difficulty as keyof typeof variants] || variants.intermediate
    );
  };

  const getPerformanceIcon = (level: string) => {
    switch (level) {
      case "high":
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case "medium":
        return <Target className="h-4 w-4 text-yellow-600" />;
      case "low":
        return <TrendingDown className="h-4 w-4 text-orange-600" />;
      case "very_low":
        return <AlertCircle className="h-4 w-4 text-red-600" />;
      default:
        return <Clock className="h-4 w-4 text-gray-600" />;
    }
  };

  // Filter and sort data
  const filteredAndSortedData = heatmapData
    .filter((topic) => {
      if (filterBy === "needs_attention")
        return topic.status === "needs_attention";
      if (filterBy === "good") return topic.status === "good";
      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case "mastery":
          return b.mastery_score - a.mastery_score;
        case "understanding":
          return b.understanding_score - a.understanding_score;
        case "students":
          return b.students_attempted - a.students_attempted;
        default:
          return 0;
      }
    });

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Konu Yeterlilik Haritası</CardTitle>
          <CardDescription>
            Konular arası performans karşılaştırması
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent"></div>
            <span className="ml-3 text-sm text-muted-foreground">
              Yükleniyor...
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
          <CardTitle>Konu Yeterlilik Haritası</CardTitle>
          <CardDescription>
            Konular arası performans karşılaştırması
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-sm text-muted-foreground mb-4">{error}</p>
            <Button onClick={fetchHeatmapData} variant="outline" size="sm">
              Yeniden Dene
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (heatmapData.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Konu Yeterlilik Haritası</CardTitle>
          <CardDescription>
            Konular arası performans karşılaştırması
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Brain className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-sm text-muted-foreground">
              Henüz konu verisi bulunmamaktadır
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Konu Yeterlilik Haritası
            </CardTitle>
            <CardDescription>
              Öğrenci performansının konu bazlı görsel analizi
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
                variant={sortBy === "mastery" ? "default" : "outline"}
                size="sm"
                onClick={() => setSortBy("mastery")}
                className="text-xs"
              >
                Yeterlilik
              </Button>
              <Button
                variant={sortBy === "students" ? "default" : "outline"}
                size="sm"
                onClick={() => setSortBy("students")}
                className="text-xs"
              >
                Katılım
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
                variant={filterBy === "needs_attention" ? "default" : "outline"}
                size="sm"
                onClick={() => setFilterBy("needs_attention")}
                className="text-xs"
              >
                Dikkat
              </Button>
              <Button
                variant={filterBy === "good" ? "default" : "outline"}
                size="sm"
                onClick={() => setFilterBy("good")}
                className="text-xs"
              >
                İyi
              </Button>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {/* Legend */}
        <div className="mb-6 p-4 bg-muted/30 rounded-lg">
          <h4 className="text-sm font-semibold mb-3">
            Renk Skalası (Anlama Skoru)
          </h4>
          <div className="flex flex-wrap gap-2 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-500 rounded"></div>
              <span>4.0+ Mükemmel</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-400 rounded"></div>
              <span>3.5+ İyi</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-yellow-400 rounded"></div>
              <span>3.0+ Orta</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-orange-400 rounded"></div>
              <span>2.0+ Zayıf</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-500 rounded"></div>
              <span>{"<"}2.0 Kritik</span>
            </div>
          </div>
        </div>

        {/* Heatmap Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filteredAndSortedData.map((topic) => (
            <div
              key={topic.topic_id}
              className="relative overflow-hidden rounded-lg border-2 border-gray-200 hover:border-gray-300 transition-colors"
            >
              {/* Main score background */}
              <div
                className={`p-4 ${getScoreColor(
                  topic.understanding_score,
                  "understanding"
                )} ${getScoreTextColor(topic.understanding_score)}`}
              >
                {/* Topic header */}
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-semibold text-sm leading-tight flex-1 pr-2">
                    {topic.topic_title}
                  </h4>
                  {getPerformanceIcon(topic.performance_level)}
                </div>

                {/* Difficulty badge */}
                <div className="mb-3">
                  <span
                    className={`px-2 py-1 rounded text-xs font-medium border ${getDifficultyBadge(
                      topic.difficulty
                    )}`}
                  >
                    {topic.difficulty === "beginner"
                      ? "Başlangıç"
                      : topic.difficulty === "intermediate"
                      ? "Orta"
                      : "İleri"}
                  </span>
                </div>

                {/* Main metrics */}
                <div className="grid grid-cols-2 gap-2 text-center">
                  <div>
                    <div className="text-lg font-bold">
                      {topic.understanding_score.toFixed(1)}
                    </div>
                    <div className="text-xs opacity-90">Anlama</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold">
                      {topic.mastery_score.toFixed(1)}
                    </div>
                    <div className="text-xs opacity-90">Yeterlilik</div>
                  </div>
                </div>
              </div>

              {/* Stats footer */}
              <div className="bg-white p-2 text-xs text-gray-600 border-t">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-1">
                    <Users className="h-3 w-3" />
                    <span>{topic.students_attempted} öğrenci</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <MessageSquare className="h-3 w-3" />
                    <span>{topic.satisfaction_score.toFixed(1)}</span>
                  </div>
                </div>
              </div>

              {/* Status indicator */}
              {topic.status === "needs_attention" && (
                <div className="absolute top-2 right-2">
                  <AlertCircle className="h-4 w-4 text-red-600" />
                </div>
              )}

              {topic.status === "good" && (
                <div className="absolute top-2 right-2">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Summary stats */}
        <div className="mt-6 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {heatmapData.length}
              </div>
              <div className="text-sm text-gray-600">Toplam Konu</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-600">
                {heatmapData.filter((t) => t.status === "good").length}
              </div>
              <div className="text-sm text-gray-600">İyi Performans</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-red-600">
                {
                  heatmapData.filter((t) => t.status === "needs_attention")
                    .length
                }
              </div>
              <div className="text-sm text-gray-600">Dikkat Gerekli</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600">
                {(
                  heatmapData.reduce(
                    (sum, t) => sum + t.understanding_score,
                    0
                  ) / heatmapData.length
                ).toFixed(2)}
              </div>
              <div className="text-sm text-gray-600">Ortalama Anlama</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TopicMasteryHeatmap;
