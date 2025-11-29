/**
 * RerankerSelector Component - Reranker enable/disable toggle and type selection
 * Handles reranker service toggle, type selection, provider information, and performance settings
 */

import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import {
  Sparkles,
  Info,
  Zap,
  Globe,
  Layers,
  CheckCircle,
  AlertTriangle,
  TrendingUp,
} from "lucide-react";
import { useRagSettings } from "../../hooks/useRagSettings";

interface RerankerSelectorProps {
  sessionId: string;
}

export const RerankerSelector: React.FC<RerankerSelectorProps> = ({
  sessionId,
}) => {
  const {
    useRerankerService,
    selectedRerankerType,
    setUseRerankerService,
    setSelectedRerankerType,
    RERANKER_TYPE_OPTIONS,
  } = useRagSettings(sessionId);

  const getRerankerTypeInfo = (type: string) => {
    switch (type) {
      case "bge-reranker-v2-m3":
        return {
          name: "BGE-Reranker-V2-M3",
          provider: "BAAI",
          icon: "🔮",
          features: ["Türkçe Optimize", "Yüksek Doğruluk", "Çok Dilli"],
          performance: { accuracy: 95, speed: 80, memory: 85 },
          description:
            "Türkçe içerik için özel olarak optimize edilmiş, yüksek doğruluk oranına sahip reranker modeli.",
          color: "bg-purple-100 text-purple-700",
        };
      case "ms-marco-minilm-l6":
        return {
          name: "MS-MARCO MiniLM-L6",
          provider: "Microsoft",
          icon: "⚡",
          features: ["Hızlı", "Hafif", "Düşük Kaynak"],
          performance: { accuracy: 85, speed: 95, memory: 95 },
          description:
            "Hızlı yanıt süresi için optimize edilmiş, hafif ve verimli reranker modeli.",
          color: "bg-blue-100 text-blue-700",
        };
      case "gte-rerank-v2":
        return {
          name: "GTE-Rerank-V2",
          provider: "Alibaba",
          icon: "🛒",
          features: ["50+ Dil", "Yüksek Kalite", "Enterprise"],
          performance: { accuracy: 90, speed: 75, memory: 80 },
          description:
            "50'den fazla dili destekleyen, enterprise düzeyinde kalite sunan reranker modeli.",
          color: "bg-orange-100 text-orange-700",
        };
      default:
        return {
          name: "Bilinmeyen Model",
          provider: "Unknown",
          icon: "🔧",
          features: ["Standart"],
          performance: { accuracy: 70, speed: 70, memory: 70 },
          description: "Model bilgisi mevcut değil.",
          color: "bg-gray-100 text-gray-700",
        };
    }
  };

  const selectedRerankerInfo = getRerankerTypeInfo(selectedRerankerType);

  const PerformanceBar: React.FC<{
    label: string;
    value: number;
    color: string;
  }> = ({ label, value, color }) => (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground w-16">{label}</span>
      <div className="flex-1 bg-gray-200 rounded-full h-1.5">
        <div
          className={`h-1.5 rounded-full ${color}`}
          style={{ width: `${value}%` }}
        />
      </div>
      <span className="text-xs text-muted-foreground w-8">{value}%</span>
    </div>
  );

  return (
    <div className="space-y-4">
      {/* Reranker Toggle */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-primary" />
            Reranker Servisi
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Toggle Switch */}
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <Label htmlFor="reranker-toggle" className="text-sm font-medium">
                Reranker Servisini Etkinleştir
              </Label>
              <p className="text-xs text-muted-foreground">
                Arama sonuçlarını yeniden sıralayarak daha alakalı sonuçlar elde
                edin
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <input
                id="reranker-toggle"
                type="checkbox"
                checked={useRerankerService}
                onChange={(e) => setUseRerankerService(e.target.checked)}
                className="w-4 h-4 text-primary border-border rounded focus:ring-primary"
              />
            </div>
          </div>

          {/* Service Status */}
          <div className="flex items-center gap-2">
            <Badge
              className={
                useRerankerService
                  ? "bg-green-100 text-green-700"
                  : "bg-gray-100 text-gray-700"
              }
            >
              {useRerankerService ? (
                <>
                  <CheckCircle className="w-3 h-3 mr-1" />
                  Aktif
                </>
              ) : (
                <>
                  <AlertTriangle className="w-3 h-3 mr-1" />
                  Pasif
                </>
              )}
            </Badge>
            {useRerankerService && (
              <span className="text-xs text-muted-foreground">
                Ayrı container'da çalışır
              </span>
            )}
          </div>

          {/* Benefits Info */}
          {!useRerankerService && (
            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
              <div className="flex items-start gap-2">
                <Info className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-blue-800 dark:text-blue-200">
                  <div className="font-medium mb-1">
                    Reranker Servisi Faydaları:
                  </div>
                  <ul className="text-xs space-y-0.5 list-disc list-inside">
                    <li>Arama sonuçlarının kalitesini %20-40 artırır</li>
                    <li>Daha alakalı içerikleri üst sıralara taşır</li>
                    <li>Kullanıcı sorularına daha doğru cevaplar</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Reranker Type Selection */}
      {useRerankerService && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Layers className="w-4 h-4 text-primary" />
              Reranker Modeli
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="reranker-type-select">Model Seçin</Label>
              <select
                id="reranker-type-select"
                value={selectedRerankerType}
                onChange={(e) => setSelectedRerankerType(e.target.value)}
                className="w-full px-4 py-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all bg-background shadow-sm"
              >
                {RERANKER_TYPE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="border-t border-border my-3" />

            {/* Selected Model Info */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Badge className={selectedRerankerInfo.color}>
                  {selectedRerankerInfo.icon} {selectedRerankerInfo.name}
                </Badge>
                <span className="text-xs text-muted-foreground">
                  by {selectedRerankerInfo.provider}
                </span>
              </div>

              <p className="text-xs text-muted-foreground">
                {selectedRerankerInfo.description}
              </p>

              {/* Features */}
              <div className="flex flex-wrap gap-1">
                {selectedRerankerInfo.features.map((feature, index) => (
                  <Badge key={index} variant="outline" className="text-xs">
                    {feature}
                  </Badge>
                ))}
              </div>

              {/* Performance Metrics */}
              <div className="space-y-2 p-3 bg-muted/30 rounded-md">
                <div className="text-sm font-medium text-foreground flex items-center gap-1">
                  <TrendingUp className="w-3 h-3" />
                  Performans Metrikleri
                </div>
                <div className="space-y-1.5">
                  <PerformanceBar
                    label="Doğruluk"
                    value={selectedRerankerInfo.performance.accuracy}
                    color="bg-green-500"
                  />
                  <PerformanceBar
                    label="Hız"
                    value={selectedRerankerInfo.performance.speed}
                    color="bg-blue-500"
                  />
                  <PerformanceBar
                    label="Verimlilik"
                    value={selectedRerankerInfo.performance.memory}
                    color="bg-purple-500"
                  />
                </div>
              </div>

              {/* Recommendation */}
              {selectedRerankerType === "bge-reranker-v2-m3" && (
                <Alert>
                  <Sparkles className="h-4 w-4 text-purple-500" />
                  <AlertDescription>
                    <span className="font-medium">Önerilen:</span> Türkçe içerik
                    için en iyi performansı sunar. Akademik ve teknik metinlerde
                    yüksek doğruluk oranı.
                  </AlertDescription>
                </Alert>
              )}

              {selectedRerankerType === "ms-marco-minilm-l6" && (
                <Alert>
                  <Zap className="h-4 w-4 text-blue-500" />
                  <AlertDescription>
                    <span className="font-medium">Hızlı:</span> Düşük gecikme
                    süresi gerektiren uygulamalar için ideal. Gerçek zamanlı
                    soru-cevap sistemleri için optimize edilmiş.
                  </AlertDescription>
                </Alert>
              )}

              {selectedRerankerType === "gte-rerank-v2" && (
                <Alert>
                  <Globe className="h-4 w-4 text-orange-500" />
                  <AlertDescription>
                    <span className="font-medium">Çok Dilli:</span> 50+ dil
                    desteği ile global uygulamalar için ideal. Enterprise
                    düzeyinde kalite ve güvenilirlik.
                  </AlertDescription>
                </Alert>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default RerankerSelector;
