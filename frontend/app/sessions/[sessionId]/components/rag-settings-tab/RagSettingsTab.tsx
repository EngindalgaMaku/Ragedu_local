/**
 * RagSettingsTab Component - Main RAG settings tab component
 * Configuration overview cards, model selectors, and save functionality
 * Turkish language interface with mobile responsive design
 */

import React from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Settings,
  Brain,
  Database,
  Sparkles,
  CheckCircle,
  AlertCircle,
  Info,
  Zap,
  Globe,
  Crown,
} from "lucide-react";
import { ModelSelector } from "./ModelSelector";
import { EmbeddingSelector } from "./EmbeddingSelector";
import { RerankerSelector } from "./RerankerSelector";
import { SettingsSaveButton } from "./SettingsSaveButton";
import { useRagSettings } from "../../hooks/useRagSettings";

interface RagSettingsTabProps {
  sessionId: string;
}

export const RagSettingsTab: React.FC<RagSettingsTabProps> = ({
  sessionId,
}) => {
  const {
    selectedProvider,
    selectedQueryModel,
    selectedEmbeddingProvider,
    selectedEmbeddingModel,
    useRerankerService,
    selectedRerankerType,
    hasUnsavedChanges,
    validateSettings,
    error,
  } = useRagSettings(sessionId);

  const validation = validateSettings();

  // Configuration overview data
  const getConfigurationStatus = () => {
    const configs = [
      {
        id: "ai_model",
        title: "AI Modeli",
        icon: <Brain className="w-4 h-4" />,
        status:
          selectedProvider && selectedQueryModel ? "configured" : "pending",
        value: selectedQueryModel
          ? `${selectedProvider} / ${selectedQueryModel}`
          : "Seçilmedi",
        description: "Metin üretimi için kullanılan AI modeli",
      },
      {
        id: "embedding",
        title: "Embedding Modeli",
        icon: <Database className="w-4 h-4" />,
        status:
          selectedEmbeddingProvider && selectedEmbeddingModel
            ? "configured"
            : "pending",
        value: selectedEmbeddingModel
          ? `${selectedEmbeddingProvider} / ${selectedEmbeddingModel}`
          : "Seçilmedi",
        description: "Metin vektörleştirme için kullanılan model",
        priority: selectedEmbeddingProvider === "alibaba",
      },
      {
        id: "reranker",
        title: "Reranker Servisi",
        icon: <Sparkles className="w-4 h-4" />,
        status: useRerankerService ? "configured" : "disabled",
        value: useRerankerService ? selectedRerankerType : "Pasif",
        description: "Arama sonuçları yeniden sıralaması",
      },
    ];

    return configs;
  };

  const configurations = getConfigurationStatus();

  const getStatusColor = (status: string) => {
    switch (status) {
      case "configured":
        return "bg-green-100 text-green-700 border-green-200";
      case "pending":
        return "bg-orange-100 text-orange-700 border-orange-200";
      case "disabled":
        return "bg-gray-100 text-gray-600 border-gray-200";
      default:
        return "bg-gray-100 text-gray-600 border-gray-200";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "configured":
        return <CheckCircle className="w-3 h-3" />;
      case "pending":
        return <AlertCircle className="w-3 h-3" />;
      case "disabled":
        return <Info className="w-3 h-3" />;
      default:
        return <Info className="w-3 h-3" />;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case "configured":
        return "Yapılandırıldı";
      case "pending":
        return "Beklemede";
      case "disabled":
        return "Pasif";
      default:
        return "Bilinmeyen";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-primary/10">
          <Settings className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-foreground">
            RAG Ayarları
          </h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            Bu ders oturumu için RAG (Retrieval-Augmented Generation) ayarlarını
            yapılandırın
          </p>
        </div>
      </div>

      {/* Configuration Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Info className="w-4 h-4 text-primary" />
            Yapılandırma Durumu
          </CardTitle>
          <CardDescription>
            Mevcut RAG ayarlarınızın genel durumu
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {configurations.map((config) => (
              <div
                key={config.id}
                className="p-3 border border-border rounded-lg bg-muted/30 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {config.icon}
                    <span className="text-sm font-medium">{config.title}</span>
                    {config.priority && (
                      <div title="Önerilen">
                        <Crown className="w-3 h-3 text-yellow-500" />
                      </div>
                    )}
                  </div>
                  <Badge className={getStatusColor(config.status)}>
                    {getStatusIcon(config.status)}
                    <span className="ml-1">{getStatusText(config.status)}</span>
                  </Badge>
                </div>
                <div className="text-xs text-muted-foreground">
                  {config.description}
                </div>
                <div className="text-sm font-medium text-foreground">
                  {config.value}
                </div>
              </div>
            ))}
          </div>

          {/* Overall Status */}
          <div className="mt-4 pt-4 border-t border-border">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-foreground">
                Genel Durum:
              </span>
              <div className="flex items-center gap-2">
                {validation.isValid ? (
                  <>
                    <Badge className="bg-green-100 text-green-700">
                      <CheckCircle className="w-3 h-3 mr-1" />
                      Hazır
                    </Badge>
                    {hasUnsavedChanges && (
                      <Badge
                        variant="outline"
                        className="border-orange-200 text-orange-700"
                      >
                        <AlertCircle className="w-3 h-3 mr-1" />
                        Kaydedilmemiş
                      </Badge>
                    )}
                  </>
                ) : (
                  <Badge className="bg-red-100 text-red-700">
                    <AlertCircle className="w-3 h-3 mr-1" />
                    Eksik Ayarlar
                  </Badge>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Selection Section */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-primary" />
          <h3 className="text-lg font-medium text-foreground">Model Seçimi</h3>
        </div>
        <ModelSelector sessionId={sessionId} />
      </div>

      {/* Embedding Selection Section */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-primary" />
          <h3 className="text-lg font-medium text-foreground">
            Embedding Yapılandırması
          </h3>
          <Badge
            variant="outline"
            className="border-yellow-200 text-yellow-700"
          >
            <Crown className="w-3 h-3 mr-1" />
            Alibaba Önerilen
          </Badge>
        </div>
        <EmbeddingSelector sessionId={sessionId} />
      </div>

      {/* Reranker Selection Section */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          <h3 className="text-lg font-medium text-foreground">
            Reranker Yapılandırması
          </h3>
          <Badge variant="outline" className="text-xs">
            İsteğe Bağlı
          </Badge>
        </div>
        <RerankerSelector sessionId={sessionId} />
      </div>

      {/* Performance Notice */}
      <Alert>
        <Zap className="h-4 w-4 text-blue-500" />
        <AlertDescription>
          <span className="font-medium">Performans İpucu:</span> En iyi sonuçlar
          için{" "}
          <Badge variant="outline" className="mx-1">
            Alibaba embedding
          </Badge>{" "}
          ve{" "}
          <Badge variant="outline" className="mx-1">
            BGE reranker
          </Badge>{" "}
          kombinasyonunu öneririz. Bu kombinasyon Türkçe içerik için optimize
          edilmiştir.
        </AlertDescription>
      </Alert>

      {/* Global Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Save Section */}
      <SettingsSaveButton sessionId={sessionId} />

      {/* Help Information */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Info className="w-4 h-4 text-primary" />
            RAG Ayarları Hakkında
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-blue-500" />
                <span className="font-medium">AI Provider & Model</span>
              </div>
              <p className="text-muted-foreground">
                Öğrenci sorularına cevap üretmek için kullanılacak yapay zeka
                modelini seçin. Groq hızlı yanıt, Ollama yerel güvenlik sağlar.
              </p>
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-green-500" />
                <span className="font-medium">Embedding Model</span>
              </div>
              <p className="text-muted-foreground">
                Dökümanları vektörlere dönüştürür. Alibaba Cloud modelleri
                Türkçe için özel olarak optimize edilmiştir.
              </p>
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-purple-500" />
                <span className="font-medium">Reranker Service</span>
              </div>
              <p className="text-muted-foreground">
                Bulunan içerikleri yeniden sıralayarak en alakalı olanları öne
                çıkarır. %20-40 daha iyi sonuçlar sağlar.
              </p>
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Globe className="w-4 h-4 text-orange-500" />
                <span className="font-medium">Türkçe Optimizasyonu</span>
              </div>
              <p className="text-muted-foreground">
                Türkçe içerik için en iyi performansı elde etmek adına özel
                olarak seçilmiş model kombinasyonları önerilir.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default RagSettingsTab;
