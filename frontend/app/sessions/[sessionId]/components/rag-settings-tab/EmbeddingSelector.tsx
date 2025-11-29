/**
 * EmbeddingSelector Component - Embedding provider and model selection
 * Handles embedding provider selection with Alibaba priority, model dropdown, dimensions info
 */

import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import {
  Layers,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Database,
  Zap,
  Globe,
  Crown,
} from "lucide-react";
import { useRagSettings } from "../../hooks/useRagSettings";

interface EmbeddingSelectorProps {
  sessionId: string;
}

export const EmbeddingSelector: React.FC<EmbeddingSelectorProps> = ({
  sessionId,
}) => {
  const {
    availableEmbeddingModels,
    selectedEmbeddingProvider,
    selectedEmbeddingModel,
    embeddingModelsLoading,
    error,
    setSelectedEmbeddingProvider,
    setSelectedEmbeddingModel,
    fetchEmbeddingModels,
    EMBEDDING_PROVIDER_OPTIONS,
  } = useRagSettings(sessionId);

  const handleProviderChange = (provider: string) => {
    setSelectedEmbeddingProvider(provider);

    // Auto-select first available model for the new provider
    if (provider === "alibaba" && availableEmbeddingModels.alibaba?.length) {
      setSelectedEmbeddingModel(availableEmbeddingModels.alibaba[0].id);
    } else if (
      provider === "ollama" &&
      availableEmbeddingModels.ollama.length
    ) {
      setSelectedEmbeddingModel(availableEmbeddingModels.ollama[0]);
    } else if (
      provider === "huggingface" &&
      availableEmbeddingModels.huggingface.length
    ) {
      setSelectedEmbeddingModel(availableEmbeddingModels.huggingface[0].id);
    } else {
      setSelectedEmbeddingModel("");
    }
  };

  const getProviderIcon = (provider: string) => {
    switch (provider) {
      case "alibaba":
        return "🛒";
      case "huggingface":
        return "🤗";
      case "ollama":
        return "🏠";
      default:
        return "🔧";
    }
  };

  const getProviderStatus = (provider: string) => {
    switch (provider) {
      case "alibaba":
        return {
          status: "premium",
          label: "Premium",
          color: "bg-yellow-100 text-yellow-700",
          description: "Yüksek kaliteli, çok boyutlu embeddings",
        };
      case "huggingface":
        return {
          status: "free",
          label: "Free",
          color: "bg-blue-100 text-blue-700",
          description: "Ücretsiz, açık kaynak modeller",
        };
      case "ollama":
        return {
          status: "local",
          label: "Local",
          color: "bg-purple-100 text-purple-700",
          description: "Yerel işlem, gizlilik öncelikli",
        };
      default:
        return {
          status: "unknown",
          label: "Unknown",
          color: "bg-gray-100 text-gray-700",
          description: "Bilinmeyen provider",
        };
    }
  };

  const getCurrentProviderModels = () => {
    switch (selectedEmbeddingProvider) {
      case "alibaba":
        return availableEmbeddingModels.alibaba || [];
      case "ollama":
        return availableEmbeddingModels.ollama.map((model) => ({
          id: model,
          name: model,
        }));
      case "huggingface":
        return availableEmbeddingModels.huggingface;
      default:
        return [];
    }
  };

  const getSelectedModelInfo = () => {
    const models = getCurrentProviderModels();

    if (selectedEmbeddingProvider === "ollama") {
      return models.find((m: any) => m.id === selectedEmbeddingModel);
    }

    return models.find((m: any) => m.id === selectedEmbeddingModel);
  };

  const providerStatus = getProviderStatus(selectedEmbeddingProvider);
  const selectedModelInfo = getSelectedModelInfo();
  const availableModelsForProvider = getCurrentProviderModels();

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Embedding Provider Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Database className="w-4 h-4 text-primary" />
            Embedding Provider
            {selectedEmbeddingProvider === "alibaba" && (
              <div title="Önerilen">
                <Crown className="w-4 h-4 text-yellow-500" />
              </div>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="embedding-provider-select">Provider Seçin</Label>
            <select
              id="embedding-provider-select"
              value={selectedEmbeddingProvider}
              onChange={(e) => handleProviderChange(e.target.value)}
              disabled={embeddingModelsLoading}
              className="w-full px-4 py-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all bg-background shadow-sm"
            >
              <option value="">Provider seçin...</option>
              {EMBEDDING_PROVIDER_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                  {option.value === "alibaba" ? " (Önerilen)" : ""}
                </option>
              ))}
            </select>
          </div>

          {/* Provider Status and Info */}
          {selectedEmbeddingProvider && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Badge className={providerStatus.color}>
                  <CheckCircle className="w-3 h-3 mr-1" />
                  {providerStatus.label}
                </Badge>
                <span className="text-sm text-muted-foreground">
                  {getProviderIcon(selectedEmbeddingProvider)}{" "}
                  {selectedEmbeddingProvider.charAt(0).toUpperCase() +
                    selectedEmbeddingProvider.slice(1)}
                </span>
              </div>

              <div className="text-xs text-muted-foreground">
                {providerStatus.description}
              </div>

              {/* Provider Features */}
              <div className="flex flex-wrap gap-1">
                {selectedEmbeddingProvider === "alibaba" && (
                  <>
                    <Badge variant="outline" className="text-xs">
                      <Zap className="w-3 h-3 mr-1" />
                      Yüksek Performans
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      <Globe className="w-3 h-3 mr-1" />
                      Çok Dilli
                    </Badge>
                  </>
                )}
                {selectedEmbeddingProvider === "huggingface" && (
                  <>
                    <Badge variant="outline" className="text-xs">
                      <Globe className="w-3 h-3 mr-1" />
                      Açık Kaynak
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      Ücretsiz
                    </Badge>
                  </>
                )}
                {selectedEmbeddingProvider === "ollama" && (
                  <>
                    <Badge variant="outline" className="text-xs">
                      🏠 Yerel
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      🔒 Güvenli
                    </Badge>
                  </>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Embedding Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Layers className="w-4 h-4 text-primary" />
            Embedding Modeli
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="embedding-model-select">Model Seçin</Label>
            <select
              id="embedding-model-select"
              value={selectedEmbeddingModel}
              onChange={(e) => setSelectedEmbeddingModel(e.target.value)}
              disabled={
                embeddingModelsLoading ||
                availableModelsForProvider.length === 0
              }
              className="w-full px-4 py-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-all bg-background shadow-sm"
            >
              <option value="">
                {embeddingModelsLoading
                  ? "Modeller yükleniyor..."
                  : availableModelsForProvider.length === 0
                  ? "Bu provider için model bulunamadı"
                  : "Model seçin..."}
              </option>
              {availableModelsForProvider.map((model: any) => {
                const modelId = typeof model === "string" ? model : model.id;
                const modelName =
                  typeof model === "string" ? model : model.name;
                const modelDesc =
                  typeof model === "object" && model.description
                    ? ` - ${model.description}`
                    : "";
                const dimensions =
                  typeof model === "object" && model.dimensions
                    ? ` (${model.dimensions}D)`
                    : "";

                return (
                  <option key={modelId} value={modelId}>
                    {modelName}
                    {modelDesc}
                    {dimensions}
                  </option>
                );
              })}
            </select>
          </div>

          {/* Model Actions */}
          <div className="flex items-center gap-2">
            {availableModelsForProvider.length === 0 &&
              !embeddingModelsLoading && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={fetchEmbeddingModels}
                  className="gap-2"
                >
                  <RefreshCw className="w-3 h-3" />
                  Modelleri Yükle
                </Button>
              )}
            {embeddingModelsLoading && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <RefreshCw className="w-3 h-3 animate-spin" />
                Yükleniyor...
              </div>
            )}
          </div>

          <div className="border-t border-border my-3" />

          {/* Selected Model Info */}
          {selectedModelInfo && (
            <div className="space-y-2">
              <div className="text-sm font-medium text-foreground">
                Model Özellikleri
              </div>
              <div className="text-xs text-muted-foreground space-y-1">
                <div>
                  <span className="font-medium">Model ID:</span>{" "}
                  {selectedModelInfo.id || selectedModelInfo.name}
                </div>
                {selectedModelInfo.name &&
                  selectedModelInfo.id !== selectedModelInfo.name && (
                    <div>
                      <span className="font-medium">İsim:</span>{" "}
                      {selectedModelInfo.name}
                    </div>
                  )}
                {(selectedModelInfo as any).description && (
                  <div>
                    <span className="font-medium">Açıklama:</span>{" "}
                    {(selectedModelInfo as any).description}
                  </div>
                )}
                {(selectedModelInfo as any).dimensions && (
                  <div>
                    <span className="font-medium">Boyutlar:</span>{" "}
                    {(selectedModelInfo as any).dimensions}D
                  </div>
                )}
                {(selectedModelInfo as any).language && (
                  <div>
                    <span className="font-medium">Dil Desteği:</span>{" "}
                    {(selectedModelInfo as any).language}
                  </div>
                )}
              </div>

              {/* Performance Indicator */}
              <div className="mt-3 p-2 bg-muted/30 rounded-md">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">
                    Embedding Kalitesi
                  </span>
                  <div className="flex items-center gap-1">
                    {selectedEmbeddingProvider === "alibaba" && (
                      <div className="flex gap-0.5">
                        {[...Array(5)].map((_, i) => (
                          <div
                            key={i}
                            className="w-2 h-2 bg-yellow-400 rounded-full"
                          />
                        ))}
                      </div>
                    )}
                    {selectedEmbeddingProvider === "huggingface" && (
                      <div className="flex gap-0.5">
                        {[...Array(4)].map((_, i) => (
                          <div
                            key={i}
                            className="w-2 h-2 bg-blue-400 rounded-full"
                          />
                        ))}
                        <div className="w-2 h-2 bg-gray-300 rounded-full" />
                      </div>
                    )}
                    {selectedEmbeddingProvider === "ollama" && (
                      <div className="flex gap-0.5">
                        {[...Array(3)].map((_, i) => (
                          <div
                            key={i}
                            className="w-2 h-2 bg-purple-400 rounded-full"
                          />
                        ))}
                        {[...Array(2)].map((_, i) => (
                          <div
                            key={i + 3}
                            className="w-2 h-2 bg-gray-300 rounded-full"
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <div className="md:col-span-2">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      )}

      {/* Alibaba Priority Notice */}
      {!selectedEmbeddingProvider && (
        <div className="md:col-span-2">
          <Alert>
            <Crown className="h-4 w-4 text-yellow-500" />
            <AlertDescription>
              <span className="font-medium">Öneri:</span> En iyi performans için{" "}
              <span className="font-medium">Alibaba</span> provider'ını
              seçmenizi öneriyoruz. Yüksek kaliteli, çok boyutlu embeddings ve
              Türkçe dil desteği sunar.
            </AlertDescription>
          </Alert>
        </div>
      )}
    </div>
  );
};

export default EmbeddingSelector;
