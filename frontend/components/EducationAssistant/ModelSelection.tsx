"use client";

import React, { useState, useEffect } from "react";

interface ModelSelectionProps {
  // Model Selection
  availableModels: any[];
  filteredModels: any[];
  selectedProvider: string;
  selectedQueryModel: string;
  modelsLoading: boolean;
  modelProviders: Record<string, any>;
  onProviderChange: (provider: string) => void;
  onModelChange: (model: string) => void;

  // Embedding Models
  selectedEmbeddingProvider: string;
  selectedEmbeddingModel: string;
  availableEmbeddingModels: {
    ollama: string[];
    huggingface: Array<{
      id: string;
      name: string;
      description?: string;
      dimensions?: number;
      language?: string;
    }>;
    alibaba?: Array<{
      id: string;
      name: string;
      description?: string;
      dimensions?: number;
      language?: string;
    }>;
  };
  embeddingModelsLoading: boolean;
  onEmbeddingProviderChange: (provider: string) => void;
  onEmbeddingModelChange: (model: string) => void;

  // Advanced Settings
  useDirectLLM: boolean;
  onDirectLLMChange: (enabled: boolean) => void;
  chainType: "stuff" | "refine" | "map_reduce";
  onChainTypeChange: (type: "stuff" | "refine" | "map_reduce") => void;
  selectedRerankerType: string;
  onRerankerTypeChange: (type: string) => void;
  useRerankerService: boolean;
  onRerankerServiceChange: (enabled: boolean) => void;

  // Session Settings
  sessionRagSettings?: any;
  selectedSessionId: string;
  isStudent: boolean;
}

export default function ModelSelection({
  availableModels,
  filteredModels,
  selectedProvider,
  selectedQueryModel,
  modelsLoading,
  modelProviders,
  onProviderChange,
  onModelChange,
  selectedEmbeddingProvider,
  selectedEmbeddingModel,
  availableEmbeddingModels,
  embeddingModelsLoading,
  onEmbeddingProviderChange,
  onEmbeddingModelChange,
  useDirectLLM,
  onDirectLLMChange,
  chainType,
  onChainTypeChange,
  selectedRerankerType,
  onRerankerTypeChange,
  useRerankerService,
  onRerankerServiceChange,
  sessionRagSettings,
  selectedSessionId,
  isStudent,
}: ModelSelectionProps) {
  // State for save functionality
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");

  // Don't show model selection for students
  if (isStudent) {
    return null;
  }

  // Save model settings to database
  const saveModelSettings = async () => {
    if (isSaving || !selectedSessionId) return;

    setIsSaving(true);
    setSaveSuccess(false);

    try {
      // Import saveSessionRagSettings API function
      const { saveSessionRagSettings } = await import("@/lib/api");

      const settings = {
        model: selectedQueryModel,
        provider: selectedProvider,
        chain_type: chainType,
        use_direct_llm: useDirectLLM,
        embedding_model: selectedEmbeddingModel,
        embedding_provider: selectedEmbeddingProvider,
        use_reranker_service: useRerankerService,
        reranker_type: selectedRerankerType as "bge" | "ms-marco",
      };

      await saveSessionRagSettings(selectedSessionId, settings);

      setSaveSuccess(true);
      setSaveMessage("Model ayarları oturuma başarıyla kaydedildi!");

      // Auto-hide success message after 3 seconds
      setTimeout(() => {
        setSaveSuccess(false);
        setSaveMessage("");
      }, 3000);
    } catch (error: any) {
      console.error("Error saving model settings:", error);
      setSaveMessage(error?.message || "Ayarlar kaydedilirken hata oluştu!");
      setTimeout(() => setSaveMessage(""), 3000);
    } finally {
      setIsSaving(false);
    }
  };

  // Get available providers
  const providers = Object.keys(modelProviders);

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden mb-6">
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 px-6 py-4 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <span className="text-2xl">🎯</span>
          <span>Model Ayarları</span>
        </h2>
        <p className="text-gray-600 text-sm mt-1">
          Sorgu ve embedding modelleri ile gelişmiş ayarları yapılandırın
        </p>
      </div>

      <div className="p-6 space-y-6">
        {/* Direct LLM Toggle */}
        <div className="flex items-center justify-between p-4 bg-amber-50 border border-amber-200 rounded-lg">
          <div>
            <div className="font-semibold text-amber-800 flex items-center gap-2">
              <span>🤖</span>
              <span>Direkt LLM Modu</span>
            </div>
            <p className="text-sm text-amber-700 mt-1">
              RAG yerine doğrudan LLM kullanın (genel sorular için)
            </p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={useDirectLLM}
              onChange={(e) => onDirectLLMChange(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-amber-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-600"></div>
          </label>
        </div>

        {/* Query Model Selection */}
        {!useDirectLLM && (
          <>
            <div className="space-y-6">
              {/* Provider Selection */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-3">
                  🏢 Model Sağlayıcı
                </label>
                <select
                  value={selectedProvider}
                  onChange={(e) => onProviderChange(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all bg-white"
                  disabled={modelsLoading}
                >
                  {providers.map((provider) => (
                    <option key={provider} value={provider}>
                      {provider.charAt(0).toUpperCase() + provider.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              {/* Model Selection */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-3">
                  🧠 Sorgu Modeli
                </label>
                <select
                  value={selectedQueryModel}
                  onChange={(e) => onModelChange(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all bg-white"
                  disabled={modelsLoading || filteredModels.length === 0}
                >
                  <option value="">Model Seçin</option>
                  {filteredModels.map((model) => (
                    <option
                      key={model.id || model.name}
                      value={model.id || model.name}
                    >
                      {model.name || model.id}{" "}
                      {model.context_length
                        ? `(${model.context_length} tokens)`
                        : ""}
                    </option>
                  ))}
                </select>
                {modelsLoading && (
                  <div className="text-xs text-gray-500 mt-1 flex items-center gap-1">
                    <div className="w-3 h-3 border border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                    <span>Modeller yükleniyor...</span>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* Embedding Model Selection */}
        <div className="border-t border-gray-200 pt-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <span>🎯</span>
            <span>Embedding Modeli</span>
          </h3>

          <div className="space-y-6">
            {/* Embedding Provider */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                🏢 Embedding Sağlayıcı
              </label>
              <select
                value={selectedEmbeddingProvider}
                onChange={(e) => onEmbeddingProviderChange(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all bg-white"
                disabled={embeddingModelsLoading}
              >
                <option value="alibaba">Alibaba Cloud</option>
                <option value="huggingface">HuggingFace</option>
                <option value="ollama">Ollama (Yerel)</option>
              </select>
            </div>

            {/* Embedding Model */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                🔧 Embedding Modeli
              </label>
              <select
                value={selectedEmbeddingModel}
                onChange={(e) => onEmbeddingModelChange(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all bg-white"
                disabled={embeddingModelsLoading}
              >
                <option value="">Model Seçin</option>
                {selectedEmbeddingProvider === "ollama" &&
                  availableEmbeddingModels.ollama.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                {selectedEmbeddingProvider === "huggingface" &&
                  availableEmbeddingModels.huggingface.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}{" "}
                      {model.dimensions ? `(${model.dimensions}D)` : ""}{" "}
                      {model.language ? `[${model.language}]` : ""}
                    </option>
                  ))}
                {selectedEmbeddingProvider === "alibaba" &&
                  availableEmbeddingModels.alibaba?.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}{" "}
                      {model.dimensions ? `(${model.dimensions}D)` : ""}{" "}
                      {model.language ? `[${model.language}]` : ""}
                    </option>
                  ))}
              </select>
              {embeddingModelsLoading && (
                <div className="text-xs text-gray-500 mt-1 flex items-center gap-1">
                  <div className="w-3 h-3 border border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                  <span>Embedding modelleri yükleniyor...</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Reranker Settings */}
        <div className="border-t border-gray-200 pt-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <span>🚀</span>
            <span>Reranker Ayarları</span>
          </h3>

          {/* Reranker Service Toggle */}
          <div className="flex items-center justify-between p-4 bg-orange-50 border border-orange-200 rounded-lg mb-4">
            <div>
              <div className="font-semibold text-orange-800 flex items-center gap-2">
                <span>🚀</span>
                <span>Reranker Servisi</span>
              </div>
              <p className="text-sm text-orange-700 mt-1">
                Gelişmiş reranker servisi kullan
              </p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={useRerankerService}
                onChange={(e) => onRerankerServiceChange(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-orange-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-orange-600"></div>
            </label>
          </div>

          {/* Reranker Type - Only show when service is enabled */}
          {useRerankerService && (
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                🎯 Reranker Tipi
              </label>
              <select
                value={selectedRerankerType}
                onChange={(e) => onRerankerTypeChange(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500 transition-all bg-white"
              >
                <option value="ms-marco">gte-rerank-v2 (Alibaba)</option>
                <option value="bge">BGE Reranker</option>
              </select>
            </div>
          )}
        </div>

        {/* Session Settings Info */}
        {sessionRagSettings && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">ℹ️</span>
              <h4 className="font-semibold text-blue-800">
                Aktif Oturum Ayarları
              </h4>
            </div>
            <div className="text-sm text-blue-700 space-y-1">
              {sessionRagSettings.model && (
                <div>
                  <strong>Model:</strong> {sessionRagSettings.model}
                </div>
              )}
              {sessionRagSettings.embedding_model && (
                <div>
                  <strong>Embedding:</strong>{" "}
                  {sessionRagSettings.embedding_model}
                </div>
              )}
              {sessionRagSettings.chain_type && (
                <div>
                  <strong>Chain:</strong> {sessionRagSettings.chain_type}
                </div>
              )}
              <div className="text-xs text-blue-600 mt-2">
                💡 Oturum ayarları varsa onlar kullanılır, yoksa yukarıdaki
                seçimler geçerlidir.
              </div>
            </div>
          </div>
        )}

        {/* Save Settings Section */}
        <div className="border-t border-gray-200 pt-6">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                <span>💾</span>
                <span>Ayarları Kaydet</span>
              </h3>
              <p className="text-sm text-gray-600 mt-1">
                Model seçimlerinizi oturuma kaydedin ve bu oturumda sorgu
                yaparken bu ayarlar kullanılsın
              </p>
            </div>

            <button
              onClick={saveModelSettings}
              disabled={isSaving}
              className={`
                px-6 py-3 rounded-lg font-semibold text-white transition-all duration-200 flex items-center gap-2
                ${
                  isSaving
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 hover:shadow-lg transform hover:scale-105"
                }
              `}
            >
              {isSaving ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Kaydediliyor...</span>
                </>
              ) : (
                <>
                  <span>💾</span>
                  <span>Kaydet</span>
                </>
              )}
            </button>
          </div>

          {/* Success Message */}
          {saveSuccess && saveMessage && (
            <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3">
              <span className="text-green-600 text-xl">✅</span>
              <div className="flex-1">
                <p className="text-green-800 font-medium">{saveMessage}</p>
                <p className="text-green-700 text-sm mt-1">
                  Bu oturumda yapılacak sorguların tümünde bu ayarlar
                  kullanılacak.
                </p>
              </div>
            </div>
          )}

          {/* Error Message */}
          {!saveSuccess && saveMessage && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
              <span className="text-red-600 text-xl">❌</span>
              <div className="flex-1">
                <p className="text-red-800 font-medium">{saveMessage}</p>
                <p className="text-red-700 text-sm mt-1">
                  Lütfen tekrar deneyin veya tarayıcı ayarlarınızı kontrol edin.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
