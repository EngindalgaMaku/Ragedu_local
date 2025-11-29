"use client";

import React, { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
  X,
  FileText,
  Sparkles,
  Save,
  RotateCcw,
  Copy,
  Check,
  AlertCircle,
  Zap,
  Brain,
  Eye,
  EyeOff,
  Hash,
  Calendar,
  User,
} from "lucide-react";
import type { Chunk } from "../../types/chunks.types";
import { useChunkImprovement } from "../../hooks/useChunksData";

export interface ChunkModalProps {
  chunk: Chunk | null;
  isOpen: boolean;
  onClose: () => void;
  onSave?: (chunkId: string, improvedContent: string) => Promise<void>;
  sessionId: string;
  disabled?: boolean;
}

interface ImprovementConfig {
  llm_provider: "ollama" | "grok" | "openai";
  llm_model: string;
  improvement_type: "clarity" | "completeness" | "coherence" | "all";
}

const IMPROVEMENT_TYPES = [
  { value: "all", label: "Genel İyileştirme", desc: "Tüm alanları iyileştir" },
  { value: "clarity", label: "Netlik", desc: "Daha anlaşılır yap" },
  { value: "completeness", label: "Tamlık", desc: "Eksik bilgileri tamamla" },
  { value: "coherence", label: "Tutarlılık", desc: "Mantıksal akışı düzenle" },
] as const;

const LLM_PROVIDERS = [
  {
    value: "ollama",
    label: "Ollama (Yerel)",
    models: ["llama3.2:3b", "mistral:7b", "phi3:mini"],
  },
  {
    value: "grok",
    label: "Grok (Cloud)",
    models: ["grok-beta", "grok-vision-beta"],
  },
  { value: "openai", label: "OpenAI", models: ["gpt-4", "gpt-3.5-turbo"] },
] as const;

export default function ChunkModal({
  chunk,
  isOpen,
  onClose,
  onSave,
  sessionId,
  disabled = false,
}: ChunkModalProps) {
  const [activeTab, setActiveTab] = useState<"view" | "improve" | "compare">(
    "view"
  );
  const [editedContent, setEditedContent] = useState("");
  const [originalContent, setOriginalContent] = useState("");
  const [improvementConfig, setImprovementConfig] = useState<ImprovementConfig>(
    {
      llm_provider: "ollama",
      llm_model: "llama3.2:3b",
      improvement_type: "all",
    }
  );
  const [showOriginal, setShowOriginal] = useState(true);
  const [isCopied, setIsCopied] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  const { improveChunk, isImproving } = useChunkImprovement(sessionId);

  // Initialize content when chunk changes
  useEffect(() => {
    if (chunk) {
      setOriginalContent(chunk.chunk_text);
      setEditedContent(chunk.chunk_text);
      setHasUnsavedChanges(false);

      // Set default tab based on chunk status
      if (chunk.chunk_metadata?.llm_improved) {
        setActiveTab("compare");
      } else {
        setActiveTab("view");
      }
    }
  }, [chunk]);

  // Track content changes
  useEffect(() => {
    if (chunk && editedContent !== chunk.chunk_text) {
      setHasUnsavedChanges(true);
    } else {
      setHasUnsavedChanges(false);
    }
  }, [editedContent, chunk]);

  const handleClose = () => {
    if (hasUnsavedChanges) {
      const confirm = window.confirm(
        "Kaydedilmemiş değişiklikleriniz var. Çıkmak istediğinizden emin misiniz?"
      );
      if (!confirm) return;
    }
    onClose();
  };

  const handleSave = async () => {
    if (!chunk || !onSave || !hasUnsavedChanges) return;

    try {
      await onSave(chunk.id, editedContent);
      setHasUnsavedChanges(false);
    } catch (error) {
      console.error("Failed to save chunk:", error);
    }
  };

  const handleImprove = async () => {
    if (!chunk) return;

    try {
      await improveChunk(chunk.id, improvementConfig);
      setActiveTab("compare");
    } catch (error) {
      console.error("Failed to improve chunk:", error);
    }
  };

  const handleReset = () => {
    if (chunk) {
      setEditedContent(chunk.chunk_text);
      setHasUnsavedChanges(false);
    }
  };

  const handleCopy = async () => {
    if (editedContent) {
      try {
        await navigator.clipboard.writeText(editedContent);
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
      } catch (error) {
        console.error("Failed to copy:", error);
      }
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleString("tr-TR");
    } catch {
      return dateString;
    }
  };

  if (!chunk) return null;

  const isImproved = chunk.chunk_metadata?.llm_improved;
  const selectedProvider = LLM_PROVIDERS.find(
    (p) => p.value === improvementConfig.llm_provider
  );

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-6xl h-[90vh] flex flex-col">
        <DialogHeader className="flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FileText className="w-5 h-5 text-primary" />
              <div>
                <DialogTitle className="text-lg">
                  Parça #{chunk.chunk_index}
                </DialogTitle>
                <p className="text-sm text-muted-foreground mt-1">
                  {chunk.document_name.replace(".md", "")}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {isImproved && (
                <Badge variant="secondary" className="gap-1">
                  <Sparkles className="w-3 h-3" />
                  İyileştirilmiş
                </Badge>
              )}

              <Badge variant="outline" className="gap-1">
                <Hash className="w-3 h-3" />
                {chunk.character_count.toLocaleString("tr-TR")} karakter
              </Badge>

              <Button variant="ghost" size="sm" onClick={handleClose}>
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </DialogHeader>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 border-b border-border">
          <Button
            variant={activeTab === "view" ? "default" : "ghost"}
            size="sm"
            onClick={() => setActiveTab("view")}
            className="gap-2"
          >
            <Eye className="w-4 h-4" />
            Görüntüle
          </Button>

          <Button
            variant={activeTab === "improve" ? "default" : "ghost"}
            size="sm"
            onClick={() => setActiveTab("improve")}
            className="gap-2"
            disabled={isImproved}
          >
            <Sparkles className="w-4 h-4" />
            İyileştir
          </Button>

          {isImproved && (
            <Button
              variant={activeTab === "compare" ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveTab("compare")}
              className="gap-2"
            >
              <Eye className="w-4 h-4" />
              Karşılaştır
            </Button>
          )}

          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="gap-1"
            >
              {isCopied ? (
                <>
                  <Check className="w-3 h-3" />
                  Kopyalandı
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  Kopyala
                </>
              )}
            </Button>

            {hasUnsavedChanges && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleSave}
                disabled={disabled}
                className="gap-1"
              >
                <Save className="w-3 h-3" />
                Kaydet
              </Button>
            )}
          </div>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-hidden">
          {activeTab === "view" && (
            <div className="h-full overflow-y-auto p-4">
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <div className="bg-muted/30 rounded-lg p-4 border border-border font-mono text-sm whitespace-pre-wrap">
                  {chunk.chunk_text}
                </div>
              </div>

              {/* Metadata */}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Temel Bilgiler</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm space-y-2">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">
                        Parça İndeksi:
                      </span>
                      <span>#{chunk.chunk_index}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">
                        Karakter Sayısı:
                      </span>
                      <span>
                        {chunk.character_count.toLocaleString("tr-TR")}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Oluşturma:</span>
                      <span>{formatDate(chunk.created_at)}</span>
                    </div>
                    {chunk.language && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Dil:</span>
                        <span>{chunk.language}</span>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {chunk.quality_metrics && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">
                        Kalite Metrikleri
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="text-sm space-y-2">
                      {chunk.quality_metrics.coherence_score && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">
                            Tutarlılık:
                          </span>
                          <span>
                            {Math.round(
                              chunk.quality_metrics.coherence_score * 100
                            )}
                            %
                          </span>
                        </div>
                      )}
                      {chunk.quality_metrics.completeness_score && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Tamlık:</span>
                          <span>
                            {Math.round(
                              chunk.quality_metrics.completeness_score * 100
                            )}
                            %
                          </span>
                        </div>
                      )}
                      {chunk.quality_metrics.readability_score && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">
                            Okunabilirlik:
                          </span>
                          <span>
                            {Math.round(
                              chunk.quality_metrics.readability_score * 100
                            )}
                            %
                          </span>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          )}

          {activeTab === "improve" && !isImproved && (
            <div className="h-full overflow-y-auto p-4">
              <div className="max-w-2xl mx-auto space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Brain className="w-5 h-5 text-violet-600" />
                      LLM İyileştirme Ayarları
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Provider Selection */}
                    <div>
                      <label className="text-sm font-medium mb-2 block">
                        AI Sağlayıcı
                      </label>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                        {LLM_PROVIDERS.map((provider) => (
                          <Button
                            key={provider.value}
                            variant={
                              improvementConfig.llm_provider === provider.value
                                ? "default"
                                : "outline"
                            }
                            onClick={() =>
                              setImprovementConfig((prev) => ({
                                ...prev,
                                llm_provider: provider.value,
                                llm_model: provider.models[0],
                              }))
                            }
                            className="h-auto p-3 text-left"
                          >
                            <div>
                              <div className="font-medium">
                                {provider.label}
                              </div>
                              <div className="text-xs opacity-70">
                                {provider.models.length} model
                              </div>
                            </div>
                          </Button>
                        ))}
                      </div>
                    </div>

                    {/* Model Selection */}
                    <div>
                      <label className="text-sm font-medium mb-2 block">
                        Model
                      </label>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {selectedProvider?.models.map((model) => (
                          <Button
                            key={model}
                            variant={
                              improvementConfig.llm_model === model
                                ? "default"
                                : "outline"
                            }
                            onClick={() =>
                              setImprovementConfig((prev) => ({
                                ...prev,
                                llm_model: model,
                              }))
                            }
                            size="sm"
                          >
                            {model}
                          </Button>
                        ))}
                      </div>
                    </div>

                    {/* Improvement Type */}
                    <div>
                      <label className="text-sm font-medium mb-2 block">
                        İyileştirme Türü
                      </label>
                      <div className="space-y-2">
                        {IMPROVEMENT_TYPES.map((type) => (
                          <Button
                            key={type.value}
                            variant={
                              improvementConfig.improvement_type === type.value
                                ? "default"
                                : "outline"
                            }
                            onClick={() =>
                              setImprovementConfig((prev) => ({
                                ...prev,
                                improvement_type: type.value,
                              }))
                            }
                            className="w-full h-auto p-3 text-left justify-start"
                          >
                            <div>
                              <div className="font-medium">{type.label}</div>
                              <div className="text-xs opacity-70">
                                {type.desc}
                              </div>
                            </div>
                          </Button>
                        ))}
                      </div>
                    </div>

                    <Button
                      onClick={handleImprove}
                      disabled={isImproving(chunk.id) || disabled}
                      className="w-full gap-2"
                    >
                      {isImproving(chunk.id) ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                          İyileştiriliyor...
                        </>
                      ) : (
                        <>
                          <Zap className="w-4 h-4" />
                          İyileştirmeyi Başlat
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          {activeTab === "compare" && isImproved && (
            <div className="h-full overflow-y-auto p-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
                <Card className="flex flex-col">
                  <CardHeader className="flex-shrink-0">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <EyeOff className="w-4 h-4" />
                      Orijinal İçerik
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 overflow-hidden">
                    <div className="h-full overflow-y-auto bg-muted/30 rounded-lg p-4 border border-border">
                      <pre className="text-sm font-mono whitespace-pre-wrap">
                        {originalContent}
                      </pre>
                    </div>
                  </CardContent>
                </Card>

                <Card className="flex flex-col">
                  <CardHeader className="flex-shrink-0">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-violet-600" />
                      İyileştirilmiş İçerik
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 overflow-hidden">
                    <Textarea
                      value={editedContent}
                      onChange={(e) => setEditedContent(e.target.value)}
                      className="h-full resize-none font-mono text-sm"
                      placeholder="İyileştirilmiş içerik..."
                    />
                  </CardContent>
                </Card>
              </div>

              {chunk.chunk_metadata && (
                <Card className="mt-4">
                  <CardHeader>
                    <CardTitle className="text-sm">
                      İyileştirme Detayları
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm space-y-2">
                    {chunk.chunk_metadata.improvement_model && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Model:</span>
                        <span>{chunk.chunk_metadata.improvement_model}</span>
                      </div>
                    )}
                    {chunk.chunk_metadata.improvement_timestamp && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">
                          İyileştirme Tarihi:
                        </span>
                        <span>
                          {formatDate(
                            chunk.chunk_metadata.improvement_timestamp
                          )}
                        </span>
                      </div>
                    )}
                    {chunk.chunk_metadata.original_length &&
                      chunk.chunk_metadata.improved_length && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">
                            Uzunluk Değişimi:
                          </span>
                          <span>
                            {chunk.chunk_metadata.original_length} →{" "}
                            {chunk.chunk_metadata.improved_length}
                            {chunk.chunk_metadata.improved_length >
                              chunk.chunk_metadata.original_length && (
                              <span className="ml-1 text-green-600">
                                (+
                                {Math.round(
                                  (chunk.chunk_metadata.improved_length /
                                    chunk.chunk_metadata.original_length -
                                    1) *
                                    100
                                )}
                                %)
                              </span>
                            )}
                          </span>
                        </div>
                      )}
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="flex-shrink-0 flex items-center justify-between pt-4 border-t border-border">
          <div className="flex items-center gap-2">
            {hasUnsavedChanges && (
              <div className="flex items-center gap-2 text-sm text-amber-600">
                <AlertCircle className="w-4 h-4" />
                Kaydedilmemiş değişiklikler
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            {hasUnsavedChanges && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleReset}
                className="gap-1"
              >
                <RotateCcw className="w-3 h-3" />
                Sıfırla
              </Button>
            )}

            <Button variant="outline" onClick={handleClose}>
              Kapat
            </Button>

            {hasUnsavedChanges && onSave && (
              <Button
                onClick={handleSave}
                disabled={disabled}
                className="gap-1"
              >
                <Save className="w-3 h-3" />
                Kaydet
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
