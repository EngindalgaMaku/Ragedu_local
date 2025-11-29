"use client";

import React from "react";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import {
  User,
  Calendar,
  Settings,
  Database,
  Zap,
  Info,
  Clock,
  FileText,
  Brain,
} from "lucide-react";

import { useSessionData, useSessionMetadata } from "../../hooks/useSessionData";
import {
  useSessionChunks,
  useSessionRagConfig,
} from "../../hooks/useSessionData";
import {
  useCurrentSession,
  useSessionSettings,
} from "../../stores/sessionStore";

interface SessionInfoProps {
  sessionId: string;
  className?: string;
}

export default function SessionInfo({
  sessionId,
  className,
}: SessionInfoProps) {
  const {
    hasSession,
    hasChunks,
    hasRagConfig,
    chunksCount,
    interactionsCount,
  } = useSessionData();

  const { chunks } = useSessionChunks();
  const { config: ragConfig } = useSessionRagConfig();

  // Get current session and settings from store - FIXED: Use proper hooks instead of require
  const currentSession = useCurrentSession();
  const sessionSettings = useSessionSettings();

  console.log(`🖼️ [SESSION INFO] Rendering with session:`, currentSession);
  console.log(
    `🖼️ [SESSION INFO] Session name: ${currentSession?.name || "No name"}`
  );
  console.log(`🖼️ [SESSION INFO] Has session: ${!!currentSession}`);

  // Format date helper
  const formatDate = (dateString?: string) => {
    if (!dateString) return "Bilinmiyor";
    try {
      return new Date(dateString).toLocaleDateString("tr-TR", {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "Bilinmiyor";
    }
  };

  // Calculate document statistics
  const documentStats = React.useMemo(() => {
    if (!chunks.length) return null;

    const documents = new Map();
    let totalCharacters = 0;
    let llmImprovedCount = 0;

    chunks.forEach((chunk) => {
      const docName = chunk.document_name;
      if (!documents.has(docName)) {
        documents.set(docName, {
          name: docName,
          chunks: 0,
          characters: 0,
          llmImproved: 0,
        });
      }

      const doc = documents.get(docName);
      doc.chunks++;
      doc.characters += chunk.character_count || 0;
      totalCharacters += chunk.character_count || 0;

      if (chunk.chunk_metadata?.llm_improved) {
        doc.llmImproved++;
        llmImprovedCount++;
      }
    });

    return {
      totalDocuments: documents.size,
      totalCharacters,
      llmImprovedCount,
      documents: Array.from(documents.values()),
      averageChunkSize: Math.round(totalCharacters / chunks.length),
    };
  }, [chunks]);

  return (
    <div className={className}>
      <div className="space-y-6">
        {/* Session Basic Info */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <User className="w-4 h-4 text-muted-foreground" />
            <Label className="text-sm font-medium">Oturum Bilgileri</Label>
          </div>

          <div className="space-y-3 pl-6">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Oturum ID:</span>
              <Badge variant="outline" className="font-mono text-xs">
                {sessionId.slice(-8)}
              </Badge>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                Oluşturulma:
              </span>
              <span className="text-sm text-foreground">
                {formatDate(currentSession?.created_at)}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                Son Güncelleme:
              </span>
              <span className="text-sm text-foreground">
                {formatDate(currentSession?.updated_at)}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Durum:</span>
              <Badge variant={hasSession ? "default" : "secondary"}>
                {hasSession ? "Aktif" : "Yükleniyor"}
              </Badge>
            </div>
          </div>
        </div>

        <div className="border-t border-border my-4"></div>

        {/* Session Statistics */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-muted-foreground" />
            <Label className="text-sm font-medium">Veri İstatistikleri</Label>
          </div>

          <div className="space-y-3 pl-6">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                Belge Parçası:
              </span>
              <Badge variant="secondary">
                {chunksCount.toLocaleString("tr-TR")}
              </Badge>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Etkileşim:</span>
              <Badge variant="secondary">
                {interactionsCount.toLocaleString("tr-TR")}
              </Badge>
            </div>

            {documentStats && (
              <>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">
                    Toplam Belge:
                  </span>
                  <Badge variant="secondary">
                    {documentStats.totalDocuments}
                  </Badge>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">
                    Toplam Karakter:
                  </span>
                  <Badge variant="secondary">
                    {documentStats.totalCharacters.toLocaleString("tr-TR")}
                  </Badge>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">
                    Ortalama Parça:
                  </span>
                  <Badge variant="secondary">
                    ~{documentStats.averageChunkSize} karakter
                  </Badge>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">
                    LLM İyileştirme:
                  </span>
                  <Badge variant="secondary" className="gap-1">
                    <Brain className="w-3 h-3" />
                    {documentStats.llmImprovedCount}
                    {chunksCount > 0 && (
                      <span className="text-xs">
                        (%
                        {Math.round(
                          (documentStats.llmImprovedCount / chunksCount) * 100
                        )}
                        )
                      </span>
                    )}
                  </Badge>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="border-t border-border my-4"></div>

        {/* Session Settings */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4 text-muted-foreground" />
            <Label className="text-sm font-medium">Oturum Ayarları</Label>
          </div>

          <div className="space-y-3 pl-6">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                Aşamalı Değerlendirme:
              </span>
              <Badge
                variant={
                  sessionSettings?.enable_progressive_assessment
                    ? "default"
                    : "outline"
                }
              >
                {sessionSettings?.enable_progressive_assessment
                  ? "Açık"
                  : "Kapalı"}
              </Badge>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                Kişiselleştirilmiş Yanıt:
              </span>
              <Badge
                variant={
                  sessionSettings?.enable_personalized_responses
                    ? "default"
                    : "outline"
                }
              >
                {sessionSettings?.enable_personalized_responses
                  ? "Açık"
                  : "Kapalı"}
              </Badge>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                Çok Boyutlu Geri Bildirim:
              </span>
              <Badge
                variant={
                  sessionSettings?.enable_multi_dimensional_feedback
                    ? "default"
                    : "outline"
                }
              >
                {sessionSettings?.enable_multi_dimensional_feedback
                  ? "Açık"
                  : "Kapalı"}
              </Badge>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                Konu Analitiği:
              </span>
              <Badge
                variant={
                  sessionSettings?.enable_topic_analytics
                    ? "default"
                    : "outline"
                }
              >
                {sessionSettings?.enable_topic_analytics ? "Açık" : "Kapalı"}
              </Badge>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">
                Emoji Geri Bildirim:
              </span>
              <Badge
                variant={
                  sessionSettings?.enable_emoji_feedback ? "default" : "outline"
                }
              >
                {sessionSettings?.enable_emoji_feedback ? "Açık" : "Kapalı"}
              </Badge>
            </div>
          </div>
        </div>

        <div className="border-t border-border my-4"></div>

        {/* RAG Configuration */}
        {hasRagConfig && ragConfig && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-muted-foreground" />
              <Label className="text-sm font-medium">RAG Konfigürasyonu</Label>
            </div>

            <div className="space-y-3 pl-6">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">
                  Embedding Model:
                </span>
                <Badge variant="outline" className="font-mono text-xs">
                  {ragConfig.embedding_model || "Varsayılan"}
                </Badge>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">
                  Chunk Strategy:
                </span>
                <Badge variant="outline">
                  {(ragConfig as any).chunk_strategy || "Semantic"}
                </Badge>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">
                  Chunk Size:
                </span>
                <Badge variant="secondary">
                  {(ragConfig as any).chunk_size || 1000}
                </Badge>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Overlap:</span>
                <Badge variant="secondary">
                  {(ragConfig as any).chunk_overlap || 200}
                </Badge>
              </div>

              {ragConfig.retrieval_method && (
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">
                    Retrieval Method:
                  </span>
                  <Badge variant="outline">{ragConfig.retrieval_method}</Badge>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Processing History */}
        {documentStats && documentStats.documents.length > 0 && (
          <>
            <div className="border-t border-border my-4"></div>
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-muted-foreground" />
                <Label className="text-sm font-medium">İşlenmiş Belgeler</Label>
              </div>

              <div className="space-y-2 pl-6 max-h-32 overflow-y-auto">
                {documentStats.documents.slice(0, 5).map((doc, index) => (
                  <div
                    key={doc.name}
                    className="flex justify-between items-center text-xs"
                  >
                    <span className="text-muted-foreground truncate flex-1 mr-2">
                      {doc.name.replace(".md", "")}
                    </span>
                    <div className="flex items-center gap-1 flex-shrink-0">
                      <Badge variant="outline" className="text-xs h-5">
                        {doc.chunks}
                      </Badge>
                      {doc.llmImproved > 0 && (
                        <Badge
                          variant="secondary"
                          className="text-xs h-5 gap-1"
                        >
                          <Brain className="w-2 h-2" />
                          {doc.llmImproved}
                        </Badge>
                      )}
                    </div>
                  </div>
                ))}
                {documentStats.documents.length > 5 && (
                  <div className="text-xs text-muted-foreground text-center pt-1">
                    ... ve {documentStats.documents.length - 5} belge daha
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* Status Summary */}
        <div className="bg-muted/30 p-3 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Info className="w-4 h-4 text-muted-foreground" />
            <Label className="text-xs font-medium text-muted-foreground">
              Sistem Durumu
            </Label>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="text-center">
              <div
                className={`font-medium ${
                  hasSession ? "text-green-600" : "text-orange-600"
                }`}
              >
                {hasSession ? "✓" : "○"} Oturum
              </div>
            </div>
            <div className="text-center">
              <div
                className={`font-medium ${
                  hasChunks ? "text-green-600" : "text-gray-500"
                }`}
              >
                {hasChunks ? "✓" : "○"} Belgeler
              </div>
            </div>
            <div className="text-center">
              <div
                className={`font-medium ${
                  hasRagConfig ? "text-green-600" : "text-gray-500"
                }`}
              >
                {hasRagConfig ? "✓" : "○"} RAG
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
