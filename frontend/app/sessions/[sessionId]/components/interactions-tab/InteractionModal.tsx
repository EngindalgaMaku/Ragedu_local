/**
 * InteractionModal Component
 * Detailed interaction viewer with full query, response, feedback, and analytics
 */

import React from "react";
import {
  Clock,
  User,
  BookOpen,
  ExternalLink,
  MessageSquare,
  X,
  Copy,
  CheckCircle,
  AlertCircle,
  Info,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import type { APRAGInteraction } from "@/lib/api";

interface InteractionModalProps {
  interaction: APRAGInteraction | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function InteractionModal({
  interaction,
  open,
  onOpenChange,
}: InteractionModalProps) {
  const [copiedText, setCopiedText] = React.useState<string | null>(null);

  if (!interaction) return null;

  // Get student name with fallback logic
  const getStudentName = () => {
    if (interaction.first_name && interaction.last_name) {
      return `${interaction.first_name} ${interaction.last_name}`;
    }
    if (interaction.first_name || interaction.last_name) {
      return interaction.first_name || interaction.last_name;
    }
    if (
      interaction.student_name &&
      !interaction.student_name.startsWith("Öğrenci (ID:")
    ) {
      return interaction.student_name;
    }
    return "Bilinmeyen Öğrenci";
  };

  const studentName = getStudentName();

  // Copy to clipboard functionality
  const handleCopy = async (text: string, type: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedText(type);
      setTimeout(() => setCopiedText(null), 2000);
    } catch (err) {
      console.error("Failed to copy to clipboard:", err);
    }
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString("tr-TR", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  // Get response to display
  const responseText =
    interaction.personalized_response || interaction.original_response;
  const hasPersonalizedResponse = !!interaction.personalized_response;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader className="space-y-3">
          <div className="flex items-start justify-between">
            <div>
              <DialogTitle className="text-xl">Etkileşim Detayları</DialogTitle>
              <DialogDescription>
                Öğrenci sorgusu ve sistem yanıtı detaylı analizi
              </DialogDescription>
            </div>
          </div>

          {/* Student and metadata info */}
          <div className="flex flex-wrap gap-3 pt-2">
            <div className="flex items-center gap-2">
              <User className="w-4 h-4 text-muted-foreground" />
              <span className="font-medium">{studentName}</span>
            </div>

            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">
                {formatTimestamp(interaction.timestamp)}
              </span>
            </div>

            {interaction.processing_time_ms && (
              <Badge variant="outline">
                {interaction.processing_time_ms}ms
              </Badge>
            )}

            {interaction.topic_info && (
              <Badge variant="secondary">
                <BookOpen className="w-3 h-3 mr-1" />
                {interaction.topic_info.title}
                {interaction.topic_info.confidence && (
                  <span className="ml-1 opacity-70">
                    ({Math.round(interaction.topic_info.confidence * 100)}%)
                  </span>
                )}
              </Badge>
            )}
          </div>
        </DialogHeader>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto space-y-6">
          {/* Question Section */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-base">
                  <MessageSquare className="w-4 h-4 text-blue-600" />
                  <span className="text-blue-600">Öğrenci Sorusu</span>
                </CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopy(interaction.query, "question")}
                  className="text-xs"
                >
                  {copiedText === "question" ? (
                    <CheckCircle className="w-3 h-3" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm max-w-none">
                <p className="text-foreground whitespace-pre-wrap text-sm leading-relaxed">
                  {interaction.query}
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Response Section */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-base">
                  <MessageSquare className="w-4 h-4 text-green-600" />
                  <span className="text-green-600">
                    {hasPersonalizedResponse
                      ? "Kişiselleştirilmiş Yanıt"
                      : "Sistem Yanıtı"}
                  </span>
                  {hasPersonalizedResponse && (
                    <Badge variant="secondary" className="text-xs">
                      ✨ APRAG
                    </Badge>
                  )}
                </CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopy(responseText, "response")}
                  className="text-xs"
                >
                  {copiedText === "response" ? (
                    <CheckCircle className="w-3 h-3" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm max-w-none">
                <div className="text-foreground whitespace-pre-wrap text-sm leading-relaxed">
                  {responseText}
                </div>
              </div>

              {/* Show original response if personalized exists */}
              {hasPersonalizedResponse && interaction.original_response && (
                <div className="mt-4 p-3 bg-muted/50 rounded-lg border-l-4 border-l-muted-foreground">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-muted-foreground">
                      Orijinal Yanıt:
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        handleCopy(interaction.original_response!, "original")
                      }
                      className="text-xs h-6"
                    >
                      {copiedText === "original" ? (
                        <CheckCircle className="w-3 h-3" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                    </Button>
                  </div>
                  <div className="text-sm text-muted-foreground whitespace-pre-wrap">
                    {interaction.original_response}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Sources Section */}
          {interaction.sources && interaction.sources.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <ExternalLink className="w-4 h-4 text-purple-600" />
                  <span className="text-purple-600">
                    Kaynaklar ({interaction.sources.length})
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {interaction.sources.map((source, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-3 p-3 bg-muted/30 rounded-lg"
                    >
                      <div className="flex-shrink-0 w-6 h-6 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center text-xs font-medium">
                        {idx + 1}
                      </div>
                      <div className="flex-1 space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">
                            {source.source}
                          </span>
                          {source.score !== undefined && (
                            <Badge variant="outline" className="text-xs">
                              {(source.score * 100).toFixed(1)}% benzerlik
                            </Badge>
                          )}
                        </div>
                        {source.chunk_text && (
                          <div className="text-xs text-muted-foreground">
                            <p className="line-clamp-2">{source.chunk_text}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Technical Details */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Info className="w-4 h-4 text-blue-600" />
                <span className="text-blue-600">Teknik Bilgiler</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium">Etkileşim ID:</span>
                  <p className="text-muted-foreground font-mono">
                    {interaction.interaction_id}
                  </p>
                </div>
                <div>
                  <span className="font-medium">Oturum ID:</span>
                  <p className="text-muted-foreground font-mono">
                    {interaction.session_id}
                  </p>
                </div>
                <div>
                  <span className="font-medium">Kullanıcı ID:</span>
                  <p className="text-muted-foreground font-mono">
                    {interaction.user_id}
                  </p>
                </div>
                {interaction.model_used && (
                  <div>
                    <span className="font-medium">Kullanılan Model:</span>
                    <p className="text-muted-foreground">
                      {interaction.model_used}
                    </p>
                  </div>
                )}
                {interaction.chain_type && (
                  <div>
                    <span className="font-medium">Zincir Türü:</span>
                    <p className="text-muted-foreground">
                      {interaction.chain_type}
                    </p>
                  </div>
                )}
                <div>
                  <span className="font-medium">Oluşturulma:</span>
                  <p className="text-muted-foreground">
                    {formatTimestamp(interaction.timestamp)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* APRAG Integration Info */}
          {hasPersonalizedResponse && (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Bu yanıt APRAG (Adaptive Personalized RAG) sistemi tarafından
                öğrencinin öğrenme profiline göre kişiselleştirilmiştir.
              </AlertDescription>
            </Alert>
          )}
        </div>

        {/* Footer with actions */}
        <div className="flex items-center justify-end gap-2 pt-4 border-t border-border">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Kapat
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default InteractionModal;
