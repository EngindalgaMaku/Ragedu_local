/**
 * InteractionCard Component
 * Individual interaction display with student query, response, and metadata
 */

import React from "react";
import {
  Clock,
  User,
  BookOpen,
  ExternalLink,
  MessageSquare,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
// Using div instead of Separator component for better compatibility
import type { APRAGInteraction } from "@/lib/api";

interface InteractionCardProps {
  interaction: APRAGInteraction;
  index: number;
  onViewDetails?: (interaction: APRAGInteraction) => void;
  showFullContent?: boolean;
  compact?: boolean;
}

export function InteractionCard({
  interaction,
  index,
  onViewDetails,
  showFullContent = false,
  compact = false,
}: InteractionCardProps) {
  // Get student name with fallback logic from original code
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

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString("tr-TR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  // Truncate text for preview
  const truncateText = (text: string, maxLength: number = 150) => {
    if (showFullContent || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
  };

  // Get response to display (personalized if available, otherwise original)
  const responseText =
    interaction.personalized_response || interaction.original_response;

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4">
        {/* Header with student info and index */}
        <div className="flex items-start gap-3 mb-3">
          <div className="flex-shrink-0 w-7 h-7 bg-primary/10 text-primary rounded-full flex items-center justify-center font-medium text-sm">
            {index}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex flex-col gap-2 mb-2">
              {/* Student info and topic */}
              <div className="flex items-center gap-2 flex-wrap">
                <div className="flex items-center gap-1 text-sm font-medium text-foreground">
                  <User className="w-3 h-3" />
                  <span>{studentName}</span>
                </div>

                {interaction.topic_info && (
                  <Badge variant="secondary" className="text-xs">
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

              {/* Timestamp and processing time */}
              <div className="flex items-center gap-2 text-xs text-muted-foreground flex-wrap">
                <div className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  <span>{formatTimestamp(interaction.timestamp)}</span>
                </div>
                {interaction.processing_time_ms && (
                  <>
                    <span>•</span>
                    <span>{interaction.processing_time_ms}ms</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="space-y-3">
          {/* Question */}
          <div>
            <div className="flex items-center gap-1 mb-1">
              <MessageSquare className="w-4 h-4 text-blue-600" />
              <p className="text-sm font-semibold text-blue-600">Soru</p>
            </div>
            <p className="text-sm text-foreground pl-5">
              {compact
                ? truncateText(interaction.query, 100)
                : interaction.query}
            </p>
          </div>

          {!compact && <div className="h-px bg-border my-3" />}

          {/* Answer */}
          <div>
            <div className="flex items-center gap-1 mb-1">
              <MessageSquare className="w-4 h-4 text-green-600" />
              <p className="text-sm font-semibold text-green-600">Cevap</p>
            </div>
            <div className="text-sm text-foreground pl-5 whitespace-pre-wrap">
              {showFullContent ? responseText : truncateText(responseText, 200)}
            </div>
          </div>

          {/* Sources */}
          {interaction.sources && interaction.sources.length > 0 && (
            <>
              {!compact && <div className="h-px bg-border my-3" />}
              <div>
                <div className="flex items-center gap-1 mb-1.5">
                  <ExternalLink className="w-4 h-4 text-purple-600" />
                  <p className="text-sm font-semibold text-purple-600">
                    Kaynaklar ({interaction.sources.length})
                  </p>
                </div>
                <div className="flex flex-wrap gap-1.5 pl-5">
                  {interaction.sources
                    .slice(0, compact ? 3 : undefined)
                    .map((source, idx) => (
                      <Badge
                        key={idx}
                        variant="outline"
                        className="text-xs"
                        title={source.source}
                      >
                        {source.source}
                        {source.score !== undefined && (
                          <span className="ml-1">
                            ({(source.score * 100).toFixed(1)}%)
                          </span>
                        )}
                      </Badge>
                    ))}
                  {compact && interaction.sources.length > 3 && (
                    <Badge
                      variant="outline"
                      className="text-xs text-muted-foreground"
                    >
                      +{interaction.sources.length - 3} daha
                    </Badge>
                  )}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Actions */}
        {(compact || onViewDetails) && (
          <div className="flex items-center justify-end gap-2 mt-4 pt-3 border-t border-border">
            {onViewDetails && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onViewDetails(interaction)}
                className="text-xs"
              >
                Detayları Görüntüle
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Compact version for list views
export function InteractionCardCompact({
  interaction,
  index,
  onViewDetails,
}: Omit<InteractionCardProps, "compact" | "showFullContent">) {
  return (
    <InteractionCard
      interaction={interaction}
      index={index}
      onViewDetails={onViewDetails}
      compact={true}
      showFullContent={false}
    />
  );
}

// Mobile version with optimized layout
export function InteractionCardMobile({
  interaction,
  index,
  onViewDetails,
}: Omit<InteractionCardProps, "compact" | "showFullContent">) {
  const studentName =
    interaction.first_name && interaction.last_name
      ? `${interaction.first_name} ${interaction.last_name}`
      : interaction.first_name ||
        interaction.last_name ||
        interaction.student_name ||
        "Bilinmeyen Öğrenci";

  return (
    <Card className="border-l-4 border-l-primary">
      <CardContent className="p-4 space-y-3">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-primary/10 text-primary rounded-full flex items-center justify-center font-medium text-xs">
              {index}
            </div>
            <span className="text-sm font-medium text-foreground">
              {studentName}
            </span>
          </div>
          <span className="text-xs text-muted-foreground">
            {new Date(interaction.timestamp).toLocaleString("tr-TR")}
          </span>
        </div>

        {/* Topic */}
        {interaction.topic_info && (
          <Badge variant="secondary" className="text-xs">
            📚 {interaction.topic_info.title}
          </Badge>
        )}

        {/* Question */}
        <div>
          <p className="text-xs font-semibold text-blue-600 mb-1">Soru</p>
          <p className="text-sm text-foreground">
            {interaction.query.length > 100
              ? interaction.query.substring(0, 100) + "..."
              : interaction.query}
          </p>
        </div>

        {/* Action */}
        {onViewDetails && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onViewDetails(interaction)}
            className="w-full justify-center text-xs"
          >
            Detayları Görüntüle
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

export default InteractionCard;
