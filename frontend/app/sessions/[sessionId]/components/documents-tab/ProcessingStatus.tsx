"use client";

import React from "react";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Loader2,
  CheckCircle,
  FileText,
  Settings,
  Clock,
  AlertCircle,
} from "lucide-react";

interface ProcessingStatusProps {
  isProcessing: boolean;
  processingStep: string;
  selectedFiles: string[];
  uploadProgress: Record<string, number>;
  className?: string;
}

export default function ProcessingStatus({
  isProcessing,
  processingStep,
  selectedFiles,
  uploadProgress,
  className,
}: ProcessingStatusProps) {
  // Calculate overall progress
  const totalProgress = React.useMemo(() => {
    if (!selectedFiles.length) return 0;

    const progressValues = selectedFiles.map(
      (file) => uploadProgress[file] || 0
    );
    const averageProgress =
      progressValues.reduce((sum, progress) => sum + progress, 0) /
      progressValues.length;

    return Math.round(averageProgress);
  }, [selectedFiles, uploadProgress]);

  // Processing step icons
  const getStepIcon = (step: string) => {
    if (step.includes("tamamlandı")) {
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    } else if (step.includes("hazırlanıyor")) {
      return <Settings className="w-5 h-5 text-blue-500" />;
    } else if (step.includes("okunuyor")) {
      return <FileText className="w-5 h-5 text-orange-500" />;
    } else if (
      step.includes("oluşturuluyor") ||
      step.includes("hesaplanıyor")
    ) {
      return <Loader2 className="w-5 h-5 text-purple-500 animate-spin" />;
    } else if (step.includes("güncelleniyor")) {
      return <Settings className="w-5 h-5 text-indigo-500" />;
    } else {
      return <Loader2 className="w-5 h-5 text-primary animate-spin" />;
    }
  };

  // Get estimated completion time
  const getEstimatedTime = () => {
    const remainingProgress = 100 - totalProgress;
    if (remainingProgress <= 0) return "Tamamlanıyor...";

    // Rough estimation based on progress
    const estimatedMinutes = Math.ceil((remainingProgress / 100) * 5);
    return `~${estimatedMinutes} dakika kaldı`;
  };

  if (!isProcessing && !processingStep) {
    return null;
  }

  return (
    <div className={className}>
      <div className="space-y-4">
        {/* Processing Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {getStepIcon(processingStep)}
            <div>
              <h3 className="font-medium text-foreground">
                {isProcessing ? "İşlem Devam Ediyor" : "İşlem Tamamlandı"}
              </h3>
              <p className="text-sm text-muted-foreground">
                {selectedFiles.length} dosya işleniyor
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge
              variant={totalProgress === 100 ? "default" : "secondary"}
              className="gap-1"
            >
              {totalProgress === 100 ? (
                <CheckCircle className="w-3 h-3" />
              ) : (
                <Clock className="w-3 h-3" />
              )}
              %{totalProgress}
            </Badge>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="space-y-2">
          <Progress value={totalProgress} className="h-2" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{processingStep}</span>
            <span>{getEstimatedTime()}</span>
          </div>
        </div>

        {/* File Progress Details */}
        {selectedFiles.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Dosya İlerlemesi
            </h4>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {selectedFiles.map((file) => {
                const progress = uploadProgress[file] || 0;
                const isCompleted = progress === 100;

                return (
                  <div
                    key={file}
                    className="flex items-center gap-3 p-2 rounded-lg bg-muted/30"
                  >
                    <div className="flex-shrink-0">
                      {isCompleted ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      ) : (
                        <Loader2 className="w-4 h-4 text-primary animate-spin" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-foreground truncate">
                        {file.replace(".md", "")}
                      </div>
                      <div className="w-full bg-muted rounded-full h-1.5 mt-1">
                        <div
                          className="bg-primary h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                    </div>
                    <div className="flex-shrink-0">
                      <span className="text-xs font-medium text-muted-foreground">
                        {progress}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Processing Steps Guide */}
        {isProcessing && (
          <Alert className="border-blue-500/20 bg-blue-500/10">
            <AlertCircle className="h-4 w-4 text-blue-600" />
            <AlertDescription className="text-sm text-blue-700">
              <div className="space-y-1">
                <p className="font-medium">İşlem Aşamaları:</p>
                <div className="text-xs space-y-0.5">
                  <div
                    className={
                      processingStep.includes("hazırlanıyor")
                        ? "font-semibold"
                        : ""
                    }
                  >
                    1. Konfigürasyon hazırlanıyor
                  </div>
                  <div
                    className={
                      processingStep.includes("okunuyor") ? "font-semibold" : ""
                    }
                  >
                    2. Dosyalar okunuyor
                  </div>
                  <div
                    className={
                      processingStep.includes("oluşturuluyor")
                        ? "font-semibold"
                        : ""
                    }
                  >
                    3. Metin parçaları oluşturuluyor
                  </div>
                  <div
                    className={
                      processingStep.includes("hesaplanıyor")
                        ? "font-semibold"
                        : ""
                    }
                  >
                    4. Embedding vektörleri hesaplanıyor
                  </div>
                  <div
                    className={
                      processingStep.includes("güncelleniyor")
                        ? "font-semibold"
                        : ""
                    }
                  >
                    5. Veritabanı güncelleniyor
                  </div>
                </div>
              </div>
            </AlertDescription>
          </Alert>
        )}

        {/* Completion Message */}
        {totalProgress === 100 && (
          <Alert className="border-green-500/20 bg-green-500/10">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <AlertDescription className="text-sm text-green-700">
              <p className="font-medium">İşlem başarıyla tamamlandı!</p>
              <p className="text-xs mt-1">
                {selectedFiles.length} dosya işlendi. Sayfa otomatik olarak
                güncellenecek.
              </p>
            </AlertDescription>
          </Alert>
        )}

        {/* Background Processing Note */}
        {isProcessing && (
          <div className="text-xs text-muted-foreground bg-muted/30 p-3 rounded-lg">
            💡 Bu sekmeyi kapatabilirsiniz - İşlem arka planda devam edecek ve
            tamamlandığında sonuçlar burada görünecektir.
          </div>
        )}
      </div>
    </div>
  );
}
