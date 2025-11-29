/**
 * SettingsSaveButton Component - Smart save button with validation
 * Handles unsaved changes detection, save progress, success/error feedback
 */

import React from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Save,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  AlertTriangle,
  Upload,
  RotateCcw,
} from "lucide-react";
import { useRagSettings } from "../../hooks/useRagSettings";

interface SettingsSaveButtonProps {
  sessionId: string;
}

export const SettingsSaveButton: React.FC<SettingsSaveButtonProps> = ({
  sessionId,
}) => {
  const {
    savingSettings,
    hasUnsavedChanges,
    error,
    success,
    saveSettings,
    resetSettings,
    validateSettings,
    clearMessages,
  } = useRagSettings(sessionId);

  const handleSave = async () => {
    // Clear any existing messages
    clearMessages();

    // Validate settings before saving
    const validation = validateSettings();
    if (!validation.isValid) {
      // Validation errors will be shown in the validation results
      return;
    }

    // Save settings
    await saveSettings();
  };

  const validation = validateSettings();

  return (
    <div className="space-y-4">
      {/* Save Actions */}
      <div className="flex items-center justify-between p-4 border-t border-border bg-muted/30">
        <div className="flex items-center gap-3">
          {/* Unsaved Changes Indicator */}
          {hasUnsavedChanges && (
            <Badge variant="outline" className="gap-1">
              <AlertTriangle className="w-3 h-3 text-orange-500" />
              Kaydedilmemiş değişiklikler
            </Badge>
          )}

          {/* Validation Status */}
          {!validation.isValid && (
            <Badge
              variant="outline"
              className="gap-1 border-red-200 text-red-700"
            >
              <AlertCircle className="w-3 h-3" />
              {validation.errors.length} hata
            </Badge>
          )}

          {validation.isValid && hasUnsavedChanges && (
            <Badge
              variant="outline"
              className="gap-1 border-green-200 text-green-700"
            >
              <CheckCircle className="w-3 h-3" />
              Ayarlar hazır
            </Badge>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Reset Button */}
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={resetSettings}
            disabled={
              savingSettings || (!hasUnsavedChanges && validation.isValid)
            }
            className="gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            Sıfırla
          </Button>

          {/* Save Button */}
          <Button
            type="button"
            onClick={handleSave}
            disabled={
              savingSettings || !hasUnsavedChanges || !validation.isValid
            }
            className="gap-2"
            size="default"
          >
            {savingSettings ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Kaydediliyor...
              </>
            ) : (
              <>
                <Save className="w-4 h-4" />
                RAG Ayarlarını Kaydet
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Validation Errors */}
      {!validation.isValid && validation.errors.length > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <div className="font-medium mb-2">
              Ayarlar kaydedilmeden önce şu hatalar düzeltilmelidir:
            </div>
            <ul className="list-disc list-inside space-y-1 text-sm">
              {validation.errors.map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {/* Success Message */}
      {success && (
        <Alert className="border-green-200 bg-green-50 dark:bg-green-900/20">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800 dark:text-green-200">
            {success}
          </AlertDescription>
        </Alert>
      )}

      {/* Error Message */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Help Text */}
      {!hasUnsavedChanges && validation.isValid && !error && !success && (
        <div className="text-center py-6">
          <div className="text-sm text-muted-foreground">
            <CheckCircle className="w-4 h-4 inline mr-2 text-green-500" />
            Tüm RAG ayarları aktif ve kaydedilmiş
          </div>
        </div>
      )}

      {/* Save Instructions */}
      {hasUnsavedChanges && validation.isValid && (
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
          <div className="flex items-start gap-2">
            <Upload className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-blue-800 dark:text-blue-200">
              <div className="font-medium mb-1">Ayarlar Kaydedilmeye Hazır</div>
              <div className="text-xs">
                Değişiklikleriniz henüz kaydedilmedi. "RAG Ayarlarını Kaydet"
                butonuna tıklayarak ayarlarınızı bu ders oturumu için kalıcı
                hale getirin.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Detailed Status for Development */}
      {process.env.NODE_ENV === "development" && (
        <div className="text-xs text-muted-foreground p-2 bg-gray-50 rounded text-center">
          Dev: Saving: {savingSettings ? "true" : "false"} | Unsaved:{" "}
          {hasUnsavedChanges ? "true" : "false"} | Valid:{" "}
          {validation.isValid ? "true" : "false"} | Errors:{" "}
          {validation.errors.length}
        </div>
      )}
    </div>
  );
};

export default SettingsSaveButton;
