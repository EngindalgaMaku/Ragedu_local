"use client";

import React, { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent } from "@/components/ui/card";
import {
  Upload,
  FileText,
  CheckCircle,
  AlertCircle,
  Settings,
  Zap,
  Home,
  Brain,
  Scissors,
  Layers,
  Database,
  HelpCircle,
  Info,
} from "lucide-react";

interface DocumentUploadSectionProps {
  sessionId: string;
  availableFiles: string[];
  processedFiles: Set<string>;
  selectedFiles: string[];
  onFileUpload: (files: string[], config?: any) => Promise<void>;
  disabled?: boolean;
  loading?: boolean;
}

export default function DocumentUploadSection({
  sessionId,
  availableFiles,
  processedFiles,
  selectedFiles,
  onFileUpload,
  disabled = false,
  loading = false,
}: DocumentUploadSectionProps) {
  const [localSelectedFiles, setLocalSelectedFiles] =
    useState<string[]>(selectedFiles);

  // Update local selected files when prop changes
  React.useEffect(() => {
    setLocalSelectedFiles(selectedFiles);
  }, [selectedFiles]);

  // Normalize filename for display
  const normalizeDisplayName = (filename: string): string => {
    return filename.replace(/\.md$/i, "");
  };

  // Handle file selection toggle
  const handleFileToggle = useCallback((filename: string) => {
    setLocalSelectedFiles((prev) =>
      prev.includes(filename)
        ? prev.filter((f) => f !== filename)
        : [...prev, filename]
    );
  }, []);

  // Handle select all toggle
  const handleSelectAll = useCallback(() => {
    if (localSelectedFiles.length === availableFiles.length) {
      setLocalSelectedFiles([]);
    } else {
      setLocalSelectedFiles([...availableFiles]);
    }
  }, [localSelectedFiles.length, availableFiles]);

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (localSelectedFiles.length === 0) {
      return;
    }

    const config = {
      chunk_strategy: "lightweight",
      chunk_size: 800,
      chunk_overlap: 100,
      embedding_model: "nomic-embed-text",
    };

    await onFileUpload(localSelectedFiles, config);
  };

  return (
    <div className="space-y-6">
      {/* File Selection */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-muted-foreground" />
            <h3 className="text-sm font-medium">Markdown Dosyaları</h3>
            {availableFiles.length > 0 && (
              <Badge variant="secondary" className="text-xs">
                {availableFiles.length} mevcut
              </Badge>
            )}
          </div>
          {availableFiles.length > 1 && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleSelectAll}
              disabled={disabled || loading}
              className="text-xs h-8"
            >
              {localSelectedFiles.length === availableFiles.length
                ? "Hiçbirini Seçme"
                : "Tümünü Seç"}
            </Button>
          )}
        </div>

        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent mx-auto mb-3"></div>
            <p className="text-sm text-muted-foreground">
              Dosyalar yükleniyor...
            </p>
          </div>
        ) : availableFiles.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm font-medium">İşlenecek dosya bulunamadı</p>
            <p className="text-xs">Tüm markdown dosyaları zaten işlenmiş</p>
          </div>
        ) : (
          <Card>
            <CardContent className="p-4">
              <div className="max-h-64 overflow-y-auto space-y-3">
                {availableFiles.map((filename) => (
                  <div
                    key={filename}
                    className="flex items-start gap-3 p-3 rounded-lg hover:bg-muted/50 transition-colors border border-transparent hover:border-border"
                  >
                    <Checkbox
                      id={`file-${filename}`}
                      checked={localSelectedFiles.includes(filename)}
                      onCheckedChange={() => handleFileToggle(filename)}
                      disabled={disabled}
                      className="mt-0.5 flex-shrink-0"
                    />
                    <label
                      htmlFor={`file-${filename}`}
                      className="flex-1 cursor-pointer"
                    >
                      <div className="font-medium text-foreground text-sm">
                        {normalizeDisplayName(filename)}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {filename}
                      </div>
                    </label>
                    <div className="flex items-center gap-1 flex-shrink-0">
                      <FileText className="w-4 h-4 text-muted-foreground" />
                    </div>
                  </div>
                ))}
              </div>

              {localSelectedFiles.length > 0 && (
                <div className="mt-4 p-3 bg-primary/10 rounded-lg">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-primary" />
                    <span className="text-sm font-medium text-primary">
                      {localSelectedFiles.length} dosya seçili
                    </span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>

      {/* Processing Settings Information */}
      <Card className="bg-gradient-to-r from-blue-50/50 to-indigo-50/50 dark:from-blue-950/20 dark:to-indigo-950/20 border-blue-200/50 dark:border-blue-800/50">
        <CardContent className="p-6">
          <div className="flex items-center gap-2 text-blue-800 dark:text-blue-200 mb-4">
            <Settings className="w-5 h-5" />
            <h3 className="text-lg font-semibold">Sistem İşleme Ayarları</h3>
          </div>
          <p className="text-sm text-blue-600 dark:text-blue-300 mb-6">
            Bu oturumdaki tüm belgeler aşağıdaki ayarlarla işlenmektedir
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Chunking Strategy */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Scissors className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <h4 className="font-semibold text-foreground">
                  Parçalama Stratejisi
                </h4>
                <HelpCircle className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="bg-blue-100/50 dark:bg-blue-900/20 rounded-lg p-3 border border-blue-200/30 dark:border-blue-700/30">
                <div className="font-medium text-blue-800 dark:text-blue-200 mb-1">
                  Lightweight Turkish
                </div>
                <p className="text-xs text-blue-600 dark:text-blue-300 leading-relaxed">
                  Türkçe dil yapısına optimize edilmiş akıllı parçalama. Cümle
                  sınırlarını korur, anlamsal bütünlüğü sağlar ve Türkçe dil
                  bilgisine uygun şekilde metin parçalar.
                </p>
              </div>
            </div>

            {/* Chunk Size */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-green-600 dark:text-green-400" />
                <h4 className="font-semibold text-foreground">Parça Boyutu</h4>
                <HelpCircle className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="bg-green-100/50 dark:bg-green-900/20 rounded-lg p-3 border border-green-200/30 dark:border-green-700/30">
                <div className="font-medium text-green-800 dark:text-green-200 mb-1">
                  Anlamsal (~400-1200 karakter)
                </div>
                <p className="text-xs text-green-600 dark:text-green-300 leading-relaxed">
                  İçerik yapısına göre dinamik boyutlandırma. Paragraf ve cümle
                  bütünlüğünü korurken optimal arama ve anlam çıkarma
                  performansı için ideal boyutta parçalar oluşturur.
                </p>
              </div>
            </div>

            {/* Chunk Overlap */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                <h4 className="font-semibold text-foreground">
                  Parça Çakışması
                </h4>
                <HelpCircle className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="bg-purple-100/50 dark:bg-purple-900/20 rounded-lg p-3 border border-purple-200/30 dark:border-purple-700/30">
                <div className="font-medium text-purple-800 dark:text-purple-200 mb-1">
                  Otomatik (cümle bazlı)
                </div>
                <p className="text-xs text-purple-600 dark:text-purple-300 leading-relaxed">
                  Parçalar arasında anlamsal süreklilik sağlar. Cümle
                  sınırlarında doğal çakışma oluşturarak bilgi kaybını önler ve
                  sorgu-cevap kalitesini artırır.
                </p>
              </div>
            </div>

            {/* Embedding Model */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                <h4 className="font-semibold text-foreground">
                  Embedding Modeli
                </h4>
                <HelpCircle className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="bg-amber-100/50 dark:bg-amber-900/20 rounded-lg p-3 border border-amber-200/30 dark:border-amber-700/30">
                <div className="font-medium text-amber-800 dark:text-amber-200 mb-1">
                  nomic-embed-text
                </div>
                <p className="text-xs text-amber-600 dark:text-amber-300 leading-relaxed">
                  Yüksek performanslı çok dilli embedding modeli. Türkçe
                  metinler için optimize edilmiş anlamsal vektör temsilciler
                  oluşturarak arama ve benzerlik hesaplama kalitesini artırır.
                </p>
              </div>
            </div>
          </div>

          {/* Additional Info */}
          <div className="mt-6 p-4 bg-gradient-to-r from-slate-100/50 to-gray-100/50 dark:from-slate-800/30 dark:to-gray-800/30 rounded-lg border border-slate-200/30 dark:border-slate-700/30">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-slate-600 dark:text-slate-400 mt-0.5 flex-shrink-0" />
              <div className="space-y-2">
                <h5 className="font-medium text-slate-800 dark:text-slate-200">
                  Önemli Bilgiler
                </h5>
                <ul className="text-xs text-slate-600 dark:text-slate-300 space-y-1 leading-relaxed">
                  <li>
                    • Bu ayarlar sistem genelinde sabitlenmiştir ve
                    değiştirilemez
                  </li>
                  <li>• Tüm belgeler aynı kalitede ve tutarlılıkta işlenir</li>
                  <li>
                    • Ayarlar Türkçe eğitim içerikleri için özel olarak optimize
                    edilmiştir
                  </li>
                  <li>
                    • İşleme süresi belge boyutuna ve karmaşıklığına göre
                    değişiklik gösterir
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Upload Button */}
      <div className="flex justify-end">
        <Button
          onClick={handleSubmit}
          disabled={disabled || loading || localSelectedFiles.length === 0}
          className="gap-2"
        >
          <Upload className="w-4 h-4" />
          {localSelectedFiles.length === 0
            ? "Dosya Seçin"
            : `${localSelectedFiles.length} Dosyayı İşle`}
        </Button>
      </div>
    </div>
  );
}
