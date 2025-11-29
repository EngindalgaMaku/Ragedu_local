"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { BookOpen, Info } from "lucide-react";
import { SessionErrorBoundary } from "../shared/ErrorBoundary";
import TopicManagementPanel from "@/components/TopicManagementPanel.backup";

interface TopicsTabWrapperProps {
  sessionId: string;
  apragEnabled: boolean;
}

export default function TopicsTabWrapper({
  sessionId,
  apragEnabled,
}: TopicsTabWrapperProps) {
  if (!apragEnabled) {
    return (
      <div className="space-y-6 p-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-muted">
            <BookOpen className="w-5 h-5 text-muted-foreground" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-foreground">
              Konu Yönetimi
            </h2>
            <p className="text-sm text-muted-foreground mt-0.5">
              APRAG özelliği etkinleştirilmemiş
            </p>
          </div>
        </div>

        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            Konu yönetimi özelliklerini kullanmak için APRAG (Adaptif
            Personalized Retrieval-Augmented Generation) özelliğinin
            etkinleştirilmesi gerekmektedir. Sistem yöneticinizle iletişime
            geçerek bu özelliği etkinleştirebilirsiniz.
          </AlertDescription>
        </Alert>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="w-5 h-5" />
              APRAG Özellikleri
              <Badge variant="outline">Pasif</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-3 border border-border rounded-lg bg-muted/30">
                <h4 className="font-medium text-foreground mb-1">
                  🧠 Otomatik Konu Çıkarımı
                </h4>
                <p className="text-sm text-muted-foreground">
                  Yüklediğiniz belgelerden otomatik olarak konular çıkarılır
                </p>
              </div>
              <div className="p-3 border border-border rounded-lg bg-muted/30">
                <h4 className="font-medium text-foreground mb-1">
                  📚 Bilgi Tabanı Oluşturma
                </h4>
                <p className="text-sm text-muted-foreground">
                  Her konu için detaylı bilgi tabanları oluşturulur
                </p>
              </div>
              <div className="p-3 border border-border rounded-lg bg-muted/30">
                <h4 className="font-medium text-foreground mb-1">
                  ❓ Soru-Cevap Üretimi
                </h4>
                <p className="text-sm text-muted-foreground">
                  Konular için otomatik soru-cevap çiftleri üretilir
                </p>
              </div>
              <div className="p-3 border border-border rounded-lg bg-muted/30">
                <h4 className="font-medium text-foreground mb-1">
                  📊 İlerleme Takibi
                </h4>
                <p className="text-sm text-muted-foreground">
                  Öğrenci ilerlemesi konu bazında takip edilir
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <SessionErrorBoundary>
      <div className="space-y-6 p-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <BookOpen className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-foreground">
              Konu & Bilgi Tabanı Yönetimi
            </h2>
            <p className="text-sm text-muted-foreground mt-0.5">
              Otomatik konu çıkarımı ve bilgi tabanı oluşturma
            </p>
          </div>
          <Badge className="bg-green-100 text-green-700">APRAG Aktif</Badge>
        </div>

        {/* Integrated TopicManagementPanel */}
        <TopicManagementPanel
          sessionId={sessionId}
          apragEnabled={apragEnabled}
        />
      </div>
    </SessionErrorBoundary>
  );
}
