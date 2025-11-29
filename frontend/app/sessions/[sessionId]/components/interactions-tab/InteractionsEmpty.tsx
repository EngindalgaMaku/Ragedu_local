/**
 * InteractionsEmpty Component
 * Empty state for when no interactions are found
 */

import React from "react";
import { MessageSquare, BookOpen, Users } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

interface InteractionsEmptyProps {
  hasFilters?: boolean;
  onClearFilters?: () => void;
}

export function InteractionsEmpty({
  hasFilters,
  onClearFilters,
}: InteractionsEmptyProps) {
  return (
    <Card className="border-dashed">
      <CardContent className="flex flex-col items-center justify-center py-16 px-6 text-center">
        <div className="mb-4 rounded-full bg-muted p-3">
          <MessageSquare className="h-8 w-8 text-muted-foreground" />
        </div>

        {hasFilters ? (
          <>
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Filtreye uygun etkileşim bulunamadı
            </h3>
            <p className="text-sm text-muted-foreground mb-4 max-w-md">
              Arama kriterlerinize uygun öğrenci sorusu bulunamadı. Filtreleri
              temizleyerek tüm etkileşimleri görebilirsiniz.
            </p>
            <button
              onClick={onClearFilters}
              className="inline-flex items-center px-4 py-2 text-sm font-medium text-primary bg-primary/10 rounded-md hover:bg-primary/20 transition-colors"
            >
              Filtreleri Temizle
            </button>
          </>
        ) : (
          <>
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Henüz soru sorulmamış
            </h3>
            <p className="text-sm text-muted-foreground mb-6 max-w-md">
              Öğrenciler bu ders oturumunda henüz soru sormamış. İlk sorular
              geldiğinde burada görüntülenecek.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-lg">
              <div className="flex items-start gap-3 p-4 bg-muted/50 rounded-lg">
                <div className="rounded-full bg-blue-100 p-2">
                  <BookOpen className="h-4 w-4 text-blue-600" />
                </div>
                <div className="text-left">
                  <p className="text-sm font-medium text-foreground">
                    İçerik Hazır
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Dökümanlar yüklendi ve analiz edildi
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3 p-4 bg-muted/50 rounded-lg">
                <div className="rounded-full bg-green-100 p-2">
                  <Users className="h-4 w-4 text-green-600" />
                </div>
                <div className="text-left">
                  <p className="text-sm font-medium text-foreground">
                    Öğrenci Bekleniyor
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Öğrencilerin soru sorması bekleniyor
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <p className="text-sm text-blue-800 dark:text-blue-200">
                <strong>İpucu:</strong> Öğrenciler sisteme giriş yapıp ders
                oturumuna katıldıktan sonra APRAG özellikli soru-cevap sistemini
                kullanarak sorularını sorabilirler.
              </p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default InteractionsEmpty;
