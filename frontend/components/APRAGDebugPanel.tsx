"use client";

import React, { useState } from "react";
import {
  AdaptiveQueryResponse,
  DocumentScore,
  PedagogicalContext,
} from "@/lib/api";
import {
  ChevronDown,
  ChevronUp,
  Info,
  BarChart3,
  Brain,
  FileText,
  Clock,
  Settings,
  Sparkles,
} from "lucide-react";

interface APRAGDebugPanelProps {
  debugData: AdaptiveQueryResponse | null;
  personalizationData?: {
    personalization_factors?: Record<string, any>;
    zpd_info?: any;
    bloom_info?: any;
    cognitive_load?: any;
    pedagogical_instructions?: string;
  } | null;
  query?: string;
}

export default function APRAGDebugPanel({
  debugData,
  personalizationData,
  query,
}: APRAGDebugPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set()
  );

  // Always show panel, but display message if no data
  if (!debugData) {
    return (
      <div className="border-t border-gray-200 bg-gray-50 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <BarChart3 className="w-4 h-4" />
            <span>APRAG Debug Panel</span>
          </div>
          <span className="text-xs text-gray-400">
            Henüz sorgu yapılmadı. İlk sorgu sonrası detaylar burada görünecek.
          </span>
        </div>
      </div>
    );
  }

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const isExpanded = (section: string) => expandedSections.has(section);

  return (
    <div className="border-t border-gray-300 bg-gray-50">
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 bg-gradient-to-r from-purple-50 to-blue-50 hover:from-purple-100 hover:to-blue-100 border-b border-gray-200 flex items-center justify-between transition-colors"
      >
        <div className="flex items-center gap-2">
          <Settings className="w-5 h-5 text-purple-600" />
          <span className="font-semibold text-gray-900">
            🔍 APRAG Debug Paneli
          </span>
          <span className="text-xs text-gray-500 bg-white px-2 py-0.5 rounded-full">
            Araştırma Modu
          </span>
        </div>
        {isOpen ? (
          <ChevronUp className="w-5 h-5 text-gray-600" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-600" />
        )}
      </button>

      {isOpen && (
        <div className="p-4 space-y-4 max-h-[600px] overflow-y-auto">
          {/* Query Info */}
          {query && (
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <h3 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
                <FileText className="w-4 h-4" />
                Öğrenci Sorgusu
              </h3>
              <p className="text-sm text-gray-700 bg-gray-50 p-2 rounded">
                {query}
              </p>
            </div>
          )}

          {/* Processing Time */}
          {debugData.processing_time_ms && (
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4 text-blue-600" />
                <h3 className="font-semibold text-gray-900">İşlem Süresi</h3>
              </div>
              <p className="text-sm text-gray-700">
                {debugData.processing_time_ms.toFixed(2)} ms (
                {(debugData.processing_time_ms / 1000).toFixed(2)} s)
              </p>
            </div>
          )}

          {/* Components Active */}
          <div className="bg-white rounded-lg p-4 border border-gray-200">
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <Settings className="w-4 h-4 text-gray-600" />
              Aktif Bileşenler
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(debugData.components_active || {}).map(
                ([key, value]) => (
                  <div
                    key={key}
                    className={`flex items-center gap-2 p-2 rounded ${
                      value
                        ? "bg-green-50 text-green-700"
                        : "bg-gray-50 text-gray-500"
                    }`}
                  >
                    <div
                      className={`w-2 h-2 rounded-full ${
                        value ? "bg-green-500" : "bg-gray-400"
                      }`}
                    />
                    <span className="text-xs font-medium capitalize">
                      {key.replace("_", " ")}
                    </span>
                  </div>
                )
              )}
            </div>
          </div>

          {/* CACS Document Scoring */}
          <div className="bg-white rounded-lg border border-gray-200">
            <button
              onClick={() => toggleSection("cacs")}
              className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-blue-600" />
                <h3 className="font-semibold text-gray-900">
                  CACS Doküman Skorlaması
                </h3>
                <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                  {debugData.top_documents?.length || 0} doküman
                </span>
              </div>
              {isExpanded("cacs") ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {isExpanded("cacs") && (
              <div className="p-4 pt-0 space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-3">
                  <div className="flex items-start gap-2">
                    <Info className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <div className="text-xs text-blue-800">
                      <p className="font-semibold mb-1">CACS (Conversation-Aware Content Scoring) Nedir?</p>
                      <p className="mb-2">
                        CACS, her dokümanı 4 farklı skorla değerlendirir ve bunları birleştirerek
                        öğrenciye en uygun dokümanları seçer:
                      </p>
                      <ul className="list-disc list-inside space-y-1 ml-2">
                        <li>
                          <strong>Base Score:</strong> RAG sisteminden gelen semantik benzerlik skoru
                          (0-1 arası). Sorgu ile dokümanın ne kadar ilgili olduğunu gösterir.
                        </li>
                        <li>
                          <strong>Personal Score:</strong> Öğrencinin geçmiş etkileşimlerine göre
                          hesaplanan kişisel skor. Öğrencinin daha önce hangi dokümanlardan faydalandığını
                          ve hangi konularda zorlandığını dikkate alır.
                        </li>
                        <li>
                          <strong>Global Score:</strong> Tüm öğrencilerden toplanan geri bildirimlere
                          göre hesaplanan genel skor. Hangi dokümanların genel olarak daha faydalı
                          olduğunu gösterir.
                        </li>
                        <li>
                          <strong>Context Score:</strong> Mevcut konuşma bağlamına göre hesaplanan
                          skor. Önceki sorular ve cevaplar dikkate alınarak dokümanın konuşma akışına
                          ne kadar uygun olduğunu ölçer.
                        </li>
                        <li>
                          <strong>Final Score:</strong> Yukarıdaki 4 skorun ağırlıklı ortalaması.
                          Bu skora göre dokümanlar sıralanır ve en yüksek skorlu dokümanlar seçilir.
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>

                {debugData.cacs_applied ? (
                  <div className="space-y-3">
                    {debugData.top_documents?.map((doc: DocumentScore, idx: number) => (
                      <div
                        key={doc.doc_id}
                        className="border border-gray-200 rounded-lg p-3 bg-gray-50"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-semibold bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                              #{doc.rank}
                            </span>
                            <span className="text-sm font-medium text-gray-900">
                              {doc.doc_id}
                            </span>
                          </div>
                          <span className="text-sm font-bold text-blue-600">
                            Final: {(doc.final_score * 100).toFixed(1)}%
                          </span>
                        </div>

                        <div className="grid grid-cols-2 gap-2 mt-2">
                          <div className="bg-white p-2 rounded border border-gray-200">
                            <div className="text-xs text-gray-500 mb-1">Base Score</div>
                            <div className="text-sm font-semibold text-gray-900">
                              {(doc.base_score * 100).toFixed(1)}%
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                              <div
                                className="bg-blue-500 h-1.5 rounded-full"
                                style={{ width: `${doc.base_score * 100}%` }}
                              />
                            </div>
                          </div>

                          <div className="bg-white p-2 rounded border border-gray-200">
                            <div className="text-xs text-gray-500 mb-1">Personal Score</div>
                            <div className="text-sm font-semibold text-gray-900">
                              {(doc.personal_score * 100).toFixed(1)}%
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                              <div
                                className="bg-green-500 h-1.5 rounded-full"
                                style={{ width: `${doc.personal_score * 100}%` }}
                              />
                            </div>
                          </div>

                          <div className="bg-white p-2 rounded border border-gray-200">
                            <div className="text-xs text-gray-500 mb-1">Global Score</div>
                            <div className="text-sm font-semibold text-gray-900">
                              {(doc.global_score * 100).toFixed(1)}%
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                              <div
                                className="bg-purple-500 h-1.5 rounded-full"
                                style={{ width: `${doc.global_score * 100}%` }}
                              />
                            </div>
                          </div>

                          <div className="bg-white p-2 rounded border border-gray-200">
                            <div className="text-xs text-gray-500 mb-1">Context Score</div>
                            <div className="text-sm font-semibold text-gray-900">
                              {(doc.context_score * 100).toFixed(1)}%
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                              <div
                                className="bg-orange-500 h-1.5 rounded-full"
                                style={{ width: `${doc.context_score * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-gray-500 italic">
                    CACS skorlaması uygulanmadı (devre dışı veya yeterli veri yok)
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Pedagogical Context */}
          <div className="bg-white rounded-lg border border-gray-200">
            <button
              onClick={() => toggleSection("pedagogical")}
              className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-600" />
                <h3 className="font-semibold text-gray-900">
                  Pedagojik Analiz
                </h3>
              </div>
              {isExpanded("pedagogical") ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {isExpanded("pedagogical") && (
              <div className="p-4 pt-0 space-y-4">
                {debugData.pedagogical_context && (
                  <>
                    {/* ZPD */}
                    <div className="border border-purple-200 rounded-lg p-3 bg-purple-50">
                      <div className="flex items-center gap-2 mb-2">
                        <h4 className="font-semibold text-purple-900">ZPD (Zone of Proximal Development)</h4>
                        <Info className="w-4 h-4 text-purple-600" />
                      </div>
                      <div className="bg-white rounded p-2 mb-2">
                        <div className="text-xs text-gray-600 mb-1">Mevcut Seviye</div>
                        <div className="text-sm font-semibold text-gray-900 capitalize">
                          {debugData.pedagogical_context.zpd_level}
                        </div>
                      </div>
                      <div className="bg-white rounded p-2 mb-2">
                        <div className="text-xs text-gray-600 mb-1">Önerilen Seviye</div>
                        <div className="text-sm font-semibold text-purple-700 capitalize">
                          {debugData.pedagogical_context.zpd_recommended}
                        </div>
                      </div>
                      <div className="bg-white rounded p-2">
                        <div className="text-xs text-gray-600 mb-1">Başarı Oranı</div>
                        <div className="text-sm font-semibold text-gray-900">
                          {(debugData.pedagogical_context.zpd_success_rate * 100).toFixed(1)}%
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                          <div
                            className="bg-purple-500 h-1.5 rounded-full"
                            style={{
                              width: `${debugData.pedagogical_context.zpd_success_rate * 100}%`,
                            }}
                          />
                        </div>
                      </div>
                      <div className="mt-2 text-xs text-purple-800 bg-purple-100 p-2 rounded">
                        <strong>Ne İşe Yarar?</strong> ZPD, öğrencinin mevcut bilgi seviyesi ile
                        öğrenebileceği maksimum seviye arasındaki "yakın gelişim alanı"nı belirler.
                        Sistem, öğrencinin başarı oranına göre zorluk seviyesini otomatik olarak
                        ayarlar. Yüksek başarı oranı varsa seviye artırılır, düşükse azaltılır.
                      </div>
                    </div>

                    {/* Bloom Taxonomy */}
                    <div className="border border-green-200 rounded-lg p-3 bg-green-50">
                      <div className="flex items-center gap-2 mb-2">
                        <h4 className="font-semibold text-green-900">Bloom Taksonomisi</h4>
                        <Info className="w-4 h-4 text-green-600" />
                      </div>
                      <div className="bg-white rounded p-2 mb-2">
                        <div className="text-xs text-gray-600 mb-1">Tespit Edilen Seviye</div>
                        <div className="text-sm font-semibold text-gray-900 capitalize">
                          {debugData.pedagogical_context.bloom_level} (Seviye {debugData.pedagogical_context.bloom_level_index})
                        </div>
                      </div>
                      <div className="mt-2 text-xs text-green-800 bg-green-100 p-2 rounded">
                        <strong>Ne İşe Yarar?</strong> Bloom Taksonomisi, öğrencinin sorusunun
                        hangi bilişsel seviyede olduğunu belirler (Hatırlama, Anlama, Uygulama,
                        Analiz, Değerlendirme, Yaratma). Sistem, cevabı bu seviyeye uygun şekilde
                        hazırlar. Örneğin, "hatırlama" seviyesindeki bir soruya basit tanımlar,
                        "analiz" seviyesindeki bir soruya ise detaylı karşılaştırmalar içeren
                        cevaplar verilir.
                      </div>
                    </div>

                    {/* Cognitive Load */}
                    <div className="border border-orange-200 rounded-lg p-3 bg-orange-50">
                      <div className="flex items-center gap-2 mb-2">
                        <h4 className="font-semibold text-orange-900">Bilişsel Yük</h4>
                        <Info className="w-4 h-4 text-orange-600" />
                      </div>
                      <div className="bg-white rounded p-2 mb-2">
                        <div className="text-xs text-gray-600 mb-1">Toplam Yük</div>
                        <div className="text-sm font-semibold text-gray-900">
                          {debugData.pedagogical_context.cognitive_load.toFixed(2)} / 1.0
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                          <div
                            className={`h-1.5 rounded-full ${
                              debugData.pedagogical_context.cognitive_load > 0.7
                                ? "bg-red-500"
                                : debugData.pedagogical_context.cognitive_load > 0.4
                                ? "bg-yellow-500"
                                : "bg-green-500"
                            }`}
                            style={{
                              width: `${debugData.pedagogical_context.cognitive_load * 100}%`,
                            }}
                          />
                        </div>
                      </div>
                      <div className="bg-white rounded p-2">
                        <div className="text-xs text-gray-600 mb-1">Basitleştirme Gerekli mi?</div>
                        <div
                          className={`text-sm font-semibold ${
                            debugData.pedagogical_context.needs_simplification
                              ? "text-red-600"
                              : "text-green-600"
                          }`}
                        >
                          {debugData.pedagogical_context.needs_simplification
                            ? "Evet"
                            : "Hayır"}
                        </div>
                      </div>
                      <div className="mt-2 text-xs text-orange-800 bg-orange-100 p-2 rounded">
                        <strong>Ne İşe Yarar?</strong> Bilişsel Yük teorisi, öğrencinin aynı anda
                        işleyebileceği bilgi miktarını ölçer. Yüksek bilişsel yük (0.7+), öğrencinin
                        cevabı anlamakta zorlanabileceği anlamına gelir. Bu durumda sistem cevabı
                        parçalara böler veya basitleştirir. Düşük yük (0.4-), öğrencinin daha
                        karmaşık bilgileri işleyebileceği anlamına gelir.
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          {/* LLM Personalization */}
          {(personalizationData?.personalization_factors ||
            personalizationData?.pedagogical_instructions) && (
            <div className="bg-white rounded-lg border border-gray-200">
              <button
                onClick={() => toggleSection("personalization")}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-pink-600" />
                  <h3 className="font-semibold text-gray-900">
                    LLM Kişiselleştirme
                  </h3>
                </div>
                {isExpanded("personalization") ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>

              {isExpanded("personalization") && (
                <div className="p-4 pt-0 space-y-4">
                  <div className="bg-pink-50 border border-pink-200 rounded-lg p-3 mb-3">
                    <div className="flex items-start gap-2">
                      <Info className="w-4 h-4 text-pink-600 mt-0.5 flex-shrink-0" />
                      <div className="text-xs text-pink-800">
                        <p className="font-semibold mb-1">LLM Kişiselleştirme Nasıl Çalışır?</p>
                        <p>
                          Sistem, yukarıdaki pedagojik analiz sonuçlarını (ZPD, Bloom, Bilişsel Yük)
                          kullanarak LLM'e özel talimatlar gönderir. LLM bu talimatlara göre orijinal
                          cevabı öğrencinin seviyesine uygun şekilde yeniden yazar.
                        </p>
                      </div>
                    </div>
                  </div>

                  {personalizationData.personalization_factors && (
                    <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
                      <h4 className="font-semibold text-gray-900 mb-2 text-sm">
                        Kişiselleştirme Faktörleri
                      </h4>
                      <div className="space-y-2">
                        {Object.entries(
                          personalizationData.personalization_factors
                        ).map(([key, value]) => (
                          <div
                            key={key}
                            className="bg-white p-2 rounded border border-gray-200"
                          >
                            <div className="text-xs text-gray-600 mb-1 capitalize">
                              {key.replace(/_/g, " ")}
                            </div>
                            <div className="text-sm font-semibold text-gray-900">
                              {typeof value === "object"
                                ? JSON.stringify(value, null, 2)
                                : String(value)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {personalizationData.pedagogical_instructions && (
                    <div className="border border-pink-200 rounded-lg p-3 bg-pink-50">
                      <h4 className="font-semibold text-pink-900 mb-2 text-sm">
                        LLM'e Gönderilen Pedagojik Talimatlar
                      </h4>
                      <div className="bg-white p-3 rounded border border-pink-200">
                        <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                          {personalizationData.pedagogical_instructions}
                        </pre>
                      </div>
                      <div className="mt-2 text-xs text-pink-800 bg-pink-100 p-2 rounded">
                        <strong>Ne İşe Yarar?</strong> Bu talimatlar, LLM'in orijinal cevabı nasıl
                        değiştireceğini belirler. Örneğin, "ZPD seviyesi: beginner" talimatı varsa,
                        LLM cevabı daha basit kelimelerle ve daha fazla örnekle yazar. "Bloom seviyesi:
                        analyze" talimatı varsa, LLM cevaba karşılaştırmalar ve analizler ekler.
                      </div>
                    </div>
                  )}

                  {/* Original vs Personalized Response Comparison */}
                  {debugData.original_response && debugData.personalized_response && (
                    <div className="border border-blue-200 rounded-lg p-3 bg-blue-50">
                      <h4 className="font-semibold text-blue-900 mb-3 text-sm">
                        Orijinal vs Kişiselleştirilmiş Cevap Karşılaştırması
                      </h4>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-white p-3 rounded border border-blue-200">
                          <div className="text-xs font-semibold text-blue-700 mb-2 flex items-center justify-between">
                            <span>Orijinal Cevap</span>
                            <span className="text-gray-500 font-normal">
                              {debugData.original_response.length} karakter
                            </span>
                          </div>
                          <div className="text-xs text-gray-700 max-h-[500px] overflow-y-auto whitespace-pre-wrap font-mono bg-gray-50 p-3 rounded border border-gray-200">
                            {debugData.original_response}
                          </div>
                        </div>
                        <div className="bg-white p-3 rounded border border-pink-200">
                          <div className="text-xs font-semibold text-pink-700 mb-2 flex items-center justify-between">
                            <span>Kişiselleştirilmiş Cevap</span>
                            <span className="text-gray-500 font-normal">
                              {debugData.personalized_response.length} karakter
                            </span>
                          </div>
                          <div className="text-xs text-gray-700 max-h-[500px] overflow-y-auto whitespace-pre-wrap font-mono bg-pink-50 p-3 rounded border border-pink-200">
                            {debugData.personalized_response}
                          </div>
                        </div>
                      </div>
                      <div className="mt-3 text-xs text-blue-800 bg-blue-100 p-2 rounded">
                        <strong>Ne İşe Yarar?</strong> Bu karşılaştırma, LLM'in cevabı nasıl
                        değiştirdiğini gösterir. Farklar genellikle şunlardan kaynaklanır:
                        <ul className="list-disc list-inside mt-1 ml-2">
                          <li>ZPD seviyesine göre zorluk ayarı</li>
                          <li>Bloom seviyesine göre açıklama derinliği</li>
                          <li>Bilişsel yüke göre parçalama/basitleştirme</li>
                          <li>Öğrencinin geçmiş etkileşimlerine göre stil ayarı</li>
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Hybrid RAG Debug Info */}
          {debugData?.hybrid_rag_debug && (
            <div className="bg-white rounded-lg border border-gray-200">
              <button
                onClick={() => toggleSection("hybrid_rag")}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Settings className="w-5 h-5 text-blue-600" />
                  <h3 className="font-semibold text-gray-900">
                    🔧 Hybrid RAG Detayları
                  </h3>
                </div>
                {isExpanded("hybrid_rag") ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>

              {isExpanded("hybrid_rag") && (
                <div className="p-4 pt-0 space-y-4">
                  {/* LLM Request Details */}
                  {debugData.hybrid_rag_debug.llm_request && (
                    <div className="border border-blue-200 rounded-lg p-3 bg-blue-50">
                      <h4 className="font-semibold text-blue-900 mb-3 text-sm">
                        🤖 LLM İstek Detayları
                      </h4>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-white p-2 rounded">
                          <div className="text-xs text-gray-600 mb-1">Model</div>
                          <div className="text-sm font-semibold text-gray-900">
                            {debugData.hybrid_rag_debug.llm_request.model || "N/A"}
                          </div>
                        </div>
                        <div className="bg-white p-2 rounded">
                          <div className="text-xs text-gray-600 mb-1">Max Tokens</div>
                          <div className="text-sm font-semibold text-gray-900">
                            {debugData.hybrid_rag_debug.llm_request.max_tokens || "N/A"}
                          </div>
                        </div>
                        <div className="bg-white p-2 rounded">
                          <div className="text-xs text-gray-600 mb-1">Temperature</div>
                          <div className="text-sm font-semibold text-gray-900">
                            {debugData.hybrid_rag_debug.llm_request.temperature || "N/A"}
                          </div>
                        </div>
                        <div className="bg-white p-2 rounded">
                          <div className="text-xs text-gray-600 mb-1">Context Length</div>
                          <div className="text-sm font-semibold text-gray-900">
                            {debugData.hybrid_rag_debug.llm_request.context_length?.toLocaleString() || "N/A"} chars
                          </div>
                        </div>
                        <div className="bg-white p-2 rounded">
                          <div className="text-xs text-gray-600 mb-1">Query Length</div>
                          <div className="text-sm font-semibold text-gray-900">
                            {debugData.hybrid_rag_debug.llm_request.query_length?.toLocaleString() || "N/A"} chars
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* CRAG Evaluation Details */}
                  {debugData.hybrid_rag_debug.crag_evaluation && (
                    <div className="border border-purple-200 rounded-lg p-3 bg-purple-50">
                      <h4 className="font-semibold text-purple-900 mb-3 text-sm">
                        🔍 CRAG Evaluation Detayları
                      </h4>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-white p-2 rounded">
                          <div className="text-xs text-gray-600 mb-1">Action</div>
                          <div className={`text-sm font-semibold ${
                            debugData.hybrid_rag_debug.crag_evaluation.action === "accept" 
                              ? "text-green-600" 
                              : debugData.hybrid_rag_debug.crag_evaluation.action === "reject"
                              ? "text-red-600"
                              : "text-yellow-600"
                          }`}>
                            {debugData.hybrid_rag_debug.crag_evaluation.action || "N/A"}
                          </div>
                        </div>
                        <div className="bg-white p-2 rounded">
                          <div className="text-xs text-gray-600 mb-1">Confidence</div>
                          <div className="text-sm font-semibold text-gray-900">
                            {debugData.hybrid_rag_debug.crag_evaluation.confidence 
                              ? (debugData.hybrid_rag_debug.crag_evaluation.confidence * 100).toFixed(1) + "%"
                              : "N/A"}
                          </div>
                        </div>
                        {debugData.hybrid_rag_debug.crag_evaluation.max_score !== undefined && (
                          <div className="bg-white p-2 rounded">
                            <div className="text-xs text-gray-600 mb-1">Max Score</div>
                            <div className="text-sm font-semibold text-gray-900">
                              {debugData.hybrid_rag_debug.crag_evaluation.max_score.toFixed(4)}
                            </div>
                          </div>
                        )}
                        {debugData.hybrid_rag_debug.crag_evaluation.avg_score !== undefined && (
                          <div className="bg-white p-2 rounded">
                            <div className="text-xs text-gray-600 mb-1">Avg Score</div>
                            <div className="text-sm font-semibold text-gray-900">
                              {debugData.hybrid_rag_debug.crag_evaluation.avg_score.toFixed(4)}
                            </div>
                          </div>
                        )}
                        {debugData.hybrid_rag_debug.crag_evaluation.filtered !== undefined && (
                          <div className="bg-white p-2 rounded col-span-2">
                            <div className="text-xs text-gray-600 mb-1">Filtered Docs</div>
                            <div className="text-sm font-semibold text-gray-900">
                              {debugData.hybrid_rag_debug.crag_evaluation.filtered}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Retrieval Details */}
                  {debugData.hybrid_rag_debug.retrieval_details && (
                    <div className="border border-green-200 rounded-lg p-3 bg-green-50">
                      <h4 className="font-semibold text-green-900 mb-3 text-sm">
                        📚 Retrieval Detayları
                      </h4>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-white p-2 rounded">
                          <div className="text-xs text-gray-600 mb-1">Chunks Retrieved</div>
                          <div className="text-sm font-semibold text-gray-900">
                            {debugData.hybrid_rag_debug.retrieval_details.chunks_retrieved || 0}
                          </div>
                        </div>
                        <div className="bg-white p-2 rounded">
                          <div className="text-xs text-gray-600 mb-1">KB Items</div>
                          <div className="text-sm font-semibold text-gray-900">
                            {debugData.hybrid_rag_debug.retrieval_details.kb_items_retrieved || 0}
                          </div>
                        </div>
                        <div className="bg-white p-2 rounded">
                          <div className="text-xs text-gray-600 mb-1">QA Pairs Matched</div>
                          <div className="text-sm font-semibold text-gray-900">
                            {debugData.hybrid_rag_debug.retrieval_details.qa_pairs_matched || 0}
                          </div>
                        </div>
                        <div className="bg-white p-2 rounded">
                          <div className="text-xs text-gray-600 mb-1">Total Merged</div>
                          <div className="text-sm font-semibold text-gray-900">
                            {debugData.hybrid_rag_debug.retrieval_details.total_merged || 0}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Response Size */}
                  {debugData.hybrid_rag_debug.response_size !== undefined && (
                    <div className="border border-orange-200 rounded-lg p-3 bg-orange-50">
                      <h4 className="font-semibold text-orange-900 mb-2 text-sm">
                        📏 Response Size
                      </h4>
                      <div className="bg-white p-3 rounded">
                        <div className="text-2xl font-bold text-orange-600">
                          {debugData.hybrid_rag_debug.response_size.toLocaleString()} chars
                        </div>
                        <div className="text-xs text-gray-600 mt-1">
                          ({((debugData.hybrid_rag_debug.response_size / 1024) * 100).toFixed(1)} KB)
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Reranker Info */}
                  {debugData.hybrid_rag_debug.reranker_used !== undefined && (
                    <div className="border border-indigo-200 rounded-lg p-3 bg-indigo-50">
                      <h4 className="font-semibold text-indigo-900 mb-2 text-sm">
                        🎯 Reranker
                      </h4>
                      <div className="bg-white p-2 rounded">
                        <div className="text-sm font-semibold text-gray-900">
                          {debugData.hybrid_rag_debug.reranker_used ? "✅ Kullanıldı" : "❌ Kullanılmadı"}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Raw Data */}
          <div className="bg-white rounded-lg border border-gray-200">
            <button
              onClick={() => toggleSection("raw")}
              className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-gray-600" />
                <h3 className="font-semibold text-gray-900">Ham Veri (JSON)</h3>
              </div>
              {isExpanded("raw") ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {isExpanded("raw") && (
              <div className="p-4 pt-0">
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                  <pre className="text-xs font-mono">
                    {JSON.stringify(
                      {
                        adaptiveQuery: debugData,
                        personalization: personalizationData,
                      },
                      null,
                      2
                    )}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

