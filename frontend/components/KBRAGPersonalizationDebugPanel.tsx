"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, Terminal, Copy, Download, Info } from "lucide-react";

interface KBRAGPersonalizationDebugPanelProps {
  debugData: {
    adaptiveQueryResult?: any;
    personalizationData?: any;
    lastQuery?: string | null;
  } | null;
}

export default function KBRAGPersonalizationDebugPanel({ debugData }: KBRAGPersonalizationDebugPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  if (!debugData || !debugData.adaptiveQueryResult) {
    return (
      <div className="border-t border-gray-200 bg-gray-50 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Terminal className="w-4 h-4" />
            <span>KBRAG & Kişiselleştirme Debug</span>
          </div>
          <span className="text-xs text-gray-400">
            Henüz sorgu yapılmadı. İlk sorgu sonrası detaylar burada görünecek.
          </span>
        </div>
      </div>
    );
  }

  const data = debugData.adaptiveQueryResult;
  const hybridDebug = data.hybrid_rag_debug || {};
  const personalization = debugData.personalizationData || {};
  
  // Get comprehensive debug from multiple sources
  let comprehensiveDebug = personalization?.comprehensive_debug;
  if (!comprehensiveDebug && hybridDebug?.comprehensive_debug) {
    comprehensiveDebug = hybridDebug.comprehensive_debug;
  }
  if (!comprehensiveDebug && data?.comprehensive_debug) {
    comprehensiveDebug = data.comprehensive_debug;
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

  // Extract personalization prompt (we need to reconstruct it)
  const reconstructPersonalizationPrompt = () => {
    const factors = personalization?.personalization_factors || {};
    const zpdInfo = personalization?.zpd_info || {};
    const bloomInfo = personalization?.bloom_info || {};
    const cognitiveLoad = personalization?.cognitive_load || {};
    const pedagogicalInstructions = personalization?.pedagogical_instructions || "";

    let prompt = `Sen bir eğitim asistanısın. Aşağıdaki cevabı öğrencinin öğrenme profiline ve pedagojik analiz sonuçlarına göre kişiselleştir.

📊 ÖĞRENCİ PROFİLİ:
- Anlama Seviyesi: ${factors.understanding_level || "N/A"}
- Zorluk Seviyesi: ${factors.difficulty_level || "N/A"}
- Açıklama Stili: ${factors.explanation_style || "N/A"}
- Örnekler Gerekli: ${factors.needs_examples ? "Evet" : "Hayır"}`;

    if (zpdInfo && Object.keys(zpdInfo).length > 0) {
      prompt += `\n\n🎯 ZPD (Zone of Proximal Development):
- Mevcut Seviye: ${zpdInfo.current_level || "N/A"}
- Önerilen Seviye: ${zpdInfo.recommended_level || "N/A"}
- Başarı Oranı: ${zpdInfo.success_rate !== undefined ? (zpdInfo.success_rate * 100).toFixed(1) + "%" : "N/A"}`;
    }

    if (bloomInfo && Object.keys(bloomInfo).length > 0) {
      prompt += `\n\n🧠 Bloom Taksonomisi:
- Tespit Edilen Seviye: ${bloomInfo.level || "N/A"} (Seviye ${bloomInfo.level_index || "N/A"})
- Güven: ${bloomInfo.confidence !== undefined ? (bloomInfo.confidence * 100).toFixed(1) + "%" : "N/A"}`;
    }

    if (cognitiveLoad && Object.keys(cognitiveLoad).length > 0) {
      prompt += `\n\n⚖️ Bilişsel Yük:
- Toplam Yük: ${cognitiveLoad.total_load !== undefined ? cognitiveLoad.total_load.toFixed(2) : "N/A"}
- Sadeleştirme Gerekli: ${cognitiveLoad.needs_simplification ? "Evet" : "Hayır"}`;
    }

    prompt += `\n\n📝 ORİJİNAL SORU:
${debugData.lastQuery || "N/A"}

📄 ORİJİNAL CEVAP:
${data.original_response || "N/A"}

⚠️ ÇOK ÖNEMLİ - DOĞRULUK KURALLARI:
- SADECE orijinal cevapta ve ders materyallerinde bulunan bilgileri kullan
- Orijinal cevapta olmayan yeni bilgiler EKLEME
- Orijinal cevabın içeriğini koru, sadece sunumunu değiştir
- Emin olmadığın bilgileri uydurma veya tahmin etme

🔧 KİŞİSELLEŞTİRME TALİMATLARI:`;

    if (factors.explanation_style === "detailed") {
      prompt += "\n- Daha detaylı açıklamalar ekle\n- Her adımı açıkça belirt";
    } else if (factors.explanation_style === "concise") {
      prompt += "\n- Daha kısa ve öz bir açıklama yap\n- Gereksiz detayları çıkar";
    }

    if (factors.needs_examples) {
      prompt += "\n- Pratik örnekler ekle\n- Günlük hayattan örnekler ver";
    }

    if (factors.difficulty_level === "beginner" || factors.difficulty_level === "elementary" || 
        (zpdInfo && zpdInfo.recommended_level === "elementary")) {
      prompt += "\n- Temel kavramları önce açıkla\n- Teknik terimleri basit dille açıkla\n- Daha basit kelimeler kullan";
    } else if (factors.difficulty_level === "advanced" || 
               (zpdInfo && zpdInfo.recommended_level === "advanced")) {
      prompt += "\n- Daha derinlemesine bilgi ver\n- İleri seviye detaylar ekle";
    }

    if (pedagogicalInstructions) {
      prompt += `\n\n🎓 PEDAGOJİK TALİMATLAR (ÇOK ÖNEMLİ - MUTLAKA UYGULA):
${pedagogicalInstructions}`;
    }

    prompt += `\n\n✅ ÖNEMLİ: Kişiselleştirilmiş cevabı SADECE TÜRKÇE olarak ver. Orijinal cevabın içeriğini koru, ancak sunumunu, detay seviyesini ve zorluk seviyesini öğrenci profiline ve pedagojik talimatlara göre ayarla. Cevabı başlık veya madde listesi olmadan, düz metin olarak ver.

⚠️ ÇOK ÖNEMLİ: Aynı bilgiyi veya cümleyi tekrar etme. Her cümle yeni bir bilgi veya açıklama içermeli. Gereksiz tekrarlardan kaçın.`;

    return prompt;
  };

  const personalizationPrompt = reconstructPersonalizationPrompt();

  return (
    <div className="border-t border-gray-300 bg-gray-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 bg-gradient-to-r from-blue-800 to-purple-800 hover:from-blue-900 hover:to-purple-900 border-b border-gray-700 flex items-center justify-between transition-colors text-white"
      >
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5" />
          <span className="font-semibold">📊 KBRAG & Kişiselleştirme Debug Paneli</span>
        </div>
        {isOpen ? (
          <ChevronUp className="w-5 h-5" />
        ) : (
          <ChevronDown className="w-5 h-5" />
        )}
      </button>

      {isOpen && (
        <div className="bg-white p-6 space-y-6 max-h-[calc(100vh-300px)] overflow-y-auto">
          
          {/* 1. KBRAG RETRIEVAL BÖLÜMÜ */}
          <div className="border border-blue-200 rounded-lg p-4 bg-blue-50">
            <button
              onClick={() => toggleSection("kbrag")}
              className="w-full flex items-center justify-between text-left"
            >
              <h3 className="text-lg font-bold text-blue-900 flex items-center gap-2">
                <Info className="w-5 h-5" />
                1. KBRAG Bilgi Çekme (Retrieval) Süreci
              </h3>
              {expandedSections.has("kbrag") ? (
                <ChevronUp className="w-5 h-5" />
              ) : (
                <ChevronDown className="w-5 h-5" />
              )}
            </button>
            
            {expandedSections.has("kbrag") && hybridDebug.retrieval_stages && (
              <div className="mt-4 space-y-4 text-sm">
                {/* Topic Classification */}
                {hybridDebug.retrieval_stages.topic_classification && (
                  <div className="bg-white p-3 rounded border border-blue-100">
                    <h4 className="font-semibold text-blue-800 mb-2">📚 Konu Sınıflandırması (Topic Classification)</h4>
                    <p className="text-gray-700 mb-2">
                      <strong>Açıklama:</strong> Soru hangi konulara ait olduğunu tespit eder.
                    </p>
                    <div className="space-y-1 text-gray-600">
                      <p>Eşleşen Konu Sayısı: <strong>{hybridDebug.retrieval_stages.topic_classification.topics_count || 0}</strong></p>
                      <p>Güven Oranı: <strong>{(Number(hybridDebug.retrieval_stages.topic_classification.confidence || 0) * 100).toFixed(1)}%</strong></p>
                      {hybridDebug.retrieval_stages.topic_classification.matched_topics?.map((topic: any, idx: number) => (
                        <div key={idx} className="ml-4 p-2 bg-gray-50 rounded">
                          <p>Konu {idx + 1}: <strong>{topic.topic_title}</strong> (Güven: {(Number(topic.confidence || 0) * 100).toFixed(1)}%)</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Chunk Retrieval */}
                {hybridDebug.retrieval_stages.chunk_retrieval && (
                  <div className="bg-white p-3 rounded border border-blue-100">
                    <h4 className="font-semibold text-blue-800 mb-2">📄 Döküman Parçaları (Chunks)</h4>
                    <p className="text-gray-700 mb-2">
                      <strong>Açıklama:</strong> Dökümanlardan en ilgili parçalar çekilir. Skor ne kadar yüksekse o kadar ilgili demektir.
                    </p>
                    <div className="space-y-1 text-gray-600">
                      <p>Çekilen Parça Sayısı: <strong>{hybridDebug.retrieval_stages.chunk_retrieval.chunks_retrieved || 0}</strong></p>
                      {hybridDebug.retrieval_stages.chunk_retrieval.chunks?.map((chunk: any, idx: number) => (
                        <div key={idx} className="ml-4 p-2 bg-gray-50 rounded">
                          <p>Parça {idx + 1}: <strong>{(Number(chunk.score || 0) * 100).toFixed(1)}%</strong> skor</p>
                          <p className="text-xs text-gray-500 mt-1">{chunk.content_preview?.substring(0, 100)}...</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* KB Retrieval */}
                {hybridDebug.retrieval_stages.kb_retrieval && (
                  <div className="bg-white p-3 rounded border border-blue-100">
                    <h4 className="font-semibold text-blue-800 mb-2">🧠 Bilgi Tabanı (Knowledge Base)</h4>
                    <p className="text-gray-700 mb-2">
                      <strong>Açıklama:</strong> Önceden hazırlanmış konu bilgileri çekilir.
                    </p>
                    <div className="space-y-1 text-gray-600">
                      <p>Çekilen KB Öğesi: <strong>{hybridDebug.retrieval_stages.kb_retrieval.kb_items_retrieved || 0}</strong></p>
                      {hybridDebug.retrieval_stages.kb_retrieval.kb_items?.map((kb: any, idx: number) => (
                        <div key={idx} className="ml-4 p-2 bg-gray-50 rounded">
                          <p>KB {idx + 1}: <strong>{kb.topic_title}</strong> (İlgililik: {(Number(kb.relevance_score || 0) * 100).toFixed(1)}%)</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Merged Results */}
                {hybridDebug.retrieval_stages.merged_results && (
                  <div className="bg-white p-3 rounded border border-blue-100">
                    <h4 className="font-semibold text-blue-800 mb-2">🔄 Birleştirilmiş Sonuçlar</h4>
                    <p className="text-gray-700 mb-2">
                      <strong>Açıklama:</strong> Tüm kaynaklardan gelen bilgiler birleştirilir.
                    </p>
                    <div className="space-y-1 text-gray-600">
                      <p>Toplam Birleştirilen: <strong>{hybridDebug.retrieval_stages.merged_results.total_merged || 0}</strong></p>
                      {hybridDebug.retrieval_stages.merged_results.by_source && (
                        <div className="ml-4 space-y-1">
                          <p>• Döküman Parçaları: {hybridDebug.retrieval_stages.merged_results.by_source.chunks || 0}</p>
                          <p>• Bilgi Tabanı: {hybridDebug.retrieval_stages.merged_results.by_source.kb || 0}</p>
                          <p>• Soru-Cevap Çiftleri: {hybridDebug.retrieval_stages.merged_results.by_source.qa_pairs || 0}</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 2. CACS PUANLAMA BÖLÜMÜ */}
          {comprehensiveDebug?.cacs_scoring && comprehensiveDebug.cacs_scoring.applied && (
            <div className="border border-purple-200 rounded-lg p-4 bg-purple-50">
              <button
                onClick={() => toggleSection("cacs")}
                className="w-full flex items-center justify-between text-left"
              >
                <h3 className="text-lg font-bold text-purple-900 flex items-center gap-2">
                  <Info className="w-5 h-5" />
                  2. CACS Puanlama Sistemi (Context-Aware Content Scoring)
                </h3>
                {expandedSections.has("cacs") ? (
                  <ChevronUp className="w-5 h-5" />
                ) : (
                  <ChevronDown className="w-5 h-5" />
                )}
              </button>
              
              {expandedSections.has("cacs") && (
                <div className="mt-4 space-y-4 text-sm">
                  <div className="bg-white p-4 rounded border border-purple-100">
                    <p className="text-gray-700 mb-4">
                      <strong>Açıklama:</strong> CACS, dökümanları öğrenci profiline göre puanlar. 
                      Her döküman için 4 farklı skor hesaplanır ve bunlar birleştirilerek final skor oluşturulur.
                    </p>
                    
                    <div className="space-y-3">
                      <p className="font-semibold text-purple-800">Puanlanan Döküman Sayısı: {comprehensiveDebug.cacs_scoring.documents_scored || 0}</p>
                      
                      {comprehensiveDebug.cacs_scoring.scoring_details?.map((doc: any, idx: number) => (
                        <div key={idx} className="bg-gray-50 p-4 rounded border border-purple-100">
                          <h4 className="font-semibold text-purple-800 mb-3">Döküman {idx + 1} (Sıra: {doc.rank})</h4>
                          
                          <div className="grid grid-cols-2 gap-3 mb-3">
                            <div className="bg-blue-100 p-2 rounded">
                              <p className="text-xs text-blue-800 font-semibold">Base Score (Temel Skor)</p>
                              <p className="text-lg font-bold text-blue-900">{(doc.base_score * 100).toFixed(1)}%</p>
                              <p className="text-xs text-gray-600 mt-1">Dökümanın soruya olan temel ilgisi</p>
                            </div>
                            
                            <div className="bg-green-100 p-2 rounded">
                              <p className="text-xs text-green-800 font-semibold">Personal Score (Kişisel Skor)</p>
                              <p className="text-lg font-bold text-green-900">{(doc.personal_score * 100).toFixed(1)}%</p>
                              <p className="text-xs text-gray-600 mt-1">Öğrencinin geçmiş etkileşimlerine göre uygunluk</p>
                            </div>
                            
                            <div className="bg-yellow-100 p-2 rounded">
                              <p className="text-xs text-yellow-800 font-semibold">Global Score (Global Skor)</p>
                              <p className="text-lg font-bold text-yellow-900">{(doc.global_score * 100).toFixed(1)}%</p>
                              <p className="text-xs text-gray-600 mt-1">Tüm öğrenciler için genel başarı oranı</p>
                            </div>
                            
                            <div className="bg-orange-100 p-2 rounded">
                              <p className="text-xs text-orange-800 font-semibold">Context Score (Bağlam Skoru)</p>
                              <p className="text-lg font-bold text-orange-900">{(doc.context_score * 100).toFixed(1)}%</p>
                              <p className="text-xs text-gray-600 mt-1">Mevcut konuşma bağlamına uygunluk</p>
                            </div>
                          </div>
                          
                          <div className="bg-purple-200 p-3 rounded border-2 border-purple-400">
                            <p className="text-xs text-purple-900 font-semibold mb-1">🎯 Final Score (Final Skor)</p>
                            <p className="text-2xl font-bold text-purple-900">{(doc.final_score * 100).toFixed(1)}%</p>
                            <p className="text-xs text-gray-700 mt-2">
                              <strong>Hesaplama:</strong> Tüm skorlar birleştirilerek hesaplanır. 
                              Bu skor dökümanların sıralanmasında kullanılır.
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 3. PEDAGOJİK ANALİZ BÖLÜMÜ */}
          {(comprehensiveDebug?.pedagogical_analysis || data.pedagogical_context) && (
            <div className="border border-green-200 rounded-lg p-4 bg-green-50">
              <button
                onClick={() => toggleSection("pedagogical")}
                className="w-full flex items-center justify-between text-left"
              >
                <h3 className="text-lg font-bold text-green-900 flex items-center gap-2">
                  <Info className="w-5 h-5" />
                  3. Pedagojik Analiz (ZPD, Bloom, Cognitive Load)
                </h3>
                {expandedSections.has("pedagogical") ? (
                  <ChevronUp className="w-5 h-5" />
                ) : (
                  <ChevronDown className="w-5 h-5" />
                )}
              </button>
              
              {expandedSections.has("pedagogical") && (
                <div className="mt-4 space-y-4 text-sm">
                  {/* ZPD */}
                  {(comprehensiveDebug?.pedagogical_analysis?.zpd || personalization?.zpd_info) && (
                    <div className="bg-white p-4 rounded border border-green-100">
                      <h4 className="font-semibold text-green-800 mb-2">🎯 ZPD (Zone of Proximal Development)</h4>
                      <p className="text-gray-700 mb-3">
                        <strong>Açıklama:</strong> Öğrencinin mevcut seviyesi ve önerilen öğrenme seviyesi belirlenir.
                      </p>
                      {(() => {
                        const zpd = comprehensiveDebug?.pedagogical_analysis?.zpd || personalization?.zpd_info || {};
                        return (
                          <div className="space-y-2 text-gray-600">
                            <div className="grid grid-cols-2 gap-3">
                              <div className="bg-blue-50 p-2 rounded">
                                <p className="text-xs font-semibold text-blue-800">Mevcut Seviye</p>
                                <p className="text-lg font-bold text-blue-900">{zpd.current_level || "N/A"}</p>
                              </div>
                              <div className="bg-green-50 p-2 rounded">
                                <p className="text-xs font-semibold text-green-800">Önerilen Seviye</p>
                                <p className="text-lg font-bold text-green-900">{zpd.recommended_level || "N/A"}</p>
                              </div>
                            </div>
                            <p>Başarı Oranı: <strong>{(Number(zpd.success_rate || 0) * 100).toFixed(1)}%</strong></p>
                            <p className="text-xs text-gray-500 mt-2">
                              <strong>Prompt'a Yansıması:</strong> Önerilen seviye → Kişiselleştirme talimatlarına eklenir 
                              (örn: "elementary" ise "Temel kavramları önce açıkla" talimatı eklenir)
                            </p>
                          </div>
                        );
                      })()}
                    </div>
                  )}

                  {/* Bloom */}
                  {(comprehensiveDebug?.pedagogical_analysis?.bloom || personalization?.bloom_info) && (
                    <div className="bg-white p-4 rounded border border-green-100">
                      <h4 className="font-semibold text-green-800 mb-2">🧠 Bloom Taksonomisi</h4>
                      <p className="text-gray-700 mb-3">
                        <strong>Açıklama:</strong> Sorunun hangi bilişsel seviyeyi gerektirdiği tespit edilir.
                      </p>
                      {(() => {
                        const bloom = comprehensiveDebug?.pedagogical_analysis?.bloom || personalization?.bloom_info || {};
                        return (
                          <div className="space-y-2 text-gray-600">
                            <div className="bg-purple-50 p-3 rounded">
                              <p className="text-sm font-semibold text-purple-800">Tespit Edilen Seviye</p>
                              <p className="text-xl font-bold text-purple-900">
                                Seviye {bloom.level_index || "N/A"}: {bloom.level || "N/A"}
                              </p>
                              <p className="text-xs text-gray-600 mt-1">Güven: {(Number(bloom.confidence || 0) * 100).toFixed(1)}%</p>
                            </div>
                            <p className="text-xs text-gray-500 mt-2">
                              <strong>Prompt'a Yansıması:</strong> Bloom seviyesi → Özel pedagojik talimatlar oluşturulur 
                              (Section 9.5'te görünen talimatlar)
                            </p>
                          </div>
                        );
                      })()}
                    </div>
                  )}

                  {/* Cognitive Load */}
                  {(comprehensiveDebug?.pedagogical_analysis?.cognitive_load || personalization?.cognitive_load) && (
                    <div className="bg-white p-4 rounded border border-green-100">
                      <h4 className="font-semibold text-green-800 mb-2">⚖️ Bilişsel Yük (Cognitive Load)</h4>
                      <p className="text-gray-700 mb-3">
                        <strong>Açıklama:</strong> Yanıtın öğrenci için ne kadar zor olduğu ölçülür.
                      </p>
                      {(() => {
                        const cogLoad = comprehensiveDebug?.pedagogical_analysis?.cognitive_load || personalization?.cognitive_load || {};
                        return (
                          <div className="space-y-2 text-gray-600">
                            <div className="bg-orange-50 p-3 rounded">
                              <p className="text-sm font-semibold text-orange-800">Toplam Bilişsel Yük</p>
                              <p className="text-xl font-bold text-orange-900">{Number(cogLoad.total_load || 0).toFixed(3)}</p>
                              <p className="text-xs text-gray-600 mt-1">
                                Sadeleştirme Gerekli: <strong>{cogLoad.needs_simplification ? "Evet" : "Hayır"}</strong>
                              </p>
                            </div>
                            <p className="text-xs text-gray-500 mt-2">
                              <strong>Prompt'a Yansıması:</strong> Yüksek ise → Simplification talimatları eklenir
                            </p>
                          </div>
                        );
                      })()}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* 4. KİŞİSELLEŞTİRME PARAMETRELERİ */}
          {personalization?.personalization_factors && (
            <div className="border border-indigo-200 rounded-lg p-4 bg-indigo-50">
              <button
                onClick={() => toggleSection("personalization")}
                className="w-full flex items-center justify-between text-left"
              >
                <h3 className="text-lg font-bold text-indigo-900 flex items-center gap-2">
                  <Info className="w-5 h-5" />
                  4. Kişiselleştirme Parametreleri
                </h3>
                {expandedSections.has("personalization") ? (
                  <ChevronUp className="w-5 h-5" />
                ) : (
                  <ChevronDown className="w-5 h-5" />
                )}
              </button>
              
              {expandedSections.has("personalization") && (
                <div className="mt-4 space-y-4 text-sm">
                  <div className="bg-white p-4 rounded border border-indigo-100">
                    <p className="text-gray-700 mb-4">
                      <strong>Açıklama:</strong> Bu parametreler öğrenci profiline göre belirlenir ve prompt'a yansır.
                    </p>
                    
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-blue-50 p-3 rounded">
                        <p className="text-xs font-semibold text-blue-800">Anlama Seviyesi</p>
                        <p className="text-lg font-bold text-blue-900">{personalization.personalization_factors.understanding_level || "N/A"}</p>
                      </div>
                      
                      <div className="bg-green-50 p-3 rounded">
                        <p className="text-xs font-semibold text-green-800">Zorluk Seviyesi</p>
                        <p className="text-lg font-bold text-green-900">{personalization.personalization_factors.difficulty_level || "N/A"}</p>
                        <p className="text-xs text-gray-600 mt-1">ZPD'den gelen öneri</p>
                      </div>
                      
                      <div className="bg-purple-50 p-3 rounded">
                        <p className="text-xs font-semibold text-purple-800">Açıklama Stili</p>
                        <p className="text-lg font-bold text-purple-900">{personalization.personalization_factors.explanation_style || "N/A"}</p>
                      </div>
                      
                      <div className="bg-yellow-50 p-3 rounded">
                        <p className="text-xs font-semibold text-yellow-800">Örnekler Gerekli</p>
                        <p className="text-lg font-bold text-yellow-900">{personalization.personalization_factors.needs_examples ? "Evet" : "Hayır"}</p>
                      </div>
                    </div>
                    
                    <div className="mt-4 p-3 bg-gray-50 rounded">
                      <p className="text-xs font-semibold text-gray-800 mb-2">Prompt'a Yansıması:</p>
                      <ul className="text-xs text-gray-700 space-y-1 ml-4">
                        <li>• <strong>Zorluk Seviyesi:</strong> "elementary" ise → "Temel kavramları önce açıkla" talimatı eklenir</li>
                        <li>• <strong>Açıklama Stili:</strong> "detailed" ise → "Daha detaylı açıklamalar ekle" talimatı eklenir</li>
                        <li>• <strong>Örnekler:</strong> "true" ise → "Pratik örnekler ekle" talimatı eklenir</li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 5. KİŞİSELLEŞTİRİLMİŞ PROMPT */}
          {personalizationPrompt && (
            <div className="border border-amber-200 rounded-lg p-4 bg-amber-50">
              <button
                onClick={() => toggleSection("prompt")}
                className="w-full flex items-center justify-between text-left"
              >
                <h3 className="text-lg font-bold text-amber-900 flex items-center gap-2">
                  <Info className="w-5 h-5" />
                  5. Kişiselleştirilmiş Prompt (LLM'e Gönderilen)
                </h3>
                {expandedSections.has("prompt") ? (
                  <ChevronUp className="w-5 h-5" />
                ) : (
                  <ChevronDown className="w-5 h-5" />
                )}
              </button>
              
              {expandedSections.has("prompt") && (
                <div className="mt-4">
                  <div className="bg-black text-green-400 font-mono text-xs p-4 rounded border border-amber-200 overflow-auto max-h-96">
                    <pre className="whitespace-pre-wrap">{personalizationPrompt}</pre>
                  </div>
                  <p className="text-xs text-gray-600 mt-2">
                    <strong>Açıklama:</strong> Bu prompt, orijinal yanıtı kişiselleştirmek için LLM'e gönderilir. 
                    Yukarıdaki tüm parametreler bu prompt'a yansır.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* 6. YANIT KARŞILAŞTIRMASI */}
          {data.original_response && data.personalized_response && (
            <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
              <button
                onClick={() => toggleSection("comparison")}
                className="w-full flex items-center justify-between text-left"
              >
                <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                  <Info className="w-5 h-5" />
                  6. Orijinal vs Kişiselleştirilmiş Yanıt
                </h3>
                {expandedSections.has("comparison") ? (
                  <ChevronUp className="w-5 h-5" />
                ) : (
                  <ChevronDown className="w-5 h-5" />
                )}
              </button>
              
              {expandedSections.has("comparison") && (
                <div className="mt-4 space-y-4 text-sm">
                  {comprehensiveDebug?.response_comparison && (
                    <div className="bg-white p-3 rounded border border-gray-200 mb-4">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <p className="text-xs font-semibold text-gray-700">Orijinal Uzunluk</p>
                          <p className="text-lg font-bold">{comprehensiveDebug.response_comparison.original_length || 0} karakter</p>
                        </div>
                        <div>
                          <p className="text-xs font-semibold text-gray-700">Kişiselleştirilmiş Uzunluk</p>
                          <p className="text-lg font-bold">{comprehensiveDebug.response_comparison.personalized_length || 0} karakter</p>
                        </div>
                        <div>
                          <p className="text-xs font-semibold text-gray-700">Fark</p>
                          <p className="text-lg font-bold">{comprehensiveDebug.response_comparison.length_difference || 0} karakter</p>
                        </div>
                        <div>
                          <p className="text-xs font-semibold text-gray-700">Benzerlik Oranı</p>
                          <p className="text-lg font-bold">
                            {comprehensiveDebug.response_comparison.similarity_ratio !== undefined 
                              ? (Number(comprehensiveDebug.response_comparison.similarity_ratio) * 100).toFixed(1) + "%"
                              : "N/A"}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-red-50 p-4 rounded border border-red-200">
                      <h4 className="font-semibold text-red-800 mb-2">Orijinal Yanıt</h4>
                      <p className="text-sm text-gray-700 whitespace-pre-wrap">{data.original_response}</p>
                    </div>
                    <div className="bg-green-50 p-4 rounded border border-green-200">
                      <h4 className="font-semibold text-green-800 mb-2">Kişiselleştirilmiş Yanıt</h4>
                      <p className="text-sm text-gray-700 whitespace-pre-wrap">{data.personalized_response}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

        </div>
      )}
    </div>
  );
}

