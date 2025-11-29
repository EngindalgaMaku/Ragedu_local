"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, User, Brain, Target, BarChart3, AlertCircle, RotateCcw, Info } from "lucide-react";

interface StudentProfileDebugCardProps {
  profileData: {
    zpd?: {
      current_level?: string;
      recommended_level?: string;
      success_rate?: number;
    };
    bloom?: {
      level?: string;
      level_index?: number;
      confidence?: number;
    };
    cognitive_load?: {
      total_load?: number;
      needs_simplification?: boolean;
    };
    personalization_factors?: {
      understanding_level?: string;
      difficulty_level?: string;
      explanation_style?: string;
    };
    profile_stats?: {
      total_interactions?: number;
      total_feedback_count?: number;
      average_understanding?: number | null;
      average_satisfaction?: number | null;
    };
  } | null;
  isLoading?: boolean;
  onReset?: () => void;
}

const StudentProfileDebugCard: React.FC<StudentProfileDebugCardProps> = ({ profileData, isLoading = false, onReset }) => {
  const [isOpen, setIsOpen] = useState(false);

  if (isLoading) {
    return (
      <div className="border border-blue-300 bg-blue-50 rounded-lg p-3 mb-4">
        <div className="flex items-center gap-2 text-sm text-blue-800">
          <Brain className="w-4 h-4 animate-pulse" />
          <span>Öğrenci profili yükleniyor...</span>
        </div>
      </div>
    );
  }

  if (!profileData) {
    return (
      <div className="border border-yellow-300 bg-yellow-50 rounded-lg p-3 mb-4">
        <div className="flex items-center gap-2 text-sm text-yellow-800">
          <AlertCircle className="w-4 h-4" />
          <span>Öğrenci profili henüz yüklenmedi. İlk sorgu sonrası parametreler görünecek.</span>
        </div>
      </div>
    );
  }

  const zpd = profileData.zpd || {};
  const bloom = profileData.bloom || {};
  const cognitiveLoad = profileData.cognitive_load || {};
  const factors = profileData.personalization_factors || {};
  const stats = profileData.profile_stats || {};

  const getZPDColor = (level?: string) => {
    const colors: Record<string, string> = {
      beginner: "bg-blue-100 text-blue-800",
      elementary: "bg-green-100 text-green-800",
      intermediate: "bg-yellow-100 text-yellow-800",
      advanced: "bg-orange-100 text-orange-800",
      expert: "bg-red-100 text-red-800",
    };
    return colors[level || ""] || "bg-gray-100 text-gray-800";
  };

  const getBloomColor = (level?: string) => {
    const colors: Record<string, string> = {
      remember: "bg-purple-100 text-purple-800",
      understand: "bg-blue-100 text-blue-800",
      apply: "bg-green-100 text-green-800",
      analyze: "bg-yellow-100 text-yellow-800",
      evaluate: "bg-orange-100 text-orange-800",
      create: "bg-red-100 text-red-800",
    };
    return colors[level || ""] || "bg-gray-100 text-gray-800";
  };

  const getLoadColor = (load?: number) => {
    if (!load) return "bg-gray-100 text-gray-800";
    if (load < 0.3) return "bg-green-100 text-green-800";
    if (load < 0.6) return "bg-yellow-100 text-yellow-800";
    return "bg-red-100 text-red-800";
  };

  return (
    <div className="border border-blue-300 bg-blue-50 rounded-lg mb-4 shadow-sm">
      <div className="px-4 py-3 flex items-center justify-between">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex-1 flex items-center justify-between hover:bg-blue-100 transition-colors rounded-lg -mx-2 px-2 py-1"
        >
          <div className="flex items-center gap-2">
            <User className="w-5 h-5 text-blue-600" />
            <span className="font-semibold text-blue-900">
              📊 Öğrenci Profil Parametreleri (Debug - Geçici)
            </span>
          </div>
          {isOpen ? (
            <ChevronUp className="w-5 h-5 text-blue-600" />
          ) : (
            <ChevronDown className="w-5 h-5 text-blue-600" />
          )}
        </button>
        {onReset && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (confirm("Öğrencinin tüm öğrenme parametrelerini sıfırlamak istediğinize emin misiniz? Bu işlem geri alınamaz.")) {
                onReset();
              }
            }}
            className="ml-2 px-3 py-1.5 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors flex items-center gap-2 text-sm"
            title="Öğrenme parametrelerini sıfırla"
          >
            <RotateCcw className="w-4 h-4" />
            <span className="hidden sm:inline">Sıfırla</span>
          </button>
        )}
      </div>

      {isOpen && (
        <div className="p-4 space-y-4 border-t border-blue-200">
          {/* ZPD Section */}
          <div className="bg-white rounded-lg p-3 border border-blue-200">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-blue-600" />
              <span className="font-semibold text-sm text-gray-700">ZPD (Zone of Proximal Development)</span>
              <div className="group relative">
                <Info className="w-3.5 h-3.5 text-gray-400 cursor-help" />
                <div className="absolute left-0 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity z-10 pointer-events-none">
                  Vygotsky'nin Yakınsal Gelişim Alanı teorisi. Öğrencinin bağımsız yapabildiği ile rehberlikle yapabileceği arasındaki optimal öğrenme bölgesini belirler. Mevcut seviye: öğrencinin şu anki yetenek seviyesi. Önerilen seviye: bir sonraki adım için ideal zorluk seviyesi.
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Mevcut Seviye:</span>
                <span className={`ml-2 px-2 py-1 rounded ${getZPDColor(zpd.current_level)}`}>
                  {zpd.current_level || "Belirlenmedi"}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Önerilen Seviye:</span>
                <span className={`ml-2 px-2 py-1 rounded ${getZPDColor(zpd.recommended_level)}`}>
                  {zpd.recommended_level || "Belirlenmedi"}
                </span>
              </div>
              <div className="col-span-2">
                <span className="text-gray-500">Başarı Oranı:</span>
                <div className="mt-1 w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all"
                    style={{ width: `${(zpd.success_rate || 0) * 100}%` }}
                  />
                </div>
                <span className="text-xs text-gray-600 ml-1">
                  {((zpd.success_rate || 0) * 100).toFixed(1)}% ({(zpd.success_rate || 0).toFixed(3)})
                </span>
              </div>
            </div>
          </div>

          {/* Bloom Taxonomy Section */}
          <div className="bg-white rounded-lg p-3 border border-blue-200">
            <div className="flex items-center gap-2 mb-2">
              <Brain className="w-4 h-4 text-purple-600" />
              <span className="font-semibold text-sm text-gray-700">Bloom Taksonomisi</span>
              <div className="group relative">
                <Info className="w-3.5 h-3.5 text-gray-400 cursor-help" />
                <div className="absolute left-0 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity z-10 pointer-events-none">
                  Bloom'un Bilişsel Taksonomisi: Öğrencinin sorduğu soruların bilişsel seviyesini belirler. L1 (Hatırlama) → L6 (Yaratma) arası seviyeler. Güven: Tespit edilen seviyenin doğruluk oranı (0-1).
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Seviye:</span>
                <span className={`ml-2 px-2 py-1 rounded ${getBloomColor(bloom.level)}`}>
                  {bloom.level || "Belirlenmedi"} (L{bloom.level_index || "?"})
                </span>
              </div>
              <div>
                <span className="text-gray-500">Güven:</span>
                <span className="ml-2 text-gray-700">
                  {((bloom.confidence || 0) * 100).toFixed(1)}% ({(bloom.confidence || 0).toFixed(3)})
                </span>
              </div>
            </div>
          </div>

          {/* Cognitive Load Section */}
          <div className="bg-white rounded-lg p-3 border border-blue-200">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-orange-600" />
              <span className="font-semibold text-sm text-gray-700">Bilişsel Yük (Cognitive Load)</span>
              <div className="group relative">
                <Info className="w-3.5 h-3.5 text-gray-400 cursor-help" />
                <div className="absolute left-0 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity z-10 pointer-events-none">
                  John Sweller'in Bilişsel Yük Teorisi. Öğrencinin işleyebileceği bilgi miktarını ölçer. 0-1 arası değer: 0-0.3 (Düşük), 0.3-0.6 (Orta), 0.6+ (Yüksek). Yüksek yük durumunda içerik sadeleştirilir.
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Toplam Yük:</span>
                <span className={`ml-2 px-2 py-1 rounded ${getLoadColor(cognitiveLoad.total_load)}`}>
                  {(cognitiveLoad.total_load || 0).toFixed(3)}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Sadeleştirme:</span>
                <span className={`ml-2 px-2 py-1 rounded ${
                  cognitiveLoad.needs_simplification 
                    ? "bg-red-100 text-red-800" 
                    : "bg-green-100 text-green-800"
                }`}>
                  {cognitiveLoad.needs_simplification ? "Gerekli" : "Gerekli Değil"}
                </span>
              </div>
            </div>
          </div>

          {/* Personalization Factors Section */}
          <div className="bg-white rounded-lg p-3 border border-blue-200">
            <div className="flex items-center gap-2 mb-2">
              <User className="w-4 h-4 text-green-600" />
              <span className="font-semibold text-sm text-gray-700">Kişiselleştirme Faktörleri</span>
              <div className="group relative">
                <Info className="w-3.5 h-3.5 text-gray-400 cursor-help" />
                <div className="absolute left-0 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity z-10 pointer-events-none">
                  Öğrencinin profilinden çıkarılan kişiselleştirme parametreleri. Bu değerler LLM prompt'una eklenerek cevaplar öğrenciye özel hale getirilir.
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Anlama Seviyesi:</span>
                <span className={`ml-2 px-2 py-1 rounded ${
                  factors.understanding_level 
                    ? getZPDColor(factors.understanding_level)
                    : "bg-gray-100 text-gray-500"
                }`}>
                  {factors.understanding_level || "Belirlenmedi"}
                </span>
                {stats.average_understanding !== null && stats.average_understanding !== undefined && (
                  <span className="ml-1 text-gray-400 text-xs">
                    (Ort: {stats.average_understanding.toFixed(2)})
                  </span>
                )}
              </div>
              <div>
                <span className="text-gray-500">Zorluk Seviyesi:</span>
                <span className={`ml-2 px-2 py-1 rounded ${
                  factors.difficulty_level 
                    ? getZPDColor(factors.difficulty_level)
                    : "bg-gray-100 text-gray-500"
                }`}>
                  {factors.difficulty_level || "Belirlenmedi"}
                </span>
              </div>
              <div className="col-span-2">
                <span className="text-gray-500">Açıklama Stili:</span>
                <span className={`ml-2 px-2 py-1 rounded ${
                  factors.explanation_style 
                    ? "bg-green-100 text-green-800"
                    : "bg-gray-100 text-gray-500"
                }`}>
                  {factors.explanation_style || "Belirlenmedi"}
                </span>
              </div>
            </div>
          </div>

          {/* Profile Statistics Section */}
          <div className="bg-white rounded-lg p-3 border border-blue-200">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-indigo-600" />
              <span className="font-semibold text-sm text-gray-700">Profil İstatistikleri</span>
              <div className="group relative">
                <Info className="w-3.5 h-3.5 text-gray-400 cursor-help" />
                <div className="absolute left-0 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity z-10 pointer-events-none">
                  Öğrencinin genel performans metrikleri. Ortalama anlama ve memnuniyet değerleri 1-5 arası ölçeklenir. Bu değerler öğrencinin öğrenme durumunu yansıtır.
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Toplam Etkileşim:</span>
                <span className="ml-2 font-semibold text-gray-700">
                  {stats.total_interactions ?? 0}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Toplam Geri Bildirim:</span>
                <span className="ml-2 font-semibold text-gray-700">
                  {stats.total_feedback_count ?? 0}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Ort. Anlama:</span>
                <span className={`ml-2 px-2 py-1 rounded ${
                  stats.average_understanding !== null && stats.average_understanding !== undefined
                    ? stats.average_understanding >= 4.0
                      ? "bg-green-100 text-green-800"
                      : stats.average_understanding >= 3.0
                      ? "bg-yellow-100 text-yellow-800"
                      : "bg-red-100 text-red-800"
                    : "bg-gray-100 text-gray-500"
                }`}>
                  {stats.average_understanding !== null && stats.average_understanding !== undefined
                    ? stats.average_understanding.toFixed(3)
                    : "Belirlenmedi"}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Ort. Memnuniyet:</span>
                <span className={`ml-2 px-2 py-1 rounded ${
                  stats.average_satisfaction !== null && stats.average_satisfaction !== undefined
                    ? stats.average_satisfaction >= 4.0
                      ? "bg-green-100 text-green-800"
                      : stats.average_satisfaction >= 3.0
                      ? "bg-yellow-100 text-yellow-800"
                      : "bg-red-100 text-red-800"
                    : "bg-gray-100 text-gray-500"
                }`}>
                  {stats.average_satisfaction !== null && stats.average_satisfaction !== undefined
                    ? stats.average_satisfaction.toFixed(3)
                    : "Belirlenmedi"}
                </span>
              </div>
            </div>
          </div>

          <div className="text-xs text-gray-500 italic pt-2 border-t border-blue-200">
            💡 Bu kart test amaçlıdır. Sistem optimize edildikten sonra kaldırılacaktır.
          </div>
        </div>
      )}
    </div>
  );
};

export default StudentProfileDebugCard;

