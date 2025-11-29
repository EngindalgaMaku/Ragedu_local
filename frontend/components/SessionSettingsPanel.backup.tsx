"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import {
  getSessionSettings,
  updateSessionSettings,
  resetSessionSettings,
  getSessionSettingsPresets,
  applySessionSettingsPreset,
  type SessionSettings,
  type SessionSettingsUpdate,
  type SessionSettingsPresetsResponse,
} from "@/lib/api";

interface SessionSettingsPanelProps {
  sessionId: string;
  userId?: string;
  className?: string;
}

interface ToggleProps {
  id: string;
  label: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

const Toggle: React.FC<ToggleProps> = ({
  id,
  label,
  description,
  checked,
  onChange,
  disabled,
}) => (
  <div className="flex items-start justify-between p-3 border border-border rounded-lg hover:bg-muted/50 transition-colors">
    <div className="flex-1 min-w-0 pr-3">
      <label
        htmlFor={id}
        className="text-sm font-medium text-foreground cursor-pointer"
      >
        {label}
      </label>
      <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
    </div>
    <div className="flex-shrink-0">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        className={`
          relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed
          ${checked ? "bg-primary" : "bg-muted"}
        `}
        onClick={() => !disabled && onChange(!checked)}
      >
        <span
          className={`
            inline-block h-3 w-3 transform rounded-full bg-white transition-transform
            ${checked ? "translate-x-5" : "translate-x-1"}
          `}
        />
      </button>
    </div>
  </div>
);

const SessionSettingsPanel: React.FC<SessionSettingsPanelProps> = ({
  sessionId,
  userId: propUserId,
  className = "",
}) => {
  const { user } = useAuth();
  const userId = propUserId || user?.id?.toString() || "";
  const [settings, setSettings] = useState<SessionSettings | null>(null);
  const [presets, setPresets] = useState<SessionSettingsPresetsResponse | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [expandedSection, setExpandedSection] = useState<string | null>("main");

  // Load settings and presets
  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [settingsResponse, presetsResponse] = await Promise.all([
        getSessionSettings(sessionId),
        getSessionSettingsPresets(sessionId),
      ]);

      setSettings(settingsResponse.settings);
      setPresets(presetsResponse);
    } catch (e: any) {
      setError(e.message || "Ayarlar yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (sessionId && userId) {
      loadData();
    }
  }, [sessionId, userId]);

  // Update a single setting
  const updateSetting = async (
    key: keyof SessionSettingsUpdate,
    value: boolean
  ) => {
    if (!settings || saving) return;

    try {
      setSaving(true);
      setError(null);

      const updates: SessionSettingsUpdate = { [key]: value };
      const response = await updateSessionSettings(sessionId, updates, userId);

      setSettings(response.settings);
      setSuccess(
        `${getSettingLabel(key)} ${
          value ? "etkinleştirildi" : "devre dışı bırakıldı"
        }`
      );

      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(null), 3000);
    } catch (e: any) {
      setError(e.message || "Ayar güncellenemedi");
    } finally {
      setSaving(false);
    }
  };

  // Apply preset
  const applyPreset = async (presetName: string) => {
    if (saving) return;

    try {
      setSaving(true);
      setError(null);

      const response = await applySessionSettingsPreset(
        sessionId,
        presetName,
        userId
      );
      setSettings(response.settings);
      setSuccess(`${getPresetDisplayName(presetName)} profili uygulandı`);

      setTimeout(() => setSuccess(null), 3000);
    } catch (e: any) {
      setError(e.message || "Profil uygulanamadı");
    } finally {
      setSaving(false);
    }
  };

  // Reset to defaults
  const resetSettings = async () => {
    if (
      !confirm(
        "Tüm ayarları varsayılan değerlere sıfırlamak istediğinizden emin misiniz?"
      )
    ) {
      return;
    }

    try {
      setSaving(true);
      setError(null);

      const response = await resetSessionSettings(sessionId, userId);
      setSettings(response.settings);
      setSuccess("Ayarlar varsayılan değerlere sıfırlandı");

      setTimeout(() => setSuccess(null), 3000);
    } catch (e: any) {
      setError(e.message || "Ayarlar sıfırlanamadı");
    } finally {
      setSaving(false);
    }
  };

  const getSettingLabel = (key: string): string => {
    const labels: Record<string, string> = {
      enable_progressive_assessment: "İlerlemeli Değerlendirme",
      enable_personalized_responses: "Kişiselleştirilmiş Yanıtlar",
      enable_multi_dimensional_feedback: "Çok Boyutlu Geri Bildirim",
      enable_topic_analytics: "Konu Analitikleri",
      enable_cacs: "CACS Belge Skorlama",
      enable_zpd: "ZPD Hesaplama",
      enable_bloom: "Bloom Taksonomisi",
      enable_cognitive_load: "Bilişsel Yük Yönetimi",
      enable_emoji_feedback: "Emoji Geri Bildirimi",
    };
    return labels[key] || key;
  };

  const getPresetDisplayName = (presetName: string): string => {
    const names: Record<string, string> = {
      conservative: "Muhafazakar",
      balanced: "Dengeli",
      advanced: "İleri Düzey",
    };
    return names[presetName] || presetName;
  };

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  if (loading) {
    return (
      <div className={`${className}`}>
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent mb-3"></div>
          <p className="text-sm text-muted-foreground">Ayarlar yükleniyor...</p>
        </div>
      </div>
    );
  }

  if (!settings) {
    return (
      <div className={`${className}`}>
        <div className="text-center py-8">
          <p className="text-sm text-muted-foreground">Ayarlar yüklenemedi</p>
          <button
            onClick={loadData}
            className="mt-2 px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm hover:bg-primary/90"
          >
            Tekrar Dene
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-foreground">
            ⚙️ Ders Oturumu Ayarları
          </h3>
          <p className="text-sm text-muted-foreground mt-0.5">
            Öğrenciler için eğitsel özellikleri kontrol edin
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={resetSettings}
            disabled={saving}
            className="px-3 py-1.5 text-xs bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Sıfırla
          </button>
        </div>
      </div>

      {/* Messages */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3">
          <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
        </div>
      )}

      {success && (
        <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md p-3">
          <p className="text-sm text-green-800 dark:text-green-200">
            {success}
          </p>
        </div>
      )}

      {/* Quick Presets */}
      {presets && (
        <div className="border border-border rounded-lg p-4">
          <h4 className="text-sm font-medium text-foreground mb-3">
            🚀 Hızlı Profiller
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {Object.entries(presets.presets).map(([key, preset]) => (
              <button
                key={key}
                onClick={() => applyPreset(key)}
                disabled={saving}
                className="p-3 text-left border border-border rounded-md hover:bg-muted/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="text-sm font-medium text-foreground">
                  {preset.name}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {preset.description}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main Educational Features */}
      <div className="border border-border rounded-lg">
        <button
          onClick={() => toggleSection("main")}
          className="w-full flex items-center justify-between p-4 hover:bg-muted/30 transition-colors"
        >
          <h4 className="text-sm font-medium text-foreground">
            📚 Ana Eğitsel Özellikler
          </h4>
          <span className="text-muted-foreground">
            {expandedSection === "main" ? "▲" : "▼"}
          </span>
        </button>

        {expandedSection === "main" && (
          <div className="p-4 pt-0 space-y-3">
            <Toggle
              id="progressive_assessment"
              label="İlerlemeli Değerlendirme"
              description="Öğrenci ilerlemesini adım adım takip eden akıllı değerlendirme sistemi"
              checked={settings.enable_progressive_assessment}
              onChange={(checked) =>
                updateSetting("enable_progressive_assessment", checked)
              }
              disabled={saving}
            />

            <Toggle
              id="personalized_responses"
              label="Kişiselleştirilmiş Yanıtlar"
              description="Öğrenci profiline göre AI destekli yanıt kişiselleştirme"
              checked={settings.enable_personalized_responses}
              onChange={(checked) =>
                updateSetting("enable_personalized_responses", checked)
              }
              disabled={saving}
            />

            <Toggle
              id="multi_dimensional_feedback"
              label="Çok Boyutlu Geri Bildirim"
              description="Anlayış, ilgililik ve açıklık boyutlarında detaylı feedback toplama"
              checked={settings.enable_multi_dimensional_feedback}
              onChange={(checked) =>
                updateSetting("enable_multi_dimensional_feedback", checked)
              }
              disabled={saving}
            />

            <Toggle
              id="topic_analytics"
              label="Konu Bazlı Analitikler"
              description="Konu tabanlı öğrenme analizi ve ilerleme takibi"
              checked={settings.enable_topic_analytics}
              onChange={(checked) =>
                updateSetting("enable_topic_analytics", checked)
              }
              disabled={saving}
            />
          </div>
        )}
      </div>

      {/* Advanced AI Components */}
      <div className="border border-border rounded-lg">
        <button
          onClick={() => toggleSection("advanced")}
          className="w-full flex items-center justify-between p-4 hover:bg-muted/30 transition-colors"
        >
          <h4 className="text-sm font-medium text-foreground">
            🤖 Gelişmiş AI Bileşenleri
          </h4>
          <span className="text-muted-foreground">
            {expandedSection === "advanced" ? "▲" : "▼"}
          </span>
        </button>

        {expandedSection === "advanced" && (
          <div className="p-4 pt-0 space-y-3">
            <Toggle
              id="cacs"
              label="CACS Belge Skorlama"
              description="Konuşma Farkında İçerik Skorlama algoritması"
              checked={settings.enable_cacs}
              onChange={(checked) => updateSetting("enable_cacs", checked)}
              disabled={saving}
            />

            <Toggle
              id="zpd"
              label="ZPD Hesaplama"
              description="Yakınsal Gelişim Alanı (Zone of Proximal Development) analizi"
              checked={settings.enable_zpd}
              onChange={(checked) => updateSetting("enable_zpd", checked)}
              disabled={saving}
            />

            <Toggle
              id="bloom"
              label="Bloom Taksonomisi Tespiti"
              description="Soru ve yanıtların Bloom seviyesi otomatik tespiti"
              checked={settings.enable_bloom}
              onChange={(checked) => updateSetting("enable_bloom", checked)}
              disabled={saving}
            />

            <Toggle
              id="cognitive_load"
              label="Bilişsel Yük Yönetimi"
              description="Yanıtların karmaşıklık analizi ve uyarlamalı basitleştirme"
              checked={settings.enable_cognitive_load}
              onChange={(checked) =>
                updateSetting("enable_cognitive_load", checked)
              }
              disabled={saving}
            />

            <Toggle
              id="emoji_feedback"
              label="Emoji Geri Bildirim"
              description="Hızlı emoji tabanlı micro-feedback toplama sistemi"
              checked={settings.enable_emoji_feedback}
              onChange={(checked) =>
                updateSetting("enable_emoji_feedback", checked)
              }
              disabled={saving}
            />
          </div>
        )}
      </div>

      {/* Settings Summary */}
      <div className="bg-muted/30 border border-border rounded-lg p-4">
        <h4 className="text-sm font-medium text-foreground mb-2">
          📊 Ayar Özeti
        </h4>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>
            Etkin özellikler:{" "}
            {Object.values(settings).filter(Boolean).length - 2} /{" "}
            {Object.keys(settings).length - 2}
          </div>
          <div>Son güncelleme: {new Date().toLocaleString("tr-TR")}</div>
          <div>
            Öğrenci deneyimi:{" "}
            {settings.enable_progressive_assessment &&
            settings.enable_personalized_responses
              ? "Tam kişiselleştirilmiş"
              : settings.enable_personalized_responses
              ? "Kısmi kişiselleştirilmiş"
              : "Standart"}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SessionSettingsPanel;
