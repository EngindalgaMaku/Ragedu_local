"use client";

import React, { useState, useEffect } from "react";
import ModernAdminLayout from "../components/ModernAdminLayout";
import { getApiUrl } from "@/lib/api";

interface APRAGSettings {
  enabled: boolean;
  global_enabled: boolean;
  session_enabled?: boolean;
  features: {
    feedback_collection: boolean;
    personalization: boolean;
    recommendations: boolean;
    analytics: boolean;
  };
}

export default function SettingsPage() {
  const [apragSettings, setApragSettings] = useState<APRAGSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    fetchApragSettings();
  }, []);

  const fetchApragSettings = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${getApiUrl()}/aprag/settings/status`);
      if (response.ok) {
        const data = await response.json();
        setApragSettings(data);
      } else {
        // Fallback to default settings if API not available
        setApragSettings({
          enabled: false,
          global_enabled: false,
          features: {
            feedback_collection: false,
            personalization: false,
            recommendations: false,
            analytics: false,
          },
        });
      }
    } catch (err) {
      console.error("Failed to fetch APRAG settings:", err);
      setError("Ayarlar yüklenemedi");
      // Set default settings on error
      setApragSettings({
        enabled: false,
        global_enabled: false,
        features: {
          feedback_collection: false,
          personalization: false,
          recommendations: false,
          analytics: false,
        },
      });
    } finally {
      setLoading(false);
    }
  };

  const updateApragSetting = async (key: string, value: boolean, scope: string = "global") => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch(`${getApiUrl()}/aprag/settings/toggle`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          enabled: value,
          scope: scope,
          flag_key: key,
        }),
      });

      if (response.ok) {
        setSuccess("Ayar başarıyla güncellendi");
        await fetchApragSettings();
        setTimeout(() => setSuccess(null), 3000);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || "Ayar güncellenemedi");
      }
    } catch (err) {
      console.error("Failed to update setting:", err);
      setError("Ayar güncellenemedi");
    } finally {
      setSaving(false);
    }
  };

  return (
    <ModernAdminLayout
      title="Ayarlar"
      description="Sistem ayarları ve konfigürasyon"
    >
      <div className="space-y-6">
        {/* Error/Success Messages */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg
                  className="h-5 w-5 text-red-400"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-red-800 dark:text-red-200">
                  {error}
                </p>
              </div>
              <div className="ml-auto pl-3">
                <button
                  onClick={() => setError(null)}
                  className="inline-flex rounded-md bg-red-50 dark:bg-red-900/20 p-1.5 text-red-500 hover:bg-red-100 dark:hover:bg-red-900/40"
                >
                  <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                      clipRule="evenodd"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}

        {success && (
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg
                  className="h-5 w-5 text-green-400"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-green-800 dark:text-green-200">
                  {success}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* APRAG Settings Card */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div className="mb-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
              APRAG Modülü
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Adaptive Personalized RAG System - Öğrencilerin öğrenme deneyimini
              kişiselleştiren ve sürekli gelişen bir sistem modülü.
            </p>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
            </div>
          ) : (
            <>
              {/* Global Enable/Disable */}
              <div className="border-b border-gray-200 dark:border-gray-700 pb-6 mb-6">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                      APRAG Modülünü Etkinleştir
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Tüm APRAG özelliklerini global olarak açmak veya kapatmak için
                      kullanın. Kapalıyken hiçbir APRAG işlemi yapılmaz ve hiçbir yerde görünmez.
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer ml-4">
                    <input
                      type="checkbox"
                      checked={apragSettings?.global_enabled || false}
                      onChange={(e) =>
                        updateApragSetting("aprag_enabled", e.target.checked, "global")
                      }
                      disabled={saving}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 dark:peer-focus:ring-indigo-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-indigo-600"></div>
                  </label>
                </div>
              </div>

              {/* Feature Settings */}
              {apragSettings?.global_enabled && (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Özellik Ayarları
                    </h3>
                    <div className="space-y-4">
                      {/* Feedback Collection */}
                      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                        <div className="flex-1">
                          <h4 className="font-medium text-gray-900 dark:text-white mb-1">
                            Geri Bildirim Toplama
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Öğrencilerden anlama seviyesi, memnuniyet ve yeterlilik
                            hakkında geri bildirim toplar.
                          </p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer ml-4">
                          <input
                            type="checkbox"
                            checked={apragSettings?.features.feedback_collection || false}
                            onChange={(e) =>
                              updateApragSetting(
                                "aprag_feedback_collection",
                                e.target.checked,
                                "global"
                              )
                            }
                            disabled={saving || !apragSettings?.global_enabled}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 dark:peer-focus:ring-indigo-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-indigo-600"></div>
                        </label>
                      </div>

                      {/* Personalization */}
                      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                        <div className="flex-1">
                          <h4 className="font-medium text-gray-900 dark:text-white mb-1">
                            Kişiselleştirme
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Öğrenci profiline göre RAG cevaplarını adapte eder ve
                            kişiselleştirir.
                          </p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer ml-4">
                          <input
                            type="checkbox"
                            checked={apragSettings?.features.personalization || false}
                            onChange={(e) =>
                              updateApragSetting(
                                "aprag_personalization",
                                e.target.checked,
                                "global"
                              )
                            }
                            disabled={saving || !apragSettings?.global_enabled}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 dark:peer-focus:ring-indigo-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-indigo-600"></div>
                        </label>
                      </div>

                      {/* Recommendations */}
                      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                        <div className="flex-1">
                          <h4 className="font-medium text-gray-900 dark:text-white mb-1">
                            Öneriler
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Öğrencilere kişiselleştirilmiş soru ve konu önerileri
                            sunar.
                          </p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer ml-4">
                          <input
                            type="checkbox"
                            checked={apragSettings?.features.recommendations || false}
                            onChange={(e) =>
                              updateApragSetting(
                                "aprag_recommendations",
                                e.target.checked,
                                "global"
                              )
                            }
                            disabled={saving || !apragSettings?.global_enabled}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 dark:peer-focus:ring-indigo-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-indigo-600"></div>
                        </label>
                      </div>

                      {/* Analytics */}
                      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                        <div className="flex-1">
                          <h4 className="font-medium text-gray-900 dark:text-white mb-1">
                            Analitik
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Öğrenci öğrenme kalıplarını analiz eder ve raporlar
                            oluşturur.
                          </p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer ml-4">
                          <input
                            type="checkbox"
                            checked={apragSettings?.features.analytics || false}
                            onChange={(e) =>
                              updateApragSetting(
                                "aprag_analytics",
                                e.target.checked,
                                "global"
                              )
                            }
                            disabled={saving || !apragSettings?.global_enabled}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 dark:peer-focus:ring-indigo-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-indigo-600"></div>
                        </label>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Info Box */}
              {!apragSettings?.global_enabled && (
                <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg
                        className="h-5 w-5 text-blue-400"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <p className="text-sm text-blue-800 dark:text-blue-200">
                        APRAG modülü şu anda kapalı. Özellik ayarlarını görmek için
                        modülü etkinleştirin. Modül kapalıyken hiçbir APRAG özelliği
                        görünmez veya çalışmaz.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </ModernAdminLayout>
  );
}



