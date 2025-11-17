"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../../contexts/AuthContext";
import {
  getProfile,
  updateProfile,
  changePassword,
  UserProfile,
} from "../../lib/api";
import Modal from "../../components/Modal";

export default function ProfilePage() {
  const router = useRouter();
  const auth = useAuth();
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);
  const [changingPassword, setChangingPassword] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Profile form data
  const [profileFormData, setProfileFormData] = useState({
    username: "",
    email: "",
    first_name: "",
    last_name: "",
  });

  // Password form data
  const [passwordFormData, setPasswordFormData] = useState({
    current_password: "",
    new_password: "",
    confirm_password: "",
  });

  const [activeTab, setActiveTab] = useState<"profile" | "password">("profile");

  // Load user profile
  useEffect(() => {
    if (!auth.isAuthenticated) {
      router.push("/login");
      return;
    }

    loadProfile();
  }, [auth.isAuthenticated, router]);

  const loadProfile = async () => {
    try {
      setLoading(true);
      const profile = await getProfile();
      setUserProfile(profile);
      setProfileFormData({
        username: profile.username,
        email: profile.email,
        first_name: profile.first_name,
        last_name: profile.last_name,
      });
    } catch (err: any) {
      setError(err.message || "Profil bilgileri yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  const handleProfileUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setUpdating(true);
      setError(null);
      setSuccess(null);

      const updated = await updateProfile(profileFormData);
      setUserProfile(updated);
      setSuccess("Profil başarıyla güncellendi!");

      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.message || "Profil güncellenemedi");
    } finally {
      setUpdating(false);
    }
  };

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();

    if (passwordFormData.new_password !== passwordFormData.confirm_password) {
      setError("Yeni şifreler eşleşmiyor");
      return;
    }

    if (passwordFormData.new_password.length < 6) {
      setError("Yeni şifre en az 6 karakter olmalıdır");
      return;
    }

    try {
      setChangingPassword(true);
      setError(null);
      setSuccess(null);

      await changePassword(
        passwordFormData.current_password,
        passwordFormData.new_password
      );
      setSuccess("Şifre başarıyla değiştirildi!");
      setPasswordFormData({
        current_password: "",
        new_password: "",
        confirm_password: "",
      });

      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      setError(err.message || "Şifre değiştirilemedi");
    } finally {
      setChangingPassword(false);
    }
  };

  const clearMessages = () => {
    setError(null);
    setSuccess(null);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-cyan-100 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-cyan-100 py-4 sm:py-6 lg:py-8">
      <div className="container mx-auto px-3 sm:px-4 max-w-4xl">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => router.back()}
            className="mb-4 px-4 py-3 rounded-lg font-medium border border-gray-200 text-gray-700 bg-white hover:bg-gray-50 cursor-pointer shadow-sm text-sm min-h-[44px]"
          >
            ← Geri Dön
          </button>
          <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900">Profil Ayarları</h1>
          <p className="text-gray-600 mt-2">
            Profil bilgilerinizi ve şifrenizi yönetin
          </p>
        </div>

        {/* Messages */}
        {success && (
          <div className="mb-6 bg-green-50 border border-green-200 text-green-800 p-4 rounded-lg">
            {success}
          </div>
        )}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 text-red-800 p-4 rounded-lg">
            {error}
            <button
              onClick={clearMessages}
              className="ml-2 text-red-600 hover:text-red-800"
            >
              ✕
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
              <div className="p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">
                  Profil Bilgileri
                </h3>
                {userProfile && (
                  <div className="space-y-4">
                    <div>
                      <div className="text-sm text-gray-500">Kullanıcı Adı</div>
                      <div className="font-semibold">
                        {userProfile.username}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">E-posta</div>
                      <div className="font-semibold">{userProfile.email}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">Rol</div>
                      <div className="font-semibold">
                        <span className="inline-block px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                          {userProfile.role_name}
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">Durum</div>
                      <div className="font-semibold">
                        <span
                          className={`inline-block px-2 py-1 rounded-full text-xs ${
                            userProfile.is_active
                              ? "bg-green-100 text-green-800"
                              : "bg-red-100 text-red-800"
                          }`}
                        >
                          {userProfile.is_active ? "Aktif" : "Pasif"}
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">Kayıt Tarihi</div>
                      <div className="text-sm">
                        {new Date(userProfile.created_at).toLocaleDateString(
                          "tr-TR"
                        )}
                      </div>
                    </div>
                    {userProfile.last_login && (
                      <div>
                        <div className="text-sm text-gray-500">Son Giriş</div>
                        <div className="text-sm">
                          {new Date(userProfile.last_login).toLocaleString(
                            "tr-TR"
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Navigation */}
              <div className="border-t border-gray-200 p-4 space-y-2">
                <button
                  onClick={() => setActiveTab("profile")}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-colors min-h-[44px] ${
                    activeTab === "profile"
                      ? "bg-indigo-100 text-indigo-700 font-medium"
                      : "text-gray-600 hover:bg-gray-100"
                  }`}
                >
                  📝 Profil Düzenle
                </button>
                <button
                  onClick={() => setActiveTab("password")}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-colors min-h-[44px] ${
                    activeTab === "password"
                      ? "bg-indigo-100 text-indigo-700 font-medium"
                      : "text-gray-600 hover:bg-gray-100"
                  }`}
                >
                  🔒 Şifre Değiştir
                </button>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-2">
            {activeTab === "profile" && (
              <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
                <div className="p-6 border-b border-gray-200">
                  <h3 className="text-xl font-bold text-gray-900">
                    Profil Bilgilerini Düzenle
                  </h3>
                </div>
                <div className="p-6">
                  <form onSubmit={handleProfileUpdate} className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Kullanıcı Adı
                        </label>
                        <input
                          type="text"
                          value={profileFormData.username}
                          onChange={(e) =>
                            setProfileFormData({
                              ...profileFormData,
                              username: e.target.value,
                            })
                          }
                          className="w-full px-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent min-h-[44px]"
                          required
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          E-posta
                        </label>
                        <input
                          type="email"
                          value={profileFormData.email}
                          onChange={(e) =>
                            setProfileFormData({
                              ...profileFormData,
                              email: e.target.value,
                            })
                          }
                          className="w-full px-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent min-h-[44px]"
                          required
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Ad
                        </label>
                        <input
                          type="text"
                          value={profileFormData.first_name}
                          onChange={(e) =>
                            setProfileFormData({
                              ...profileFormData,
                              first_name: e.target.value,
                            })
                          }
                          className="w-full px-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent min-h-[44px]"
                          required
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Soyad
                        </label>
                        <input
                          type="text"
                          value={profileFormData.last_name}
                          onChange={(e) =>
                            setProfileFormData({
                              ...profileFormData,
                              last_name: e.target.value,
                            })
                          }
                          className="w-full px-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent min-h-[44px]"
                          required
                        />
                      </div>
                    </div>
                    <div className="pt-4">
                      <button
                        type="submit"
                        disabled={updating}
                        className="w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white px-4 py-3 rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all min-h-[44px]"
                      >
                        {updating ? "Güncelleniyor..." : "Profil Güncelle"}
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}

            {activeTab === "password" && (
              <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
                <div className="p-6 border-b border-gray-200">
                  <h3 className="text-xl font-bold text-gray-900">
                    Şifre Değiştir
                  </h3>
                </div>
                <div className="p-6">
                  <form onSubmit={handlePasswordChange} className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Mevcut Şifre
                      </label>
                      <input
                        type="password"
                        value={passwordFormData.current_password}
                        onChange={(e) =>
                          setPasswordFormData({
                            ...passwordFormData,
                            current_password: e.target.value,
                          })
                        }
                        className="w-full px-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent min-h-[44px]"
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Yeni Şifre
                      </label>
                      <input
                        type="password"
                        value={passwordFormData.new_password}
                        onChange={(e) =>
                          setPasswordFormData({
                            ...passwordFormData,
                            new_password: e.target.value,
                          })
                        }
                        className="w-full px-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent min-h-[44px]"
                        required
                        minLength={6}
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        En az 6 karakter olmalıdır
                      </p>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Yeni Şifre (Tekrar)
                      </label>
                      <input
                        type="password"
                        value={passwordFormData.confirm_password}
                        onChange={(e) =>
                          setPasswordFormData({
                            ...passwordFormData,
                            confirm_password: e.target.value,
                          })
                        }
                        className="w-full px-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent min-h-[44px]"
                        required
                      />
                      {passwordFormData.new_password &&
                        passwordFormData.confirm_password &&
                        passwordFormData.new_password !==
                          passwordFormData.confirm_password && (
                          <p className="text-xs text-red-500 mt-1">
                            Şifreler eşleşmiyor
                          </p>
                        )}
                    </div>
                    <div className="pt-4">
                      <button
                        type="submit"
                        disabled={
                          changingPassword ||
                          passwordFormData.new_password !==
                            passwordFormData.confirm_password
                        }
                        className="w-full bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 text-white px-4 py-3 rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all min-h-[44px]"
                      >
                        {changingPassword
                          ? "Değiştiriliyor..."
                          : "Şifreyi Değiştir"}
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
