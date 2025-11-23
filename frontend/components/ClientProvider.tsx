"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, useRef } from "react";
import {
  setGlobalApiUrl,
  getProfile,
  updateProfile,
  changePassword,
  UserProfile,
} from "@/lib/api";
import MobileMenu from "./MobileMenu";
import { BackendProvider, useBackend } from "@/contexts/BackendContext";
import { useAuth } from "@/contexts/AuthContext";
import LoadingSpinner from "./LoadingSpinner";
import Modal from "./Modal";

// This is the inner component that handles the actual UI and logic after providers are set up.
function ClientProviderInner({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const isLoginPage = pathname === "/login";

  const { apiUrl } = useBackend();
  const { isAuthenticated, isLoading, logout } = useAuth();

  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [profileFormData, setProfileFormData] = useState({
    username: "",
    email: "",
    first_name: "",
    last_name: "",
  });
  const [passwordFormData, setPasswordFormData] = useState({
    old_password: "",
    new_password: "",
    confirm_password: "",
  });
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileSuccess, setProfileSuccess] = useState<string | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsUserMenuOpen(false);
      }
    };
    if (isUserMenuOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isUserMenuOpen]);

  // Effect to set the global API URL for the api client.
  useEffect(() => {
    setGlobalApiUrl(apiUrl);
  }, [apiUrl]);

  // Global error handler for browser File System API errors
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      // Suppress "illegal path" errors from File System Access API
      if (
        event.message?.includes("illegal path") ||
        event.message?.includes("Dosya sistemi eklenemiyor") ||
        event.error?.message?.includes("illegal path") ||
        event.error?.message?.includes("Dosya sistemi eklenemiyor")
      ) {
        event.preventDefault();
        console.warn(
          "File system access cancelled or invalid path - this is normal"
        );
        return false;
      }
    };

    const handleRejection = (event: PromiseRejectionEvent) => {
      // Suppress "illegal path" errors from File System Access API
      const errorMessage =
        event.reason?.message || event.reason?.toString() || "";
      if (
        errorMessage.includes("illegal path") ||
        errorMessage.includes("Dosya sistemi eklenemiyor")
      ) {
        event.preventDefault();
        console.warn(
          "File system access cancelled or invalid path - this is normal"
        );
        return false;
      }
    };

    window.addEventListener("error", handleError);
    window.addEventListener("unhandledrejection", handleRejection);

    return () => {
      window.removeEventListener("error", handleError);
      window.removeEventListener("unhandledrejection", handleRejection);
    };
  }, []);

  // Effect to handle auth state changes and protect routes.
  useEffect(() => {
    // When loading is finished, if the user is not authenticated and not on the login page,
    // redirect them to the login page.
    if (!isLoading && !isAuthenticated && !isLoginPage) {
      router.replace("/login");
    }
  }, [isLoading, isAuthenticated, isLoginPage, router]);

  // While the AuthContext is restoring the session, show a global loading spinner.
  // This prevents a flash of un-styled or protected content.
  if (isLoading) {
    return <LoadingSpinner />;
  }

  // If we are on the login page, we don't need to render the main layout.
  // Just render the children, which will be the login page itself.
  if (isLoginPage) {
    return <>{children}</>;
  }

  // After loading, if the user is still not authenticated, we are about to redirect.
  // Return null to prevent rendering anything before the redirect happens.
  if (!isAuthenticated) {
    return null;
  }

  // If we've made it this far, the user is authenticated.
  // Render the full application layout with header, footer, and content.
  return (
    <div className="min-h-full">
      <MobileMenu
        isOpen={isMobileMenuOpen}
        onClose={() => setIsMobileMenuOpen(false)}
        onLogout={logout} // Use the correct logout function from AuthContext
      />

      {!pathname.startsWith("/admin") && !pathname.startsWith("/student") && (
        <header className="bg-card border-b border-border">
          <div className="w-full px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-3">
                {/* Eğitim Logosu */}
                <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-lg shadow-lg">
                  <svg
                    className="w-6 h-6 text-white"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M12 3L1 9l4 2.18v6L12 21l7-3.82v-6l2-1.09V17h2V9L12 3zm6.82 6L12 12.72 5.18 9 12 5.28 18.82 9zM17 15.99l-5 2.73-5-2.73v-3.72L12 15l5-2.73v3.72z" />
                  </svg>
                </div>

                {/* Başlık */}
                <div className="flex flex-col">
                  <span className="font-bold text-lg text-gray-900">
                    RAG Temelli Eğitim Asistanı
                  </span>
                  <span className="text-sm text-gray-600">
                    Yapay Zeka Destekli Öğretim Platformu
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-4">
                {/* User Menu Dropdown */}
                <div className="hidden md:block relative" ref={menuRef}>
                  <button
                    onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 rounded-lg shadow-lg hover:shadow-xl transition-all duration-200"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013 3v1"
                      />
                    </svg>
                    <span>Çıkış Yap</span>
                    <svg
                      className={`w-4 h-4 transition-transform duration-200 ${
                        isUserMenuOpen ? "rotate-180" : ""
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </button>

                  {/* Dropdown Menu */}
                  {isUserMenuOpen && (
                    <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl border border-gray-200 py-1 z-50">
                      <button
                        onClick={async () => {
                          try {
                            setProfileLoading(true);
                            const profile = await getProfile();
                            setUserProfile(profile);
                            setProfileFormData({
                              username: profile.username,
                              email: profile.email,
                              first_name: profile.first_name,
                              last_name: profile.last_name,
                            });
                            setIsUserMenuOpen(false);
                            setIsProfileModalOpen(true);
                          } catch (e: any) {
                            setProfileError(
                              e.message || "Profil bilgileri yüklenemedi"
                            );
                          } finally {
                            setProfileLoading(false);
                          }
                        }}
                        className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-2"
                      >
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                          />
                        </svg>
                        Profil Düzenle
                      </button>
                      <div className="border-t border-gray-200 my-1"></div>
                      <button
                        onClick={() => {
                          setIsUserMenuOpen(false);
                          logout();
                        }}
                        className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
                      >
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013 3v1"
                          />
                        </svg>
                        Çıkış Yap
                      </button>
                    </div>
                  )}
                </div>

                <button
                  onClick={() => setIsMobileMenuOpen(true)}
                  className="md:hidden p-2 rounded-md text-muted-foreground hover:bg-accent"
                >
                  <span className="sr-only">Menüyü Aç</span>
                  <svg
                    className="h-6 w-6"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 6h16M4 12h16m-7 6h7"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </header>
      )}

      {pathname.startsWith("/admin") ||
      pathname.startsWith("/student") ||
      pathname === "/" ? (
        children
      ) : (
        <main className="w-full px-4 sm:px-6 lg:px-10 py-8 min-h-[calc(100vh-8rem)] animate-fade-in">
          <div className="w-full">{children}</div>
        </main>
      )}

      {/* Profile Settings Modal */}
      {isProfileModalOpen && (
        <Modal
          isOpen={isProfileModalOpen}
          onClose={() => {
            setIsProfileModalOpen(false);
            setProfileError(null);
            setProfileSuccess(null);
            setPasswordFormData({
              old_password: "",
              new_password: "",
              confirm_password: "",
            });
          }}
          title="Profil Ayarları"
        >
          <div className="space-y-6">
            {profileSuccess && (
              <div className="bg-green-50 border border-green-200 text-green-800 p-4 rounded-lg">
                {profileSuccess}
              </div>
            )}
            {profileError && (
              <div className="bg-red-50 border border-red-200 text-red-800 p-4 rounded-lg">
                {profileError}
              </div>
            )}

            {/* Profile Information */}
            {userProfile && (
              <div className="mb-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <div className="text-sm text-gray-600 mb-2">
                  Kullanıcı Bilgileri
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-500">Rol</div>
                    <div className="font-semibold">{userProfile.role_name}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Hesap Durumu</div>
                    <div className="font-semibold">
                      {userProfile.is_active ? "Aktif" : "Pasif"}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Edit Profile Form */}
            <div>
              <h3 className="text-lg font-semibold mb-4 text-gray-800">
                Profil Bilgileri
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
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
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    placeholder="Kullanıcı adınız"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
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
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
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
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
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
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    />
                  </div>
                </div>
                <button
                  onClick={async () => {
                    try {
                      setProfileLoading(true);
                      setProfileError(null);
                      setProfileSuccess(null);
                      const updated = await updateProfile(profileFormData);
                      setUserProfile(updated);
                      setProfileSuccess("Profil başarıyla güncellendi!");
                      setTimeout(() => {
                        setIsProfileModalOpen(false);
                        setProfileSuccess(null);
                      }, 2000);
                    } catch (e: any) {
                      setProfileError(e.message || "Profil güncellenemedi");
                    } finally {
                      setProfileLoading(false);
                    }
                  }}
                  disabled={profileLoading}
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white px-4 py-3 rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all"
                >
                  {profileLoading ? "Güncelleniyor..." : "Profil Güncelle"}
                </button>
              </div>
            </div>

            {/* Change Password Form */}
            <div className="border-t pt-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">
                Şifre Değiştir
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Eski Şifre
                  </label>
                  <input
                    type="password"
                    value={passwordFormData.old_password}
                    onChange={(e) =>
                      setPasswordFormData({
                        ...passwordFormData,
                        old_password: e.target.value,
                      })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
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
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
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
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
                {passwordFormData.new_password &&
                  passwordFormData.confirm_password &&
                  passwordFormData.new_password !==
                    passwordFormData.confirm_password && (
                    <div className="text-red-600 text-sm">
                      Şifreler eşleşmiyor
                    </div>
                  )}
                <button
                  onClick={async () => {
                    if (
                      passwordFormData.new_password !==
                      passwordFormData.confirm_password
                    ) {
                      setProfileError("Şifreler eşleşmiyor");
                      return;
                    }
                    if (passwordFormData.new_password.length < 6) {
                      setProfileError("Yeni şifre en az 6 karakter olmalıdır");
                      return;
                    }
                    try {
                      setProfileLoading(true);
                      setProfileError(null);
                      setProfileSuccess(null);
                      await changePassword(
                        passwordFormData.old_password,
                        passwordFormData.new_password
                      );
                      setProfileSuccess("Şifre başarıyla değiştirildi!");
                      setPasswordFormData({
                        old_password: "",
                        new_password: "",
                        confirm_password: "",
                      });
                      setTimeout(() => {
                        setProfileSuccess(null);
                      }, 3000);
                    } catch (e: any) {
                      setProfileError(e.message || "Şifre değiştirilemedi");
                    } finally {
                      setProfileLoading(false);
                    }
                  }}
                  disabled={
                    profileLoading ||
                    passwordFormData.new_password !==
                      passwordFormData.confirm_password
                  }
                  className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white px-4 py-3 rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {profileLoading ? "Değiştiriliyor..." : "Şifreyi Değiştir"}
                </button>
              </div>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}

// This is the main export. It wraps the inner component with the BackendProvider.
// The AuthProvider from the context is already wrapping this component in layout.tsx.
export default function ClientProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <BackendProvider>
      <ClientProviderInner>{children}</ClientProviderInner>
    </BackendProvider>
  );
}
