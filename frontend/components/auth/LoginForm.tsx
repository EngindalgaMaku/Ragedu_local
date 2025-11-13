/**
 * RAG Eğitim Asistanı için Geliştirilmiş Giriş Formu Bileşeni
 * Kapsamlı doğrulama ve hata yönetimi içeren modern, erişilebilir giriş formu
 */

"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  LoginFormProps,
  LoginCredentials,
  ApiError,
  ValidationErrors,
  ValidationRules,
} from "../../types/auth";
import { useLogin } from "../../hooks/useAuth";

// ===== DOĞRULAMA KURALLARI =====

const validationRules: ValidationRules = {
  username: {
    required: true,
    minLength: 3,
    maxLength: 50,
    pattern: /^[a-zA-Z0-9_@.-]+$/,
    custom: (value: string) => {
      if (value.includes("@")) {
        // Email validation
        const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailPattern.test(value)) {
          return "Lütfen geçerli bir e-posta adresi girin";
        }
      }
      return null;
    },
  },
  email: {
    required: true,
    pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  },
  password: {
    required: true,
    minLength: 6,
    maxLength: 128,
  },
  firstName: {
    required: true,
    minLength: 1,
    maxLength: 100,
  },
  lastName: {
    required: true,
    minLength: 1,
    maxLength: 100,
  },
};

// ===== DOĞRULAMA FONKSİYONLARI =====

function validateField(value: string, rules: any): string | null {
  if (rules.required && !value.trim()) {
    return "Bu alan zorunludur";
  }

  if (value && rules.minLength && value.length < rules.minLength) {
    return `Minimum ${rules.minLength} karakter olmalıdır`;
  }

  if (value && rules.maxLength && value.length > rules.maxLength) {
    return `Maksimum ${rules.maxLength} karakter olabilir`;
  }

  if (value && rules.pattern && !rules.pattern.test(value)) {
    return "Geçersiz format";
  }

  if (value && rules.custom) {
    return rules.custom(value);
  }

  return null;
}

function validateForm(credentials: LoginCredentials): ValidationErrors {
  const errors: ValidationErrors = {};

  errors.username = validateField(
    credentials.username,
    validationRules.username
  );
  errors.password = validateField(
    credentials.password,
    validationRules.password
  );

  return errors;
}

// ===== ANA GİRİŞ FORMU BİLEŞENİ =====

const LoginForm: React.FC<LoginFormProps> = ({
  onLogin,
  onError,
  redirectTo = "/",
  showRememberMe = false,
  className = "",
}) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login, isLoading, error, clearError } = useLogin();

  // Form state
  const [credentials, setCredentials] = useState<LoginCredentials>({
    username: "",
    password: "",
  });

  const [validationErrors, setValidationErrors] = useState<ValidationErrors>(
    {}
  );
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  // Get redirect URL from query params
  const redirectUrl = searchParams?.get("redirect") || redirectTo;

  // ===== FORM YÖNETİCİLERİ =====

  const handleInputChange = useCallback(
    (field: keyof LoginCredentials, value: string) => {
      setCredentials((prev) => ({ ...prev, [field]: value }));

      if (validationErrors[field]) {
        setValidationErrors((prev) => ({ ...prev, [field]: null }));
      }

      if (error) {
        clearError();
      }
    },
    [validationErrors, error, clearError]
  );

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      console.log("LoginForm: Form submitted with credentials:", credentials);

      // Clear any previous errors
      clearError();

      const errors = validateForm(credentials);
      const hasErrors = Object.values(errors).some((error) => error !== null);

      if (hasErrors) {
        console.log("LoginForm: Validation errors:", errors);
        setValidationErrors(errors);
        return;
      }

      try {
        console.log("LoginForm: Calling login function...");
        const success = await login(credentials, redirectUrl);
        console.log("LoginForm: Login result:", success);

        if (!success) {
          console.log("LoginForm: Login failed");
          // Login failed - error state is already set by useLogin hook
          // The error will be rendered by renderError()
          // Force a small delay to ensure error state is updated
          await new Promise((resolve) => setTimeout(resolve, 100));
          return;
        }

        console.log("LoginForm: Login successful");
        if (success && onLogin) {
          onLogin({} as any);
        }
      } catch (err: any) {
        console.error("LoginForm: Login error caught:", err);
        // This catch block handles unexpected errors
        // Normal login failures are handled by useLogin hook's error state
        if (onError) {
          onError(err);
        }
      }
    },
    [credentials, login, redirectUrl, onLogin, onError, clearError, error]
  );

  // ===== EFEKTLER =====

  // Debug: Monitor error state changes (development only)
  useEffect(() => {
    if (process.env.NODE_ENV === "development" && error) {
      console.log("LoginForm - Error detected:", error);
    }
  }, [error]);

  // Note: Error clearing is handled in handleInputChange

  // ===== GÖRSELLEŞTİRME YARDIMCILARI =====

  const renderInputField = (
    field: keyof LoginCredentials,
    label: string,
    type: string = "text",
    placeholder?: string,
    autoComplete?: string
  ) => {
    const hasError = validationErrors[field] || error?.status === 400;
    const fieldId = `login-${field}`;
    const hasValue = credentials[field].length > 0;

    return (
      <div className="space-y-2">
        <label
          htmlFor={fieldId}
          className={`
            block text-sm font-semibold transition-colors duration-200
            ${hasError ? "text-red-600" : "text-gray-700"}
          `}
        >
          {label}
          <span className="text-red-500 ml-1">*</span>
        </label>
        <div className="relative group">
          {field === "username" && (
            <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 group-hover:text-gray-600 transition-colors duration-200">
              <svg
                className="w-5 h-5"
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
            </div>
          )}

          {field === "password" && (
            <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 group-hover:text-gray-600 transition-colors duration-200">
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                />
              </svg>
            </div>
          )}

          <input
            id={fieldId}
            type={field === "password" && showPassword ? "text" : type}
            value={credentials[field]}
            onChange={(e) => handleInputChange(field, e.target.value)}
            placeholder={placeholder}
            autoComplete={autoComplete}
            disabled={isLoading}
            className={`
              w-full px-12 py-4 text-base rounded-xl border-2 transition-all duration-300
              bg-white/80 backdrop-blur-sm
              ${
                hasError
                  ? "border-red-300 bg-red-50/50 focus:border-red-500 focus:ring-red-500/20"
                  : hasValue
                  ? "border-green-300 bg-green-50/30 focus:border-blue-500 focus:ring-blue-500/20"
                  : "border-gray-200 focus:border-blue-500 focus:ring-blue-500/20"
              }
              focus:outline-none focus:ring-4 focus:bg-white
              disabled:opacity-50 disabled:cursor-not-allowed
              placeholder:text-gray-400
              group-hover:border-gray-300
              transform hover:scale-[1.01] focus:scale-[1.02]
            `}
            aria-invalid={hasError ? "true" : "false"}
            aria-describedby={hasError ? `${fieldId}-error` : undefined}
            spellCheck={false}
            autoCapitalize="none"
            data-lpignore="true"
          />

          {hasValue && !hasError && (
            <div className="absolute right-12 top-1/2 transform -translate-y-1/2 text-green-500">
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M5 13l4 4L19 7"
                />
              </svg>
            </div>
          )}

          {field === "password" && (
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              disabled={isLoading}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 p-2 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
              aria-label={showPassword ? "Şifreyi Gizle" : "Şifreyi Göster"}
              title={showPassword ? "Şifreyi Gizle" : "Şifreyi Göster"}
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                {showPassword ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.878 9.878L3 3m6.878 6.878L21 21"
                  />
                ) : (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                  />
                )}
              </svg>
            </button>
          )}

          {isLoading && field === "password" && (
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-500 border-t-transparent"></div>
            </div>
          )}
        </div>

        {validationErrors[field] && (
          <div className="flex items-start space-x-2 mt-2">
            <svg
              className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p
              id={`${fieldId}-error`}
              className="text-sm text-red-600 font-medium"
              role="alert"
            >
              {validationErrors[field]}
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderError = () => {
    if (!error) return null;

    const getErrorMessage = (apiError: ApiError): string => {
      if (apiError.status === 401) {
        return "Hatalı kullanıcı adı veya şifre";
      }
      if (apiError.status === 429) {
        return "Çok fazla giriş denemesi. Lütfen bir süre bekleyip tekrar deneyin.";
      }
      if (apiError.status === 500) {
        return "Sunucu hatası. Lütfen daha sonra tekrar deneyin.";
      }
      if (apiError.status === 0) {
        return "Ağ hatası. Lütfen internet bağlantınızı kontrol edin.";
      }
      return apiError.message || "Giriş başarısız. Lütfen tekrar deneyin.";
    };

    return (
      <div
        className="p-4 mb-4 bg-red-50 border-2 border-red-300 rounded-lg shadow-sm"
        role="alert"
        aria-live="polite"
      >
        <div className="flex items-start gap-3">
          <svg
            className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div className="flex-1">
            <p className="text-sm font-semibold text-red-800 mb-1">
              Giriş Başarısız
            </p>
            <p className="text-sm text-red-700">{getErrorMessage(error)}</p>
          </div>
        </div>
      </div>
    );
  };

  const renderDemoCredentials = () => {
    return null;
  };

  // ===== ANA GÖRSELLEŞTİRME =====

  return (
    <div className={`w-full max-w-md mx-auto ${className}`}>
      <div className="bg-white/95 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 p-8 space-y-8 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-indigo-500/5 pointer-events-none"></div>

        <div className="text-center relative z-10">
          <div className="relative">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-2xl transform hover:scale-105 transition-all duration-300">
              <svg
                className="w-12 h-12 text-white"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M12 1.5a.75.75 0 01.75.75V3h3.5a.75.75 0 01.75.75v15a.75.75 0 01-.75.75H6.75a.75.75 0 01-.75-.75V3.75A.75.75 0 016.75 3H11V2.25a.75.75 0 01.75-.75zM12 12.5a1 1 0 100-2 1 1 0 000 2z" />
                <path d="M4.5 3.75a.75.75 0 00-.75.75v15a3 3 0 003 3h12a3 3 0 003-3V3.75a3 3 0 00-3-3H6.75a3 3 0 00-3 3V3.75z" />
              </svg>
            </div>
            <div className="absolute inset-0 w-20 h-20 mx-auto rounded-3xl animate-ping bg-blue-400/20"></div>
            <div className="absolute inset-0 w-20 h-20 mx-auto rounded-3xl animate-pulse bg-indigo-400/10"></div>
          </div>

          <h1 className="text-3xl font-bold text-gray-800 mb-3">
            Akıllı Eğitim Asistanı
          </h1>

          <p className="text-gray-600 text-lg">Oturum Açın</p>
        </div>

        {renderDemoCredentials()}

        {renderError()}

        <form
          onSubmit={handleSubmit}
          className="space-y-6 relative z-10"
          noValidate
        >
          {renderInputField(
            "username",
            "Kullanıcı Adı veya E-posta",
            "text",
            "Kullanıcı adınızı veya e-postanızı girin",
            "username"
          )}

          {renderInputField(
            "password",
            "Şifre",
            "password",
            "Şifrenizi girin",
            "current-password"
          )}

          {showRememberMe && (
            <div className="flex items-center">
              <input
                id="remember-me"
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                disabled={isLoading}
                className="h-4 w-4 text-primary focus:ring-primary/20 border-border rounded"
              />
              <label
                htmlFor="remember-me"
                className="ml-2 block text-sm text-foreground"
              >
                Beni Hatırla
              </label>
            </div>
          )}

          <button
            type="submit"
            disabled={
              isLoading || !credentials.username || !credentials.password
            }
            className={`
              group relative w-full py-4 px-6 text-base font-semibold rounded-2xl
              transition-all duration-300 min-h-[56px] overflow-hidden
              transform hover:scale-[1.02] active:scale-[0.98]
              disabled:transform-none disabled:cursor-not-allowed
              ${
                isLoading
                  ? "bg-gradient-to-r from-blue-400 to-indigo-400 text-white cursor-wait"
                  : !credentials.username || !credentials.password
                  ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                  : "bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white shadow-lg hover:shadow-xl"
              }
              focus:outline-none focus:ring-4 focus:ring-blue-500/20
              flex items-center justify-center gap-3
            `}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-700 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>

            <div className="relative z-10 flex items-center gap-3">
              {isLoading ? (
                <>
                  <div className="relative">
                    <div className="animate-spin rounded-full h-6 w-6 border-2 border-white border-t-transparent"></div>
                    <div className="absolute inset-0 animate-ping rounded-full h-6 w-6 border border-white/50"></div>
                  </div>
                  <span className="animate-pulse">Oturum Açılıyor...</span>
                </>
              ) : (
                <>
                  <svg
                    className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-300"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1"
                    />
                  </svg>
                  <span className="group-hover:tracking-wider transition-all duration-300">
                    Oturum Aç
                  </span>
                </>
              )}
            </div>

            {!isLoading && credentials.username && credentials.password && (
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-700 ease-in-out"></div>
            )}
          </button>

          <div className="text-center">
            <div className="inline-flex items-center space-x-2 text-xs text-gray-500">
              <svg
                className="w-3 h-3 text-green-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                />
              </svg>
              <span>256-bit SSL şifreleme ile korunmaktadır</span>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};

export const CompactLoginForm: React.FC<Omit<LoginFormProps, "className">> = (
  props
) => <LoginForm {...props} className="max-w-sm" />;

export const FullLoginForm: React.FC<LoginFormProps> = (props) => (
  <LoginForm {...props} showRememberMe />
);

export default LoginForm;
