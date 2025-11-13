"use client";

import React, { createContext, useContext, useEffect, useState } from "react";

export type Theme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

interface ThemeContextType {
  theme: Theme;
  resolvedTheme: ResolvedTheme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: React.ReactNode;
  defaultTheme?: Theme;
  storageKey?: string;
}

export function ThemeProvider({
  children,
  defaultTheme = "system",
  storageKey = "admin-theme",
}: ThemeProviderProps) {
  // Initialize theme from localStorage on client side
  const getInitialTheme = (): Theme => {
    if (typeof window === "undefined") return defaultTheme;
    const savedTheme = localStorage.getItem(storageKey) as Theme;
    if (savedTheme && ["light", "dark", "system"].includes(savedTheme)) {
      return savedTheme;
    }
    return defaultTheme;
  };

  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  
  // Calculate initial resolved theme
  const getInitialResolvedTheme = (): ResolvedTheme => {
    if (typeof window === "undefined") return "light";
    const initialTheme = getInitialTheme();
    if (initialTheme === "system") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
    }
    return initialTheme as ResolvedTheme;
  };

  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>(getInitialResolvedTheme);

  // Initialize theme from localStorage on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem(storageKey) as Theme;
    if (savedTheme && ["light", "dark", "system"].includes(savedTheme)) {
      setTheme(savedTheme);
    }
  }, [storageKey]);

  // Update resolved theme based on theme preference and system preference
  useEffect(() => {
    const updateResolvedTheme = () => {
      let resolved: ResolvedTheme;

      if (theme === "system") {
        resolved = window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
      } else {
        resolved = theme as ResolvedTheme;
      }

      setResolvedTheme(resolved);

      // Update DOM
      const root = document.documentElement;
      root.classList.remove("light", "dark");
      root.classList.add(resolved);

      // Update meta theme-color
      const metaThemeColor = document.querySelector('meta[name="theme-color"]');
      if (metaThemeColor) {
        metaThemeColor.setAttribute(
          "content",
          resolved === "dark" ? "#0f172a" : "#ffffff"
        );
      }
    };

    // Run immediately on mount
    if (typeof window !== "undefined") {
      updateResolvedTheme();
    }

    // Listen for system theme changes
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = () => {
      if (theme === "system") {
        updateResolvedTheme();
      }
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [theme]);

  // Save theme to localStorage
  useEffect(() => {
    localStorage.setItem(storageKey, theme);
  }, [theme, storageKey]);

  const handleSetTheme = (newTheme: Theme) => {
    setTheme(newTheme);
  };

  const toggleTheme = () => {
    if (resolvedTheme === "light") {
      setTheme("dark");
    } else {
      setTheme("light");
    }
  };

  const value: ThemeContextType = {
    theme,
    resolvedTheme,
    setTheme: handleSetTheme,
    toggleTheme,
  };

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}

export default ThemeContext;
