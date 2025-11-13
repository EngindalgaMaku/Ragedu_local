"use client";
import React, { createContext, useContext } from "react";
import { URLS } from "@/config/ports";

export type BackendType = "local" | "cloud";

interface BackendContextType {
  backendType: BackendType;
  apiUrl: string;
}

const BackendContext = createContext<BackendContextType | undefined>(undefined);

// Use centralized configuration
const API_URL = URLS.API_GATEWAY;

// Determine backend type based on URL
const BACKEND_TYPE: BackendType =
  API_URL.includes("localhost") || API_URL.includes("127.0.0.1")
    ? "local"
    : "cloud";

export function BackendProvider({ children }: { children: React.ReactNode }) {
  const value: BackendContextType = {
    backendType: BACKEND_TYPE,
    apiUrl: API_URL,
  };

  return (
    <BackendContext.Provider value={value}>{children}</BackendContext.Provider>
  );
}

export function useBackend() {
  const context = useContext(BackendContext);
  if (context === undefined) {
    throw new Error("useBackend must be used within a BackendProvider");
  }
  return context;
}
