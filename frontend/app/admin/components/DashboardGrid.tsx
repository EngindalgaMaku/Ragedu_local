"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Responsive, WidthProvider, Layout } from "react-grid-layout";

const ResponsiveGridLayout = WidthProvider(Responsive);

interface DashboardGridProps {
  children: React.ReactNode[];
  className?: string;
}

type LayoutsType = {
  lg: Layout[];
  md: Layout[];
  sm: Layout[];
  xs: Layout[];
  xxs: Layout[];
};

const defaultLayouts: LayoutsType = {
  lg: [
    { i: "stats", x: 0, y: 0, w: 12, h: 4, minW: 6, minH: 4 },
    { i: "activity", x: 0, y: 4, w: 8, h: 8, minW: 6, minH: 6 },
    { i: "quick-actions", x: 8, y: 4, w: 4, h: 8, minW: 3, minH: 6 },
    { i: "system-health", x: 0, y: 12, w: 12, h: 6, minW: 6, minH: 4 },
  ],
  md: [
    { i: "stats", x: 0, y: 0, w: 10, h: 4, minW: 6, minH: 4 },
    { i: "activity", x: 0, y: 4, w: 10, h: 8, minW: 6, minH: 6 },
    { i: "quick-actions", x: 0, y: 12, w: 10, h: 8, minW: 6, minH: 6 },
    { i: "system-health", x: 0, y: 20, w: 10, h: 6, minW: 6, minH: 4 },
  ],
  sm: [
    { i: "stats", x: 0, y: 0, w: 6, h: 4, minW: 6, minH: 4 },
    { i: "activity", x: 0, y: 4, w: 6, h: 8, minW: 6, minH: 6 },
    { i: "quick-actions", x: 0, y: 12, w: 6, h: 8, minW: 6, minH: 6 },
    { i: "system-health", x: 0, y: 20, w: 6, h: 6, minW: 6, minH: 4 },
  ],
  xs: [
    { i: "stats", x: 0, y: 0, w: 4, h: 4, minW: 4, minH: 4 },
    { i: "activity", x: 0, y: 4, w: 4, h: 8, minW: 4, minH: 6 },
    { i: "quick-actions", x: 0, y: 12, w: 4, h: 8, minW: 4, minH: 6 },
    { i: "system-health", x: 0, y: 20, w: 4, h: 6, minW: 4, minH: 4 },
  ],
  xxs: [
    { i: "stats", x: 0, y: 0, w: 2, h: 4, minW: 2, minH: 4 },
    { i: "activity", x: 0, y: 4, w: 2, h: 8, minW: 2, minH: 6 },
    { i: "quick-actions", x: 0, y: 12, w: 2, h: 8, minW: 2, minH: 6 },
    { i: "system-health", x: 0, y: 20, w: 2, h: 6, minW: 2, minH: 4 },
  ],
};

const STORAGE_KEY = "admin-dashboard-layout";

export default function DashboardGrid({
  children,
  className = "",
}: DashboardGridProps) {
  const [layouts, setLayouts] = useState(defaultLayouts);
  const [mounted, setMounted] = useState(false);
  const [isCustomizing, setIsCustomizing] = useState(false);

  // Load saved layouts from localStorage on mount
  useEffect(() => {
    setMounted(true);
    const savedLayouts = localStorage.getItem(STORAGE_KEY);
    if (savedLayouts) {
      try {
        const parsedLayouts = JSON.parse(savedLayouts);
        setLayouts({ ...defaultLayouts, ...parsedLayouts });
      } catch (error) {
        console.warn(
          "Kaydedilmiş kontrol paneli yerleşimlerini ayrıştırma başarısız:",
          error
        );
      }
    }
  }, []);

  // Save layout changes to localStorage
  const handleLayoutChange = (
    layout: Layout[],
    layouts: { [key: string]: Layout[] }
  ) => {
    if (!mounted) return;

    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(layouts));
      // Cast to LayoutsType since we know it contains the required keys
      setLayouts(layouts as LayoutsType);
    } catch (error) {
      console.warn("Kontrol paneli yerleşimlerini kaydetme başarısız:", error);
    }
  };

  // Reset to default layout
  const resetLayout = () => {
    localStorage.removeItem(STORAGE_KEY);
    setLayouts(defaultLayouts);
  };

  // Toggle customization mode
  const toggleCustomization = () => {
    setIsCustomizing(!isCustomizing);
  };

  // Grid breakpoints and columns
  const breakpoints = { lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 };
  const cols = { lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 };

  // Enhanced children with keys for grid layout
  const gridChildren = useMemo(() => {
    const widgetKeys = ["stats", "activity", "quick-actions", "system-health"];

    return React.Children.map(children, (child, index) => {
      const key = widgetKeys[index] || `widget-${index}`;

      return (
        <div
          key={key}
          className={
            isCustomizing ? "ring-2 ring-indigo-300 ring-opacity-50" : ""
          }
        >
          {child}
        </div>
      );
    });
  }, [children, isCustomizing]);

  if (!mounted) {
    return (
      <div className="space-y-6">
        {React.Children.map(children, (child, index) => (
          <div key={index} className="animate-pulse">
            <div className="bg-gray-200 dark:bg-gray-700 rounded-lg h-64"></div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={`w-full ${className}`} style={{ minWidth: 0 }}>
      {/* Dashboard Controls */}
      <div className="flex items-center justify-between mb-6 p-4 bg-white/50 dark:bg-gray-800/50 rounded-lg border border-gray-200/50 dark:border-gray-700/50">
        <div className="flex items-center space-x-3">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Kontrol Paneli Yerleşimi
          </h2>
          {isCustomizing && (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">
              Özelleştirme Modu
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={toggleCustomization}
            className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
              isCustomizing
                ? "bg-indigo-600 text-white hover:bg-indigo-700"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
            }`}
          >
            {isCustomizing ? "Yerleşimi Kaydet" : "Özelleştir"}
          </button>

          <button
            onClick={resetLayout}
            className="px-3 py-1.5 text-sm font-medium text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors"
          >
            Sıfırla
          </button>
        </div>
      </div>

      {/* Responsive Grid Layout */}
      <div className="w-full" style={{ minWidth: 0 }}>
        <ResponsiveGridLayout
          className="layout"
          layouts={layouts}
          breakpoints={breakpoints}
          cols={cols}
          rowHeight={30}
          margin={[16, 16]}
          containerPadding={[0, 0]}
          onLayoutChange={handleLayoutChange}
          isDraggable={isCustomizing}
          isResizable={isCustomizing}
          useCSSTransforms={true}
          preventCollision={false}
          compactType="vertical"
          autoSize={true}
          measureBeforeMount={true}
          width={undefined}
        >
          {gridChildren}
        </ResponsiveGridLayout>
      </div>

      {/* Customization Help */}
      {isCustomizing && (
        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
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
              <h3 className="text-sm font-medium text-blue-800 dark:text-blue-200">
                Özelleştirme Modu Aktif
              </h3>
              <div className="mt-2 text-sm text-blue-700 dark:text-blue-300">
                <ul className="list-disc pl-5 space-y-1">
                  <li>
                    Widget'ları yeniden konumlandırmak için başlıklarından
                    sürükleyin
                  </li>
                  <li>
                    Widget'ları sağ alt köşesinden sürükleyerek yeniden
                    boyutlandırın
                  </li>
                  <li>Değişiklikler tarayıcınıza otomatik olarak kaydedilir</li>
                  <li>
                    Özelleştirmeyi bitirdiğinizde "Yerleşimi Kaydet"e tıklayın
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
