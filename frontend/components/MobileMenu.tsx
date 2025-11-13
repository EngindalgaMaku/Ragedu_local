"use client";

import Link from "next/link";

interface MobileMenuProps {
  isOpen: boolean;
  onClose: () => void;
  onLogout: () => void;
}

export default function MobileMenu({
  isOpen,
  onClose,
  onLogout,
}: MobileMenuProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 z-40 md:hidden" onClick={onClose}>
      <div
        className="fixed top-0 right-0 h-full w-72 bg-background shadow-lg p-6 z-50 animate-slide-in-right"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-8">
          <h2 className="text-lg font-semibold">Menü</h2>
          <button onClick={onClose} className="p-2 -mr-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="w-6 h-6"
            >
              <path d="M12.0007 10.5865L16.9504 5.63672L18.3646 7.05093L13.4149 12.0007L18.3646 16.9504L16.9504 18.3646L12.0007 13.4149L7.05093 18.3646L5.63672 16.9504L10.5865 12.0007L5.63672 7.05093L7.05093 5.63672L12.0007 10.5865Z" />
            </svg>
          </button>
        </div>

        <div className="flex flex-col space-y-4">
          <div className="flex flex-col p-4 bg-muted/50 rounded-lg">
            <span className="text-sm font-semibold text-foreground">
              Engin DALGA
            </span>
            <span className="text-xs text-muted-foreground">
              Danışman: Serkan BALLI
            </span>
          </div>

          <button
            onClick={() => {
              onLogout();
              onClose();
            }}
            className="w-full inline-flex items-center justify-center space-x-2 rounded-md bg-destructive/10 text-destructive hover:bg-destructive/20 transition-colors px-3 py-3"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="w-5 h-5"
            >
              <path d="M10.75 3.5a.75.75 0 0 1 .75-.75h6a.75.75 0 0 1 .75.75v17a.75.75 0 0 1-.75.75h-6a.75.75 0 0 1-.75-.75v-3a.75.75 0 0 1 1.5 0v2.25h4.5V4.25h-4.5V6.5a.75.75 0 0 1-1.5 0v-3Z" />
              <path d="M3.22 12.53a.75.75 0 0 1 0-1.06l3-3a.75.75 0 1 1 1.06 1.06L5.81 11h7.44a.75.75 0 0 1 0 1.5H5.81l1.47 1.47a.75.75 0 1 1-1.06 1.06l-3-3Z" />
            </svg>
            <span className="font-medium">Çıkış Yap</span>
          </button>
        </div>
      </div>
    </div>
  );
}
