import "./globals.css";
import type { Metadata } from "next";
import ClientProvider from "@/components/ClientProvider";
import { AuthProvider } from "@/contexts/AuthContext";
import { ThemeProvider } from "@/contexts/ThemeContext";

export const metadata: Metadata = {
  title: "RAG Eğitim Platformu",
  description: "Yapay Zeka Destekli Kişiselleştirilmiş Eğitim Platformu",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="tr" className="h-full" suppressHydrationWarning>
      <body className="h-full bg-background text-foreground font-sans antialiased overflow-x-hidden">
        <ThemeProvider>
          <AuthProvider>
            <ClientProvider>{children}</ClientProvider>
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
