"use client";
import { useRouter, usePathname } from "next/navigation";

export default function LogoutButton() {
  const router = useRouter();
  const pathname = usePathname();
  if (pathname === "/login") return null;
  const onLogout = () => {
    try {
      localStorage.removeItem("isAuthenticated");
      sessionStorage.clear();
    } catch {}
    router.replace("/login");
  };
  return (
    <button
      onClick={onLogout}
      className="fixed top-3 right-3 inline-flex items-center justify-center rounded-md bg-red-600 px-3 py-2 text-white shadow hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-offset-2 z-50"
      title="Çıkış Yap"
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
    </button>
  );
}
