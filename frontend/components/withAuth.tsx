"use client";
import { useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";

const withAuth = <P extends object>(
  WrappedComponent: React.ComponentType<P>
) => {
  const AuthComponent = (props: P) => {
    const router = useRouter();
    const pathname = usePathname();

    useEffect(() => {
      const isAuthenticated = localStorage.getItem("isAuthenticated");
      if (!isAuthenticated && pathname !== "/login") {
        router.replace("/login");
      }
    }, [router, pathname]);

    if (pathname === "/login") {
      return <WrappedComponent {...props} />;
    }

    const isAuthenticated =
      typeof window !== "undefined"
        ? localStorage.getItem("isAuthenticated")
        : null;

    if (!isAuthenticated) {
      return null; // veya bir yükleme ekranı gösterilebilir
    }

    return <WrappedComponent {...props} />;
  };

  return AuthComponent;
};

export default withAuth;
