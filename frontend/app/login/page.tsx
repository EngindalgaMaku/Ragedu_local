"use client";

import LoginForm from "@/components/auth/LoginForm";

// Force dynamic rendering
export const dynamic = 'force-dynamic';

export default function LoginPage() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="w-full max-w-md">
        <LoginForm />
      </div>
    </div>
  );
}
