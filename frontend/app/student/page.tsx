"use client";

import React from "react";
import { useAuth } from "@/contexts/AuthContext";
import StudentDashboard from "@/components/StudentDashboard";

export default function StudentPage() {
  const { user } = useAuth();

  if (!user) return null;

  return (
    <div className="py-6">
      <StudentDashboard userId={user.id.toString()} />
    </div>
  );
}

