"use client";

import React from "react";
import ModernAdminLayout from "./components/ModernAdminLayout";
import ModernDashboard from "./components/ModernDashboard";

export default function AdminDashboard() {
  return (
    <ModernAdminLayout
      title="Dashboard"
      description="Sistem durumu ve genel bakış"
    >
      <ModernDashboard />
    </ModernAdminLayout>
  );
}
