"use client";

import React from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { UserPlus, Shield, Users, Settings, ChevronRight } from "lucide-react";

interface QuickAction {
  title: string;
  description: string;
  href: string;
  icon: React.ComponentType<any>;
  color: string;
}

const quickActions: QuickAction[] = [
  {
    title: "Kullanıcı Oluştur",
    description: "Sisteme yeni bir kullanıcı ekle",
    href: "/admin/users?action=create",
    color: "from-blue-500 to-blue-600",
    icon: UserPlus,
  },
  {
    title: "Rolleri Yönet",
    description: "Kullanıcı rollerini ve izinlerini yapılandır",
    href: "/admin/roles",
    color: "from-purple-500 to-purple-600",
    icon: Shield,
  },
  {
    title: "Aktif Oturumlar",
    description: "Kullanıcı oturumlarını izle ve yönet",
    href: "/admin/sessions",
    color: "from-green-500 to-green-600",
    icon: Users,
  },
  {
    title: "Sistem Ayarları",
    description: "Sistem genelindeki ayarları yapılandır",
    href: "/admin/settings",
    color: "from-gray-500 to-gray-600",
    icon: Settings,
  },
];

export default function ModernQuickActionsWidget() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg font-semibold">Hızlı Eylemler</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {quickActions.map((action, index) => {
          const IconComponent = action.icon;
          return (
            <Link key={index} href={action.href} className="block">
              <div
                className={`group relative overflow-hidden rounded-lg bg-gradient-to-r ${action.color} p-4 text-white transition-all duration-300 hover:scale-[1.02] hover:shadow-lg`}
              >
                <div className="flex items-center gap-3">
                  <div className="rounded-lg bg-white/20 p-2 group-hover:bg-white/30 transition-colors">
                    <IconComponent className="h-5 w-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="font-semibold text-white text-sm mb-1">
                      {action.title}
                    </h4>
                    <p className="text-white/80 text-xs line-clamp-1">
                      {action.description}
                    </p>
                  </div>
                  <ChevronRight className="h-4 w-4 text-white/70 group-hover:text-white group-hover:translate-x-1 transition-all flex-shrink-0" />
                </div>
              </div>
            </Link>
          );
        })}
      </CardContent>
    </Card>
  );
}
