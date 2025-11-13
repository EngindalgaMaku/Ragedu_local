"use client";

import React from "react";

const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex items-center justify-center h-screen">
      <div className="text-2xl font-semibold text-gray-500">Loading...</div>
    </div>
  );
};

export default LoadingSpinner;
