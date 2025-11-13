/** @type {import('next').NextConfig} */

const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  experimental: {
    // Remove deprecated experimental features that might cause issues in Next.js 15
  },
  // Ensure proper handling of environment variables in production
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
  // Optimize for production builds
  compiler: {
    removeConsole: process.env.NODE_ENV === "production",
  },
  // Handle potential image optimization issues in Docker
  images: {
    unoptimized: true,
  },
  // Disable static optimization for dynamic content
  generateBuildId: async () => {
    return `build-${Date.now()}`;
  },
  // Proxy API requests to backend
  async rewrites() {
    const isDocker =
      process.env.DOCKER_ENV === "true" ||
      process.env.NODE_ENV === "production";

    // API Gateway URL - Environment variable'lardan alınır
    // Google Cloud Run için: NEXT_PUBLIC_API_URL tam URL olmalı (https://api-gateway-xxx.run.app)
    // Docker için: service name veya localhost kullanılır
    // If NEXT_PUBLIC_API_URL is a relative path like "/api", use fallback logic
    const apiUrl = (() => {
      if (
        process.env.NEXT_PUBLIC_API_URL &&
        process.env.NEXT_PUBLIC_API_URL.startsWith("http")
      ) {
        return process.env.NEXT_PUBLIC_API_URL;
      }

      const apiGatewayHost =
        process.env.API_GATEWAY_HOST ||
        (isDocker ? "api-gateway" : "localhost");
      const apiGatewayPort =
        process.env.API_GATEWAY_PORT ||
        process.env.API_GATEWAY_INTERNAL_PORT ||
        "8000";
      // Check if host is a full URL (Cloud Run)
      if (
        apiGatewayHost.startsWith("http://") ||
        apiGatewayHost.startsWith("https://")
      ) {
        return apiGatewayHost;
      }
      return isDocker
        ? `http://${apiGatewayHost}:${apiGatewayPort}`
        : `http://localhost:${apiGatewayPort}`;
    })();

    console.log("🔧 Next.js API proxy configured for:", apiUrl);
    console.log("🐳 Docker mode:", isDocker);
    console.log("🌐 Environment:", process.env.NODE_ENV || "development");

    return [
      {
        source: "/api/:path*",
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
