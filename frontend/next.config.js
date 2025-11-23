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
    // Pass frontend CORS origins to backend
    FRONTEND_CORS_ORIGINS: (() => {
      // Import ports config to get CORS_ORIGINS
      try {
        const path = require("path");
        const configPath = path.join(__dirname, "config", "ports.ts");
        delete require.cache[configPath]; // Clear cache for fresh computation
        const { CORS_ORIGINS } = require("./config/ports.ts");
        return CORS_ORIGINS.join(",");
      } catch (e) {
        console.warn("Could not load CORS_ORIGINS from ports.ts:", e.message);
        // Fallback CORS origins for production server
        return [
          "http://localhost:3000",
          "http://46.62.254.131:3000",
          "http://46.62.254.131:8000",
          "http://api-gateway:8000",
          "http://auth-service:8006",
        ].join(",");
      }
    })(),
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

      // Get host from environment variables - prioritize NEXT_PUBLIC_ vars
      // For server-side (Docker): use internal service name
      // For client-side (browser): use localhost or external URL
      const apiGatewayHost =
        process.env.API_GATEWAY_INTERNAL_URL ? 
          process.env.API_GATEWAY_INTERNAL_URL.replace('http://', '').split(':')[0] :
        process.env.NEXT_PUBLIC_API_HOST ||
        process.env.API_GATEWAY_HOST ||
        (isDocker ? "api-gateway" : "localhost");

      const apiGatewayPort =
        process.env.API_GATEWAY_INTERNAL_URL ?
          process.env.API_GATEWAY_INTERNAL_URL.replace('http://', '').split(':')[1] || "8000" :
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

      // For server-side rendering in Docker, use internal service name
      // For client-side (browser), use localhost
      const finalUrl = `http://${apiGatewayHost}:${apiGatewayPort}`;
      
      // Log for debugging
      console.log("🔧 Next.js API proxy configured for:", finalUrl);
      console.log("🐳 Docker mode:", isDocker);
      console.log("🌐 Environment:", process.env.NODE_ENV || "development");
      console.log("🔗 API Gateway Host:", apiGatewayHost);
      console.log("🔗 API Gateway Port:", apiGatewayPort);
      
      return finalUrl;
    })();

    return [
      {
        source: "/api/:path*",
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
