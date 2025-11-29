/** @type {import('next').NextConfig} */

const nextConfig = {
  reactStrictMode: process.env.NODE_ENV === "production", // Only in production
  output: "standalone",
  // Optimize memory usage
  swcMinify: true,
  experimental: {
    // Remove deprecated experimental features that might cause issues in Next.js 15
  },
  // Reduce memory usage in development
  webpack: (config, { dev, isServer }) => {
    if (dev && !isServer) {
      // Reduce memory usage in dev mode
      config.optimization = {
        ...config.optimization,
        removeAvailableModules: false,
        removeEmptyChunks: false,
        splitChunks: false,
      };
    }
    return config;
  },
  // Ensure proper handling of environment variables in production
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
    // Pass frontend CORS origins to backend
    FRONTEND_CORS_ORIGINS: (() => {
      // Build CORS origins from environment variables
      // Cannot require TypeScript files in next.config.js, so build manually
      const corsOrigins = [];

      // Add from CORS_ORIGINS env var
      if (process.env.CORS_ORIGINS) {
        corsOrigins.push(
          ...process.env.CORS_ORIGINS.split(",").map((o) => o.trim())
        );
      }

      // Add API Gateway URL
      if (process.env.NEXT_PUBLIC_API_URL) {
        corsOrigins.push(process.env.NEXT_PUBLIC_API_URL);
      } else {
        const apiPort = process.env.API_GATEWAY_PORT || "8000";
        corsOrigins.push(`http://localhost:${apiPort}`);
        corsOrigins.push(`http://api-gateway:${apiPort}`);
      }

      // Add Auth Service URL
      if (process.env.NEXT_PUBLIC_AUTH_URL) {
        corsOrigins.push(process.env.NEXT_PUBLIC_AUTH_URL);
      } else {
        const authPort = process.env.AUTH_SERVICE_PORT || "8006";
        corsOrigins.push(`http://localhost:${authPort}`);
        corsOrigins.push(`http://auth-service:${authPort}`);
      }

      // Add Frontend URL
      const frontendPort =
        process.env.FRONTEND_PORT || process.env.PORT || "3000";
      corsOrigins.push(`http://localhost:${frontendPort}`);
      corsOrigins.push(`http://frontend:${frontendPort}`);

      // Add external server IPs if configured
      if (
        process.env.NEXT_PUBLIC_API_URL &&
        process.env.NEXT_PUBLIC_API_URL.includes("46.62.254.131")
      ) {
        corsOrigins.push("http://46.62.254.131:3000");
        corsOrigins.push("http://46.62.254.131:8000");
        corsOrigins.push("http://46.62.254.131:8006");
        corsOrigins.push("http://46.62.254.131:8007");
      }

      // Remove duplicates and return
      return [...new Set(corsOrigins)].join(",");
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
      const apiGatewayHost = process.env.API_GATEWAY_INTERNAL_URL
        ? process.env.API_GATEWAY_INTERNAL_URL.replace("http://", "").split(
            ":"
          )[0]
        : process.env.NEXT_PUBLIC_API_HOST ||
          process.env.API_GATEWAY_HOST ||
          (isDocker ? "api-gateway" : "localhost");

      const apiGatewayPort = process.env.API_GATEWAY_INTERNAL_URL
        ? process.env.API_GATEWAY_INTERNAL_URL.replace("http://", "").split(
            ":"
          )[1] || "8000"
        : process.env.API_GATEWAY_PORT ||
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
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
