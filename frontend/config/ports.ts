/**
 * Merkezi Port Konfigürasyon - Frontend
 * Tüm servislerin port bilgileri burada tanımlanır
 */

// Ana Servis Portları - Environment variable'lardan alınır
export const PORTS = {
  API_GATEWAY: parseInt(
    process.env.NEXT_PUBLIC_API_GATEWAY_PORT ||
      process.env.API_GATEWAY_PORT ||
      "8000"
  ),
  AUTH_SERVICE: parseInt(
    process.env.NEXT_PUBLIC_AUTH_SERVICE_PORT ||
      process.env.AUTH_SERVICE_PORT ||
      "8006"
  ),
  FRONTEND: parseInt(
    process.env.NEXT_PUBLIC_FRONTEND_PORT || process.env.PORT || "3000"
  ),

  // Mikroservis Portları - Environment variable'lardan alınır
  DOCUMENT_PROCESSOR: parseInt(process.env.DOCUMENT_PROCESSOR_PORT || "8003"),
  MODEL_INFERENCE: parseInt(process.env.MODEL_INFERENCE_PORT || "8002"),
  CHROMADB: parseInt(process.env.CHROMADB_PORT || "8004"),
  MARKER_API: parseInt(process.env.MARKER_API_PORT || "8090"),
} as const;

// URL Builder Functions - Google Cloud Run compatible
export function getServiceUrl(
  service: keyof typeof PORTS,
  host: string = "localhost",
  useDockerNames: boolean = false
): string {
  // Check if host is already a full URL (Cloud Run)
  if (host.startsWith("http://") || host.startsWith("https://")) {
    return host;
  }

  const dockerNames = {
    API_GATEWAY: "api-gateway",
    AUTH_SERVICE: "auth-service",
    FRONTEND: "frontend",
    DOCUMENT_PROCESSOR: "document-processing-service",
    MODEL_INFERENCE: "model-inference-service",
    CHROMADB: "chromadb-service",
    MARKER_API: "marker-api",
  };

  const actualHost = useDockerNames ? dockerNames[service] : host;
  const port = PORTS[service];

  // Docker için internal port kullan - environment variable'dan alınır
  const dockerPort =
    useDockerNames && service === "API_GATEWAY"
      ? parseInt(
          process.env.API_GATEWAY_INTERNAL_PORT ||
            process.env.API_GATEWAY_PORT ||
            "8000"
        )
      : port;

  // Check if we should use HTTPS (only for Cloud Run, not Docker)
  // Docker'da her zaman HTTP kullan, Cloud Run'da environment variable'dan kontrol et
  const isDocker = process.env.DOCKER_ENV === "true" || useDockerNames;
  const isCloudRun =
    process.env.NEXT_PUBLIC_API_URL &&
    process.env.NEXT_PUBLIC_API_URL.startsWith("https://");
  const protocol = isCloudRun && !isDocker ? "https" : "http";

  return `${protocol}://${actualHost}:${dockerPort}`;
}

// Sık kullanılan URL'ler - Environment değişkenlerinden override edilebilir
// Environment detection with proper Client/Server side handling
const isServerSide = typeof window === "undefined";
const isDockerServerEnv = process.env.DOCKER_ENV === "true" && isServerSide;
const isBrowserEnv = typeof window !== "undefined";

// Get the actual host with priority: Browser External IP > Docker Service Names > Localhost
const getActualHost = () => {
  // BROWSER ENVIRONMENT: Always use external IP/hostname for client-side API calls
  if (isBrowserEnv) {
    console.log(
      "🌐 Browser environment detected, using external IP for API calls"
    );

    // Use external IP from environment variables
    if (
      process.env.NEXT_PUBLIC_API_URL &&
      process.env.NEXT_PUBLIC_API_URL.startsWith("http")
    ) {
      const hostname = new URL(process.env.NEXT_PUBLIC_API_URL).hostname;
      console.log(
        `📡 Using external hostname from NEXT_PUBLIC_API_URL: ${hostname}`
      );
      return hostname;
    }

    if (process.env.NEXT_PUBLIC_API_HOST) {
      console.log(
        `📡 Using external hostname from NEXT_PUBLIC_API_HOST: ${process.env.NEXT_PUBLIC_API_HOST}`
      );
      return process.env.NEXT_PUBLIC_API_HOST;
    }

    // Fallback to localhost for development browser testing
    console.warn(
      "⚠️ No external host configured, falling back to localhost for browser"
    );
    return "localhost";
  }

  // SERVER-SIDE DOCKER ENVIRONMENT: Use Docker service names for internal communication
  if (isDockerServerEnv) {
    console.log(
      "🐳 Docker server environment detected, using Docker service names"
    );
    if (process.env.API_GATEWAY_HOST) {
      return process.env.API_GATEWAY_HOST;
    }
    return "api-gateway";
  }

  // SERVER-SIDE NON-DOCKER: Use localhost or configured host
  console.log("🖥️ Non-Docker server environment detected");
  return process.env.API_GATEWAY_HOST || "localhost";
};

const actualHost = getActualHost();

export const URLS = {
  API_GATEWAY: isBrowserEnv
    ? process.env.NEXT_PUBLIC_API_URL ||
      getServiceUrl("API_GATEWAY", actualHost, false) // Browser: Always use external IP
    : isDockerServerEnv
    ? getServiceUrl("API_GATEWAY", actualHost, true) // Server Docker: Service names
    : getServiceUrl("API_GATEWAY", actualHost, false), // Server Non-Docker: localhost
  AUTH_SERVICE: isBrowserEnv
    ? process.env.NEXT_PUBLIC_AUTH_URL ||
      getServiceUrl("AUTH_SERVICE", actualHost, false) // Browser: Always use external IP
    : isDockerServerEnv
    ? getServiceUrl("AUTH_SERVICE", actualHost, true) // Server Docker: Service names
    : getServiceUrl("AUTH_SERVICE", actualHost, false), // Server Non-Docker: localhost
  FRONTEND: isBrowserEnv
    ? process.env.NEXT_PUBLIC_FRONTEND_URL ||
      getServiceUrl("FRONTEND", actualHost, false) // Browser: Always use external IP
    : isDockerServerEnv
    ? getServiceUrl("FRONTEND", actualHost, true) // Server Docker: Service names
    : getServiceUrl("FRONTEND", actualHost, false), // Server Non-Docker: localhost
} as const;

// Docker için URL'ler (Backward compatibility - now handled in main URLS)
export const DOCKER_URLS = {
  API_GATEWAY: getServiceUrl("API_GATEWAY", "api-gateway", true),
  AUTH_SERVICE: getServiceUrl("AUTH_SERVICE", "auth-service", true),
} as const;

// CORS için allowed origins - Environment variable'dan override edilebilir
const corsOriginsFromEnv = process.env.CORS_ORIGINS
  ? process.env.CORS_ORIGINS.split(",").map((origin) => origin.trim())
  : [];

export const CORS_ORIGINS = [
  ...corsOriginsFromEnv,
  URLS.API_GATEWAY,
  URLS.AUTH_SERVICE,
  URLS.FRONTEND,
  // Docker service names (for internal server communication)
  `http://api-gateway:${PORTS.API_GATEWAY}`,
  `http://auth-service:${PORTS.AUTH_SERVICE}`,
  `http://frontend:${PORTS.FRONTEND}`,
  // External access (for browser requests to server)
  ...(process.env.NEXT_PUBLIC_API_URL
    ? [
        process.env.NEXT_PUBLIC_API_URL,
        process.env.NEXT_PUBLIC_API_URL.replace(
          `:${PORTS.API_GATEWAY}`,
          `:${PORTS.FRONTEND}`
        ),
      ]
    : []),
  ...(process.env.NEXT_PUBLIC_AUTH_URL
    ? [process.env.NEXT_PUBLIC_AUTH_URL]
    : []),
  // Local development only (not for production server)
  ...(process.env.NODE_ENV === "development"
    ? [
        `http://127.0.0.1:${PORTS.FRONTEND}`,
        `http://localhost:${PORTS.FRONTEND}`,
        `http://host.docker.internal:${PORTS.FRONTEND}`,
        `http://host.docker.internal:${PORTS.API_GATEWAY}`,
      ]
    : []),
  // Backward compatibility
  DOCKER_URLS.API_GATEWAY,
  DOCKER_URLS.AUTH_SERVICE,
];

// Health check URL'leri
export const HEALTH_URLS = {
  API_GATEWAY: `${URLS.API_GATEWAY}/health`,
  AUTH_SERVICE: `${URLS.AUTH_SERVICE}/health`,
  FRONTEND: URLS.FRONTEND,
} as const;

// Debug için configuration'ı logla
console.log("🔧 Frontend Port Configuration DEBUG:", {
  "Environment Variables": {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
    NEXT_PUBLIC_AUTH_URL: process.env.NEXT_PUBLIC_AUTH_URL,
    NEXT_PUBLIC_API_HOST: process.env.NEXT_PUBLIC_API_HOST,
    API_GATEWAY_HOST: process.env.API_GATEWAY_HOST,
    DOCKER_ENV: process.env.DOCKER_ENV,
    NODE_ENV: process.env.NODE_ENV,
  },
  "Computed Values": {
    actualHost: actualHost,
    isBrowserEnv: isBrowserEnv,
    isServerSide: isServerSide,
    isDockerServerEnv: isDockerServerEnv,
    "API Gateway": URLS.API_GATEWAY,
    "Auth Service": URLS.AUTH_SERVICE,
    Frontend: URLS.FRONTEND,
  },
  "CORS Origins": CORS_ORIGINS.slice(0, 3).join(", ") + "...",
});

// CommonJS export for next.config.js compatibility - Removed to fix ES Modules issue
// module.exports = {
//   PORTS,
//   getServiceUrl,
//   URLS,
//   DOCKER_URLS,
//   CORS_ORIGINS,
//   HEALTH_URLS,
// };

export default URLS;
