/**
 * Merkezi Port Konfigürasyon - Frontend
 * Tüm servislerin port bilgileri burada tanımlanır
 */

// Ana Servis Portları - Environment variable'lardan alınır
export const PORTS = {
  API_GATEWAY: parseInt(process.env.NEXT_PUBLIC_API_GATEWAY_PORT || process.env.API_GATEWAY_PORT || "8000"),
  AUTH_SERVICE: parseInt(process.env.NEXT_PUBLIC_AUTH_SERVICE_PORT || process.env.AUTH_SERVICE_PORT || "8006"),
  FRONTEND: parseInt(process.env.NEXT_PUBLIC_FRONTEND_PORT || process.env.PORT || "3000"),

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
  const dockerPort = useDockerNames && service === "API_GATEWAY" 
    ? parseInt(process.env.API_GATEWAY_INTERNAL_PORT || process.env.API_GATEWAY_PORT || "8000")
    : port;

  // Check if we should use HTTPS (only for Cloud Run, not Docker)
  // Docker'da her zaman HTTP kullan, Cloud Run'da environment variable'dan kontrol et
  const isDocker = process.env.DOCKER_ENV === "true" || useDockerNames;
  const isCloudRun = process.env.NEXT_PUBLIC_API_URL && process.env.NEXT_PUBLIC_API_URL.startsWith("https://");
  const protocol = isCloudRun && !isDocker ? "https" : "http";
  
  return `${protocol}://${actualHost}:${dockerPort}`;
}

// Sık kullanılan URL'ler - Environment değişkenlerinden override edilebilir
// Docker'da çalışırken localhost kullan, Cloud Run'da environment variable'dan al
const isDockerEnv = process.env.DOCKER_ENV === "true" || typeof window === "undefined";
export const URLS = {
  API_GATEWAY: process.env.NEXT_PUBLIC_API_URL || getServiceUrl("API_GATEWAY", "localhost", false),
  AUTH_SERVICE:
    process.env.NEXT_PUBLIC_AUTH_URL || getServiceUrl("AUTH_SERVICE", "localhost", false),
  FRONTEND: process.env.NEXT_PUBLIC_FRONTEND_URL || getServiceUrl("FRONTEND", "localhost", false),
} as const;

// Docker için URL'ler
export const DOCKER_URLS = {
  API_GATEWAY: getServiceUrl("API_GATEWAY", "localhost", true),
  AUTH_SERVICE: getServiceUrl("AUTH_SERVICE", "localhost", true),
} as const;

// CORS için allowed origins - Environment variable'dan override edilebilir
const corsOriginsFromEnv = process.env.CORS_ORIGINS 
  ? process.env.CORS_ORIGINS.split(',').map(origin => origin.trim())
  : [];

export const CORS_ORIGINS = [
  ...corsOriginsFromEnv,
  URLS.API_GATEWAY,
  URLS.AUTH_SERVICE,
  URLS.FRONTEND,
  `http://127.0.0.1:${PORTS.FRONTEND}`,
  `http://host.docker.internal:${PORTS.FRONTEND}`,
  `http://host.docker.internal:${PORTS.API_GATEWAY}`,
  DOCKER_URLS.API_GATEWAY,
  `http://frontend:${PORTS.FRONTEND}`,
  DOCKER_URLS.API_GATEWAY,
] as const;

// Health check URL'leri
export const HEALTH_URLS = {
  API_GATEWAY: `${URLS.API_GATEWAY}/health`,
  AUTH_SERVICE: `${URLS.AUTH_SERVICE}/health`,
  FRONTEND: URLS.FRONTEND,
} as const;

// Debug için configuration'ı logla
if (process.env.NODE_ENV === "development") {
  console.log("🔧 Frontend Port Configuration:", {
    "API Gateway": URLS.API_GATEWAY,
    "Auth Service": URLS.AUTH_SERVICE,
    Frontend: URLS.FRONTEND,
    "CORS Origins": CORS_ORIGINS.slice(0, 3).join(", ") + "...",
  });
}

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
