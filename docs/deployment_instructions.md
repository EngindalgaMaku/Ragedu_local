# RAG Education Assistant - Deployment Instructions

## Overview

This document provides comprehensive instructions for deploying the RAG Education Assistant in different environments, from local development to production deployment.

## System Requirements

### Minimum Requirements

- **CPU**: 2 cores, 2.4 GHz
- **RAM**: 4 GB (8 GB recommended)
- **Storage**: 10 GB free space
- **OS**: Linux, macOS, or Windows with WSL2

### Recommended Requirements

- **CPU**: 4+ cores, 3.0 GHz
- **RAM**: 8+ GB (16 GB for production)
- **Storage**: 50+ GB SSD
- **Network**: Stable internet connection for external model services

### Software Dependencies

- **Docker**: 20.10+ and Docker Compose 2.0+
- **Node.js**: 18+ (for frontend development)
- **Python**: 3.9+ (for backend development)
- **Git**: For version control

## Development Deployment

### 1. Repository Setup

```bash
# Clone the repository
git clone <repository-url>
cd rag3_for_colab

# Verify directory structure
ls -la
```

### 2. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env.development

# Edit environment file with development settings
nano .env.development
```

**Development Environment Variables:**

```bash
# Database Configuration
DATABASE_PATH=data/rag_assistant.db

# JWT Configuration (Development - generate your own!)
JWT_SECRET_KEY=dev-secret-key-change-in-production-min-32-chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Service URLs (Development)
AUTH_SERVICE_URL=http://localhost:8006
API_GATEWAY_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000

# CORS Configuration (Development)
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
CORS_CREDENTIALS=true

# Rate Limiting (Relaxed for development)
RATE_LIMIT_RPM=120
RATE_LIMIT_BURST=20

# Debug Settings
DEBUG=true
LOG_LEVEL=debug

# External API Keys (Optional - for AI models)
GROQ_API_KEY=your-groq-api-key-here
DOCSTRANGE_API_KEY=your-docstrange-api-key-here
```

### 3. Database Initialization

```bash
# Initialize database with migrations and default data
python src/database/create_admin.py

# Verify database creation
ls -la data/
sqlite3 data/rag_assistant.db ".tables"
```

### 4. Frontend Setup

```bash
# Install Node.js dependencies
cd frontend
npm install

# Copy environment file for frontend
cp .env.local.example .env.local

# Edit frontend environment
nano .env.local
```

**Frontend Environment (.env.local):**

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_AUTH_SERVICE_URL=http://localhost:8006
NEXT_PUBLIC_APP_NAME=RAG Education Assistant
NEXT_PUBLIC_DEBUG=true
```

### 5. Service Dependencies

```bash
# Install Python dependencies for auth service
cd services/auth_service
pip install -r requirements.txt

# Install Python dependencies for API gateway
cd ../../src/api
pip install -r requirements.txt
```

### 6. Start Development Services

**Option A: Using Startup Script**

```bash
# Start all services with dependency management
python scripts/start_all_services.py --mode development
```

**Option B: Manual Startup**

```bash
# Terminal 1: Start auth service
cd services/auth_service
python main.py

# Terminal 2: Start API gateway
cd src/api
python main.py

# Terminal 3: Start frontend
cd frontend
npm run dev
```

### 7. Verify Development Setup

```bash
# Run system validation
python scripts/run_system_validation.py --skip-integration

# Run integration tests (after services are running)
python scripts/run_system_validation.py

# Access the application
# - Frontend: http://localhost:3000
# - API Gateway: http://localhost:8000/docs
# - Auth Service: http://localhost:8006/docs
```

## Docker Development Deployment

### 1. Docker Compose Setup

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Docker Environment

The `docker-compose.yml` automatically uses environment variables from `.env.development`:

```yaml
version: "3.8"

services:
  auth-service:
    build:
      context: .
      dockerfile: services/auth_service/Dockerfile
    ports:
      - "8006:8006"
    environment:
      - DATABASE_PATH=/data/rag_assistant.db
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    volumes:
      - ./data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8006/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  api-gateway:
    build:
      context: .
      dockerfile: Dockerfile.gateway.local
    ports:
      - "8000:8000"
    environment:
      - AUTH_SERVICE_URL=http://auth-service:8006
    depends_on:
      - auth-service
    volumes:
      - ./data:/data

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - api-gateway
```

### 3. Docker Validation

```bash
# Check service health
docker-compose ps

# Test API endpoints
curl http://localhost:8006/health
curl http://localhost:8000/health
curl http://localhost:3000

# View logs for specific service
docker-compose logs auth-service
```

## Production Deployment

### 1. Server Preparation

**Ubuntu/Debian:**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install nginx (reverse proxy)
sudo apt install nginx certbot python3-certbot-nginx -y

# Create application directory
sudo mkdir -p /opt/rag-assistant
sudo chown $USER:$USER /opt/rag-assistant
cd /opt/rag-assistant
```

### 2. Production Environment Setup

```bash
# Clone repository
git clone <repository-url> .

# Create production environment file
cp .env.example .env.production

# Generate secure JWT secret
openssl rand -hex 32

# Edit production environment
sudo nano .env.production
```

**Production Environment Variables:**

```bash
# Database Configuration
DATABASE_PATH=/data/rag_assistant.db

# JWT Configuration (SECURE!)
JWT_SECRET_KEY=<generated-secure-32-char-key>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Service URLs (Production)
AUTH_SERVICE_URL=https://yourdomain.com/auth
API_GATEWAY_URL=https://yourdomain.com/api
FRONTEND_URL=https://yourdomain.com

# CORS Configuration (Strict)
CORS_ORIGINS=https://yourdomain.com
CORS_CREDENTIALS=true

# Rate Limiting (Strict)
RATE_LIMIT_RPM=60
RATE_LIMIT_BURST=10

# Security Settings
DEBUG=false
LOG_LEVEL=info

# External Services
GROQ_API_KEY=your-production-groq-api-key
DOCSTRANGE_API_KEY=your-production-docstrange-api-key

# SSL/TLS
SSL_CERT_PATH=/etc/ssl/certs/yourdomain.com.crt
SSL_KEY_PATH=/etc/ssl/private/yourdomain.com.key
```

### 3. SSL Certificate Setup

```bash
# Using Let's Encrypt (recommended)
sudo certbot --nginx -d yourdomain.com

# Or upload existing certificates
sudo mkdir -p /etc/ssl/certs /etc/ssl/private
sudo cp your-cert.crt /etc/ssl/certs/yourdomain.com.crt
sudo cp your-key.key /etc/ssl/private/yourdomain.com.key
sudo chmod 644 /etc/ssl/certs/yourdomain.com.crt
sudo chmod 600 /etc/ssl/private/yourdomain.com.key
```

### 4. Nginx Configuration

```bash
# Create nginx configuration
sudo nano /etc/nginx/sites-available/rag-assistant
```

**Nginx Configuration:**

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/ssl/certs/yourdomain.com.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.com.key;

    # SSL Security
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubdomains";

    # Frontend (Next.js)
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # API Gateway
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Auth Service
    location /auth/ {
        proxy_pass http://localhost:8006/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # File upload size limit
    client_max_body_size 50M;
}
```

```bash
# Enable site and restart nginx
sudo ln -s /etc/nginx/sites-available/rag-assistant /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 5. Database Initialization (Production)

```bash
# Create data directory with proper permissions
sudo mkdir -p /opt/rag-assistant/data
sudo chown $USER:$USER /opt/rag-assistant/data

# Initialize production database
python src/database/create_admin.py

# Disable demo accounts for production
python src/database/create_admin.py --disable-demo-users

# Set secure admin password
python src/database/create_admin.py --reset-admin-password
```

### 6. Production Docker Deployment

```bash
# Use production docker-compose
docker-compose -f docker-compose.prod.yml up -d --build

# Verify services
docker-compose -f docker-compose.prod.yml ps

# Check logs
docker-compose -f docker-compose.prod.yml logs -f
```

### 7. Production Validation

```bash
# Run comprehensive system validation
python scripts/run_system_validation.py --save-report production_validation.json

# Test all endpoints
curl -k https://yourdomain.com/health
curl -k https://yourdomain.com/auth/health
curl -k https://yourdomain.com/api/health

# Load testing (optional)
# Install: sudo apt install apache2-utils
ab -n 100 -c 10 https://yourdomain.com/api/health
```

## Container Registry Deployment

### 1. Build and Tag Images

```bash
# Build auth service image
docker build -t rag-assistant/auth-service:latest -f services/auth_service/Dockerfile .

# Build API gateway image
docker build -t rag-assistant/api-gateway:latest -f Dockerfile.gateway.local .

# Build frontend image
docker build -t rag-assistant/frontend:latest -f frontend/Dockerfile frontend/

# Tag for registry
docker tag rag-assistant/auth-service:latest your-registry.com/rag-assistant/auth-service:v1.0.0
docker tag rag-assistant/api-gateway:latest your-registry.com/rag-assistant/api-gateway:v1.0.0
docker tag rag-assistant/frontend:latest your-registry.com/rag-assistant/frontend:v1.0.0
```

### 2. Push to Registry

```bash
# Login to registry
docker login your-registry.com

# Push images
docker push your-registry.com/rag-assistant/auth-service:v1.0.0
docker push your-registry.com/rag-assistant/api-gateway:v1.0.0
docker push your-registry.com/rag-assistant/frontend:v1.0.0
```

### 3. Deploy from Registry

```yaml
# docker-compose.registry.yml
version: "3.8"

services:
  auth-service:
    image: your-registry.com/rag-assistant/auth-service:v1.0.0
    ports:
      - "8006:8006"
    environment:
      - DATABASE_PATH=/data/rag_assistant.db
    volumes:
      - ./data:/data

  api-gateway:
    image: your-registry.com/rag-assistant/api-gateway:v1.0.0
    ports:
      - "8000:8000"
    depends_on:
      - auth-service
    volumes:
      - ./data:/data

  frontend:
    image: your-registry.com/rag-assistant/frontend:v1.0.0
    ports:
      - "3000:3000"
    depends_on:
      - api-gateway
```

```bash
# Deploy from registry
docker-compose -f docker-compose.registry.yml up -d
```

## Kubernetes Deployment

### 1. Create Kubernetes Manifests

**Namespace:**

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-assistant
```

**ConfigMap:**

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-assistant
data:
  DATABASE_PATH: "/data/rag_assistant.db"
  AUTH_SERVICE_URL: "http://auth-service:8006"
  API_GATEWAY_URL: "http://api-gateway:8000"
  CORS_ORIGINS: "https://yourdomain.com"
  DEBUG: "false"
```

**Secret:**

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-assistant
type: Opaque
stringData:
  JWT_SECRET_KEY: "your-secure-jwt-secret-key"
  GROQ_API_KEY: "your-groq-api-key"
```

**Persistent Volume:**

```yaml
# k8s/pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: rag-data-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: /opt/rag-assistant/data
```

**Auth Service Deployment:**

```yaml
# k8s/auth-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auth-service
  namespace: rag-assistant
spec:
  replicas: 2
  selector:
    matchLabels:
      app: auth-service
  template:
    metadata:
      labels:
        app: auth-service
    spec:
      containers:
        - name: auth-service
          image: your-registry.com/rag-assistant/auth-service:v1.0.0
          ports:
            - containerPort: 8006
          envFrom:
            - configMapRef:
                name: rag-config
            - secretRef:
                name: rag-secrets
          volumeMounts:
            - name: data-volume
              mountPath: /data
          livenessProbe:
            httpGet:
              path: /health
              port: 8006
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8006
            initialDelaySeconds: 5
            periodSeconds: 5
      volumes:
        - name: data-volume
          persistentVolumeClaim:
            claimName: rag-data-pvc
```

### 2. Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n rag-assistant
kubectl get services -n rag-assistant

# View logs
kubectl logs -f deployment/auth-service -n rag-assistant
```

## Monitoring and Maintenance

### 1. Health Monitoring

```bash
# Create monitoring script
cat > /opt/rag-assistant/monitor.sh << 'EOF'
#!/bin/bash
# Health monitoring script

LOG_FILE="/var/log/rag-assistant-monitor.log"
EMAIL="admin@yourdomain.com"

check_service() {
    local service_url=$1
    local service_name=$2

    if curl -f -s $service_url > /dev/null; then
        echo "$(date): $service_name is healthy" >> $LOG_FILE
        return 0
    else
        echo "$(date): $service_name is DOWN" >> $LOG_FILE
        echo "$service_name is down" | mail -s "RAG Assistant Alert" $EMAIL
        return 1
    fi
}

# Check all services
check_service "http://localhost:8006/health" "Auth Service"
check_service "http://localhost:8000/health" "API Gateway"
check_service "http://localhost:3000" "Frontend"

# Database backup (daily)
if [ $(date +%H) -eq 2 ]; then
    cd /opt/rag-assistant
    docker-compose exec auth-service sqlite3 /data/rag_assistant.db ".backup /data/backup_$(date +%Y%m%d).db"
    echo "$(date): Database backup created" >> $LOG_FILE
fi
EOF

chmod +x /opt/rag-assistant/monitor.sh

# Add to crontab (run every 5 minutes)
echo "*/5 * * * * /opt/rag-assistant/monitor.sh" | crontab -
```

### 2. Log Management

```bash
# Configure log rotation
sudo nano /etc/logrotate.d/rag-assistant
```

```
/var/log/rag-assistant-monitor.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    create 644 root root
}
```

### 3. Updates and Maintenance

```bash
# Create update script
cat > /opt/rag-assistant/update.sh << 'EOF'
#!/bin/bash
# Update deployment script

cd /opt/rag-assistant

# Pull latest code
git pull origin main

# Backup database
docker-compose exec auth-service sqlite3 /data/rag_assistant.db ".backup /data/pre_update_backup.db"

# Rebuild and restart services
docker-compose down
docker-compose up -d --build

# Wait for services to start
sleep 30

# Validate deployment
python scripts/run_system_validation.py --save-report post_update_validation.json

echo "Update completed at $(date)"
EOF

chmod +x /opt/rag-assistant/update.sh
```

## Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check logs
docker-compose logs auth-service
docker-compose logs api-gateway

# Check ports
netstat -tlnp | grep :8006
netstat -tlnp | grep :8000

# Check environment variables
docker-compose exec auth-service env | grep JWT_SECRET_KEY
```

#### Database Issues

```bash
# Check database file permissions
ls -la data/rag_assistant.db

# Verify database integrity
sqlite3 data/rag_assistant.db "PRAGMA integrity_check;"

# Recreate database if corrupted
mv data/rag_assistant.db data/rag_assistant.db.corrupted
python src/database/create_admin.py
```

#### SSL Certificate Issues

```bash
# Check certificate expiration
openssl x509 -in /etc/ssl/certs/yourdomain.com.crt -text -noout | grep "Not After"

# Renew Let's Encrypt certificate
sudo certbot renew --nginx

# Test SSL configuration
sudo nginx -t
```

#### Performance Issues

```bash
# Check resource usage
docker stats

# Check database performance
sqlite3 data/rag_assistant.db "PRAGMA optimize;"

# Monitor response times
curl -w "@curl-format.txt" -o /dev/null -s https://yourdomain.com/api/health
```

### Recovery Procedures

#### Complete System Recovery

```bash
# Stop all services
docker-compose down

# Restore from backup
cp data/backup_20240101.db data/rag_assistant.db

# Restart services
docker-compose up -d

# Validate system
python scripts/run_system_validation.py
```

This comprehensive deployment guide should enable successful deployment of the RAG Education Assistant in any environment, from development to production.
