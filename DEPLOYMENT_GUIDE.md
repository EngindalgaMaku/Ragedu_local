# 🚀 RAG Education Assistant - Server Deployment Guide

## 📋 GitHub Setup Analysis

### ✅ Current Status

- **Repository**: Connected to `https://github.com/EngindalgaMaku/Ragedu_local.git`
- **Branch**: `main` (up to date with origin)
- **Working Directory**: Clean (no uncommitted changes)
- **Git Status**: All files properly tracked and committed

### 🛡️ Security Assessment

- **✅ GOOD**: `.env` file is properly ignored by Git
- **✅ GOOD**: Comprehensive `.gitignore` file excludes sensitive data
- **✅ GOOD**: No sensitive information in Git history
- **✅ GOOD**: `.env.example` template created for safe sharing

## 🏗️ Pre-Deployment Checklist

### 1. 🔑 Environment Configuration

Before deploying, ensure you have:

1. **Copy environment template:**

   ```bash
   cp .env.example .env
   ```

2. **Update API keys in `.env`:**
   - `GROQ_API_KEY`: Your Groq API key
   - `DOCSTRANGE_API_KEY`: Your DocStrange API key
   - `JWT_SECRET_KEY`: Generate a secure key with `openssl rand -hex 32`

### 2. 🔐 Security Requirements

- [ ] Strong JWT secret key generated
- [ ] API keys obtained and secured
- [ ] Server firewall configured
- [ ] SSL/HTTPS certificates ready (for production)

### 3. 🖥️ Server Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 20GB+ free space
- **Ports**: 80, 443, 3000, 8000-8007 available

## 📥 Deployment Steps

### Step 1: Install Docker and Docker Compose

**For Ubuntu 20.04/22.04:**

```bash
# Update package index
sudo apt update

# Install prerequisites
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index again
sudo apt update

# Install Docker Engine
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose (latest version)
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add your user to docker group (optional, to run without sudo)
sudo usermod -aG docker $USER

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Verify installation
docker --version
docker-compose --version

# Apply group changes (important!)
newgrp docker
```

**Important**: After running these commands, you might need to log out and log back in, or run `newgrp docker` for the group changes to take effect.

### Step 2: Clone Repository on Server

```bash
# Clone the repository (if not already done)
git clone https://github.com/EngindalgaMaku/Ragedu_local.git
cd Ragedu_local

# Make sure you're on the main branch
git checkout main
git pull origin main
```

### Step 3: Environment Setup

```bash
# Copy and configure environment
cp .env.example .env

# Edit with your actual values
nano .env
# or
vim .env
```

**Critical values to update:**

```env
# Generate with: openssl rand -hex 32
JWT_SECRET_KEY=your-actual-secure-jwt-secret-here

# Your actual API keys
GROQ_API_KEY=your-actual-groq-api-key
DOCSTRANGE_API_KEY=your-actual-docstrange-key

# For production, update these:
CORS_ORIGINS=https://yourdomain.com
NEXT_PUBLIC_API_URL=https://yourdomain.com/api
NEXT_PUBLIC_AUTH_URL=https://yourdomain.com/auth
NEXT_PUBLIC_DEMO_MODE=false
NODE_ENV=production
```

### Step 4: Create Required Directories

```bash
# Create data directories with proper permissions
sudo mkdir -p /var/lib/rag-assistant/{chroma,database}
sudo chown -R $USER:$USER /var/lib/rag-assistant
chmod 755 /var/lib/rag-assistant/{chroma,database}
```

### Step 5: Development Deployment

For development/testing:

```bash
# Start all services
docker-compose up -d --build

# Check service status
docker-compose ps

# View logs if needed
docker-compose logs -f
```

### Step 6: Production Deployment

For production with enhanced security:

```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d --build

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

## 🔧 Service Configuration

### Port Mapping (Development)

- **Frontend**: http://server-ip:3000
- **API Gateway**: http://server-ip:8000 (internal)
- **Auth Service**: http://server-ip:8006 (internal)
- **ChromaDB**: http://server-ip:8004 (internal)
- All other services are internal only

### Port Mapping (Production)

- **HTTP**: http://server-ip:80 → Nginx reverse proxy
- **HTTPS**: https://server-ip:443 → Nginx reverse proxy
- All backend services are internal only

## 🏥 Health Checks

### Verify Deployment

```bash
# Check all containers are running
docker-compose ps

# Test API Gateway
curl http://localhost:8000/health

# Test Auth Service
curl http://localhost:8006/health

# Test Frontend
curl http://localhost:3000
```

### Service Dependencies

The services start in this order:

1. ChromaDB Service
2. Auth Service (with health check)
3. Document Processing Services
4. Model Inference Service
5. APRAG Service (depends on Auth)
6. API Gateway (depends on all services)
7. Frontend (depends on API Gateway)

## 🚨 Troubleshooting

### Common Issues

**1. Port Conflicts**

```bash
# Check what's using ports
sudo netstat -tulpn | grep :8000
sudo netstat -tulpn | grep :3000

# Kill conflicting processes
sudo fuser -k 8000/tcp
sudo fuser -k 3000/tcp
```

**2. Permission Issues**

```bash
# Fix data directory permissions
sudo chown -R $USER:docker /var/lib/rag-assistant
chmod -R 755 /var/lib/rag-assistant
```

**3. Memory Issues**

```bash
# Check available memory
free -h

# Monitor Docker resource usage
docker stats

# If needed, increase swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**4. Container Won't Start**

```bash
# Check specific container logs
docker-compose logs auth-service
docker-compose logs api-gateway

# Restart specific service
docker-compose restart auth-service

# Rebuild if needed
docker-compose up -d --build auth-service
```

## 🔄 Updating the Application

### Pull Latest Changes

```bash
# Stop services
docker-compose down

# Pull latest code
git pull origin main

# Rebuild and start
docker-compose up -d --build
```

### Database Migration

If database schema changes:

```bash
# Backup database first
cp /var/lib/rag-assistant/database/rag_assistant.db /var/lib/rag-assistant/database/rag_assistant.db.backup

# Restart services (migrations run automatically)
docker-compose restart auth-service
```

## 🔐 Production Security Considerations

### 1. Environment Variables

- Never commit `.env` files to Git
- Use strong, unique JWT secrets
- Rotate API keys regularly
- Use environment-specific values

### 2. Network Security

- Configure firewall to restrict access
- Use HTTPS in production
- Implement rate limiting
- Monitor for suspicious activity

### 3. Data Protection

- Regular database backups
- Secure file permissions
- Monitor disk usage
- Implement log rotation

### 4. SSL/HTTPS Setup

For production, configure SSL certificates:

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Configure nginx.conf for SSL
# Update CORS_ORIGINS and NEXT_PUBLIC_API_URL to use https://
```

## 🔍 Monitoring

### Service Health

```bash
# Monitor all services
docker-compose ps

# Check resource usage
docker stats

# Monitor logs
docker-compose logs -f --tail=50
```

### Performance Monitoring

- Monitor CPU and memory usage
- Check disk space regularly
- Monitor response times
- Track error rates

## 📞 Support

If you encounter issues:

1. **Check logs**: `docker-compose logs -f service-name`
2. **Verify configuration**: Ensure all environment variables are set
3. **Check dependencies**: Ensure all services are healthy
4. **Review documentation**: Check service-specific documentation

---

## 📝 Quick Commands Reference

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart specific service
docker-compose restart service-name

# View logs
docker-compose logs -f service-name

# Check service status
docker-compose ps

# Update and restart
git pull && docker-compose up -d --build

# Production deployment
docker-compose -f docker-compose.prod.yml up -d --build
```

---

**🎉 Your RAG Education Assistant is ready for deployment!**
