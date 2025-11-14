#!/bin/bash
# URGENT CORS FIX - Complete Container Rebuild
# Bu script backend servislerini tamamen rebuild eder

echo "🚨 URGENT CORS FIX - Backend Services Rebuild"
echo "==============================================="

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}1. Stopping all backend services...${NC}"
docker-compose stop api-gateway auth-service aprag-service

echo -e "${YELLOW}2. Removing containers to force environment reload...${NC}"
docker-compose rm -f api-gateway auth-service aprag-service

echo -e "${YELLOW}3. Removing images to force complete rebuild...${NC}"
docker rmi -f rag3_for_local-api-gateway rag3_for_local-auth-service rag3_for_local-aprag-service 2>/dev/null || true

echo -e "${YELLOW}4. Rebuilding with --no-cache (this takes a few minutes)...${NC}"
docker-compose build --no-cache api-gateway auth-service aprag-service

echo -e "${YELLOW}5. Starting services with fresh environment...${NC}"
echo -e "${BLUE}Starting auth-service first...${NC}"
docker-compose up -d auth-service
sleep 10

echo -e "${BLUE}Starting aprag-service...${NC}"
docker-compose up -d aprag-service  
sleep 10

echo -e "${BLUE}Starting api-gateway...${NC}"
docker-compose up -d api-gateway
sleep 5

echo -e "${GREEN}6. Verifying CORS environment variables...${NC}"
echo -e "${BLUE}API Gateway CORS_ORIGINS:${NC}"
docker-compose exec api-gateway printenv CORS_ORIGINS

echo -e "${BLUE}Auth Service CORS_ORIGINS:${NC}"
docker-compose exec auth-service printenv CORS_ORIGINS

echo -e "${BLUE}APRAG Service CORS_ORIGINS:${NC}"
docker-compose exec aprag-service printenv CORS_ORIGINS

echo -e "${GREEN}7. Testing CORS with curl...${NC}"
echo -e "${BLUE}Testing API Gateway CORS:${NC}"
curl -H "Origin: http://46.62.254.131:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: authorization,content-type" \
     -X OPTIONS http://46.62.254.131:8000/health -v 2>&1 | grep -E "(Access-Control|HTTP/)" || echo "CORS headers not found"

echo -e "${BLUE}Testing Auth Service CORS:${NC}"  
curl -H "Origin: http://46.62.254.131:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: authorization,content-type" \
     -X OPTIONS http://46.62.254.131:8006/health -v 2>&1 | grep -E "(Access-Control|HTTP/)" || echo "CORS headers not found"

echo -e "${GREEN}8. Container Status:${NC}"
docker-compose ps api-gateway auth-service aprag-service

echo -e "${GREEN}✅ CORS Rebuild Complete!${NC}"
echo -e "${GREEN}✅ Frontend should now work at http://46.62.254.131:3000${NC}"
echo -e "${YELLOW}⚠️ If CORS errors persist, check browser developer console for specific error details${NC}"