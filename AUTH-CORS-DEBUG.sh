#!/bin/bash
# AUTH SERVICE CORS DEBUG - Environment Variable Analysis

echo "🔍 AUTH SERVICE CORS DEBUG"
echo "=========================="

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}1. Checking docker-compose.yml CORS_ORIGINS definition...${NC}"
grep -A 2 -B 2 "CORS_ORIGINS=" rag3_for_local/docker-compose.yml | head -10

echo -e "\n${BLUE}2. Checking current environment variables in containers...${NC}"

echo -e "${YELLOW}API Gateway CORS_ORIGINS:${NC}"
docker-compose exec api-gateway printenv CORS_ORIGINS || echo "❌ Failed to read CORS_ORIGINS from api-gateway"

echo -e "${YELLOW}Auth Service CORS_ORIGINS:${NC}"
docker-compose exec auth-service printenv CORS_ORIGINS || echo "❌ Failed to read CORS_ORIGINS from auth-service"

echo -e "${YELLOW}APRAG Service CORS_ORIGINS:${NC}"
docker-compose exec aprag-service printenv CORS_ORIGINS || echo "❌ Failed to read CORS_ORIGINS from aprag-service"

echo -e "\n${BLUE}3. Testing CORS manually...${NC}"

echo -e "${YELLOW}Testing Auth Service OPTIONS request:${NC}"
curl -H "Origin: http://46.62.254.131:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: authorization,content-type" \
     -X OPTIONS http://46.62.254.131:8006/health -I 2>/dev/null | grep -i "access-control\|http/" || echo "❌ No CORS headers found"

echo -e "${YELLOW}Testing API Gateway OPTIONS request:${NC}"
curl -H "Origin: http://46.62.254.131:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: authorization,content-type" \
     -X OPTIONS http://46.62.254.131:8000/health -I 2>/dev/null | grep -i "access-control\|http/" || echo "❌ No CORS headers found"

echo -e "\n${BLUE}4. Environment Variable Verification in docker-compose context...${NC}"

echo -e "${YELLOW}Current shell CORS_ORIGINS (if set):${NC}"
echo "CORS_ORIGINS='$CORS_ORIGINS'"

echo -e "${YELLOW}Setting CORS_ORIGINS manually and restarting auth-service...${NC}"

export CORS_ORIGINS="http://localhost:3000,http://localhost:8000,http://host.docker.internal:3000,http://frontend:3000,http://api-gateway:8000,http://46.62.254.131:3000,http://46.62.254.131:8000,http://auth-service:8006"

echo "Manually set CORS_ORIGINS to: $CORS_ORIGINS"

echo -e "${BLUE}5. Quick restart with explicit environment...${NC}"
docker-compose stop auth-service
docker-compose up -d auth-service

sleep 5

echo -e "${YELLOW}Testing after restart:${NC}"
curl -H "Origin: http://46.62.254.131:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: authorization,content-type" \
     -X OPTIONS http://46.62.254.131:8006/health -I 2>/dev/null | grep -i "access-control\|http/" || echo "❌ Still no CORS headers found"

echo -e "\n${GREEN}✅ Debug Complete!${NC}"
echo -e "${YELLOW}💡 Next step: If still failing, the problem is in the auth service CORS configuration code${NC}"