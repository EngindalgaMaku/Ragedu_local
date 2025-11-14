#!/bin/bash
# FINAL CORS FIX - Complete rebuild with code changes

echo "🚨 FINAL CORS FIX - Complete Backend Rebuild"
echo "============================================="

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}This script will:${NC}"
echo "1. ✅ Stop all backend services"
echo "2. ✅ Remove containers and images for clean rebuild"  
echo "3. ✅ Rebuild with updated CORS configurations"
echo "4. ✅ Start services with proper external IP support"
echo "5. ✅ Verify CORS is working"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."

echo -e "${BLUE}Step 1: Stopping backend services...${NC}"
docker-compose stop api-gateway auth-service aprag-service

echo -e "${BLUE}Step 2: Removing containers and images...${NC}"
docker-compose rm -f api-gateway auth-service aprag-service
docker rmi -f rag3_for_local-api-gateway rag3_for_local-auth-service rag3_for_local-aprag-service 2>/dev/null || true

echo -e "${BLUE}Step 3: Rebuilding with updated CORS code (--no-cache)...${NC}"
echo -e "${YELLOW}This may take 3-5 minutes...${NC}"
docker-compose build --no-cache api-gateway auth-service aprag-service

echo -e "${BLUE}Step 4: Starting services...${NC}"
echo -e "${YELLOW}Starting auth-service first...${NC}"
docker-compose up -d auth-service
sleep 15

echo -e "${YELLOW}Starting aprag-service...${NC}"
docker-compose up -d aprag-service
sleep 10

echo -e "${YELLOW}Starting api-gateway...${NC}"
docker-compose up -d api-gateway
sleep 10

echo -e "${BLUE}Step 5: Verifying services are running...${NC}"
docker-compose ps api-gateway auth-service aprag-service

echo -e "${BLUE}Step 6: Testing CORS headers...${NC}"

echo -e "${YELLOW}Testing Auth Service (port 8006):${NC}"
CORS_TEST_AUTH=$(curl -s -H "Origin: http://46.62.254.131:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: authorization,content-type" \
     -X OPTIONS http://46.62.254.131:8006/health -I 2>/dev/null)

if echo "$CORS_TEST_AUTH" | grep -q "Access-Control-Allow-Origin"; then
    echo -e "${GREEN}✅ Auth Service CORS is working!${NC}"
    echo "$CORS_TEST_AUTH" | grep "Access-Control"
else
    echo -e "${RED}❌ Auth Service CORS still not working${NC}"
    echo "Response headers:"
    echo "$CORS_TEST_AUTH"
fi

echo ""
echo -e "${YELLOW}Testing API Gateway (port 8000):${NC}"
CORS_TEST_API=$(curl -s -H "Origin: http://46.62.254.131:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: authorization,content-type" \
     -X OPTIONS http://46.62.254.131:8000/health -I 2>/dev/null)

if echo "$CORS_TEST_API" | grep -q "Access-Control-Allow-Origin"; then
    echo -e "${GREEN}✅ API Gateway CORS is working!${NC}"
    echo "$CORS_TEST_API" | grep "Access-Control"
else
    echo -e "${RED}❌ API Gateway CORS still not working${NC}"
    echo "Response headers:"
    echo "$CORS_TEST_API"
fi

echo ""
echo -e "${BLUE}Step 7: Final verification with actual GET request...${NC}"

echo -e "${YELLOW}Testing actual GET request to auth service health:${NC}"
HEALTH_TEST=$(curl -s -H "Origin: http://46.62.254.131:3000" \
     http://46.62.254.131:8006/health 2>/dev/null)

if echo "$HEALTH_TEST" | grep -q "healthy"; then
    echo -e "${GREEN}✅ Auth Service health endpoint responding!${NC}"
    echo "Response: $HEALTH_TEST"
else
    echo -e "${RED}❌ Auth Service health endpoint not responding${NC}"
fi

echo ""
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}🎉 CORS FIX COMPLETE!${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Open browser to http://46.62.254.131:3000"
echo "2. Open Developer Console (F12)"
echo "3. Check if CORS errors are gone"
echo ""
echo -e "${YELLOW}If CORS errors persist, check:${NC}"
echo "- Browser cache (try hard refresh: Ctrl+F5)"
echo "- Network tab in developer tools for actual response headers"
echo "- Container logs: docker-compose logs auth-service"