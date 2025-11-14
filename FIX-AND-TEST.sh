#!/bin/bash

echo "🚨 HIZLI DÜZELTİP TEST - 30 saniyede çözüm"
echo "========================================="

# Stop everything
echo "⏹️  Stopping auth service..."
docker-compose stop auth-service
docker-compose rm -f auth-service

# Minimal CORS fix in main.py
echo "🔧 Minimal CORS fix uygulanıyor..."
cat > temp_cors_fix.py << 'EOF'
# Replace CORS section in main.py
import re

with open('rag3_for_local/services/auth_service/main.py', 'r') as f:
    content = f.read()

# Find and replace CORS middleware section
cors_replacement = '''# CORS middleware - Allow all for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)'''

# Replace the CORS section
pattern = r'# EMERGENCY CORS.*?return \{"status": "OK"\}'
content = re.sub(pattern, cors_replacement, content, flags=re.DOTALL)

with open('rag3_for_local/services/auth_service/main.py', 'w') as f:
    f.write(content)
EOF

python3 temp_cors_fix.py
rm temp_cors_fix.py

echo "🔄 Rebuilding..."
docker-compose build --no-cache auth-service
docker-compose up -d auth-service

echo "⏳ Waiting 10 seconds..."
sleep 10

echo "🧪 Testing..."
curl -s "http://46.62.254.131:8006/health" && echo "✅ Service running" || echo "❌ Service failed"

echo
echo "🧪 CORS test:"
curl -s -X OPTIONS "http://46.62.254.131:8006/admin/users/2/password" \
  -H "Origin: http://46.62.254.131:3000" \
  -H "Access-Control-Request-Method: PATCH" \
  -H "Access-Control-Request-Headers: Content-Type,Authorization" \
  -w "HTTP Status: %{http_code}\n" -o /dev/null

echo
echo "✅ READY! Test şifre değiştirme şimdi"