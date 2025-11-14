#!/bin/bash

echo "🚨 EMERGENCY AUTH CORS FIX - Allow all origins"
echo "============================================="

# Create temporary CORS bypass
cat > rag3_for_local/services/auth_service/cors_bypass.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def setup_emergency_cors(app: FastAPI):
    """Emergency CORS setup - allows everything"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=False,  # Disable credentials for wildcard
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )
    
    @app.options("/{full_path:path}")
    async def options_handler():
        return {"status": "ok"}
EOF

# Backup original main.py
cp rag3_for_local/services/auth_service/main.py rag3_for_local/services/auth_service/main.py.backup

# Create emergency main.py with bypass
sed -i 's/from fastapi.middleware.cors import CORSMiddleware/from cors_bypass import setup_emergency_cors\nfrom fastapi.middleware.cors import CORSMiddleware/' rag3_for_local/services/auth_service/main.py

# Replace CORS setup section
sed -i '/# CORS middleware/,/)/c\
# Emergency CORS setup\
setup_emergency_cors(app)' rag3_for_local/services/auth_service/main.py

echo "⏹️  Stopping auth service..."
docker-compose stop auth-service
docker-compose rm -f auth-service

echo "🔄 Emergency rebuild..."
docker-compose build --no-cache auth-service
docker-compose up -d auth-service

echo "⏳ Waiting..."
sleep 5

echo "✅ Emergency CORS fix applied!"
echo "🧪 Test şifre değişikliği şimdi"