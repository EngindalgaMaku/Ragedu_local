#!/usr/bin/env python3
"""
Smart startup script for RAG3 API
Switches between minimal and full API based on environment and health checks.
"""

import os
import sys
import time
import subprocess
import importlib.util
from pathlib import Path

def check_dependencies():
    """Check if all dependencies are available for full API"""
    try:
        # Test critical imports that might fail
        import sqlite3
        import json
        import tempfile
        from pathlib import Path
        
        # Test FastAPI basics
        from fastapi import FastAPI
        from pydantic import BaseModel
        
        print("✅ Basic dependencies available")
        return True, "basic"
        
    except ImportError as e:
        print(f"❌ Basic dependency missing: {e}")
        return False, str(e)

def check_advanced_dependencies():
    """Check if advanced dependencies are available"""
    try:
        # Test more complex imports
        from src.services.session_manager import professional_session_manager
        from src.app_logic import get_selected_provider
        
        print("✅ Advanced dependencies available") 
        return True, "advanced"
        
    except ImportError as e:
        print(f"⚠️ Advanced dependency missing: {e}")
        return False, str(e)

def start_minimal_api():
    """Start minimal API version"""
    print("🚀 Starting MINIMAL API version...")
    try:
        import uvicorn
        from src.api.main_minimal import app
        
        # Cloud Run requires listening on 0.0.0.0 with PORT environment variable
        port = int(os.environ.get("PORT", 8080))
        print(f"📍 Starting on 0.0.0.0:{port}")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("⏹️ Minimal API stopped")
    except Exception as e:
        print(f"❌ Minimal API failed: {e}")
        sys.exit(1)

def start_full_api():
    """Start full API version"""
    print("🚀 Starting FULL API version...")
    try:
        import uvicorn
        from src.api.main import app
        
        # Cloud Run requires listening on 0.0.0.0 with PORT environment variable
        port = int(os.environ.get("PORT", 8080))
        print(f"📍 Starting on 0.0.0.0:{port}")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("⏹️ Full API stopped")
    except Exception as e:
        print(f"❌ Full API failed: {e}")
        # Fallback to minimal
        print("🔄 Falling back to minimal API...")
        start_minimal_api()

def main():
    """Main startup logic"""
    print("🎯 RAG3 API Smart Startup")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🌍 Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    print(f"🔌 Port: {os.environ.get('PORT', '8080')}")
    
    # Check what we can run
    basic_ok, basic_msg = check_dependencies()
    if not basic_ok:
        print(f"💥 Fatal: Cannot start any API version - {basic_msg}")
        sys.exit(1)
    
    advanced_ok, advanced_msg = check_advanced_dependencies()
    
    # Decide which version to run based on environment and dependencies
    force_minimal = os.environ.get("FORCE_MINIMAL_API", "false").lower() == "true"
    is_cloud = os.environ.get("ENVIRONMENT", "").lower() in ["production", "cloud"]
    
    if force_minimal:
        print("🎛️ FORCE_MINIMAL_API=true - using minimal version")
        start_minimal_api()
    elif not advanced_ok and is_cloud:
        print("☁️ Cloud environment + missing dependencies = minimal API")
        start_minimal_api()
    elif advanced_ok:
        print("💪 All dependencies available - using full API")
        start_full_api()
    else:
        print("🛡️ Safety fallback - using minimal API") 
        start_minimal_api()

if __name__ == "__main__":
    main()