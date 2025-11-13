#!/usr/bin/env python3
"""
Start API Gateway Server
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Change to src directory for proper imports
os.chdir(src_dir)

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting API Gateway Server...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[:3]}")
    
    try:
        # Import and check the app
        from api.main import app
        print("✅ API app imported successfully")
        
        # Start the server
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,  # Changed from 8081 to 8000 to match frontend expectations
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()