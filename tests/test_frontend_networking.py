#!/usr/bin/env python3
"""
Test frontend networking - Docker container connectivity issue
"""
import requests
import subprocess
import json

def test_host_vs_container_networking():
    """Test if the issue is Docker networking - localhost vs container names"""
    print("🔍 Testing Docker networking connectivity...")
    
    # Test 1: localhost:8000 (what frontend is trying)
    print("\n1️⃣ Testing localhost:8000 (what frontend container tries)...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ localhost:8000 accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ localhost:8000 NOT accessible: {e}")
    
    # Test 2: Check if Docker containers are running
    print("\n2️⃣ Checking Docker containers...")
    try:
        result = subprocess.run(
            ['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'],
            capture_output=True, text=True, timeout=10
        )
        print("Docker containers:")
        print(result.stdout)
    except Exception as e:
        print(f"❌ Could not check Docker containers: {e}")
    
    # Test 3: Test container-to-container networking
    print("\n3️⃣ Testing container-to-container networking...")
    try:
        # Execute curl from inside the frontend container
        result = subprocess.run([
            'docker', 'exec', 'rag3-frontend', 
            'curl', '-s', 'http://localhost:8000/health'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Frontend container can reach localhost:8000")
            print(f"Response: {result.stdout}")
        else:
            print("❌ Frontend container CANNOT reach localhost:8000")
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Could not test from frontend container: {e}")
    
    # Test 4: Test the correct container networking approach
    print("\n4️⃣ Testing correct container networking (api-gateway:8080)...")
    try:
        result = subprocess.run([
            'docker', 'exec', 'rag3-frontend',
            'curl', '-s', 'http://api-gateway:8080/health'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Frontend container CAN reach api-gateway:8080")
            print(f"Response: {result.stdout}")
        else:
            print("❌ Frontend container cannot reach api-gateway:8080")
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Could not test api-gateway networking: {e}")

def test_frontend_environment():
    """Check what URL the frontend is actually using"""
    print("\n5️⃣ Checking frontend environment variables...")
    try:
        result = subprocess.run([
            'docker', 'exec', 'rag3-frontend',
            'printenv', 'NEXT_PUBLIC_API_URL'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print(f"Frontend API URL: {result.stdout.strip()}")
        else:
            print("❌ Could not get frontend environment")
    except Exception as e:
        print(f"❌ Could not check frontend environment: {e}")

def main():
    print("🚀 Frontend Networking Test - Docker Container Connectivity")
    print("=" * 70)
    
    test_host_vs_container_networking()
    test_frontend_environment()
    
    print("\n" + "=" * 70)
    print("📊 DIAGNOSIS:")
    print("🔍 If frontend container cannot reach localhost:8000,")
    print("   then the issue is Docker networking configuration!")
    print("💡 Solution: Update frontend environment to use api-gateway:8080")

if __name__ == "__main__":
    main()