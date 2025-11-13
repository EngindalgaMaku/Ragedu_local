#!/usr/bin/env python3
"""
Script to start all microservices in the correct order
"""

import subprocess
import time
import os
import sys

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing required dependencies...")
    try:
        # Install ChromaDB and other dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", "chromadb", "langchain", "langdetect", "regex"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_service(service_name, command, cwd=None):
    """Start a service and return the process"""
    try:
        print(f"🚀 Starting {service_name}...")
        if cwd:
            process = subprocess.Popen(command, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ {service_name} started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start {service_name}: {e}")
        return None

def main():
    """Main function to start all services"""
    print("=" * 60)
    print("RAG3 Microservices Startup Script")
    print("=" * 60)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Continuing without installing dependencies...")
    
    # Start services in order
    services = []
    
    # 1. Start API Gateway (main service)
    api_gateway = start_service(
        "API Gateway",
        "python main.py",
        cwd="src/api"
    )
    if api_gateway:
        services.append(("API Gateway", api_gateway))
        time.sleep(3)  # Give API Gateway time to start
    
    # 2. Start PDF Processing Service
    pdf_service = start_service(
        "PDF Processing Service", 
        "python main.py", 
        cwd="services/pdf_processing_service"
    )
    if pdf_service:
        services.append(("PDF Processing Service", pdf_service))
        time.sleep(3)  # Give service time to start
    
    # 3. Start Document Processing Service
    doc_service = start_service(
        "Document Processing Service", 
        "python main.py", 
        cwd="services/document_processing_service"
    )
    if doc_service:
        services.append(("Document Processing Service", doc_service))
        time.sleep(3)  # Give service time to start
    
    # 4. Start Model Inference Service
    model_service = start_service(
        "Model Inference Service", 
        "python main.py", 
        cwd="services/model_inference_service"
    )
    if model_service:
        services.append(("Model Inference Service", model_service))
    
    # Wait a moment for all services to initialize
    print("\n⏳ Waiting for services to initialize...")
    time.sleep(5)
    
    # Display status
    print("\n" + "=" * 60)
    print("SERVICE STATUS")
    print("=" * 60)
    
    for name, process in services:
        if process.poll() is None:
            print(f"✅ {name}: RUNNING (PID: {process.pid})")
        else:
            print(f"❌ {name}: STOPPED")
    
    print("\n📋 To stop all services, press Ctrl+C")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        for name, process in services:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️  {name} force killed")
        print("👋 All services stopped. Goodbye!")

if __name__ == "__main__":
    main()