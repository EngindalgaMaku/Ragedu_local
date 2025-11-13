import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'document_processing_service'))

try:
    from services.document_processing_service.main import app
    print("Service imported successfully")
    
    # Test importing the processor
    from services.document_processing_service.main import processor
    print("Processor imported successfully")
    
    # Test health check function
    import asyncio
    async def test_health():
        from services.document_processing_service.main import health_check
        result = await health_check()
        print(f"Health check result: {result}")
    
    # Run the async function
    asyncio.run(test_health())
    print("Service test completed successfully")
    
except Exception as e:
    print(f"Error testing service: {e}")
    import traceback
    traceback.print_exc()