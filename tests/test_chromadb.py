try:
    import chromadb
    print("ChromaDB imported successfully")
    print(f"ChromaDB version: {chromadb.__version__}")
    
    # Test creating a client
    client = chromadb.Client()
    print("ChromaDB client created successfully")
    
except ImportError as e:
    print(f"Failed to import ChromaDB: {e}")
except Exception as e:
    print(f"Error with ChromaDB: {e}")