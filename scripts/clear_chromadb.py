#!/usr/bin/env python3
"""
Script to clear all ChromaDB collections
Use with caution - this will delete all vector data!
"""

import sys
import os

# Add parent directory to path to import chromadb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  chromadb package not installed. Install with: pip install chromadb")

CHROMADB_HOST = "localhost"
CHROMADB_PORT = 8004

def get_chroma_client():
    """Get ChromaDB HttpClient"""
    if not CHROMADB_AVAILABLE:
        raise ImportError("chromadb package not available")
    
    try:
        client = chromadb.HttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT,
            settings=Settings(
                anonymized_telemetry=False,
                chroma_client_auth_provider=None,
            )
        )
        return client
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB at {CHROMADB_HOST}:{CHROMADB_PORT}: {e}")
        raise

def list_collections():
    """List all ChromaDB collections"""
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        return [{"name": c.name, "id": c.id, "count": c.count()} for c in collections]
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []

def delete_collection(collection_name: str):
    """Delete a specific collection"""
    try:
        client = get_chroma_client()
        client.delete_collection(name=collection_name)
        return True, f"Collection '{collection_name}' deleted successfully"
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg or "not found" in error_msg.lower():
            return True, f"Collection '{collection_name}' does not exist (already deleted)"
        return False, error_msg

def clear_all_collections(confirm: bool = False):
    """Clear all ChromaDB collections"""
    if not confirm:
        print("⚠️  WARNING: This will delete ALL ChromaDB collections!")
        print("⚠️  This action cannot be undone!")
        response = input("Type 'YES' to confirm: ")
        if response != "YES":
            print("❌ Operation cancelled")
            return
    
    print("\n📋 Listing all collections...")
    collections = list_collections()
    
    if not collections:
        print("✅ ChromaDB is already empty - no collections found")
        return
    
    print(f"\n📊 Found {len(collections)} collection(s):")
    for i, coll in enumerate(collections, 1):
        coll_name = coll.get('name', coll.get('id', 'unknown'))
        print(f"  {i}. {coll_name}")
    
    print(f"\n🗑️  Deleting {len(collections)} collection(s)...")
    
    deleted_count = 0
    failed_count = 0
    
    for coll in collections:
        coll_name = coll.get('name', coll.get('id', 'unknown'))
        print(f"  Deleting '{coll_name}'...", end=" ")
        success, result = delete_collection(coll_name)
        if success:
            print("✅")
            deleted_count += 1
        else:
            print(f"❌ Error: {result}")
            failed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Deleted: {deleted_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  📦 Total: {len(collections)}")
    
    # Verify
    print("\n🔍 Verifying...")
    remaining = list_collections()
    if not remaining:
        print("✅ ChromaDB is now empty!")
    else:
        print(f"⚠️  Warning: {len(remaining)} collection(s) still remain:")
        for coll in remaining:
            coll_name = coll.get('name', coll.get('id', 'unknown'))
            print(f"  - {coll_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear all ChromaDB collections")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list collections, don't delete"
    )
    parser.add_argument(
        "--host",
        default=CHROMADB_HOST,
        help=f"ChromaDB host (default: {CHROMADB_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=CHROMADB_PORT,
        help=f"ChromaDB port (default: {CHROMADB_PORT})"
    )
    
    args = parser.parse_args()
    CHROMADB_HOST = args.host
    CHROMADB_PORT = args.port
    
    if args.list_only:
        print("📋 Listing ChromaDB collections...\n")
        collections = list_collections()
        if not collections:
            print("✅ ChromaDB is empty - no collections found")
        else:
            print(f"📊 Found {len(collections)} collection(s):")
            for i, coll in enumerate(collections, 1):
                coll_name = coll.get('name', coll.get('id', 'unknown'))
                print(f"  {i}. {coll_name}")
    else:
        clear_all_collections(confirm=args.yes)

