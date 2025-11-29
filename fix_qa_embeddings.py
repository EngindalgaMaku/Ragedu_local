#!/usr/bin/env python3
"""
Fix QA Embeddings Script
This script fixes QA similarity cache entries that were created with the wrong embedding model (nomic-embed-text).
It clears the cache so that new embeddings will be generated with the correct model (text-embedding-v4).
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Database path - try multiple locations
DB_PATH = os.getenv("APRAG_DB_PATH", None)
if not DB_PATH:
    # Try common locations
    possible_paths = [
        str(project_root / "services" / "aprag_service" / "data" / "rag_assistant.db"),
        str(project_root / "services" / "aprag_service" / "data" / "aprag.db"),
        str(project_root / "services" / "aprag_service" / "database" / "aprag.db"),
        "/app/data/rag_assistant.db",  # Docker container path
    ]
    for path in possible_paths:
        if os.path.exists(path):
            DB_PATH = path
            break
    if not DB_PATH:
        DB_PATH = possible_paths[0]  # Default to first path

# Default embedding model
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v4")

def fix_qa_similarity_cache():
    """
    Clear QA similarity cache entries that were created with nomic-embed-text
    or other non-default embedding models.
    """
    print(f"🔧 Fixing QA similarity cache...")
    print(f"📁 Database: {DB_PATH}")
    print(f"🎯 Target embedding model: {DEFAULT_EMBEDDING_MODEL}")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check current cache entries
        cursor.execute("""
            SELECT COUNT(*) as total,
                   COUNT(CASE WHEN embedding_model != ? THEN 1 END) as wrong_model,
                   COUNT(CASE WHEN embedding_model IS NULL THEN 1 END) as null_model
            FROM qa_similarity_cache
        """, (DEFAULT_EMBEDDING_MODEL,))
        
        row = cursor.fetchone()
        total = row[0] if row else 0
        wrong_model = row[1] if row else 0
        null_model = row[2] if row else 0
        
        print(f"\n📊 Current cache statistics:")
        print(f"   Total entries: {total}")
        print(f"   Wrong model (not {DEFAULT_EMBEDDING_MODEL}): {wrong_model}")
        print(f"   Null model: {null_model}")
        
        if total == 0:
            print("✅ No cache entries found. Nothing to fix.")
            return True
        
        # Clear all cache entries that don't use the default model
        # Also clear entries with "semantic_similarity" (old bug where wrong value was stored)
        cursor.execute("""
            DELETE FROM qa_similarity_cache
            WHERE embedding_model != ? 
               OR embedding_model IS NULL
               OR embedding_model = 'semantic_similarity'
        """, (DEFAULT_EMBEDDING_MODEL,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        print(f"\n✅ Deleted {deleted_count} cache entries with wrong/null embedding model")
        
        # Show remaining entries
        cursor.execute("SELECT COUNT(*) FROM qa_similarity_cache")
        remaining = cursor.fetchone()[0]
        print(f"📊 Remaining cache entries: {remaining}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error fixing QA similarity cache: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_fix():
    """
    Verify that the fix was successful
    """
    print(f"\n🔍 Verifying fix...")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if any wrong model entries remain
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM qa_similarity_cache
            WHERE embedding_model != ? OR embedding_model IS NULL
        """, (DEFAULT_EMBEDDING_MODEL,))
        
        row = cursor.fetchone()
        wrong_count = row[0] if row else 0
        
        if wrong_count == 0:
            print("✅ Verification passed: All cache entries use the correct embedding model")
            return True
        else:
            print(f"⚠️ Warning: {wrong_count} cache entries still have wrong/null embedding model")
            return False
        
    except Exception as e:
        print(f"❌ Error verifying fix: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """
    Main function
    """
    print("=" * 60)
    print("QA Embeddings Fix Script")
    print("=" * 60)
    print()
    
    # Check if running non-interactively (e.g., in Docker)
    import sys
    non_interactive = os.getenv("NON_INTERACTIVE", "false").lower() == "true" or not sys.stdin.isatty()
    
    if not non_interactive:
        # Ask for confirmation
        try:
            response = input(f"⚠️  This will delete QA similarity cache entries that don't use '{DEFAULT_EMBEDDING_MODEL}'.\n   Continue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("❌ Aborted by user")
                return
        except (EOFError, KeyboardInterrupt):
            print("❌ Aborted")
            return
    else:
        print(f"⚠️  Running in non-interactive mode. Will delete QA similarity cache entries that don't use '{DEFAULT_EMBEDDING_MODEL}'.")
    
    # Fix the cache
    success = fix_qa_similarity_cache()
    
    if success:
        # Verify the fix
        verify_fix()
        print("\n✅ Fix completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Restart aprag-service")
        print("   2. New QA similarity calculations will use the correct embedding model")
    else:
        print("\n❌ Fix failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

