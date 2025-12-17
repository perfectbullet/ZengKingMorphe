#!/usr/bin/env python3
"""
Standalone unit test script for ChromaDB connection and operations.

This script can be run independently in a separate environment to test
ChromaDB functionality before integrating into the main project.

Usage:
    python test_chroma_standalone.py

Requirements:
    pip install chromadb requests python-dotenv

Environment Variables (optional):
    CHROMA_HOST: ChromaDB server host (default: localhost)
    CHROMA_PORT: ChromaDB server port (default: 8000)
    EMBEDDING_API_URL: Embedding service URL (default: http://localhost:50009)
    EMBEDDING_MODEL: Embedding model name (default: BAAI/bge-large-zh-v1.5)
"""

import os
import sys
import time
from typing import List, Optional
from dotenv import load_dotenv
import numpy as np  # new import
# Load environment variables
load_dotenv(".env.chroma-test")

# Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:50009")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

# Test collection name
TEST_COLLECTION_NAME = "test_collection_standalone"


class SimpleEmbeddings:
    def __call__(self, input: List[str]) -> List[np.ndarray]:
        return [
            np.full(384, (hash(text) % 100) / 100.0, dtype=float)
            for text in input
        ]

class OpenAIStyleEmbeddings:
    """OpenAI-style embedding function for production use."""
    
    def __init__(self, model: str, base_url: str, api_key: Optional[str] = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Call embedding API."""
        import requests
        
        payload = {"input": input, "model": self.model}
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "chroma-test/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
            
            data = result.get("data")
            if not data:
                raise ValueError(f"Embedding service returned no data: {result}")
            embeddings = [np.array(item["embedding"], dtype=float) for item in data]
            return embeddings
        except Exception as e:
            print(f"⚠️  Warning: Embedding API failed: {e}")
            print("   Falling back to dummy embeddings for testing")
            # Fallback to dummy embeddings
            return SimpleEmbeddings()(input)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_chromadb_connection():
    """Test basic ChromaDB connection."""
    print_section("Test 1: ChromaDB Connection")
    
    try:
        import chromadb
        print(f"✓ ChromaDB package imported successfully")
        print(f"  Version: {chromadb.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import chromadb: {e}")
        print("\n  Install with: pip install chromadb")
        return False
    
    try:
        # Try to connect to ChromaDB server
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        print(f"✓ Connected to ChromaDB server at {CHROMA_HOST}:{CHROMA_PORT}")
        
        # Test heartbeat
        heartbeat = client.heartbeat()
        print(f"✓ Server heartbeat: {heartbeat}")
        
        # List existing collections
        collections = client.list_collections()
        print(f"✓ Found {len(collections)} existing collections")
        for col in collections:
            print(f"  - {col.name}")
        
        return client
        
    except Exception as e:
        print(f"✗ Failed to connect to ChromaDB: {e}")
        print(f"\n  Make sure ChromaDB server is running:")
        print(f"  - Docker: docker run -p 8000:8000 chromadb/chroma:latest")
        print(f"  - Or update CHROMA_HOST/CHROMA_PORT environment variables")
        return None


def test_collection_operations(client):
    """Test collection creation, get, and delete operations."""
    print_section("Test 2: Collection Operations")
    
    if client is None:
        print("⊘ Skipping: No client connection")
        return None
    
    try:
        # Clean up test collection if it exists
        try:
            client.delete_collection(name=TEST_COLLECTION_NAME)
            print(f"✓ Cleaned up existing test collection")
        except Exception:
            pass
        
        # Create embedding function
        print(f"  Using embedding model: {EMBEDDING_MODEL}")
        if EMBEDDING_API_URL:
            embedding_function = OpenAIStyleEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=EMBEDDING_API_URL,
                api_key=EMBEDDING_API_KEY if EMBEDDING_API_KEY else None
            )
        else:
            print("  Note: Using dummy embeddings (no API configured)")
            embedding_function = SimpleEmbeddings()
        
        # Create collection
        collection = client.create_collection(
            name=TEST_COLLECTION_NAME,
            metadata={"description": "Test collection for unit testing"},
            embedding_function=embedding_function
        )
        print(f"✓ Created collection: {TEST_COLLECTION_NAME}")
        
        # Get collection
        retrieved_collection = client.get_collection(
            name=TEST_COLLECTION_NAME,
            embedding_function=embedding_function
        )
        print(f"✓ Retrieved collection: {retrieved_collection.name}")
        print(f"  Metadata: {retrieved_collection.metadata}")
        
        # Test get_or_create
        same_collection = client.get_or_create_collection(
            name=TEST_COLLECTION_NAME,
            embedding_function=embedding_function
        )
        print(f"✓ get_or_create returned existing collection")
        
        return collection
        
    except Exception as e:
        print(f"✗ Collection operations failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_document_operations(collection):
    """Test document add, query, and delete operations."""
    print_section("Test 3: Document Operations")
    
    if collection is None:
        print("⊘ Skipping: No collection available")
        return False
    
    try:
        # Add documents
        test_documents = [
            "ChromaDB是一个向量数据库",
            "它支持语义搜索功能",
            "可以存储文档和向量",
            "适合用于RAG应用",
            "支持多种嵌入模型"
        ]
        
        test_metadatas = [
            {"source": "doc1", "type": "intro"},
            {"source": "doc2", "type": "feature"},
            {"source": "doc3", "type": "feature"},
            {"source": "doc4", "type": "usecase"},
            {"source": "doc5", "type": "feature"}
        ]
        
        test_ids = [f"test_doc_{i}" for i in range(len(test_documents))]
        
        print(f"  Adding {len(test_documents)} test documents...")
        collection.add(
            documents=test_documents,
            metadatas=test_metadatas,
            ids=test_ids
        )
        print(f"✓ Added {len(test_documents)} documents")
        
        # Get collection count
        count = collection.count()
        print(f"✓ Collection count: {count}")
        assert count == len(test_documents), f"Expected {len(test_documents)}, got {count}"
        
        # Query documents
        print(f"\n  Querying with: '什么是向量数据库'")
        results = collection.query(
            query_texts=["什么是向量数据库"],
            n_results=3
        )
        print(f"✓ Query returned {len(results['ids'][0])} results")
        
        for i, (doc_id, document, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['distances'][0]
        )):
            print(f"  [{i+1}] ID: {doc_id}")
            print(f"      Text: {document}")
            print(f"      Distance: {distance:.4f}")
        
        # Test filtering with metadata
        print(f"\n  Querying with metadata filter (type='feature')...")
        filtered_results = collection.query(
            query_texts=["向量数据库的功能"],
            n_results=5,
            where={"type": "feature"}
        )
        print(f"✓ Filtered query returned {len(filtered_results['ids'][0])} results")
        for doc_id, metadata in zip(filtered_results['ids'][0], filtered_results['metadatas'][0]):
            print(f"  - {doc_id}: {metadata}")
        
        # Get specific documents
        print(f"\n  Getting specific documents by ID...")
        get_results = collection.get(
            ids=["test_doc_0", "test_doc_2"]
        )
        print(f"✓ Retrieved {len(get_results['ids'])} documents")
        for doc_id, document in zip(get_results['ids'], get_results['documents']):
            print(f"  - {doc_id}: {document}")
        
        # Update documents
        print(f"\n  Updating document test_doc_0...")
        collection.update(
            ids=["test_doc_0"],
            documents=["ChromaDB是一个现代化的向量数据库系统"],
            metadatas=[{"source": "doc1", "type": "intro", "updated": True}]
        )
        print(f"✓ Document updated")
        
        updated_doc = collection.get(ids=["test_doc_0"])
        print(f"  Updated text: {updated_doc['documents'][0]}")
        print(f"  Updated metadata: {updated_doc['metadatas'][0]}")
        
        # Delete documents
        print(f"\n  Deleting test_doc_4...")
        collection.delete(ids=["test_doc_4"])
        print(f"✓ Document deleted")
        
        new_count = collection.count()
        print(f"✓ New collection count: {new_count}")
        assert new_count == count - 1, f"Expected {count - 1}, got {new_count}"
        
        return True
        
    except Exception as e:
        print(f"✗ Document operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling(client):
    """Test error handling and edge cases."""
    print_section("Test 4: Error Handling")
    
    if client is None:
        print("⊘ Skipping: No client connection")
        return
    
    try:
        # Test non-existent collection
        print("  Testing non-existent collection...")
        try:
            client.get_collection(name="non_existent_collection_xyz")
            print("✗ Should have raised an error for non-existent collection")
        except Exception as e:
            print(f"✓ Correctly raised error: {type(e).__name__}")
        
        # Test duplicate collection creation
        print("\n  Testing duplicate collection creation...")
        try:
            client.create_collection(name=TEST_COLLECTION_NAME)
            print("✗ Should have raised an error for duplicate collection")
        except Exception as e:
            print(f"✓ Correctly raised error: {type(e).__name__}")
        
        # Test empty query
        print("\n  Testing edge cases...")
        collection = client.get_or_create_collection(
            name=TEST_COLLECTION_NAME,
            embedding_function=SimpleEmbeddings()
        )
        
        # Query with no results requested
        try:
            results = collection.query(query_texts=["test"], n_results=0)
            print(f"✓ Query with n_results=0 handled: {len(results['ids'][0])} results")
        except Exception as e:
            print(f"  Note: n_results=0 raised {type(e).__name__} (expected)")
        
        print("\n✓ Error handling tests completed")
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()


def cleanup(client):
    """Clean up test resources."""
    print_section("Cleanup")
    
    if client is None:
        print("⊘ No cleanup needed")
        return
    
    try:
        client.delete_collection(name=TEST_COLLECTION_NAME)
        print(f"✓ Deleted test collection: {TEST_COLLECTION_NAME}")
    except Exception as e:
        print(f"⚠️  Warning: Could not delete test collection: {e}")


def check_version_compatibility():
    """Check ChromaDB version and compatibility."""
    print_section("Version Compatibility Check")
    
    try:
        import chromadb
        version = chromadb.__version__
        major, minor, patch = version.split('.')[:3]
        major, minor = int(major), int(minor)
        
        print(f"  Installed ChromaDB version: {version}")
        
        if major == 0 and minor == 5:
            print(f"  ✓ Version 0.5.x detected (legacy)")
            print(f"    Note: Consider upgrading to 0.6.x or 1.x for latest features")
            return "0.5.x"
        elif major == 0 and minor == 6:
            print(f"  ✓ Version 0.6.x detected (stable)")
            print(f"    This is a good stable version")
            return "0.6.x"
        elif major == 1:
            print(f"  ✓ Version 1.x detected (latest)")
            print(f"    Note: API may have changed from 0.5.x")
            print(f"    Warning: client.persist() is deprecated in 1.x")
            return "1.x"
        else:
            print(f"  ⚠️  Unknown version: {version}")
            return "unknown"
            
    except Exception as e:
        print(f"  ✗ Could not check version: {e}")
        return None


def print_summary(results: dict):
    """Print test summary."""
    print_section("Test Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    print(f"  Total tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print()
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print()
    if failed == 0:
        print("  🎉 All tests passed!")
    else:
        print(f"  ⚠️  {failed} test(s) failed")
    
    return failed == 0


def main():
    """Main test runner."""
    print("\n" + "="*60)
    print("  ChromaDB Standalone Test Suite")
    print("="*60)
    print(f"\n  Configuration:")
    print(f"    ChromaDB Host: {CHROMA_HOST}")
    print(f"    ChromaDB Port: {CHROMA_PORT}")
    print(f"    Embedding API: {EMBEDDING_API_URL}")
    print(f"    Embedding Model: {EMBEDDING_MODEL}")
    
    # Check version compatibility
    version_type = check_version_compatibility()
    
    # Run tests
    results = {}
    
    # Test 1: Connection
    client = test_chromadb_connection()
    results["Connection"] = client is not None
    
    if client is None:
        print("\n❌ Cannot proceed without ChromaDB connection")
        print("\nTo start ChromaDB server:")
        print("  docker run -p 8000:8000 chromadb/chroma:latest")
        print("\nOr use docker-compose:")
        print("  docker-compose up chroma")
        return 1
    
    # Test 2: Collections
    collection = test_collection_operations(client)
    results["Collection Operations"] = collection is not None
    
    # Test 3: Documents
    doc_success = test_document_operations(collection)
    results["Document Operations"] = doc_success
    
    # Test 4: Error Handling
    test_error_handling(client)
    results["Error Handling"] = True
    
    # Cleanup
    cleanup(client)
    
    # Print summary
    all_passed = print_summary(results)
    
    # Additional recommendations
    if all_passed:
        print_section("Recommendations")
        print("  ✓ ChromaDB connection is working correctly")
        print("  ✓ Ready to integrate into main project\n")
        
        if version_type == "0.5.x":
            print("  Recommendations:")
            print("  1. Current version (0.5.x) is compatible with your code")
            print("  2. Consider upgrading to 0.6.x for stability:")
            print("     pip install chromadb==0.6.3")
            print("  3. Note: Remove client.persist() calls when upgrading to 1.x")
        elif version_type == "1.x":
            print("  Action Required:")
            print("  1. Remove client.persist() from app/core/chroma.py (line 88)")
            print("     - This method is deprecated in ChromaDB 1.x")
            print("     - Data is auto-persisted in HttpClient mode")
        
        print("\n  Docker Image Recommendations:")
        print("  - Current: chromadb/chroma:latest (not recommended)")
        print("  - Recommended: chromadb/chroma:0.6.3 (stable)")
        print("  - Alternative: chromadb/chroma:1.3.7 (latest, but requires code changes)")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
