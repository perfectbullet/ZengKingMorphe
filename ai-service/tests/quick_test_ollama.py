"""Quick test for OllamaEmbeddings fix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.embeddings import OllamaEmbeddings

print("Testing OllamaEmbeddings fix...")

try:
    embedder = OllamaEmbeddings(
        model="bge-large-zh-v1.5-2k:latest",
        base_url="http://192.168.8.233:11434"
    )
    
    # Test single embedding
    print("Test 1: Single text")
    text = "测试文本"
    result = embedder.embed_query(text)
    print(f"✓ Result type: {type(result)}")
    print(f"✓ Result length: {len(result)}")
    
    # Test batch embedding
    print("\nTest 2: Batch texts")
    texts = ["文本1", "文本2"]
    results = embedder.embed_documents(texts)
    print(results)
    print(f"✓ Results type: {type(results)}")
    print(f"✓ Results count: {len(results)}")
    print(f"✓ First result type: {type(results[0])}")
    print(f"✓ First result length: {len(results[0])}")
    
    print("\n✅ All tests passed! Fix successful.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
