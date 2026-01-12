"""
Test Ollama embedding integration.
Validates that Ollama embeddings work correctly with the configured model.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.embeddings import OllamaEmbeddings
from app.core.config import settings


def print_separator(char="=", length=80):
    """Print separator line."""
    print(char * length)


def test_ollama_config():
    """Test Ollama configuration."""
    print_separator()
    print("🧪 测试Ollama Embedding配置")
    print_separator()
    
    print(f"\n当前配置:")
    print(f"  EMBEDDING_TYPE: {settings.embedding_type}")
    print(f"  OLLAMA_BASE_URL: {settings.ollama_base_url}")
    print(f"  EMBEDDING_OLLAMA_MODEL: {settings.embedding_ollama_model}")
    
    # Validate configuration
    print(f"\n✅ 验证结果:")
    
    if settings.embedding_type == "ollama":
        print(f"  ✓ EMBEDDING_TYPE: ollama (正确)")
    else:
        print(f"  ⚠️ EMBEDDING_TYPE: {settings.embedding_type} (不是ollama)")
        return False
    
    if settings.ollama_base_url:
        print(f"  ✓ OLLAMA_BASE_URL: {settings.ollama_base_url}")
    else:
        print(f"  ❌ OLLAMA_BASE_URL 未配置")
        return False
    
    if settings.embedding_ollama_model:
        print(f"  ✓ EMBEDDING_OLLAMA_MODEL: {settings.embedding_ollama_model}")
    else:
        print(f"  ❌ EMBEDDING_OLLAMA_MODEL 未配置")
        return False
    
    print_separator()
    return True


def test_ollama_embeddings():
    """Test Ollama embeddings functionality."""
    print_separator()
    print("🧪 测试Ollama Embedding功能")
    print_separator()
    
    try:
        # Create embedder instance
        embedder = OllamaEmbeddings(
            model=settings.embedding_ollama_model,
            base_url=settings.ollama_base_url,
            max_tokens=512
        )
        
        print(f"\n创建Embedder实例:")
        print(f"  模型: {embedder.model}")
        print(f"  URL: {embedder.base_url}")
        print(f"  Max tokens: {embedder.max_tokens}")
        print(f"  Max chars: {embedder.max_chars}")
        
        # Test single text embedding
        print(f"\n📝 测试单文本向量化:")
        test_text = "这是一个测试文本，用于验证Ollama embedding功能。"
        print(f"  输入: {test_text}")
        
        embedding = embedder.embed_query(test_text)
        
        print(f"  ✓ 向量维度: {len(embedding)}")
        print(f"  ✓ 向量预览: [{embedding[0]:.6f}, {embedding[1]:.6f}, ..., {embedding[-1]:.6f}]")
        
        # Validate embedding
        if len(embedding) > 0:
            print(f"  ✓ 向量化成功")
        else:
            print(f"  ❌ 向量为空")
            return False
        
        # Test batch embedding
        print(f"\n📚 测试批量向量化:")
        test_texts = [
            "首饰制作的雕蜡工艺",
            "铸造工艺的基本原理",
            "珠宝设计与加工"
        ]
        print(f"  输入文本数: {len(test_texts)}")
        
        embeddings = embedder.embed_documents(test_texts)
        
        print(f"  ✓ 返回向量数: {len(embeddings)}")
        print(f"  ✓ 向量维度: {len(embeddings[0]) if embeddings else 0}")
        
        # Test truncation
        print(f"\n✂️ 测试文本截断:")
        long_text = "这是一个非常长的测试文本。" * 50
        print(f"  原始长度: {len(long_text)} 字符")
        
        truncated = embedder._truncate_text(long_text)
        print(f"  截断后长度: {len(truncated)} 字符")
        print(f"  是否截断: {'是' if len(truncated) < len(long_text) else '否'}")
        
        if len(truncated) <= embedder.max_chars:
            print(f"  ✓ 截断功能正常")
        else:
            print(f"  ❌ 截断失败")
            return False
        
        print_separator()
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print_separator()
        return False


def test_ollama_connection():
    """Test connection to Ollama server."""
    print_separator()
    print("🔌 测试Ollama服务连接")
    print_separator()
    
    import requests
    
    try:
        # Test Ollama API
        url = f"{settings.ollama_base_url}/api/tags"
        print(f"\n连接测试:")
        print(f"  URL: {url}")
        
        response = requests.get(url, timeout=5.0)
        response.raise_for_status()
        
        result = response.json()
        models = result.get("models", [])
        
        print(f"  ✓ 连接成功")
        print(f"  ✓ 可用模型数: {len(models)}")
        
        # Check if our embedding model is available
        model_names = [m.get("name") for m in models]
        print(f"\n可用模型列表:")
        for name in model_names:
            is_current = "← 当前使用" if name == settings.embedding_ollama_model else ""
            print(f"  - {name} {is_current}")
        
        if settings.embedding_ollama_model in model_names:
            print(f"\n  ✓ 配置的模型 {settings.embedding_ollama_model} 可用")
            return True
        else:
            print(f"\n  ⚠️ 配置的模型 {settings.embedding_ollama_model} 未找到")
            print(f"  建议: 运行 ollama pull {settings.embedding_ollama_model}")
            return False
        
    except requests.exceptions.ConnectionError:
        print(f"\n  ❌ 无法连接到Ollama服务")
        print(f"  请确认Ollama服务运行在 {settings.ollama_base_url}")
        return False
    except Exception as e:
        print(f"\n  ❌ 连接测试失败: {str(e)}")
        return False
    finally:
        print_separator()


def main():
    """Run all tests."""
    print_separator("=")
    print("🚀 Ollama Embedding集成测试")
    print_separator("=")
    
    # Run tests
    test1_passed = test_ollama_config()
    
    if not test1_passed:
        print("\n⚠️ 配置验证失败，跳过后续测试")
        print_separator("=")
        return False
    
    test2_passed = test_ollama_connection()
    test3_passed = False
    
    if test2_passed:
        test3_passed = test_ollama_embeddings()
    else:
        print("\n⚠️ 服务连接失败，跳过功能测试")
    
    # Summary
    print_separator("=")
    print("📊 测试总结")
    print_separator("=")
    
    print(f"\n配置验证: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"服务连接: {'✅ 通过' if test2_passed else '❌ 失败'}")
    print(f"功能测试: {'✅ 通过' if test3_passed else '❌ 跳过/失败'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    
    if all_passed:
        print("\n🎉 所有测试通过！Ollama embedding已就绪")
        print("\n📌 后续步骤:")
        print("  1. 重启AI服务: docker-compose restart ai-service")
        print("  2. 上传文档测试向量化")
        print("  3. 观察日志确认使用Ollama embeddings")
    else:
        print("\n⚠️ 部分测试失败")
        if not test2_passed:
            print("\n故障排查:")
            print(f"  1. 确认Ollama服务运行: curl {settings.ollama_base_url}/api/tags")
            print(f"  2. 确认模型已拉取: ollama list")
            print(f"  3. 如需拉取模型: ollama pull {settings.embedding_ollama_model}")
    
    print_separator("=")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
