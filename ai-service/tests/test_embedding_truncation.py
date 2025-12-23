"""
Test SiliconFlow embedding token truncation.
Validates that long texts are properly truncated to fit token limits.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.embeddings import SiliconFlowEmbeddings


def print_separator(char="=", length=80):
    """Print separator line."""
    print(char * length)


def test_truncation():
    """Test text truncation logic."""
    print_separator()
    print("🧪 测试文本截断功能")
    print_separator()
    
    # Create embedder instance
    embedder = SiliconFlowEmbeddings(
        model="BAAI/bge-large-zh-v1.5",
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1/embeddings",
        max_tokens=512
    )
    
    print(f"\n配置参数:")
    print(f"  max_tokens: {embedder.max_tokens}")
    print(f"  max_chars: {embedder.max_chars} (安全字符限制)")
    
    # Test cases
    test_texts = [
        ("短文本", "这是一个短文本，不需要截断。"),
        ("中等长度", "这是一个中等长度的文本。" * 20),
        ("超长文本", "这是一个非常长的文本，需要被截断以适应token限制。" * 50),
        ("极长英文", "This is a very long English text that needs to be truncated. " * 100),
    ]
    
    print_separator("-", 60)
    print("\n测试结果:")
    
    for name, text in test_texts:
        truncated = embedder._truncate_text(text)
        was_truncated = len(truncated) < len(text)
        
        print(f"\n📝 {name}:")
        print(f"  原始长度: {len(text)} 字符")
        print(f"  截断后长度: {len(truncated)} 字符")
        print(f"  是否截断: {'✅ 是' if was_truncated else '❌ 否'}")
        
        if was_truncated:
            print(f"  预览: {truncated[:100]}...")
        else:
            print(f"  预览: {truncated[:100]}")
    
    print_separator("-", 60)
    
    # Validate max_chars calculation
    print("\n✅ 验证结果:")
    
    if embedder.max_chars == 256:
        print(f"  ✓ max_chars 计算正确: {embedder.max_chars} (512 tokens / 2)")
    else:
        print(f"  ⚠️ max_chars 异常: {embedder.max_chars} (应为256)")
    
    # Test batch truncation
    long_text = "长文本测试。" * 100
    batch_texts = [long_text, "短文本", long_text]
    truncated_batch = [embedder._truncate_text(t) for t in batch_texts]
    
    truncated_count = sum(1 for orig, trunc in zip(batch_texts, truncated_batch) if len(orig) > len(trunc))
    print(f"  ✓ 批量截断测试: {truncated_count}/{len(batch_texts)} 个文本被截断")
    
    # Validate all truncated texts fit limit
    all_fit = all(len(t) <= embedder.max_chars for t in truncated_batch)
    if all_fit:
        print(f"  ✓ 所有截断文本均在 {embedder.max_chars} 字符限制内")
    else:
        print(f"  ❌ 部分文本超过字符限制！")
    
    print_separator()
    
    return all_fit


def test_chunk_size_config():
    """Test chunk size configuration."""
    print_separator()
    print("🧪 测试Chunk Size配置")
    print_separator()
    
    from app.core.config import settings
    
    print(f"\n当前配置:")
    print(f"  CHUNK_SIZE: {settings.chunk_size} 字符")
    print(f"  CHUNK_OVERLAP: {settings.chunk_overlap} 字符")
    
    # Validate chunk size
    print(f"\n✅ 验证结果:")
    
    if settings.chunk_size <= 256:
        print(f"  ✓ CHUNK_SIZE: {settings.chunk_size} (安全值，适用于512 token限制)")
    elif settings.chunk_size <= 384:
        print(f"  ⚠️ CHUNK_SIZE: {settings.chunk_size} (可能接近512 token限制)")
    else:
        print(f"  ❌ CHUNK_SIZE: {settings.chunk_size} (可能超过512 token限制！)")
        return False
    
    # Estimate token count
    estimated_tokens_zh = settings.chunk_size * 2  # Worst case for Chinese
    estimated_tokens_en = settings.chunk_size / 4  # Average for English
    
    print(f"\n预估Token数:")
    print(f"  中文文本: ~{int(estimated_tokens_zh)} tokens (1字符=2tokens)")
    print(f"  英文文本: ~{int(estimated_tokens_en)} tokens (1单词=1token)")
    
    if estimated_tokens_zh <= 512:
        print(f"  ✓ 中文文本预估在512 token限制内")
    else:
        print(f"  ❌ 中文文本可能超过512 token限制！")
    
    print_separator()
    
    return estimated_tokens_zh <= 512


def main():
    """Run all tests."""
    print_separator("=")
    print("🚀 SiliconFlow Embedding Token限制测试")
    print_separator("=")
    
    test1_passed = test_truncation()
    test2_passed = test_chunk_size_config()
    
    print_separator("=")
    print("📊 测试总结")
    print_separator("=")
    
    print(f"\n截断功能测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"Chunk配置测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！")
        print("\n📌 建议:")
        print("  1. 重启AI服务: docker-compose restart ai-service")
        print("  2. 重新上传失败的文档")
        print("  3. 观察日志确认无token超限错误")
    else:
        print("\n⚠️ 部分测试失败，请检查配置！")
    
    print_separator("=")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
