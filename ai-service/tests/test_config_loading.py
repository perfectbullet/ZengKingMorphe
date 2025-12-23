"""
Test configuration loading from .env file.
Validates that environment variables are correctly loaded into settings.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings


def print_separator(char="=", length=80):
    """Print separator line."""
    print(char * length)


def main():
    """Test configuration loading."""
    print_separator()
    print("🧪 测试配置加载")
    print_separator()
    
    # Test embedding configuration
    print("\n📦 Embedding 配置:")
    print(f"  EMBEDDING_TYPE: {settings.embedding_type}")
    print(f"  EMBEDDING_MODEL: {settings.embedding_model}")
    print(f"  EMBEDDING_BASE_URL: {settings.embedding_base_url}")
    print(f"  EMBEDDING_API_URL: {settings.embedding_api_url}")
    print(f"  EMBEDDING_API_KEY: {'***' + settings.embedding_api_key[-10:] if settings.embedding_api_key else 'None'}")
    print(f"  SILICONFLOW_API_KEY: {'***' + settings.siliconflow_api_key[-10:] if settings.siliconflow_api_key else 'None'}")
    
    # Test LLM configuration
    print("\n🤖 LLM 配置:")
    print(f"  USE_OLLAMA: {settings.use_ollama}")
    if settings.use_ollama:
        print(f"  OLLAMA_BASE_URL: {settings.ollama_base_url}")
        print(f"  OLLAMA_MODEL: {settings.ollama_model}")
        print(f"  OLLAMA_GRADER_MODEL: {settings.ollama_grader_model}")
    else:
        print(f"  OPENAI_API_BASE: {settings.openai_api_base}")
        print(f"  OPENAI_MODEL: {settings.openai_model}")
        print(f"  OPENAI_GRADER_MODEL: {settings.openai_grader_model}")
        print(f"  OPENAI_API_KEY: {'***' + settings.openai_api_key[-10:] if settings.openai_api_key else 'None'}")
    
    # Test web search configuration
    print("\n🌐 联网查询配置:")
    print(f"  WEB_SEARCH_ENABLED: {settings.web_search_enabled}")
    print(f"  WEB_SEARCH_MAX_RESULTS: {settings.web_search_max_results}")
    print(f"  TAVILY_API_KEY: {'***' + settings.tavily_api_key[-10:] if settings.tavily_api_key else 'None'}")
    print(f"  REALTIME_QUERY_ENABLED: {settings.realtime_query_enabled}")
    
    # Test database configuration
    print("\n💾 数据库配置:")
    print(f"  MongoDB URI: {settings.mongodb_uri}")
    print(f"  MongoDB DB: {settings.mongodb_db_name}")
    print(f"  Chroma Host: {settings.chroma_host}:{settings.chroma_port}")
    print(f"  ES Host: {settings.es_host}:{settings.es_port}")
    
    # Test service configuration
    print("\n⚙️ 服务配置:")
    print(f"  API Host: {settings.api_host}:{settings.api_port}")
    print(f"  API Workers: {settings.api_workers}")
    print(f"  Log Level: {settings.log_level}")
    print(f"  Debug: {settings.debug}")
    
    # Validation
    print_separator()
    print("✅ 验证结果:")
    
    issues = []
    
    # Check embedding type
    if settings.embedding_type not in ["openai_style", "siliconflow"]:
        issues.append(f"⚠️ EMBEDDING_TYPE 无效: {settings.embedding_type} (应为 'openai_style' 或 'siliconflow')")
    else:
        print(f"  ✓ EMBEDDING_TYPE: {settings.embedding_type}")
    
    # Check embedding API key based on type
    if settings.embedding_type == "siliconflow":
        if not settings.siliconflow_api_key and not settings.embedding_api_key:
            issues.append("⚠️ SiliconFlow 模式需要设置 SILICONFLOW_API_KEY 或 EMBEDDING_API_KEY")
        else:
            print(f"  ✓ SiliconFlow API Key: 已配置")
    
    # Check web search
    if settings.web_search_enabled and not settings.tavily_api_key:
        issues.append("⚠️ 联网查询已启用但 TAVILY_API_KEY 未配置")
    else:
        print(f"  ✓ 联网查询配置: {'启用' if settings.web_search_enabled else '禁用'}")
    
    # Check LLM
    if not settings.use_ollama and not settings.openai_api_key:
        issues.append("⚠️ OpenAI 模式需要设置 OPENAI_API_KEY")
    else:
        print(f"  ✓ LLM 配置: {'Ollama' if settings.use_ollama else 'OpenAI'}")
    
    if issues:
        print_separator()
        print("❌ 发现配置问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✓ 所有配置验证通过")
    
    print_separator()
    
    return len(issues) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
