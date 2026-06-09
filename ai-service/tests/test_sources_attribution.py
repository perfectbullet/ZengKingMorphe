"""
Test script to validate source attribution in chat responses.
Tests both RAG sources and web search sources.
"""
import asyncio
import httpx
import json
from datetime import datetime


# Test configuration
API_BASE_URL = "http://localhost:8000"
# API_BASE_URL = "http://localhost:8100"  # Use this for Docker deployment

TEST_CASES = [
    {
        "name": "RAG Knowledge Query",
        "request": {
            "user_id": "test_user_001",
            "employee_id": "hutao",
            "query": "雕蜡工艺的基本原理是什么？",
            "context": {"test": True}
        },
        "expected": {
            "has_rag_sources": True,
            "has_web_sources": False
        }
    },
    {
        "name": "Realtime Query (Web Search)",
        "request": {
            "user_id": "test_user_002",
            "employee_id": "hutao",
            "query": "今天北京的天气怎么样？",
            "context": {"test": True}
        },
        "expected": {
            "has_rag_sources": False,
            "has_web_sources": True
        }
    },
    {
        "name": "General Query (Hybrid)",
        "request": {
            "user_id": "test_user_003",
            "employee_id": "hutao",
            "query": "首饰制作的主要工艺有哪些？",
            "context": {"test": True}
        },
        "expected": {
            "has_rag_sources": True,
            "has_web_sources": False
        }
    }
]


def print_separator(char="=", length=80):
    """Print separator line."""
    print(char * length)


def print_sources(sources: dict):
    """Pretty print source attribution."""
    rag_sources = sources.get("rag_sources", [])
    web_sources = sources.get("web_sources", [])
    
    if rag_sources:
        print(f"\n📚 RAG文档来源 ({len(rag_sources)}条):")
        print_separator("-", 60)
        for src in rag_sources:
            print(f"  排名: {src['rank']}")
            print(f"  文档ID: {src['doc_id']}")
            print(f"  知识库ID: {src['kb_id']}")
            print(f"  评分: {src['score']:.4f}")
            if 'chunk_index' in src:
                print(f"  分块索引: {src['chunk_index']}")
            print(f"  内容摘要: {src['content_snippet'][:100]}...")
            print_separator("-", 60)
    else:
        print("\n📚 RAG文档来源: 无")
    
    if web_sources:
        print(f"\n🌐 联网查询来源 ({len(web_sources)}条):")
        print_separator("-", 60)
        for src in web_sources:
            print(f"  排名: {src['rank']}")
            print(f"  标题: {src['title']}")
            print(f"  URL: {src['url']}")
            print(f"  评分: {src['score']:.4f}")
            print_separator("-", 60)
    else:
        print("\n🌐 联网查询来源: 无")


async def test_chat_message(client: httpx.AsyncClient, test_case: dict):
    """Test /api/chat/message endpoint."""
    print_separator()
    print(f"🧪 测试用例: {test_case['name']}")
    print(f"📝 查询: {test_case['request']['query']}")
    print_separator()
    
    try:
        response = await client.post(
            f"{API_BASE_URL}/api/chat/message",
            json=test_case['request'],
            timeout=60.0
        )
        
        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"响应: {response.text}")
            return False
        
        result = response.json()
        data = result.get("data", {})
        
        # Print basic info
        print(f"\n✅ 请求成功!")
        print(f"会话ID: {data.get('conversation_id', 'N/A')}")
        print(f"意图: {data.get('intent', 'N/A')}")
        print(f"置信度: {data.get('confidence', 0.0):.2f}")
        print(f"知识库使用: {data.get('kb_used', [])}")
        print(f"联网查询: {'是' if data.get('web_search_used', False) else '否'}")
        
        # Print answer
        print(f"\n💬 AI回答:")
        print_separator("-", 60)
        print(data.get('answer', 'N/A')[:300] + "...")
        print_separator("-", 60)
        
        # Print sources
        sources = data.get('sources', {})
        if sources:
            print_sources(sources)
            
            # Validate expectations
            expected = test_case['expected']
            has_rag = len(sources.get('rag_sources', [])) > 0
            has_web = len(sources.get('web_sources', [])) > 0
            
            print(f"\n🔍 验证结果:")
            if expected.get('has_rag_sources') == has_rag:
                print(f"  ✅ RAG来源: 符合预期 (预期: {expected.get('has_rag_sources')}, 实际: {has_rag})")
            else:
                print(f"  ⚠️ RAG来源: 不符合预期 (预期: {expected.get('has_rag_sources')}, 实际: {has_rag})")
            
            if expected.get('has_web_sources') == has_web:
                print(f"  ✅ 联网来源: 符合预期 (预期: {expected.get('has_web_sources')}, 实际: {has_web})")
            else:
                print(f"  ⚠️ 联网来源: 不符合预期 (预期: {expected.get('has_web_sources')}, 实际: {has_web})")
        else:
            print("\n⚠️ 警告: 响应中没有sources字段!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_openai_completion(client: httpx.AsyncClient):
    """Test OpenAI-compatible endpoint."""
    print_separator()
    print(f"🧪 测试OpenAI兼容接口")
    print_separator()

    request_data = {
        "model": "qwen3:14b",
        "messages": [
            {"role": "user", "content": "首饰铸造工艺的优缺点是什么？"}
        ],
        "stream": True,  # API only supports streaming
        "employee_id": "hutao",
        "user_id": "test_openai_user"
    }

    try:
        response = await client.post(
            f"{API_BASE_URL}/api/chat/v1/chat/completions",
            json=request_data,
            timeout=60.0
        )

        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"响应: {response.text}")
            return False

        # Consume streaming response
        full_content = ""
        async for line in response.aiter_lines():
            if not line or line.startswith(":"):
                continue
            if line == "data: [DONE]":
                break
            if line.startswith("data: "):
                line = line[6:]
            try:
                chunk_data = json.loads(line)
                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                    delta = chunk_data["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_content += delta["content"]
            except json.JSONDecodeError:
                pass

        print(f"\n✅ 请求成功!")
        print(f"\n💬 AI回答:")
        print_separator("-", 60)
        print(full_content[:300] + "...")
        print_separator("-", 60)

        # Note: Detailed metadata extraction from streaming chunks would require
        # additional parsing of the done chunk
        print(f"\n⚠️ 注意: 流式响应的元数据需要从完成chunk中解析")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print_separator("=")
    print("🚀 开始测试来源归属功能")
    print(f"⏰ 测试时间: {datetime.now().isoformat()}")
    print(f"🔗 API地址: {API_BASE_URL}")
    print_separator("=")
    
    async with httpx.AsyncClient() as client:
        # Test standard chat endpoint
        success_count = 0
        total_count = len(TEST_CASES)
        
        for test_case in TEST_CASES:
            if await test_chat_message(client, test_case):
                success_count += 1
            await asyncio.sleep(1)  # Rate limiting
        
        # Test OpenAI endpoint
        print("\n")
        if await test_openai_completion(client):
            success_count += 1
        total_count += 1
    
    # Summary
    print_separator("=")
    print(f"📊 测试总结")
    print(f"总计: {total_count} 个测试")
    print(f"成功: {success_count} 个")
    print(f"失败: {total_count - success_count} 个")
    print(f"成功率: {(success_count/total_count)*100:.1f}%")
    print_separator("=")


if __name__ == "__main__":
    asyncio.run(main())
