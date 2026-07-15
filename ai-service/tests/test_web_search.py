"""
Test web search functionality.
"""
import asyncio
import sys
import os
from pathlib import Path

from dotenv import load_dotenv


# Always load the service-local environment file, regardless of the directory
# from which this diagnostic script is invoked.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}
WEB_SEARCH_TIMEOUT = int(os.getenv("WEB_SEARCH_TIMEOUT", "5"))
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_ONLY_FOR_REALTIME = os.getenv(
    "WEB_SEARCH_ONLY_FOR_REALTIME", "false"
).lower() in {"1", "true", "yes", "on"}

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:

    from langchain_community.tools.tavily_search import TavilySearchResults
    
    async def test_tavily_search():
        """Test Tavily search tool."""
        
        # Check if API key is configured
        if not TAVILY_API_KEY:
            print("❌ TAVILY_API_KEY not configured in .env file")
            return

        if not WEB_SEARCH_ENABLED:
            print("ℹ️  WEB_SEARCH_ENABLED=false; web search test skipped")
            return

        print(f"✅ Tavily API Key found: {TAVILY_API_KEY[8:]}...")
        print(f"✅ Web search enabled: {WEB_SEARCH_ENABLED}")
        print(f"✅ Search timeout: {WEB_SEARCH_TIMEOUT}s")
        print(f"✅ Max results: {WEB_SEARCH_MAX_RESULTS}")
        print(f"✅ Only for realtime: {WEB_SEARCH_ONLY_FOR_REALTIME}")

        # Initialize search tool
        try:
            search_tool = TavilySearchResults(
                max_results=WEB_SEARCH_MAX_RESULTS,
                search_depth="basic",  # 添加此参数
                tavily_api_key=TAVILY_API_KEY,
                )
            print("✅ TavilySearchResults initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize TavilySearchResults: {e}")
            return
        
        # Test queries
        test_queries = [
            "今天北京的天气怎么样",
            "最新的AI技术新闻",
            "Python编程语言的特点"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"🔍 Testing query: {query}")
            print(f"{'='*60}")
            
            try:
                # Perform search
                results = await asyncio.wait_for(
                    search_tool.ainvoke({"query": query}),
                    timeout=WEB_SEARCH_TIMEOUT,
                )

                if not results:
                    print("⚠️  No results returned")
                    continue

                # 先判断 results 类型
                if not isinstance(results, list):
                    results_str = str(results)
                    print(f"\n⚠️  Unexpected results type: {type(results).__name__}")
                    print(f"   Results: {results_str}")

                    # 检查配额错误 (432)
                    if "432" in results_str:
                        print("\n❌ API Quota/Insufficient Funds Error Detected!")
                        print("   ⚠️  Error: 432 - Quota exceeded or insufficient balance")
                        print("   ⚠️  Possible causes:")
                        print("      - API quota has been exhausted")
                        print("      - Account balance is insufficient")
                        print("      - Subscription plan limit reached")
                        print("\n   💡 Solutions:")
                        print("      1. Check your Tavily account at https://tavily.com")
                        print("      2. Upgrade your subscription plan")
                        print("      3. Wait for quota reset (monthly cycle)")
                        print("      4. Temporarily disable web search: WEB_SEARCH_ENABLED=false")
                    # 检查认证错误 (401)
                    elif "401" in results_str or "Unauthorized" in results_str:
                        print("\n❌ API Authentication Error Detected!")
                        print("   ⚠️  Error: 401 Unauthorized")
                        print("   ⚠️  Possible causes:")
                        print("      - API key is expired or invalid")
                        print("      - API key has been revoked")
                        print("      - Incorrect API key format")
                        print(f"\n   📋 Current key prefix: {TAVILY_API_KEY[:12]}...")
                    continue

                print(f"✅ Found {len(results)} results:")

                # 检查列表中的认证错误
                has_auth_error = False
                for result in results:
                    if not isinstance(result, dict):
                        result_str = str(result)
                        if "401" in result_str or "Unauthorized" in result_str:
                            print("\n❌ API Authentication Error Detected!")
                            print("   ⚠️  Error: 401 Unauthorized")
                            print("   ⚠️  Possible causes:")
                            print("      - API key is expired or invalid")
                            print("      - API key has been revoked")
                            print("      - Incorrect API key format")
                            print(f"\n   📋 Current key prefix: {TAVILY_API_KEY[:12]}...")
                            has_auth_error = True
                            break

                if has_auth_error:
                    continue

                # Display results
                valid_count = 0
                for i, result in enumerate(results[:WEB_SEARCH_MAX_RESULTS], 1):
                    # Skip non-dict results (e.g., exceptions or error strings)
                    if not isinstance(result, dict):
                        print(f"\n📄 Result {i}: ⚠️  Invalid result type: {type(result).__name__}")
                        if hasattr(result, '__str__'):
                            result_str = str(result)
                            # Truncate very long error strings
                            if len(result_str) > 100:
                                result_str = result_str[:100] + "..."
                            print(f"   Value: {result_str}")
                        continue

                    valid_count += 1
                    print(f"\n📄 Result {i}:")
                    print(f"   Title: {result.get('title', 'N/A')}")
                    print(f"   URL: {result.get('url', 'N/A')}")
                    print(f"   Score: {result.get('score', 'N/A')}")
                    content = result.get('content', 'N/A')
                    print(f"   Content: {content[:200]}..." if len(content) > 200 else f"   Content: {content}")

                if valid_count == 0:
                    print("\n⚠️  No valid results found (all results were invalid format)")
                    
            except Exception as e:
                print(f"❌ Search failed: {e}")
                import traceback
                traceback.print_exc()

    if __name__ == "__main__":
        print("🚀 Starting Tavily Web Search Test\n")
        asyncio.run(test_tavily_search())
        print("\n✨ Test completed")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
