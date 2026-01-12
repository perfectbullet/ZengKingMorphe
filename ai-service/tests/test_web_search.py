"""
Test web search functionality.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app.core.config import settings
    from langchain_community.tools.tavily_search import TavilySearchResults
    
    async def test_tavily_search():
        """Test Tavily search tool."""
        
        # Check if API key is configured
        if not settings.tavily_api_key:
            print("❌ TAVILY_API_KEY not configured in .env file")
            return
        
        print(f"✅ Tavily API Key found: {settings.tavily_api_key[:8]}...")
        print(f"✅ Web search enabled: {settings.web_search_enabled}")
        print(f"✅ Max results: {settings.web_search_max_results}")
        
        # Initialize search tool
        try:
            search_tool = TavilySearchResults(k=settings.web_search_max_results)
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
                results = await search_tool.ainvoke({"query": query})
                
                if not results:
                    print("⚠️  No results returned")
                    continue
                
                print(f"✅ Found {len(results)} results:")
                
                # Display results
                for i, result in enumerate(results, 1):
                    print(f"\n📄 Result {i}:")
                    print(f"   Title: {result.get('title', 'N/A')}")
                    print(f"   URL: {result.get('url', 'N/A')}")
                    print(f"   Score: {result.get('score', 'N/A')}")
                    print(f"   Content: {result.get('content', 'N/A')[:200]}...")
                    
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
