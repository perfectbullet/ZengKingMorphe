"""
Test script for streaming chunks storage and query functionality.

This script demonstrates:
1. How to make a streaming chat request (OpenAI-style)
2. How to query stored chunks by various filters

Usage:
    D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe test_stream_chunks.py
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

# API Configuration
BASE_URL = "http://192.168.8.230:8100"  # Change to 8000 if running locally
API_KEY = "test-key"  # Add your API key if auth is enabled

# Test data
TEST_USER_ID = "test_user_001"
TEST_EMPLOYEE_ID = "hutao"
TEST_SESSION_ID = f"sess_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


async def test_streaming_chat():
    """
    Test streaming chat endpoint and verify chunks are saved to MongoDB.
    """
    print("\n=== Testing Streaming Chat (OpenAI-style) ===")
    
    url = f"{BASE_URL}/api/chat/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "qwen3:32b",
        "messages": [
            {"role": "user", "content": "原神里面的巴巴托斯是谁？"}
        ],
        "stream": True,
        "employee_id": TEST_EMPLOYEE_ID,
        "user_id": TEST_USER_ID,
        "session_id": TEST_SESSION_ID
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Error: {response.status} - {error_text}")
                    return None
                
                ttfb = time.perf_counter() - start_time
                print(f"✅ Connection established (status: {response.status}) in {ttfb*1000:.2f}ms")
                print(f"📝 Session ID: {TEST_SESSION_ID}")
                print(f"👤 User ID: {TEST_USER_ID}")
                print(f"🤖 Employee ID: {TEST_EMPLOYEE_ID}")
                print("\n📦 Receiving chunks:\n")
                
                chat_id = None
                chunk_count = 0
                full_content = ""
                first_token_latency = None
                
                # Read SSE stream
                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    
                    if not line_text or line_text.startswith(":"):
                        continue
                    
                    if line_text == "data: [DONE]" or line_text == "[DONE]":
                        print("\n✅ Stream completed: [DONE]")
                        break
                    
                    # Remove "data: " prefix if present
                    if line_text.startswith("data: "):
                        line_text = line_text[6:]
                    
                    try:
                        chunk_data = json.loads(line_text)
                        chunk_count += 1
                        
                        # Extract chat_id from first chunk
                        if chat_id is None and "id" in chunk_data:
                            chat_id = chunk_data["id"]
                            print(f"💬 Chat ID: {chat_id}\n")
                        
                        # Process chunk based on type
                        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                            delta = chunk_data["choices"][0].get("delta", {})
                            
                            if "role" in delta:
                                print(f"[Chunk {chunk_count}] Role: {delta['role']}")
                            
                            if "content" in delta:
                                if first_token_latency is None and delta["content"]:
                                    first_token_latency = time.perf_counter() - start_time
                                    print(f"⏱️ First token latency: {first_token_latency*1000:.2f}ms\n")
                                
                                content = delta["content"]
                                full_content += content
                                print(content, end="", flush=True)
                            
                            finish_reason = chunk_data["choices"][0].get("finish_reason")
                            if finish_reason:
                                print(f"\n\n[Chunk {chunk_count}] Finish reason: {finish_reason}")
                                
                                # Print metadata if available
                                if "metadata" in chunk_data:
                                    metadata = chunk_data["metadata"]
                                    print("\n📊 Metadata:")
                                    print(f"  - Conversation ID: {metadata.get('conversation_id')}")
                                    print(f"  - Confidence: {metadata.get('confidence')}")
                                    print(f"  - KB Used: {metadata.get('kb_used')}")
                                    print(f"  - Web Search: {metadata.get('web_search_used')}")
                        
                        elif "error" in chunk_data:
                            print(f"\n❌ Error chunk: {chunk_data['error']}")
                            return None
                        
                    except json.JSONDecodeError as e:
                        print(f"\n⚠️ Failed to parse chunk: {line_text[:100]}")
                        continue
                
                print("\n\n📊 Summary:")
                print(f"  - Total chunks received: {chunk_count}")
                print(f"  - Content length: {len(full_content)} chars")
                print(f"  - First token latency: {f'{first_token_latency*1000:.2f}ms' if first_token_latency else 'N/A'}")
                print(f"  - Chat ID: {chat_id}")
                
                return chat_id
                
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_query_chunks(chat_id=None):
    """
    Test chunk query endpoint with various filters.
    """
    print("\n\n=== Testing Chunk Query Endpoint ===")
    
    url = f"{BASE_URL}/api/chat/stream/chunks"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # Test Case 1: Query by session_id
    print("\n📋 Test Case 1: Query by session_id")
    params = {
        "session_id": TEST_SESSION_ID,
        "page": 1,
        "page_size": 20
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    data = result.get("data", {})
                    chunks = data.get("chunks", [])
                    pagination = data.get("pagination", {})
                    
                    print(f"✅ Found {len(chunks)} chunks")
                    print(f"📄 Pagination: Page {pagination.get('page')}/{pagination.get('total_pages')}, Total: {pagination.get('total_count')}")
                    
                    # Show chunk types distribution
                    chunk_types = {}
                    for chunk in chunks:
                        chunk_type = chunk.get("chunk_type", "unknown")
                        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    
                    print(f"📊 Chunk types: {chunk_types}")
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {response.status} - {error_text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test Case 2: Query by user_id and employee_id
    print("\n📋 Test Case 2: Query by user_id and employee_id")
    params = {
        "user_id": TEST_USER_ID,
        "employee_id": TEST_EMPLOYEE_ID,
        "page": 1,
        "page_size": 200
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    data = result.get("data", {})
                    chunks = data.get("chunks", [])
                    pagination = data.get("pagination", {})
                    
                    print(f"✅ Found {len(chunks)} chunks for user {TEST_USER_ID}")
                    print(f"📄 Total count: {pagination.get('total_count')}")
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {response.status} - {error_text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test Case 3: Query by chat_id (if available)
    if chat_id:
        print(f"\n📋 Test Case 3: Query by chat_id ({chat_id})")
        params = {
            "chat_id": chat_id,
            "page": 1,
            "page_size": 100
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        data = result.get("data", {})
                        chunks = data.get("chunks", [])
                        
                        print(f"✅ Found {len(chunks)} chunks for chat {chat_id}")
                        
                        # Show detailed info for each chunk
                        print("\n📦 Chunk details:")
                        for i, chunk in enumerate(chunks, 1):
                            chunk_type = chunk.get("chunk_type")
                            sequence = chunk.get("sequence")
                            timestamp = chunk.get("timestamp", "N/A")
                            # print(f"  [{i}] Seq: {sequence}, Type: {chunk_type}, Time: {timestamp}")
                    else:
                        error_text = await response.text()
                        print(f"❌ Error: {response.status} - {error_text}")
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # Test Case 4: Query only 'token' chunks
    print("\n📋 Test Case 4: Query only 'token' type chunks")
    params = {
        "session_id": TEST_SESSION_ID,
        "chunk_type": "token",
        "page": 1,
        "page_size": 100
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    data = result.get("data", {})
                    chunks = data.get("chunks", [])
                    
                    print(f"✅ Found {len(chunks)} token chunks")
                    
                    # Reconstruct content from token chunks
                    if chunks:
                        reconstructed_content = ""
                        for chunk in reversed(chunks):  # Reverse to get chronological order
                            chunk_data = chunk.get("chunk_data", {})
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    reconstructed_content += delta["content"]
                        
                        print(f"\n📝 Reconstructed content ({len(reconstructed_content)} chars):")
                        print(f"{reconstructed_content[:200]}...")
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {response.status} - {error_text}")
    except Exception as e:
        print(f"❌ Exception: {e}")


async def main():
    """
    Main test flow.
    """
    print("=" * 60)
    print("Stream Chunks Storage & Query Test")
    print("=" * 60)
    
    # Step 1: Test streaming chat
    chat_id = await test_streaming_chat()
    
    # Wait a bit for data to be fully saved
    await asyncio.sleep(2)
    
    # Step 2: Test chunk queries
    await test_query_chunks(chat_id)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
