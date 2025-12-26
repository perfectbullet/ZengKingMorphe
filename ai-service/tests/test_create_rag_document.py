"""
Test script for create_rag_document_with_segment endpoint.
"""
import asyncio
import aiohttp
import json


async def test_create_rag_document():
    """Test creating RAG document with custom segment configuration."""
    
    # Test endpoint
    base_url = "http://localhost:8100"  # Docker mapped port
    endpoint = f"{base_url}/api/knowledge_base/documents/create_with_segment"
    
    # Test data matching Java platform format
    test_request = {
        "team_id": 40,
        "dataset_id": 3,
        "resource_id": 48907,
        "document_name": "测试文档-自定义分段.txt",
        "segment_flag": 1,  # Use custom segmentation
        "segment_vo": {
            "team_id": 40,
            "dataset_document_id": 16,
            "is_space_flag": 1,  # Enable space/newline/tab removal
            "is_menu_flag": 0,   # Disable TOC removal (for simple test)
            "segment_type": 1,   # Identifier-based splitting
            "is_segment_union_flag": 1,  # Enable segment merging
            "segment_union_max_length": 700,
            "segment_identifier_type": 0,  # System default identifiers
            "identifier_default": "1111111",  # All identifiers enabled
            "identifier_customize": ""
        },
        "rag_data_set_id": "kb_316a7dbc75d0",  # Existing knowledge base
        "resource_url": "https://raw.githubusercontent.com/example/test.txt"  # Replace with valid URL
    }
    
    print("=" * 80)
    print("Testing create_rag_document_with_segment endpoint")
    print("=" * 80)
    print(f"\nEndpoint: {endpoint}")
    print(f"\nRequest payload:")
    print(json.dumps(test_request, indent=2, ensure_ascii=False))
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=test_request, timeout=aiohttp.ClientTimeout(total=30)) as response:
                status_code = response.status
                response_data = await response.json()
                
                print(f"\n{'='*80}")
                print(f"Response Status: {status_code}")
                print(f"{'='*80}")
                print(json.dumps(response_data, indent=2, ensure_ascii=False))
                
                if status_code == 200:
                    print(f"\n✓ Success! Task ID: {response_data.get('data', {}).get('task_id')}")
                    
                    # Test getting task status
                    task_id = response_data.get('data', {}).get('task_id')
                    if task_id:
                        await asyncio.sleep(2)  # Wait for processing to start
                        status_endpoint = f"{base_url}/api/knowledge_base/documents/tasks/{task_id}"
                        
                        async with session.get(status_endpoint) as status_response:
                            status_data = await status_response.json()
                            print(f"\n{'='*80}")
                            print(f"Task Status:")
                            print(f"{'='*80}")
                            print(json.dumps(status_data, indent=2, ensure_ascii=False))
                else:
                    print(f"\n✗ Failed with status {status_code}")
    
    except aiohttp.ClientError as e:
        print(f"\n✗ Network error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")

# Simple test with minimal config
simple_request = {
    "team_id": 1,
    "dataset_id": 1,
    "resource_id": 1001,
    "document_name": "simple_test.txt",
    "segment_flag": 0,  # Auto segmentation (use defaults)
    "rag_data_set_id": "kb_316a7dbc75d0",
    "resource_url": "https://www.example.com/test.txt"  # Replace with real URL
}

async def test_with_simple_url():
    """Test with a simple text file URL."""
    
    base_url = "http://localhost:8100"
    endpoint = f"{base_url}/api/knowledge_base/documents/create_with_segment"
    

    print("\n" + "=" * 80)
    print("Testing with AUTO segmentation (segment_flag=0)")
    print("=" * 80)
    print(json.dumps(simple_request, indent=2, ensure_ascii=False))
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=simple_request, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response_data = await response.json()
                print(f"\nResponse:")
                print(json.dumps(response_data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\n✗ Error: {e}")


async def test_custom_identifiers():
    """Test with custom identifiers."""
    
    base_url = "http://localhost:8100"
    endpoint = f"{base_url}/api/knowledge_base/documents/create_with_segment"
    
    custom_request = {
        "team_id": 2,
        "dataset_id": 2,
        "resource_id": 2002,
        "document_name": "custom_identifier_test.txt",
        "segment_flag": 1,
        "segment_vo": {
            "team_id": 2,
            "dataset_document_id": 20,
            "is_space_flag": 1,
            "is_menu_flag": 0,
            "segment_type": 1,
            "is_segment_union_flag": 1,
            "segment_union_max_length": 500,
            "segment_identifier_type": 1,  # Custom identifiers
            "identifier_default": "",
            "identifier_customize": "###,===,---"  # Custom separators
        },
        "rag_data_set_id": "kb_316a7dbc75d0",
        "resource_url": "https://www.example.com/custom.txt"
    }
    
    print("\n" + "=" * 80)
    print("Testing with CUSTOM identifiers")
    print("=" * 80)
    print(json.dumps(custom_request, indent=2, ensure_ascii=False))
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=simple_request, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response_data = await response.json()
                print(f"\nResponse:")
                print(json.dumps(response_data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RAG Document Creation Test Suite                                           ║
║  Tests the /api/knowledge_base/documents/create_with_segment endpoint       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Prerequisites:
1. AI service running on http://localhost:8100
2. Knowledge base 'kb_316a7dbc75d0' exists
3. Valid resource URL in test data

Note: Replace resource_url with a real accessible URL before testing
    """)
    
    # Run tests
    asyncio.run(test_create_rag_document())
    # asyncio.run(test_with_simple_url())
    # asyncio.run(test_custom_identifiers())
