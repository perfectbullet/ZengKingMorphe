"""
Test async document upload functionality.
"""
import requests
import time

# Configuration
BASE_URL = "http://localhost:8100"
API_KEY = "test-api-key"

def test_async_upload():
    """Test async document upload with task tracking."""
    
    print("=== Testing Async Document Upload ===\n")
    
    # Step 1: Create a knowledge base
    print("1. Creating knowledge base...")
    kb_response = requests.post(
        f"{BASE_URL}/api/knowledge_base/create",
        headers={"X-API-Key": API_KEY},
        json={
            "name": "Test KB for Async Upload",
            "description": "Testing async upload functionality",
            "category": "test"
        }
    )
    
    if kb_response.status_code != 200:
        print(f"❌ Failed to create KB: {kb_response.text}")
        return
    
    kb_data = kb_response.json()
    kb_id = kb_data["kb_id"]
    print(f"✅ Created KB: {kb_id}\n")
    
    # Step 2: Upload a test document in async mode
    print("2. Uploading document in async mode...")
    
    # Create a test file
    test_content = "This is a test document.\n\nIt has multiple paragraphs.\n\nEach paragraph should be chunked separately."
    
    files = {
        'files': ('test_doc.txt', test_content, 'text/plain')
    }
    
    data = {
        'kb_id': kb_id,
        'category': 'test',
        'async_mode': 'true'
    }
    
    upload_response = requests.post(
        f"{BASE_URL}/api/knowledge_base/documents/upload",
        headers={"X-API-Key": API_KEY},
        files=files,
        data=data
    )
    
    if upload_response.status_code != 200:
        print(f"❌ Upload failed: {upload_response.text}")
        return
    
    upload_data = upload_response.json()
    task_ids = upload_data.get("tasks", [])
    
    if not task_ids:
        print("❌ No task IDs returned")
        return
    
    task_id = task_ids[0]
    print(f"✅ Task submitted: {task_id}\n")
    
    # Step 3: Poll task status
    print("3. Monitoring task progress...")
    
    max_wait = 60  # seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        status_response = requests.get(
            f"{BASE_URL}/api/knowledge_base/documents/tasks/{task_id}",
            headers={"X-API-Key": API_KEY}
        )
        
        if status_response.status_code != 200:
            print(f"❌ Failed to get status: {status_response.text}")
            break
        
        status_data = status_response.json()
        task = status_data.get("task", {})
        
        status = task.get("status")
        progress = task.get("progress", 0)
        processed = task.get("processed_chunks", 0)
        total = task.get("total_chunks", 0)
        
        print(f"   Status: {status} | Progress: {progress}% | Chunks: {processed}/{total}")
        
        if status == "completed":
            doc_id = task.get("doc_id")
            print(f"\n✅ Task completed! Document ID: {doc_id}")
            break
        elif status == "failed":
            error = task.get("error_message")
            print(f"\n❌ Task failed: {error}")
            break
        elif status == "cancelled":
            print("\n⚠️ Task was cancelled")
            break
        
        time.sleep(2)
    else:
        print("\n⏱️ Timeout waiting for task completion")
    
    # Step 4: Test task cancellation (optional)
    print("\n4. Testing task cancellation...")
    
    # Upload another document
    files2 = {
        'files': ('test_cancel.txt', "Cancel me!", 'text/plain')
    }
    
    upload_response2 = requests.post(
        f"{BASE_URL}/api/knowledge_base/documents/upload",
        headers={"X-API-Key": API_KEY},
        files=files2,
        data={'kb_id': kb_id, 'async_mode': 'true'}
    )
    
    if upload_response2.status_code == 200:
        task_id2 = upload_response2.json()["tasks"][0]
        print(f"   Submitted task: {task_id2}")
        
        # Try to cancel it immediately
        cancel_response = requests.delete(
            f"{BASE_URL}/api/knowledge_base/documents/tasks/{task_id2}",
            headers={"X-API-Key": API_KEY}
        )
        
        if cancel_response.status_code == 200:
            print(f"✅ Task cancelled successfully")
        else:
            print(f"⚠️ Cancel response: {cancel_response.text}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_async_upload()
