"""
测试数字员工信息集成功能
"""
import requests
import json

BASE_URL = "http://localhost:8100"
API_KEY = "test-api-key"

def test_chat_with_employee():
    """测试带员工配置的对话"""
    
    print("=== 测试数字员工对话集成 ===\n")
    
    # 测试请求
    chat_request = {
        "user_id": "test_user_001",
        "employee_id": "hutao",  # 使用胡桃员工配置
        "query": "如何重置密码？",  # FAQ 问题
        "context": {}
    }
    
    print("1. 发送聊天请求...")
    print(f"   用户问题: {chat_request['query']}")
    print(f"   员工ID: {chat_request['employee_id']}\n")
    
    response = requests.post(
        f"{BASE_URL}/api/chat/message",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        },
        json=chat_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ 请求成功!")
        print(f"\n回复内容:")
        print(f"  {result['data'].get('answer', '无回复')}\n")
        print(f"置信度: {result['data'].get('confidence', 0)}")
        print(f"使用知识库: {result['data'].get('kb_used', [])}")
        print(f"FAQ匹配: {result['data'].get('faq_matched')}")
        print(f"响应时间: {result['data'].get('response_time_ms', 0)}ms")
    else:
        print(f"❌ 请求失败: {response.status_code}")
        print(response.text)

def test_stream_chat():
    """测试流式对话"""
    
    print("\n=== 测试流式对话 ===\n")
    
    chat_request = {
        "user_id": "test_user_001",
        "employee_id": "hutao",
        "query": "介绍一下雕蜡工艺",
        "context": {}
    }
    
    print("发送流式请求...")
    
    response = requests.post(
        f"{BASE_URL}/api/chat/stream",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        },
        json=chat_request,
        stream=True
    )
    
    if response.status_code == 200:
        print("✅ 流式响应:\n")
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data = json.loads(line_text[6:])
                    
                    if data['type'] == 'start':
                        print(f"[开始] session_id: {data['session_id']}")
                    elif data['type'] == 'progress':
                        print(f"[进度] 执行节点: {data['node']}")
                    elif data['type'] == 'token':
                        print(data['content'], end='', flush=True)
                    elif data['type'] == 'done':
                        print(f"\n\n[完成] conversation_id: {data['conversation_id']}")
                        print(f"使用知识库: {data.get('kb_used', [])}")
                    elif data['type'] == 'error':
                        print(f"\n[错误] {data['message']}")
    else:
        print(f"❌ 请求失败: {response.status_code}")

def test_employee_config():
    """测试员工配置查询"""
    
    print("\n=== 测试员工配置 ===\n")
    
    response = requests.get(
        f"{BASE_URL}/api/employee/hutao",
        headers={"X-API-Key": API_KEY}
    )
    
    if response.status_code == 200:
        config = response.json()
        print("✅ 员工配置:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
    else:
        print(f"❌ 查询失败: {response.status_code}")

if __name__ == "__main__":
    print("注意: 请确保服务已启动 (docker-compose up -d)\n")
    
    try:
        # 测试员工配置
        test_employee_config()
        
        # 测试同步对话
        test_chat_with_employee()
        
        # 测试流式对话
        test_stream_chat()
        
        print("\n=== 所有测试完成 ===")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ 无法连接到服务，请检查服务是否启动")
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
