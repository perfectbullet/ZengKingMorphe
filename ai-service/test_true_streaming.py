"""
测试真正的流式输出 vs 假流式输出
通过测量首字节时间(TTFB)和平均token间隔来验证
"""
import asyncio
import aiohttp
import time
import json

async def test_streaming_endpoint(url: str, query: str, name: str):
    """测试流式端点并测量性能指标"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"查询: {query}")
    print(f"{'='*60}\n")
    
    payload = {
        "employee_id": "hutao",
        "messages": [
            {
            "content": query,
            "role": "user"
            }
        ],
        "model": "qwen3:32b",
        "session_id": "sess_20251218_abc123",
        "stream": True,
        "user_id": "user_123456"
    }
    
    start_time = time.time()
    ttfb = None  # Time to first byte
    token_count = 0
    token_times = []
    last_token_time = None
    full_response = ""
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                print(f"状态码: {response.status}")
                
                async for line in response.content:
                    current_time = time.time()
                    
                    # 首字节时间
                    if ttfb is None:
                        ttfb = current_time - start_time
                        print(f"⏱️  首字节时间 (TTFB): {ttfb:.3f}s\n")
                        last_token_time = current_time
                        continue
                    
                    # 解析SSE数据
                    if not line:
                        continue

                    try:
                        data_str = line.decode('utf-8').strip()
                        if not data_str:
                            continue
                        
                        # 移除 "data: " 前缀
                        if data_str.startswith("data:"):
                            data_str = data_str[5:].strip()
                        
                        if not data_str or data_str == "[DONE]":
                            continue
                        
                        data = json.loads(data_str)
                        
                        # 处理不同格式的token
                        token = None
                        if data.get("type") == "token":
                            token = data.get("content", "")
                        elif "choices" in data:
                            delta = data["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                        
                        if token:
                            token_count += 1
                            interval = current_time - last_token_time
                            token_times.append(interval)
                            last_token_time = current_time
                            
                            # 实时显示token
                            print(token, end="", flush=True)
                            full_response += token
                            
                            # 显示token间隔（前10个）
                            if token_count <= 10:
                                print(f"  [{interval:.3f}s]", end="", flush=True)
                    
                    except json.JSONDecodeError:
                        continue
        
        total_time = time.time() - start_time
        
        # 统计分析
        print(f"\n\n{'='*60}")
        print("📊 性能分析:")
        print(f"{'='*60}")
        print(f"首字节时间 (TTFB):     {ttfb:.3f}s")
        print(f"总响应时间:            {total_time:.3f}s")
        print(f"总token数:             {token_count}")
        
        if token_times:
            avg_interval = sum(token_times) / len(token_times)
            min_interval = min(token_times)
            max_interval = max(token_times)
            
            print("\nToken间隔统计:")
            print(f"  平均间隔:            {avg_interval:.3f}s ({1/avg_interval:.1f} tokens/s)")
            print(f"  最小间隔:            {min_interval:.3f}s")
            print(f"  最大间隔:            {max_interval:.3f}s")
            
            # 判断是否为真流式
            print("\n🔍 流式输出判断:")
            if avg_interval < 0.1:
                print(f"✅ 真实流式输出 (平均间隔 {avg_interval*1000:.0f}ms < 100ms)")
                print("   → LLM正在逐token生成")
            elif avg_interval < 0.5:
                print(f"⚠️  可能的真实流式 (平均间隔 {avg_interval*1000:.0f}ms)")
                print("   → 需要检查网络延迟")
            else:
                print(f"❌ 假流式输出 (平均间隔 {avg_interval*1000:.0f}ms > 500ms)")
                print("   → 答案已预先生成，按句子/块分发")
        
        print(f"\n完整回答长度: {len(full_response)} 字符")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主测试函数"""
    base_url = "http://192.168.8.230:8100"  # Docker端口
    
    # 测试查询
    query = "北京天气怎么样"
    
    # 测试两个流式端点
    await test_streaming_endpoint(
        f"{base_url}/api/chat/openai/chat/completions",
        query,
        "自定义流式API (/stream)"
    )
    
    await asyncio.sleep(2)
    
    # await test_streaming_endpoint(
    #     f"{base_url}/api/chat/openai/chat/completions",
    #     query,
    #     "OpenAI兼容API (/openai/chat/completions)"
    # )

    await asyncio.sleep(2)
    query2 = "分别介绍雕蜡与铸造工艺基本原理"
    await test_streaming_endpoint(
        f"{base_url}/api/chat/openai/chat/completions",
        query2,
        "OpenAI兼容API (/openai/chat/completions)"
    )


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          真实流式输出 vs 假流式输出 性能测试                    ║
╚══════════════════════════════════════════════════════════════╝

测试指标:
  • TTFB (Time To First Byte): 首字节时间
  • Token平均间隔: 判断是否为真实流式的关键指标
  
判断标准:
  ✅ 真实流式: 平均间隔 < 100ms (LLM逐token生成)
  ❌ 假流式:   平均间隔 > 500ms (预生成后分块发送)
  
""")
    
    asyncio.run(main())
