"""
测试 BGE-Large 的最大长度限制
"""
from openai import OpenAI
import time


def test_length_limits():
    """测试不同长度中文字符的嵌入生成"""

    client = OpenAI(
        api_key="not-needed",
        base_url="http://192.168.8.233:8093/v1",
        timeout=120,
    )

    # 生成测试用的中文字符
    chars = "人工智能是计算机科学的一个分支机器学习深度学习神经网络自然语言处理计算机视觉数据挖掘大数据云计算物联网区块链元宇宙数字孪生边缘计算量子计算算法编程软件硬件网络通信操作系统数据库前端后端全栈开发敏捷开发DevOps持续集成持续部署"

    # 测试长度：100, 200, 300, 400, 500 字符
    test_lengths = [100, 200, 300, 400, 500]

    print("=" * 70)
    print("BGE-Large 长度限制测试 (最大 512 tokens)")
    print("中文字符约 2-3 tokens/字符，理论最大约 200-250 字符")
    print("=" * 70)

    results = []

    for length in test_lengths:
        print(f"\n测试 {length} 个中文字符...")

        # 生成指定长度的文本
        text = (chars * ((length // len(chars)) + 1))[:length]
        print(f"  实际文本长度: {len(text)} 个字符")

        try:
            start_time = time.time()

            response = client.embeddings.create(
                model="BAAI/BGE-large",
                input=text,
                encoding_format="float",
            )

            elapsed = time.time() - start_time
            embedding = response.data[0].embedding

            print(f"  ✓ 成功! 耗时: {elapsed:.2f}秒")
            print(f"    嵌入维度: {len(embedding)}")
            print(f"    前5个值: {embedding[:5]}")

            results.append({
                'length': length,
                'success': True,
                'time': elapsed,
                'embedding_dim': len(embedding)
            })

        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ 失败! 错误: {error_msg[:100]}")

            # 提取实际 token 限制信息
            if "maximum context length" in error_msg:
                import re
                match = re.search(r'maximum context length is (\d+) tokens', error_msg)
                if match:
                    actual_limit = match.group(1)
                    print(f"    模型实际最大上下文: {actual_limit} tokens")

            results.append({
                'length': length,
                'success': False,
                'error': error_msg[:100]
            })

    # 打印汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"{'长度':<10} {'状态':<10} {'耗时(秒)':<15} {'嵌入维度':<10}")
    print("-" * 70)
    for r in results:
        status = "✓ 成功" if r['success'] else "✗ 失败"
        time_str = f"{r['time']:.2f}" if r.get('time') else "N/A"
        dim_str = f"{r.get('embedding_dim', 'N/A')}" if r['success'] else "N/A"
        print(f"{r['length']:<10} {status:<10} {time_str:<15} {dim_str:<10}")

    # 分析结论
    print("\n" + "=" * 70)
    print("结论分析")
    print("=" * 70)

    successful_lengths = [r['length'] for r in results if r['success']]
    failed_lengths = [r['length'] for r in results if not r['success']]

    if successful_lengths:
        max_successful = max(successful_lengths)
        print(f"✓ 成功的最大长度: {max_successful} 字符")
        print(f"  估算 tokens/字符: {512 / max_successful:.2f}")

    if failed_lengths:
        min_failed = min(failed_lengths)
        print(f"✗ 失败的最小长度: {min_failed} 字符")
        print(f"  估算 tokens/字符: {512 / min_failed:.2f}")

    print(f"\n推荐: 中文文本最大长度约 {successful_lengths[-1] if successful_lengths else 200} 字符")

    return results


if __name__ == "__main__":
    test_length_limits()
