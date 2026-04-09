"""测试 ThinkTagBuffer 使用真实的流式输出数据。"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.think_tag_buffer import ThinkTagBuffer


def test_with_stream_out():
    """使用 /home/zj/ZengKingMorphe/stream.out 文件进行测试。"""
    buffer = ThinkTagBuffer()
    output = []
    stream_out_path = '/home/zj/ZengKingMorphe/stream.out'

    if not os.path.exists(stream_out_path):
        print(f"错误: 文件不存在: {stream_out_path}")
        return False

    with open(stream_out_path, 'r') as f:
        for line in f:
            token = line.rstrip('\n')
            if not token:
                continue

            filtered = buffer.add(token)
            if filtered:
                output.append(filtered)

    # 验证结果
    result = ''.join(output)

    # 应该不包含 think 标签（检查原始标签，因为 stream.out 中的 \n 是字面字符）
    # stream.out 中的 think 标签格式是：
    # <th
    # ink
    # >\n  (这里的 \n 是字面字符)
    # ...
    # </
    # think
    # >\n\n

    # 检查是否过滤掉了 think 内容
    if '嗯，我现在要找的是椭圆' in result:
        print(f"✗ 测试失败: 结果中包含 think 标签内的内容")
        return False

    # 应该包含实际输出内容
    expected_contents = [
        '椭圆的标准方程为',
        '\\frac',
        'sqrt{7}',
    ]

    for expected in expected_contents:
        if expected not in result:
            print(f"✗ 测试失败: 结果中不包含期望内容 '{expected}'")
            print(f"实际结果: {result[:500]}")
            return False

    print("✓ 测试通过！")
    print(f"过滤后内容长度: {len(result)} 字符")
    print(f"前 200 字符: {result[:200]}")
    print(f"后 200 字符: {result[-200:]}")
    return True


if __name__ == "__main__":
    success = test_with_stream_out()
    sys.exit(0 if success else 1)
