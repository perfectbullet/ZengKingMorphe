r"""
测试 LaTeX 公式分割问题

模拟流式输出场景，测试 sentence buffer 如何处理各种 LaTeX 公式：
- \(...\) 内联公式
- $...$ 内联公式
- 公式内部的 ) 和 . 不应导致切分
"""
import sys
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, '/home/zj/ZengKingMorphe/ai-service')

from app.utils.sentence_buffer import SentenceBuffer

# 测试输入文件路径
TEST_INPUT_DIR = Path(__file__).parent / "test_input"
TEST_INPUT_FILES = [
    "test_bracket_formula_issue_input3.txt",
]


def load_test_input(filename: str) -> str:
    """从 test_input 文件夹加载测试输入数据"""
    filepath = TEST_INPUT_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"测试输入文件不存在: {filepath}")
    return filepath.read_text(encoding="utf-8")


def run_test_with_input(input_name: str, input_content: str, max_chars: int = 100):
    """通用测试函数，使用指定的输入内容运行测试"""
    buffer = SentenceBuffer(max_chars=max_chars, max_wait_seconds=0.5)
    segments_collected = []

    print("=" * 80)
    print(f"测试场景：模拟 LLM 逐个 token 输出 ({input_name})")
    print("=" * 80)
    print()

    # 模拟流式输出：逐两个字符添加
    for i in range(0, len(input_content), 2):
        token = input_content[i:i+2]
        segment = buffer.add(token)
        if segment:
            segments_collected.append(segment)

    # 输出最后剩余内容
    if buffer.buffer:
        segments_collected.append(buffer.buffer)

    print(f"segments 数量: {len(segments_collected)}")
    print("-" * 80)
    print("buffer.add 处理后的 segments:")
    print("-" * 80)
    for i, seg in enumerate(segments_collected):
        print(f"[{i:2d}] {repr(seg)}")

    return segments_collected


if __name__ == "__main__":
    # 检查测试输入文件是否存在
    missing_files = []
    for filename in TEST_INPUT_FILES:
        if not (TEST_INPUT_DIR / filename).exists():
            missing_files.append(filename)

    if missing_files:
        print(f"错误：测试输入文件不存在: {', '.join(missing_files)}")
        print(f"请确保测试输入文件位于: {TEST_INPUT_DIR}")
        sys.exit(1)

    # 运行所有测试
    for i, filename in enumerate(TEST_INPUT_FILES, 1):
        if i > 1:
            print("\n" * 3)

        input_name = f"INPUT{i}"
        input_content = load_test_input(filename)
        run_test_with_input(input_name, input_content)
