"""Unit tests for ThinkTagBuffer."""
import pytest
from app.utils.think_tag_buffer import ThinkTagBuffer


def test_think_tag_sequence():
    """测试实际的 think 标签序列。"""
    buffer = ThinkTagBuffer()

    # 正常内容
    result = buffer.add("Hello ")
    assert result == "Hello "

    # 开始标签 token 序列
    result = buffer.add("<th")
    assert result is None
    result = buffer.add("ink")
    assert result is None
    result = buffer.add(">\n")
    assert result is None

    # think 内容（应该被过滤）
    result = buffer.add("This is thinking")
    assert result is None

    # 结束标签 token 序列
    result = buffer.add("</")
    assert result is None
    result = buffer.add("think")
    assert result is None
    result = buffer.add(">\n\n")
    assert result is None

    # think 后的内容
    result = buffer.add(" World!")
    assert result == " World!"


def test_no_think_tags():
    """测试没有 think 标签的内容。"""
    buffer = ThinkTagBuffer()
    result = buffer.add("Hello world!")
    assert result == "Hello world!"


def test_partial_open_tag():
    """测试不完整的开始标签。"""
    buffer = ThinkTagBuffer()
    result = buffer.add("<")
    assert result is None
    result = buffer.add("div>")
    assert result == "<div>"


def test_flush():
    """测试 flush 方法。"""
    buffer = ThinkTagBuffer()
    result = buffer.add("Before")

    result = buffer.add("<th")
    assert result is None
    result = buffer.add("ink")
    assert result is None
    result = buffer.add(">")
    assert result is None

    # 现在 buffer 在 inside 状态
    buffer.add("hidden")  # 被丢弃
    buffer.add("</")      # 进入 detect_close
    buffer.add("think")
    buffer.add(">")

    # 现在回到 outside 状态
    result = buffer.add("After")
    assert result == "After"

    # flush 应该返回空（所有内容都已输出）
    result = buffer.flush()
    assert result is None


def test_is_in_think_tag():
    """测试 is_in_think_tag 方法。"""
    buffer = ThinkTagBuffer()
    assert not buffer.is_in_think_tag()

    buffer.add("<th")
    assert not buffer.is_in_think_tag()

    buffer.add("ink")
    assert not buffer.is_in_think_tag()

    buffer.add(">")
    assert buffer.is_in_think_tag()

    buffer.add("content")
    assert buffer.is_in_think_tag()

    buffer.add("</think>")
    assert not buffer.is_in_think_tag()


def test_consecutive_adds():
    """测试连续添加内容。"""
    buffer = ThinkTagBuffer()
    result = buffer.add("A")
    assert result == "A"

    result = buffer.add("B")
    assert result == "B"

    result = buffer.add("C")
    assert result == "C"


def test_empty_token():
    """测试空 token。"""
    buffer = ThinkTagBuffer()
    result = buffer.add("")
    assert result is None

    result = buffer.add("content")
    assert result == "content"


# =============================================================================
# think 单段计时测试
# =============================================================================


class FakeClock:
    """可控时钟，用于单元测试 wall-clock 计时。"""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds: float):
        self.t += seconds


def test_think_timing_without_think():
    """无 think 标签时，think_time_ms 为 0。"""
    clock = FakeClock()
    buffer = ThinkTagBuffer(time_fn=clock)

    assert buffer.add("hello") == "hello"
    clock.advance(1.0)
    assert buffer.add(" world") == " world"

    meta = buffer.get_timing_metadata()

    assert meta["has_think"] is False
    assert meta["think_time_ms"] == 0
    assert meta["closed"] is False


def test_think_timing_single_closed_block():
    """正常闭合的单段 think，think_time_ms 为首 <think> 到首 </think> 的时长。"""
    clock = FakeClock()
    buffer = ThinkTagBuffer(time_fn=clock)

    assert buffer.add("before") == "before"

    buffer.add("<think>")
    clock.advance(1.25)
    buffer.add("reasoning")
    clock.advance(0.75)
    buffer.add("</think>")

    assert buffer.add("after") == "after"

    meta = buffer.get_timing_metadata()

    assert meta["has_think"] is True
    assert meta["think_time_ms"] == 2000
    assert meta["closed"] is True


def test_think_timing_split_tags():
    """标签被切分时也能正确计时（开始=完整识别 <think>，结束=完整识别 </think>）。"""
    clock = FakeClock()
    buffer = ThinkTagBuffer(time_fn=clock)

    buffer.add("<th")
    clock.advance(0.1)
    buffer.add("ink")
    clock.advance(0.1)
    buffer.add(">")

    clock.advance(3.0)

    buffer.add("</")
    clock.advance(0.1)
    buffer.add("think")
    clock.advance(0.1)
    buffer.add(">")

    meta = buffer.get_timing_metadata()

    assert meta["has_think"] is True
    assert meta["think_time_ms"] == 3200
    assert meta["closed"] is True


def test_think_timing_unclosed_block():
    """未闭合 think，think_time_ms 为 None。"""
    clock = FakeClock()
    buffer = ThinkTagBuffer(time_fn=clock)

    buffer.add("<think>")
    clock.advance(2.5)
    buffer.add("reasoning without close")

    meta = buffer.get_timing_metadata()

    assert meta["has_think"] is True
    assert meta["think_time_ms"] is None
    assert meta["closed"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
