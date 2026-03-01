"""
Test SentenceBuffer timing behavior to verify single-char segment fix.
"""
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.sentence_buffer import SentenceBuffer

def test_first_token_resets_timer():
    """
    Test that the first token resets the timer, preventing immediate timeout flush.

    This simulates the scenario where:
    1. SentenceBuffer is created (last_flush_time = T0)
    2. Delay occurs (workflow processing)
    3. First token arrives after > max_wait_seconds
    4. Should NOT immediately flush due to timeout
    """
    buffer = SentenceBuffer(max_wait_seconds=0.5)

    # Simulate workflow delay - buffer is created but no tokens yet
    time.sleep(0.6)  # Exceeds max_wait_seconds

    # First token arrives - should reset timer, not immediately flush
    result = buffer.add("奇")

    # Should NOT flush immediately even though time elapsed since creation
    # The timer was reset when the first token arrived
    assert result is None, "First token should not cause immediate flush despite workflow delay"
    assert buffer.get_buffer_length() == 1
    assert buffer.buffer == "奇"

def test_still_respects_timeout_after_tokens():
    """
    Test that timeout still works after tokens have been accumulated.
    """
    buffer = SentenceBuffer(max_wait_seconds=0.3, max_chars=100)

    # Add some tokens
    buffer.add("测试")
    time.sleep(0.4)  # Exceeds max_wait_seconds

    # Should flush due to timeout
    result = buffer.add("文本")
    assert result == "测试文本", "Should flush after timeout when tokens exist"

def test_normal_sentence_splitting_still_works():
    """
    Test that normal sentence splitting logic is not affected.
    """
    buffer = SentenceBuffer(max_chars=100)

    result = buffer.add("这是第一句话。")
    assert result == "这是第一句话。", "Should split at sentence end"

    result = buffer.add("这是第二句话，")
    assert result is None, "Should not split at comma when below threshold"

    result = buffer.add("继续一些文本。")
    assert "继续一些文本。" in result, "Should split at sentence end"

def test_empty_buffer_condition():
    """
    Test that timer resets only when buffer is truly empty.
    """
    buffer = SentenceBuffer(max_wait_seconds=0.5)

    # First token - buffer was empty
    buffer.add("测")
    time.sleep(0.6)

    # Add more - buffer was NOT empty, so timer wasn't reset
    # Since time elapsed, should flush
    result = buffer.add("试")
    assert result is not None, "Should flush after timeout when buffer wasn't empty"

def test_display_formula_not_split():
    """
    Test that complete display formulas ($$...$$) spanning max_chars are not split.

    This ensures that formulas with newlines and long content are kept together.
    """
    import asyncio

    buffer = SentenceBuffer(max_chars=100, max_wait_seconds=0.5)

    # A long display formula with newlines (107 chars)
    formula = r'$$\n(x + y)^3 = \binom{3}{0} x^3 y^0 + \binom{3}{1} x^2 y^1 + \binom{3}{2} x^1 y^2 + \binom{3}{3} x^0 y^3\n$$'

    # Add characters one by one
    for char in formula:
        result = buffer.add(char)
        # Should not split in the middle of the formula
        if result:
            assert result.startswith("$$") and result.endswith("$$"), \
                f"Formula was split incorrectly: {repr(result[:30])}..."

    # Flush to get the complete formula
    async def get_final():
        final = await buffer.flush(is_final=True)
        return final

    final = asyncio.run(get_final())
    assert final is not None, "Should have content after flush"
    assert final.content == formula, "Formula should be complete"
    assert final.has_formula, "Should detect formula"

def test_display_formula_with_double_newline_not_split():
    """
    Test that display formulas with double newline before closing $$ are not split.

    This is the actual case from the logs where $$\n\n...$$ was being split.
    """
    import asyncio

    buffer = SentenceBuffer(max_chars=100, max_wait_seconds=0.5)

    # Formula with double newline before closing $$ (109 chars)
    formula = r'$$\n(x + y)^3 = \binom{3}{0} x^3 y^0 + \binom{3}{1} x^2 y^1 + \binom{3}{2} x^1 y^2 + \binom{3}{3} x^0\n y^3\n\n$$'

    # Add characters one by one
    segments = []
    for char in formula:
        result = buffer.add(char)
        if result:
            segments.append(result)

    # Should not have split the formula in the middle
    for seg in segments:
        if seg.startswith("$$") and not seg.endswith("$$"):
            assert False, f"Formula was split in the middle: {repr(seg[:50])}..."

    # Flush to get the complete formula
    async def get_final():
        final = await buffer.flush(is_final=True)
        return final

    final = asyncio.run(get_final())
    assert final is not None, "Should have content after flush"
    assert final.content == formula, "Formula should be complete"

if __name__ == "__main__":
    print("Running SentenceBuffer timing tests...")

    print("\n1. Testing first token resets timer...")
    test_first_token_resets_timer()
    print("   PASSED: First token correctly resets timer")

    print("\n2. Testing timeout still works after tokens...")
    test_still_respects_timeout_after_tokens()
    print("   PASSED: Timeout still works correctly")

    print("\n3. Testing normal sentence splitting...")
    test_normal_sentence_splitting_still_works()
    print("   PASSED: Normal splitting works")

    print("\n4. Testing empty buffer condition...")
    test_empty_buffer_condition()
    print("   PASSED: Empty buffer condition works")

    print("\n✅ All tests passed!")
