"""
Tests for the LaTeX formula to voice conversion service.

Tests cover:
- Empty and delimiter-only inputs (streaming scenarios)
- Simple formula conversions
- Complex mathematical expressions
- No-formula text (should return as-is)
"""
from unittest.mock import MagicMock

import pytest

from app.services.revise_llm import (
    convert_formula_to_voice,
    get_revise_llm,
    _is_empty_or_delimiter_only,
    _has_any_formula_marker,
    _extract_latex_formulas,
    LATEX_FORMULA_PATTERN,
    convert_math_sentence_to_voice,
)

def _mock_llm_stream(text: str):
    """Helper to create a mock LLM with astream returning the given text."""
    chunks = []
    for char in text:
        chunk = MagicMock()
        chunk.content = char
        chunks.append(chunk)

    async def _gen():
        for chunk in chunks:
            yield chunk

    class AsyncIteratorMock:
        def __aiter__(self):
            return _gen()

    llm = MagicMock()
    llm.astream.return_value = AsyncIteratorMock()
    return llm

@pytest.fixture
def mock_llm():
    """Mock LLM instance for testing."""
    llm = MagicMock()
    llm.astream = MagicMock()
    return llm

@pytest.fixture(scope="module")
def real_llm():
    """Real LLM instance for integration tests (module scope for reuse).

    Note: We return the coroutine function and let tests await it.
    """
    return get_revise_llm

class TestFastPathChecks:
    """Test fast-path checks that avoid LLM calls."""

    def test_empty_string(self):
        """Empty string should return as-is."""
        assert _is_empty_or_delimiter_only("") is True

    def test_whitespace_only(self):
        """Whitespace-only string should return as-is."""
        assert _is_empty_or_delimiter_only("   ") is True
        assert _is_empty_or_delimiter_only("\n\t") is True

    def test_delimiter_only_dollar(self):
        """Single dollar delimiter should return as-is."""
        assert _is_empty_or_delimiter_only("$") is True
        assert _is_empty_or_delimiter_only(" $ ") is True

    def test_delimiter_only_double_dollar(self):
        """Double dollar delimiter should return as-is."""
        assert _is_empty_or_delimiter_only("$$") is True
        assert _is_empty_or_delimiter_only(" $$ ") is True

    def test_delimiter_only_paren(self):
        """Paren LaTeX delimiters should return as-is."""
        assert _is_empty_or_delimiter_only(r"\(") is True
        assert _is_empty_or_delimiter_only(r"\)") is True
        assert _is_empty_or_delimiter_only(r"\( \)") is True

    def test_delimiter_only_bracket(self):
        """Bracket LaTeX delimiters should return as-is."""
        assert _is_empty_or_delimiter_only(r"\[") is True
        assert _is_empty_or_delimiter_only(r"\]") is True
        assert _is_empty_or_delimiter_only(r"\[ \]") is True

    def test_mixed_empty_delimiters(self):
        """Mixed empty delimiters should return as-is."""
        assert _is_empty_or_delimiter_only(r"$\( \[$") is True

    def test_non_empty_text(self):
        """Non-empty text should not be caught by empty check."""
        assert _is_empty_or_delimiter_only("hello") is False
        assert _is_empty_or_delimiter_only("$x$") is False
        assert _is_empty_or_delimiter_only(r"\( x \)") is False

    def test_no_formula_markers(self):
        """Plain text without formula markers."""
        assert _has_any_formula_marker("hello world") is False
        assert _has_any_formula_marker("这是一个测试") is False

    def test_has_dollar_marker(self):
        """Dollar sign indicates formula."""
        assert _has_any_formula_marker("$x$") is True
        assert _has_any_formula_marker("formula is $x^2$") is True

    def test_has_backslash_marker(self):
        """Only formulas WITH delimiters are detected, not raw LaTeX commands."""
        # Raw LaTeX commands without delimiters should NOT be detected
        # (they are incomplete and handled by sentence buffer in streaming)
        assert _has_any_formula_marker(r"\frac{a}{b}") is False
        assert _has_any_formula_marker("use \\sqrt for root") is False
        # But formulas WITH delimiters should be detected
        assert _has_any_formula_marker(r"$\frac{a}{b}$") is True
        assert _has_any_formula_marker(r"formula is $\sqrt{x}$") is True

    def test_has_latex_commands(self):
        """Only complete formulas WITH delimiters are detected."""
        # Raw LaTeX commands without delimiters should NOT be detected
        assert _has_any_formula_marker(r"\sum_{i=1}^{n}") is False
        assert _has_any_formula_marker(r"\int_0^1") is False
        assert _has_any_formula_marker(r"\sqrt{x}") is False
        # But formulas WITH delimiters should be detected
        assert _has_any_formula_marker(r"$\sum_{i=1}^{n}$") is True
        assert _has_any_formula_marker(r"\(\int_0^1\)") is True
        assert _has_any_formula_marker(r"$$\sqrt{x}$$") is True

    def test_detects_simple_formulas(self):
        """Simple formulas without complex LaTeX commands should still be detected."""
        # This was the bug: simple formulas like (x+y)^3 were being skipped
        assert _has_any_formula_marker(r"$x+y$") is True
        assert _has_any_formula_marker(r"$x^2$") is True
        assert _has_any_formula_marker(r"$$ (x + y)^3 = x^3 + 3x^2y + 3xy^2 + y^3 $$") is True
        assert _has_any_formula_marker(r"\(a + b = c\)") is True

class TestGetReviseLLM:
    """Test LLM instance creation."""

    @pytest.mark.asyncio
    async def test_get_ollama_llm(self, monkeypatch):
        """Should create Ollama LLM instance."""
        monkeypatch.setenv("REVISE_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_REVISE_MODEL", "qwen2.5:7b")

        llm = await get_revise_llm()
        assert llm is not None
        assert llm.model == "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_get_siliconflow_llm(self, monkeypatch):
        """Should create SiliconFlow LLM instance.

        NOTE: This test is skipped since get_revise_llm() is currently
        hardcoded to use ollama provider for containerized environment.
        """
        pytest.skip("get_revise_llm() hardcoded to ollama for containerized env")

    @pytest.mark.asyncio
    async def test_siliconflow_missing_key(self, monkeypatch):
        """Should raise error when API key is missing.

        NOTE: This test is skipped since get_revise_llm() is currently
        hardcoded to use ollama provider for containerized environment.
        """
        pytest.skip("get_revise_llm() hardcoded to ollama for containerized env")

class TestConvertFormulaToVoice:
    """Test the main conversion function."""

    @pytest.mark.asyncio
    async def test_empty_delimiter_fast_path(self, monkeypatch):
        """Empty delimiter should use fast path, no LLM call."""
        # This test verifies the fast path works without mocking the LLM
        result = await convert_formula_to_voice("$$", None)
        assert result == "$$"

    @pytest.mark.asyncio
    async def test_whitespace_fast_path(self, monkeypatch):
        """Whitespace should use fast path."""
        result = await convert_formula_to_voice("   ", None)
        assert result == "   "

    @pytest.mark.asyncio
    async def test_paren_delimiter_fast_path(self, monkeypatch):
        """Empty paren delimiters should use fast path."""
        result = await convert_formula_to_voice(r"\(", None)
        assert result == r"\("

    @pytest.mark.asyncio
    async def test_no_formula_fast_path(self, monkeypatch):
        """Text without formula markers should use fast path."""
        result = await convert_formula_to_voice("这是普通文本", None)
        assert result == "这是普通文本"

    @pytest.mark.asyncio
    async def test_convert_single_inline_formula(self):
        """Convert single inline formula $x^2$."""
        mock_llm = _mock_llm_stream("x的平方")

        result = await convert_formula_to_voice("公式是 $x^2$", mock_llm)
        assert "x的平方" in result
        assert "$x^2$" not in result  # Formula should be replaced
        assert "公式是" in result  # Regular text should be preserved

    @pytest.mark.asyncio
    async def test_convert_fraction_formula(self):
        """Convert fraction formula \\frac{a}{b}."""
        mock_llm = _mock_llm_stream("a 除以 b")

        result = await convert_formula_to_voice("结果是 $\\frac{a}{b}$", mock_llm)
        assert "a 除以 b" in result
        assert r"$\frac{a}{b}$" not in result

    @pytest.mark.asyncio
    async def test_convert_multiple_formulas(self):
        """Convert multiple formulas in text."""
        responses = ["x的平方", "y的立方"]

        async def _make_response_stream(text):
            for c in text:
                chunk = MagicMock()
                chunk.content = c
                yield chunk

        class AsyncIteratorMock:
            def __init__(self, text):
                async def _gen():
                    for c in text:
                        chunk = MagicMock()
                        chunk.content = c
                        yield chunk
                self._gen = _gen

            def __aiter__(self):
                return self._gen()

        llm = MagicMock()
        llm.astream = MagicMock(side_effect=lambda *args, **kwargs: AsyncIteratorMock(responses.pop(0)))

        result = await convert_formula_to_voice("$x^2$ 加上 $$y^3$$", llm)
        # Both formulas should be replaced
        assert "$x^2$" not in result
        assert "$$y^3$$" not in result
        assert "x的平方" in result
        assert "y的立方" in result

    @pytest.mark.asyncio
    async def test_mixed_text_and_formulas(self):
        """Convert formulas in mixed text."""
        mock_llm = _mock_llm_stream("m 乘以 c 的平方")

        result = await convert_formula_to_voice("爱因斯坦方程是 $E = mc^2$", mock_llm)
        assert "爱因斯坦方程是" in result
        assert "$E = mc^2$" not in result
        assert "m 乘以 c 的平方" in result or "mc的平方" in result

    @pytest.mark.asyncio
    async def test_paren_delimiter_formula(self):
        """Convert formula with \\(...\\) delimiters."""
        mock_llm = _mock_llm_stream("x 平方")

        result = await convert_formula_to_voice(r"公式是 \(x^2\) 完整的", mock_llm)
        assert r"\(x^2\)" not in result
        assert "x 平方" in result

    @pytest.mark.asyncio
    async def test_bracket_delimiter_formula(self):
        """Convert formula with \\[...\\] delimiters."""
        mock_llm = _mock_llm_stream("x 的平方")

        result = await convert_formula_to_voice(r"显示公式 \[x^2\] 结束", mock_llm)
        assert r"\[x^2\]" not in result

    @pytest.mark.asyncio
    async def test_formula_with_spaces(self):
        """Convert formula with spaces around delimiters."""
        mock_llm = _mock_llm_stream("x 的平方")

        result = await convert_formula_to_voice("公式是 $ x^2 $ 完成", mock_llm)
        # Formula extraction includes spaces around delimiters
        # Replacement should work correctly
        assert "$ x^2 $" not in result

    @pytest.mark.asyncio
    async def test_formula_with_chinese_text(self):
        """Convert formula with Chinese text."""
        mock_llm = _mock_llm_stream("根号下 x 平方 加 y 平方")

        result = await convert_formula_to_voice(r"距离公式是 $\sqrt{x^2 + y^2}$", mock_llm)
        assert r"$\sqrt{x^2 + y^2}$" not in result
        assert "距离公式是" in result

class TestSentenceBufferIntegration:
    """Integration tests with SentenceBuffer for formula boundary protection."""

    def test_sentence_buffer_detects_paren_delimiters(self):
        """SentenceBuffer should detect \\( and \\) delimiters."""
        from app.utils.sentence_buffer import SentenceBuffer

        # Test with unclosed paren formula
        buffer = SentenceBuffer()
        text = r"这是一个公式 \(x^2"
        assert buffer._has_latex_formula(text) is True

        # Test with closed paren formula
        text = r"这是一个公式 \(x^2\) 完成了"
        assert buffer._has_latex_formula(text) is True

    def test_sentence_buffer_unclosed_paren(self):
        """SentenceBuffer should detect unclosed \\( delimiter."""
        from app.utils.sentence_buffer import SentenceBuffer

        buffer = SentenceBuffer()
        unclosed = buffer._get_unclosed_delimiter_type(r"公式 \(x^2 还没结束")
        assert unclosed == r"\("

    def test_sentence_buffer_unclosed_bracket(self):
        """SentenceBuffer should detect unclosed \\[ delimiter."""
        from app.utils.sentence_buffer import SentenceBuffer

        buffer = SentenceBuffer()
        unclosed = buffer._get_unclosed_delimiter_type(r"公式 \[x^2 还没结束")
        assert unclosed == r"\["

    def test_sentence_buffer_position_in_paren_formula(self):
        """SentenceBuffer should detect position inside \\(...\\) formula."""
        from app.utils.sentence_buffer import SentenceBuffer

        buffer = SentenceBuffer()
        text = r"公式 \(x^2\) 后面内容"

        # Position inside the formula
        assert buffer._is_in_latex_formula(text, 5) is True  # at x
        assert buffer._is_in_latex_formula(text, 6) is True  # at ^
        # Position outside the formula
        assert buffer._is_in_latex_formula(text, 10) is False  # after \)

    def test_sentence_buffer_all_delimiter_counts(self):
        """SentenceBuffer should count all delimiter types."""
        from app.utils.sentence_buffer import SentenceBuffer

        buffer = SentenceBuffer()
        counts = buffer._count_all_latex_delimiters(r"\(x\) $$y$$ $z$")

        assert counts["paren_open"] == 1
        assert counts["paren_close"] == 1
        assert counts["display_dollar"] == 2  # $$ counts as 2
        assert counts["inline_dollar"] == 2   # $z$ has 2 $
        assert counts["bracket_open"] == 0
        assert counts["bracket_close"] == 0

class TestFormulaExtraction:
    """Test LaTeX formula extraction functionality."""

    def test_extract_single_inline_formula(self):
        """Extract single inline formula $...$."""
        text = "爱因斯坦方程是 $E = mc^2$，非常著名"
        formulas = _extract_latex_formulas(text)
        assert len(formulas) == 1
        assert formulas[0][0] == "$E = mc^2$"
        # Positions are based on Unicode character count
        # Position 8 is '$' (after space), position 18 is after '$'
        assert formulas[0][1] == 8  # start position
        assert formulas[0][2] == 18  # end position
        # Verify formula content by reconstruction
        start, end = formulas[0][1], formulas[0][2]
        assert text[start:end] == "$E = mc^2$"

    def test_extract_multiple_formulas(self):
        """Extract multiple formulas from text."""
        text = "第一个公式 $x^2$ 和第二个公式 $$y^3$$ 以及第三个 $\\frac{a}{b}$"
        formulas = _extract_latex_formulas(text)
        assert len(formulas) == 3
        assert formulas[0][0] == "$x^2$"
        assert formulas[1][0] == "$$y^3$$"
        assert formulas[2][0] == r"$\frac{a}{b}$"

    def test_extract_paren_formula(self):
        """Extract formulas with \\(...\\) delimiters."""
        text = r"公式是 \(x^2 + y^2\) 完整的"
        formulas = _extract_latex_formulas(text)
        assert len(formulas) == 1
        assert r"\(x^2 + y^2\)" in formulas[0][0]

    def test_extract_bracket_formula(self):
        """Extract formulas with \\[...\\] delimiters."""
        text = r"显示公式是 \[ \int_0^1 x dx \] 完整的"
        formulas = _extract_latex_formulas(text)
        assert len(formulas) == 1
        assert r"\[" in formulas[0][0]

    def test_extract_multiline_formula(self):
        """Extract multi-line formulas (matrices)."""
        text = r"""
        矩阵是:
        $$
        \begin{pmatrix}
        a & b \\
        c & d
        \end{pmatrix}
        $$
        这是矩阵
        """
        formulas = _extract_latex_formulas(text)
        assert len(formulas) == 1
        formula = formulas[0][0]
        assert r"\begin{pmatrix}" in formula
        assert r"\end{pmatrix}" in formula

    def test_extract_no_formulas(self):
        """Return empty list when no formulas found."""
        text = "这是普通文本，没有任何公式"
        formulas = _extract_latex_formulas(text)
        assert len(formulas) == 0

    def test_extract_incomplete_formulas_ignored(self):
        """Incomplete formulas (unclosed delimiters) should not be extracted."""
        # Only closed formulas are extracted
        text = "完整公式 $x^2$ 和不完整公式 $y^3"
        formulas = _extract_latex_formulas(text)
        assert len(formulas) == 1
        assert formulas[0][0] == "$x^2$"

    def test_extract_nested_quotes(self):
        """Handle formulas with quotes inside."""
        text = r'公式是 "$E = mc^2$" 很有名'
        formulas = _extract_latex_formulas(text)
        assert len(formulas) == 1
        assert formulas[0][0] == "$E = mc^2$"

class TestConvertFormulaToVoiceIntegration:
    """Integration tests with real LLM."""

    @pytest.mark.asyncio
    async def test_convert_single_formula_real_llm(self, real_llm):
        """Convert single inline formula using real LLM."""
        llm = await real_llm()
        input_text = "公式是 $x^2$"
        print(f"\n=== 测试: 转换单个公式 ===")
        print(f"输入: {input_text}")

        result = await convert_formula_to_voice(input_text, llm)

        print(f"输出: {result}\n")

        assert "$x^2$" not in result

    @pytest.mark.asyncio
    async def test_convert_fraction_formula_real_llm(self, real_llm):
        """Convert fraction formula using real LLM."""
        llm = await real_llm()
        input_text = "结果是 $\\frac{a}{b}$"
        print(f"\n=== 测试: 转换分数公式 ===")
        print(f"输入: {input_text}")

        result = await convert_formula_to_voice(input_text, llm)

        print(f"输出: {result}\n")

        assert r"$\frac{a}{b}$" not in result

    @pytest.mark.asyncio
    async def test_convert_multiple_formulas_real_llm(self, real_llm):
        """Convert multiple formulas using real LLM."""
        llm = await real_llm()
        input_text = "$x^2$ 加上 $$y^3$$"
        print(f"\n=== 测试: 转换多个公式 ===")
        print(f"输入: {input_text}")

        result = await convert_formula_to_voice(input_text, llm)

        print(f"输出: {result}\n")

        assert "$x^2$" not in result
        assert "$$y^3$$" not in result

    @pytest.mark.asyncio
    async def test_mixed_text_and_formulas_real_llm(self, real_llm):
        """Convert Einstein equation using real LLM."""
        llm = await real_llm()
        input_text = "爱因斯坦方程是 $E = mc^2$"
        print(f"\n=== 测试: 爱因斯坦方程 ===")
        print(f"输入: {input_text}")

        result = await convert_formula_to_voice(input_text, llm)

        print(f"输出: {result}\n")

        assert "$E = mc^2$" not in result

    @pytest.mark.asyncio
    async def test_paren_delimiter_formula_real_llm(self, real_llm):
        """Convert formula with \\(...\\) delimiters using real LLM."""
        llm = await real_llm()
        input_text = r"公式是 \(x^2\) 完整的"
        print(f"\n=== 测试: 括号定界符公式 ===")
        print(f"输入: {input_text}")

        result = await convert_formula_to_voice(input_text, llm)

        print(f"输出: {result}\n")

        assert r"\(x^2\)" not in result

    @pytest.mark.asyncio
    async def test_sqrt_formula_real_llm(self, real_llm):
        """Convert square root formula using real LLM."""
        llm = await real_llm()
        input_text = r"距离公式是 $\sqrt{x^2 + y^2}$"
        print(f"\n=== 测试: 根号公式 ===")
        print(f"输入: {input_text}")

        result = await convert_formula_to_voice(input_text, llm)

        print(f"输出: {result}\n")
        assert r"$\sqrt{x^2 + y^2}$" not in result

class TestConvertMathSentenceToVoice:
    """Test the convert_math_sentence_to_voice function."""

    @pytest.mark.skip("LLM integration test - requires running Ollama")
    @pytest.mark.asyncio
    async def test_convert_math_sentence_with_symbols(self):
        """Should convert math symbols using LLM (requires running Ollama)."""
        pass

class TestFormulaConversionWithOperators:
    """Test formula conversion with operator translation (from actual logs)."""

    @pytest.mark.asyncio
    async def test_binomial_formula_with_operators_real_llm(self, real_llm):
        """Test binomial theorem formula conversion with operators using real LLM.

        Input formula (from logs): $(a + b)^n = \\sum_{k=0}^{n} \\binom{n}{k} a^{n-k} b^k$

        Expected behavior:
        - + should be converted to 加
        - = should be converted to 等于
        - Output should NOT contain backslashes
        - Output should NOT contain + operator
        """
        llm = await real_llm()
        input_text = "$(a + b)^n = \\sum_{k=0}^{n} \\binom{n}{k} a^{n-k} b^k$"

        print(f"\n=== 测试: 二项式公式运算符转换 ===")
        print(f"输入: {input_text}")

        result = await convert_formula_to_voice(input_text, llm)

        print(f"输出: {result}")
        print(f"包含反斜杠: {chr(92) in result}")
        print(f"包含+: {(chr(43) in result)}")
        print(f"包含加: {'加' in result}")
        print(f"包含等于: {'等于' in result}")

        # Critical assertions based on log issues
        assert "\\" not in result, "输出不应包含反斜杠"
        assert "+" not in result, "输出不应包含未转换的 + 运算符"

        # Expected conversions
        assert "加" in result, "应包含 '加' (转换为 +)"
        assert "等于" in result or "等" in result, "应包含 '等于' (转换为 =)"

        # Original formula should be replaced
        assert "$(a + b)^n" not in result, "原公式应被替换"
        assert "$$" not in result, "不应包含定界符"

    @pytest.mark.asyncio
    async def test_simple_operator_addition(self, real_llm):
        """Test simple addition operator conversion."""
        llm = await real_llm()
        input_text = "$a + b$"

        print(f"\n=== 测试: 简单加法转换 ===")
        print(f"输入: {input_text}")

        result = await convert_formula_to_voice(input_text, llm)

        print(f"输出: {result}")

        assert "+" not in result, "输出不应包含未转换的 + 运算符"
        assert "加" in result, "应包含 '加'"
        assert "$" not in result, "不应包含定界符"

    @pytest.mark.asyncio
    async def test_complex_formula_all_operators(self, real_llm):
        """Test formula with multiple operators: +, -, *, /, =."""
        llm = await real_llm()
        input_text = "$(x + y) * (x - y) = x^2 - y^2$"

        print(f"\n=== 测试: 多运算符公式 ===")
        print(f"输入: {input_text}")

        result = await convert_formula_to_voice(input_text, llm)

        print(f"输出: {result}")

        # No raw operators should remain
        assert "+" not in result, "输出不应包含未转换的 +"
        assert "-" not in result or result.count("-") <= 1, "输出不应包含未转换的 - (n-k允许)"
        assert "*" not in result, "输出不应包含未转换的 *"
        assert "=" not in result, "输出不应包含未转换的 ="

        # All operators should be converted to Chinese
        assert "加" in result, "应包含 '加'"
        assert "乘以" in result or "乘" in result, "应包含 '乘'"
        assert "等于" in result or "等" in result, "应包含 '等于'"

    @pytest.mark.asyncio
    async def test_summation_with_operator(self, real_llm):
        """Test summation formula with + operator."""
        llm = await real_llm()
        input_text = "$\\sum_{i=1}^{n} (a_i + b_i)$"

        print(f"\n=== 测试: 求和公式中的运算符 ===")
        print(f"输入: {input_text}")

        result = await convert_formula_to_voice(input_text, llm)

        print(f"输出: {result}")

        assert "+" not in result, "输出不应包含未转换的 +"
        assert "加" in result, "应包含 '加'"
        assert "求和" in result, "应包含 '求和'"
        assert "\\" not in result, "输出不应包含反斜杠"
