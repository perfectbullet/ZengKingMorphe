"""Sentence Buffer for streaming text with intelligent segmentation.

Supports:
- Sentence-level punctuation-based splitting (highest priority)
- Character limit forced splitting
- Time limit forced splitting
- Comma-based splitting (lower priority than sentence endings)
- LaTeX formula integrity preservation
"""
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class SentenceSegment:
    """A single sentence segment ready for output."""
    content: str
    has_formula: bool
    is_final: bool = False

    def __len__(self) -> int:
        return len(self.content)


SplitReason = Literal[
    "sentence_end",
    "comma",
    "char_limit_space",
    "char_limit_forced",
    "char_limit_before_formula",
    "no_split",
    "unclosed_formula",
    "formula_at_start",
    "waiting_for_complete_formula",
    "latex_closing",
]


class SentenceBuffer:
    r"""
    Sentence buffer with intelligent segmentation and dual output support.

    Features:
    1. Punctuation-based splitting (。！？.!?\n) - highest priority
    2. Character limit splitting - forced when exceeded
    3. Time limit splitting - forced when timeout
    4. Comma-based splitting (，；、,;) - lower priority
    5. LaTeX formula integrity - $$...$$, $...$, \(...\), \[...\] protected
    6. Formula detection - automatic detection in sentences
    """

    # Punctuation patterns
    SENTENCE_END_PATTERN = re.compile(r'([。！？.!?\n])')
    COMMA_PATTERN = re.compile(r'([，；、,;])')

    # LaTeX patterns for formula detection and integrity
    CLOSED_FORMULA_PATTERN = re.compile(
        r'\$\$[\s\S]+?\$\$|'
        r'\$[^$\n]+?\$|'
        r'\\\([\s\S]+?\\\)|'
        r'\\\[[\s\S]+?\\\]'
    )
    ESCAPED_DOLLAR_PATTERN = re.compile(r'\\\$')

    # LaTeX delimiter patterns for counting
    DELIMITER_PATTERNS = {
        'paren_open': re.compile(r'\\\('),
        'paren_close': re.compile(r'\\\)'),
        'bracket_open': re.compile(r'\\\['),
        'bracket_close': re.compile(r'\\\]'),
    }

    # LaTeX delimiter pairs (opening, closing, length of opening)
    DELIMITER_PAIRS = [
        ('$$', '$$', 2),
        (r'\(', r'\)', 2),
        (r'\[', r'\]', 2),
    ]

    # Range to search backwards for whitespace when splitting at char limit
    SPACE_SEARCH_RANGE = 20

    def __init__(
        self,
        max_chars: int = 200,
        max_wait_seconds: float = 0.5,
        comma_split_threshold: int = 30,
    ):
        """Initialize the sentence buffer.

        Args:
            max_chars: Maximum characters before forced split
            max_wait_seconds: Maximum seconds to wait before forced output
            comma_split_threshold: Minimum chars accumulated before comma split
        """
        self.buffer = ""
        self.max_chars = max_chars
        self.max_wait_seconds = max_wait_seconds
        self.comma_split_threshold = comma_split_threshold
        self.last_flush_time = time.time()
        self.is_flushed = True

    def _count_all_latex_delimiters(self, text: str) -> dict:
        r"""Count all LaTeX formula delimiters in text.

        Returns:
            Dict with counts for each delimiter type:
            - display_dollar: Number of $$ delimiters
            - inline_dollar: Number of $ delimiters (excluding $$)
            - paren_open: Number of \( delimiters
            - paren_close: Number of \) delimiters
            - bracket_open: Number of \[ delimiters
            - bracket_close: Number of \] delimiters
        """
        text_clean = self.ESCAPED_DOLLAR_PATTERN.sub('', text)

        display_dollar = text_clean.count('$$')
        remaining = text_clean.replace('$$', '')
        inline_dollar = remaining.count('$')

        paren_open = len(self.DELIMITER_PATTERNS['paren_open'].findall(text))
        paren_close = len(self.DELIMITER_PATTERNS['paren_close'].findall(text))
        bracket_open = len(self.DELIMITER_PATTERNS['bracket_open'].findall(text))
        bracket_close = len(self.DELIMITER_PATTERNS['bracket_close'].findall(text))

        return {
            "display_dollar": display_dollar,
            "inline_dollar": inline_dollar,
            "paren_open": paren_open,
            "paren_close": paren_close,
            "bracket_open": bracket_open,
            "bracket_close": bracket_close,
        }

    def _count_dollar_delimiters(self, text: str) -> Tuple[int, int]:
        """Count LaTeX dollar delimiters in text.

        Returns:
            Tuple of (display_count, inline_count) where:
            - display_count: Number of $$ delimiters (should be even for closed formulas)
            - inline_count: Number of $ delimiters excluding $$
        """
        counts = self._count_all_latex_delimiters(text)
        return counts["display_dollar"], counts["inline_dollar"]

    def _is_in_latex_formula(self, text: str, position: int) -> bool:
        r"""Check if a position is within a LaTeX formula delimiter.

        Supports: $$...$$, $...$, \(...\), \[...\]

        Args:
            text: The full text to check
            position: The character position to check

        Returns:
            True if position is within any LaTeX formula
        """
        counts = self._count_all_latex_delimiters(text[:position])
        return (
            counts["display_dollar"] % 2 == 1
            or counts["inline_dollar"] % 2 == 1
            or (counts["paren_open"] > counts["paren_close"])
            or (counts["bracket_open"] > counts["bracket_close"])
        )

    def _get_unclosed_delimiter_type(self, text: str) -> Optional[str]:
        r"""Check if text has unclosed LaTeX formula delimiters.

        Supports: $$...$$, $...$, \(...\), \[...\]

        Returns:
            The unclosed delimiter type ("$", "$$", r"\(", r"\[") or None if all closed.
        """
        counts = self._count_all_latex_delimiters(text)

        if counts["display_dollar"] % 2 != 0:
            return '$$'
        if counts["bracket_open"] > counts["bracket_close"]:
            return r'\['
        if counts["paren_open"] > counts["paren_close"]:
            return r'\('
        if counts["inline_dollar"] % 2 != 0:
            return '$'
        return None

    def _find_last_standalone_dollar(self, text: str) -> int:
        """Find the last standalone $ delimiter (not part of $$).

        Returns:
            Position of the last standalone $, or -1 if not found.
        """
        without_double = text.replace('$$', '')
        idx = without_double.rfind('$')

        if idx < 0:
            return -1

        # Map position back to original text, accounting for $$ pairs
        original_idx = 0
        remaining = idx
        while remaining > 0:
            if text[original_idx:original_idx + 2] == '$$':
                original_idx += 2
                remaining -= 2
            else:
                original_idx += 1
                remaining -= 1

        # Only return if not escaped
        if original_idx > 0 and text[original_idx - 1] != '\\':
            return original_idx

        return -1

    def _find_formula_boundary_split(self, text: str, delimiter: str) -> Tuple[int, str]:
        r"""Find a safe split position when dealing with unclosed formulas.

        Args:
            text: Text to find split position in
            delimiter: The unclosed delimiter type ("$", "$$", r"\(", r"\[")

        Returns:
            Tuple of (position, reason)
            position: -1 means no valid split, split at formula start means we should wait
        """
        if delimiter == '$':
            pos = self._find_last_standalone_dollar(text)
        else:
            pos = text.rfind(delimiter)

        if pos > 0:
            return pos, "char_limit_before_formula"
        if pos == 0:
            return -1, "formula_at_start"
        return -1, "no_split"

    def _find_space_near_limit(self, text: str, limit: int) -> Tuple[int, str]:
        """Find a whitespace position near a given limit.

        Returns:
            Tuple of (position, reason), or (-1, "no_split") if not found.
        """
        search_start = max(0, limit - self.SPACE_SEARCH_RANGE)

        for i in range(limit, search_start, -1):
            if i < len(text) and text[i] in ' \n\t':
                if not self._is_in_latex_formula(text, i):
                    return i, "char_limit_space"

        return -1, "no_split"

    def _would_split_complete_formula(self, text: str, split_pos: int) -> bool:
        """Check if splitting at a position would break a complete formula.

        Returns:
            True if splitting would break a complete formula
        """
        for opening, closing, _ in self.DELIMITER_PAIRS:
            last_opening = text.rfind(opening, 0, split_pos)
            if last_opening >= 0:
                next_closing = text.find(closing, split_pos)
                if next_closing > split_pos:
                    return True

        # Handle inline $ (not $$)
        last_dollar = text.rfind('$', 0, split_pos)
        if last_dollar >= 0 and not text.startswith('$$', last_dollar - 1):
            next_dollar = text.find('$', split_pos)
            if next_dollar > split_pos and not text.startswith('$$', next_dollar - 1):
                return True

        return False

    def _find_latex_closing_delimiter(self, text: str, unclosed_type: str) -> Tuple[int, str]:
        r"""Find closing delimiter for an unclosed LaTeX formula.

        Args:
            text: Text to search in
            unclosed_type: The unclosed delimiter type ("$", "$$", r"\(", r"\[")

        Returns:
            Tuple of (position, reason), or (-1, "no_closing") if not found.
            Position is index AFTER closing delimiter (for splitting).
        """
        closing_map = {
            '$': ('$', 1),
            '$$': ('$$', 2),
            r'\(': (r'\)', 2),
            r'\[': (r'\]', 2),
        }

        closing_info = closing_map.get(unclosed_type)
        if not closing_info:
            return -1, "no_closing"

        closing, skip = closing_info
        pos = text.rfind(closing)
        if pos < 0:
            return -1, "no_closing"

        # For inline $, verify it's the closing dollar
        if unclosed_type == '$':
            text_before = text[:pos].replace('$$', '')
            if text_before.count('$') % 2 == 1:
                return pos + 1, "latex_closing"
            return -1, "no_closing"

        # For $$, verify it's the closing pair
        if unclosed_type == '$$':
            if text[:pos].count('$$') % 2 == 1:
                return pos + 2, "latex_closing"
            return -1, "no_closing"

        # For \( and \[, verify matching opening exists
        opening = unclosed_type
        if text.rfind(opening, 0, pos) >= 0:
            return pos + 2, "latex_closing"

        return -1, "no_closing"

    def _find_safe_split_position(self, text: str) -> Tuple[int, SplitReason]:
        """Find a safe position to split text, respecting LaTeX formula boundaries.

        Returns:
            Tuple of (split_position, split_reason)
            split_position: -1 if no split should occur
        """
        unclosed_delimiter = self._get_unclosed_delimiter_type(text)
        has_complete = self._has_complete_formula(text)

        # Extend limit when we have an unclosed formula without a complete one
        extended_limit = self.max_chars
        if unclosed_delimiter and not has_complete and len(text) >= self.max_chars:
            extended_limit = self.max_chars * 2

        if '$$' in text[:20]:
            logger.debug(
                f"[SentenceBuffer] text_len={len(text)}, unclosed_delimiter={unclosed_delimiter}, "
                f"has_complete={has_complete}, extended_limit={extended_limit}, "
                f"text_start={repr(text[:30])}, text_end={repr(text[-30:])}"
            )

        # Priority 1: Find closing delimiter for unclosed formulas
        if unclosed_delimiter:
            closing_pos, reason = self._find_latex_closing_delimiter(text, unclosed_delimiter)
            if closing_pos > 0:
                return closing_pos, reason

            if len(text) < extended_limit:
                return -1, "unclosed_formula"

        # Priority 2: Sentence ending punctuation
        for match in reversed(list(self.SENTENCE_END_PATTERN.finditer(text))):
            pos = match.end()
            matched_char = match.group(1)

            # Skip escaped closing parenthesis
            if matched_char == ')' and pos > 1 and text[pos - 2] == '\\':
                continue

            if not self._is_in_latex_formula(text, pos - 1):
                if has_complete and pos < len(text) and self._would_split_complete_formula(text, pos):
                    continue
                return pos, "sentence_end"

        # Priority 3: Comma splitting
        if not unclosed_delimiter and len(text) >= self.comma_split_threshold:
            for match in reversed(list(self.COMMA_PATTERN.finditer(text))):
                pos = match.end()
                if not self._is_in_latex_formula(text, pos - 1):
                    return pos, "comma"

        # Priority 4: Forced split at (extended) max_chars
        if len(text) >= extended_limit:
            if has_complete and unclosed_delimiter is None:
                potential_split = min(extended_limit, len(text))
                if self._would_split_complete_formula(text, potential_split):
                    return -1, "waiting_for_complete_formula"

            if unclosed_delimiter:
                pos, reason = self._find_formula_boundary_split(text, unclosed_delimiter)
                if pos > 0:
                    return pos, reason
                if reason == "formula_at_start":
                    return -1, "unclosed_formula"

            pos, reason = self._find_space_near_limit(text, extended_limit)
            if pos > 0:
                return pos, reason

            return min(extended_limit, len(text)), "char_limit_forced"

        return -1, "no_split"

    @staticmethod
    def _has_complete_formula(text: str) -> bool:
        """Check if text contains at least one complete LaTeX formula.

        A complete formula is a properly matched pair of delimiters WITH content.
        The formula must contain at least one letter or digit between delimiters.

        Returns:
            True if text contains at least one complete formula with content
        """
        def has_letter_or_digit(content: str) -> bool:
            return bool(re.search(r'[a-zA-Z0-9]', content))

        # Check display formulas $$...$$
        for opening, closing, skip in SentenceBuffer.DELIMITER_PAIRS:
            pos = text.find(opening)
            if pos >= 0:
                closing_pos = text.find(closing, pos + skip)
                if closing_pos >= 0:
                    content = text[pos + skip:closing_pos]
                    if has_letter_or_digit(content):
                        return True

        # Check inline formulas $...$ (excluding $$)
        if '$$' not in text:
            pos = text.find('$')
            if pos >= 0:
                next_pos = text.find('$', pos + 1)
                if next_pos >= 0:
                    content = text[pos + 1:next_pos]
                    if has_letter_or_digit(content):
                        return True

        return False

    @staticmethod
    def _is_punctuation_only(text: str) -> bool:
        """检查文本是否只包含标点符号和空白字符.

        Args:
            text: Text to check

        Returns:
            True if text contains only punctuation and whitespace characters
        """
        # 只包含空白字符和常见标点符号
        return bool(re.match(r'^[\s\n\r。！？.,;:!?\-—\*\•]+$', text))

    @staticmethod
    def _is_short_prefix_segment(text: str, min_length: int = 5) -> bool:
        """检查是否是过短的前缀 segment（如 '1.', '2.', 'a.'）.

        这类 segment 对语音合成不友好，需要合并到相邻的 segment 中。

        Args:
            text: Text to check
            min_length: Minimum length threshold (default: 5)

        Returns:
            True if segment is short and consists of digits/letters + punctuation
        """
        # 长度小于阈值且只包含数字/字母+单个标点
        # 模式：以数字或字母开头，以标点结尾，中间没有其他内容
        if len(text) >= min_length:
            return False
        # 检查模式：数字或字母 + 单个标点
        return bool(re.match(r'^[0-9a-zA-Z]+[.：:：]$', text))

    @staticmethod
    def _merge_punctuation_segments(segments: list[str]) -> list[str]:
        """合并单独的标点符号 segment 和短前缀 segment 到相邻的 segment 中.

        规则:
        1. 标点-only segment 合并到前一个 segment
        2. 短前缀 segment (如 '1.', '2.') 后面跟着 '\n'，一起合并到后一个
        3. 如果是第一个且是标点/短前缀，合并到后一个 segment
        4. 连续多个需要合并的 segment 统一处理

        Args:
            segments: List of segments to process

        Returns:
            List of segments with punctuation-only and short prefix segments merged
        """
        if not segments:
            return []

        # 标记哪些 segment 需要被合并
        # 标点-only: 直接合并到前一个
        # 短前缀: 优先合并到后一个，因为它是前缀
        to_merge_back = [SentenceBuffer._is_punctuation_only(s) for s in segments]
        to_merge_forward = [SentenceBuffer._is_short_prefix_segment(s) for s in segments]

        result = []
        i = 0
        n = len(segments)

        while i < n:
            if to_merge_back[i]:
                # 标点-only segment，合并到前一个
                if result:
                    result[-1] += segments[i]
                else:
                    # 没有前一个，收集连续的标点 segment 并找下一个合并
                    collected = [segments[i]]
                    j = i + 1
                    while j < n and to_merge_back[j]:
                        collected.append(segments[j])
                        j += 1
                    if j < n:
                        result.append(''.join(collected) + segments[j])
                        i = j + 1
                        continue
                    else:
                        result.extend(collected)
                        i = n
                        continue
                i += 1
            elif to_merge_forward[i]:
                # 短前缀 segment，需要合并到后一个
                if i + 1 < n:
                    # 检查下一个 segment 是否是 '\n'
                    prefix = segments[i]
                    j = i + 1
                    # 如果后面跟着 '\n'，把 '\n' 也合并进来
                    if j < n and segments[j] == '\n':
                        prefix += '\n'
                        j += 1
                    # 继续检查是否还有连续短前缀
                    while j < n and to_merge_forward[j]:
                        prefix += segments[j]
                        j += 1
                    if j < n:
                        result.append(prefix + segments[j])
                        i = j + 1
                    else:
                        result.append(prefix)
                        i = n
                else:
                    # 没有后一个，保留
                    result.append(segments[i])
                    i += 1
            else:
                result.append(segments[i])
                i += 1

        return result

    @staticmethod
    def _has_latex_formula(text: str) -> bool:
        r"""Detect if text contains LaTeX formulas (closed or unclosed).

        Supports: $...$, $$...$$, \(...\), \[...\]

        Returns:
            True if text contains any LaTeX formula (closed or unclosed)
        """
        if SentenceBuffer.CLOSED_FORMULA_PATTERN.search(text):
            return True

        text_clean = SentenceBuffer.ESCAPED_DOLLAR_PATTERN.sub('', text)

        # Check for unclosed display formulas
        if text_clean.count('$$') % 2 != 0:
            return True

        # Check for unclosed inline formulas
        remaining = text_clean.replace('$$', '')
        if remaining.count('$') % 2 != 0:
            return True

        # Check for unclosed paren/bracket delimiters
        for opening, closing in [(r'\(', r'\)'), (r'\[', r'\]')]:
            if text.count(opening) != text.count(closing):
                return True

        return False

    def _flush_buffer(self) -> str:
        """Clear and return the current buffer content."""
        content = self.buffer
        self.buffer = ""
        self.last_flush_time = time.time()
        self.is_flushed = True
        return content

    def add(self, token: str) -> Optional[str]:
        """Add a token to the buffer and return a segment if ready to flush.

        Args:
            token: Token string to add

        Returns:
            Segment string if ready to flush, None otherwise
        """
        if not self.buffer:
            self.last_flush_time = time.time()

        self.buffer += token
        self.is_flushed = False

        split_pos, split_reason = self._find_safe_split_position(self.buffer)

        if split_pos > 0 and '$$' in self.buffer[:10]:
            logger.warning(
                f"[SentenceBuffer.add] SPLITTING formula! buffer_len={len(self.buffer)}, "
                f"split_pos={split_pos}, reason={split_reason}, "
                f"segment_starts_with_$$={self.buffer[:split_pos].startswith('$$')}, "
                f"segment_ends_with_$$={self.buffer[:split_pos].endswith('$$')}"
            )

        if split_pos > 0:
            segment = self.buffer[:split_pos]
            self.buffer = self.buffer[split_pos:]
            self.last_flush_time = time.time()
            self.is_flushed = True
            return segment

        elapsed = time.time() - self.last_flush_time
        if self.buffer and elapsed > self.max_wait_seconds:
            return self._flush_buffer()

        return None

    async def flush(self, is_final: bool = False) -> Optional[SentenceSegment]:
        """Flush remaining buffer content.

        Args:
            is_final: True if this is the final flush

        Returns:
            SentenceSegment if buffer has content, None otherwise
        """
        if not self.buffer:
            return None

        content = self._flush_buffer()
        has_formula = self._has_latex_formula(content)

        return SentenceSegment(
            content=content,
            has_formula=has_formula,
            is_final=is_final
        )

    def has_pending_content(self) -> bool:
        """Check if buffer has pending content."""
        return bool(self.buffer)

    def get_buffer_length(self) -> int:
        """Get current buffer length."""
        return len(self.buffer)


def has_latex_formula(text: str) -> bool:
    """Detect if text contains LaTeX formulas (closed or unclosed).

    This is a convenience function for external use.

    Args:
        text: Text to check

    Returns:
        True if text contains $...$, $$...$$, or unclosed $/$$
    """
    return SentenceBuffer._has_latex_formula(text)
