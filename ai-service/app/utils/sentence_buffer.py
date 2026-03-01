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
    r"""Sentence buffer with intelligent segmentation and dual output support.

    Features:
    1. Punctuation-based splitting (。！？.!?\n) - highest priority
    2. Character limit splitting - forced when exceeded
    3. Time limit splitting - forced when timeout
    4. Comma-based splitting (，；、,;) - lower priority
    5. LaTeX formula integrity - $$...$$, $...$, \(...\), \[...\] protected
    6. Formula detection - automatic detection in sentences
    """

    SENTENCE_END_PATTERN = re.compile(r'([。！？.!?\n])')
    COMMA_PATTERN = re.compile(r'([，；、,;])')

    CLOSED_FORMULA_PATTERN = re.compile(
        r'\$\$[\s\S]+?\$\$|'
        r'\$[^$\n]+?\$|'
        r'\\\([\s\S]+?\\\)|'
        r'\\\[[\s\S]+?\\\]'
    )
    ESCAPED_DOLLAR_PATTERN = re.compile(r'\\\$')

    DELIMITER_PATTERNS = {
        'paren_open': re.compile(r'\\\('),
        'paren_close': re.compile(r'\\\)'),
        'bracket_open': re.compile(r'\\\['),
        'bracket_close': re.compile(r'\\\]'),
    }

    DELIMITER_PAIRS = [
        ('$$', '$$', 2),
        (r'\(', r'\)', 2),
        (r'\[', r'\]', 2),
    ]

    SPACE_SEARCH_RANGE = 20

    def __init__(
        self,
        max_chars: int = 200,
        max_wait_seconds: float = 0.5,
        comma_split_threshold: int = 30,
    ):
        self.buffer = ""
        self._pending_merge = ""  # 缓存待合并的短 segment
        self.max_chars = max_chars
        self.max_wait_seconds = max_wait_seconds
        self.comma_split_threshold = comma_split_threshold
        self.last_flush_time = time.time()
        self.is_flushed = True

    def _count_all_latex_delimiters(self, text: str) -> dict:
        """Count all LaTeX formula delimiters in text.

        Returns:
            Dict with counts for each delimiter type.
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
            Tuple of (display_count, inline_count).
        """
        counts = self._count_all_latex_delimiters(text)
        return counts["display_dollar"], counts["inline_dollar"]

    def _is_in_latex_formula(self, text: str, position: int) -> bool:
        r"""Check if a position is within a LaTeX formula delimiter.

        Supports: $$...$$, $...$, \(...\), \[...\]
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

        Returns:
            The unclosed delimiter type ("$", "$$", r"\(", r"\[") or None.
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
        double_count = 0
        while remaining > 0:
            next_slice = text[original_idx:original_idx + 2]
            if next_slice == '$$':
                original_idx += 2
                remaining -= 2
                double_count += 1
            else:
                original_idx += 1
                remaining -= 1

        # Only return if not escaped and not part of $$
        if original_idx > 0 and text[original_idx - 1] != '\\':
            return original_idx

        return -1

    def _find_formula_boundary_split(self, text: str, delimiter: str) -> Tuple[int, str]:
        """Find a safe split position when dealing with unclosed formulas.

        Returns:
            Tuple of (position, reason).
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
            True if splitting would break a complete formula.
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
        """Find closing delimiter for an unclosed LaTeX formula.

        Returns:
            Tuple of (position, reason), or (-1, "no_closing") if not found.
            Position is index AFTER closing delimiter.
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

        if unclosed_type == '$':
            text_before = text[:pos].replace('$$', '')
            if text_before.count('$') % 2 == 1:
                return pos + 1, "latex_closing"
            return -1, "no_closing"

        if unclosed_type == '$$':
            if text[:pos].count('$$') % 2 == 1:
                return pos + 2, "latex_closing"
            return -1, "no_closing"

        opening = unclosed_type
        if text.rfind(opening, 0, pos) >= 0:
            return pos + 2, "latex_closing"

        return -1, "no_closing"

    def _find_safe_split_position(self, text: str) -> Tuple[int, SplitReason]:
        """Find a safe position to split text, respecting LaTeX formula boundaries.

        Returns:
            Tuple of (split_position, split_reason).
        """
        unclosed_delimiter = self._get_unclosed_delimiter_type(text)
        has_complete = self._has_complete_formula(text)

        extended_limit = self.max_chars
        if unclosed_delimiter and not has_complete and len(text) >= self.max_chars:
            extended_limit = self.max_chars * 2

        if '$$' in text[:20]:
            logger.debug(
                f"[SentenceBuffer] text_len={len(text)}, unclosed_delimiter={unclosed_delimiter}, "
                f"has_complete={has_complete}, extended_limit={extended_limit}"
            )

        if unclosed_delimiter:
            closing_pos, reason = self._find_latex_closing_delimiter(text, unclosed_delimiter)
            if closing_pos > 0:
                return closing_pos, reason

            if len(text) < extended_limit:
                return -1, "unclosed_formula"

        for match in reversed(list(self.SENTENCE_END_PATTERN.finditer(text))):
            pos = match.end()
            matched_char = match.group(1)

            if matched_char == ')' and pos > 1 and text[pos - 2] == '\\':
                continue

            if not self._is_in_latex_formula(text, pos - 1):
                if has_complete and pos < len(text) and self._would_split_complete_formula(text, pos):
                    continue
                return pos, "sentence_end"

        if not unclosed_delimiter and len(text) >= self.comma_split_threshold:
            for match in reversed(list(self.COMMA_PATTERN.finditer(text))):
                pos = match.end()
                if not self._is_in_latex_formula(text, pos - 1):
                    return pos, "comma"

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

        # Debug logging: check if not splitting due to unclosed formula
        if unclosed_delimiter:
            logger.debug(
                f"[SentenceBuffer] Not splitting due to unclosed formula: "
                f"delimiter={unclosed_delimiter!r}, buffer_len={len(text)}, "
                f"buffer_end={repr(text[-50:] if len(text) > 50 else text)}"
            )

        return -1, "no_split"

    @staticmethod
    def _has_complete_formula(text: str) -> bool:
        """Check if text contains at least one complete LaTeX formula.

        A complete formula has properly matched delimiters with alphanumeric content.
        """
        def has_letter_or_digit(content: str) -> bool:
            return bool(re.search(r'[a-zA-Z0-9]', content))

        for opening, closing, skip in SentenceBuffer.DELIMITER_PAIRS:
            pos = text.find(opening)
            if pos >= 0:
                closing_pos = text.find(closing, pos + skip)
                if closing_pos >= 0:
                    content = text[pos + skip:closing_pos]
                    if has_letter_or_digit(content):
                        return True

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
        """Check if text contains only punctuation and whitespace characters."""
        return bool(re.match(r'^[\s\n\r。！？.,;:!?\-—\*\•]+$', text))

    @staticmethod
    def _is_short_prefix_segment(text: str, min_length: int = 5) -> bool:
        """Check if segment is a short prefix like '1.', '2.', 'a.'.

        These segments should be merged with adjacent segments for TTS.
        """
        if len(text) >= min_length:
            return False
        return bool(re.match(r'^[0-9a-zA-Z]+[.：:：]$', text))

    @staticmethod
    def _is_meaningful_segment(text: str, min_length: int = 8) -> bool:
        """检查 segment 是否有意义，值得独立输出。

        有意义的 segment 满足以下任一条件：
        1. 长度 >= min_length 且包含至少一个中文字符或英文字母
        2. 包含完整的句子（有内容，不只是标点/空格）

        无意义的 segment：
        - 纯空白/换行
        - 只包含标点符号
        - 短前缀如 "1.", "2.", "a." 后面没有实际内容
        - 省略号片段如 ".-" 等

        Returns:
            True 如果 segment 有意义，可以独立输出
            False 如果 segment 太短或没有实际内容，应该继续积累
        """
        # 空字符串无意义
        if not text:
            return False

        # 如果长度 >= min_length 且包含至少一个中文字符或英文字母，有意义
        if len(text) >= min_length:
            has_letter = bool(re.search(r'[a-zA-Z\u4e00-\u9fff]', text))
            return has_letter

        # 检查是否是短前缀（如 "1.", "2."）后跟空白
        if re.match(r'^[0-9a-zA-Z]+[.：:：]\s*$', text):
            return False

        # 检查是否只是标点和空白（包括省略号片段）
        if re.match(r'^[\s\n\r。！？.,;:!?\-—\*\•\'\"]+$', text):
            return False

        # 其他短内容视为无意义
        return False

    @staticmethod
    def _merge_punctuation_segments(segments: list[str]) -> list[str]:
        """Merge punctuation-only and short prefix segments into adjacent segments.

        Rules:
        1. Punctuation-only segments merge into the previous segment
        2. Short prefix segments (like '1.', '2.') followed by '\n' merge into the next
        3. If first segment needs merging, merge into the next
        4. Multiple consecutive segments are handled together
        """
        if not segments:
            return []

        n = len(segments)
        is_punct = [SentenceBuffer._is_punctuation_only(s) for s in segments]
        is_prefix = [SentenceBuffer._is_short_prefix_segment(s) for s in segments]

        result = []
        i = 0

        while i < n:
            if is_punct[i]:
                if result:
                    result[-1] += segments[i]
                    i += 1
                else:
                    # Collect consecutive punctuation segments at start
                    j = i
                    while j < n and is_punct[j]:
                        j += 1
                    if j < n:
                        result.append(''.join(segments[i:j]) + segments[j])
                    else:
                        result.extend(segments[i:j])
                    i = j
            elif is_prefix[i]:
                if i + 1 < n:
                    prefix = segments[i]
                    j = i + 1
                    # Include following newline if present
                    if j < n and segments[j] == '\n':
                        prefix += '\n'
                        j += 1
                    # Include consecutive prefix segments
                    while j < n and is_prefix[j]:
                        prefix += segments[j]
                        j += 1
                    if j < n:
                        result.append(prefix + segments[j])
                    else:
                        result.append(prefix)
                    i = j + 1
                else:
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
        """
        if SentenceBuffer.CLOSED_FORMULA_PATTERN.search(text):
            return True

        text_clean = SentenceBuffer.ESCAPED_DOLLAR_PATTERN.sub('', text)

        if text_clean.count('$$') % 2 != 0:
            return True

        remaining = text_clean.replace('$$', '')
        if remaining.count('$') % 2 != 0:
            return True

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

        Returns:
            Segment string if ready to flush, None otherwise.
        """
        if not self.buffer:
            self.last_flush_time = time.time()

        self.buffer += token
        self.is_flushed = False

        split_pos, split_reason = self._find_safe_split_position(self.buffer)

        if split_pos > 0 and '$$' in self.buffer[:10]:
            logger.warning(
                f"[SentenceBuffer.add] SPLITTING formula! buffer_len={len(self.buffer)}, "
                f"split_pos={split_pos}, reason={split_reason}"
            )

        if split_pos > 0 and self._get_unclosed_delimiter_type(self.buffer):
            logger.debug(
                f"[SentenceBuffer] Splitting with unclosed delimiter: "
                f"buffer_len={len(self.buffer)}, split_pos={split_pos}, "
                f"reason={split_reason}, unclosed={self._get_unclosed_delimiter_type(self.buffer)!r}"
            )

        if split_pos > 0:
            segment = self.buffer[:split_pos]
            self.buffer = self.buffer[split_pos:]
            self.last_flush_time = time.time()
            self.is_flushed = True

            # 检查 segment 是否有意义
            if not self._is_meaningful_segment(segment):
                self._pending_merge += segment
                return None  # 不立即返回，等待更多内容

            # 如果有待合并的短 segment，先合并
            if self._pending_merge:
                segment = self._pending_merge + segment
                self._pending_merge = ""

            return segment

        elapsed = time.time() - self.last_flush_time

        # 如果有未闭合的公式，不应用超时逻辑，继续等待
        unclosed_delimiter = self._get_unclosed_delimiter_type(self.buffer)
        if self.buffer and elapsed > self.max_wait_seconds and not unclosed_delimiter:
            # 超时也需要合并待缓存的内容
            content = self._flush_buffer()
            if self._pending_merge:
                content = self._pending_merge + content
                self._pending_merge = ""
            return content

        # 添加调试日志：超时但未闭合公式的情况
        if self.buffer and elapsed > self.max_wait_seconds and unclosed_delimiter:
            logger.debug(
                f"[SentenceBuffer] Timeout bypassed: elapsed={elapsed:.2f}s, "
                f"unclosed_delimiter={unclosed_delimiter!r}, buffer_len={len(self.buffer)}"
            )

        return None

    async def flush(self, is_final: bool = False) -> Optional[SentenceSegment]:
        """Flush remaining buffer content.

        Returns:
            SentenceSegment if buffer has content, None otherwise.
        """
        if not self.buffer and not self._pending_merge:
            return None

        content = self._flush_buffer()

        # 合并 pending_merge 和 buffer（即使 pending_merge 只有 '\n' 或 '1.' 也要返回）
        if self._pending_merge:
            content = self._pending_merge + content
            self._pending_merge = ""

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

    Convenience function for external use.
    """
    return SentenceBuffer._has_latex_formula(text)
