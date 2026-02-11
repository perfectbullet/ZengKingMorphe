#!/usr/bin/env python3
"""
LaTeX 公式提取器模块。

可复用的 LaTeX 公式提取组件，支持流式处理。

使用示例:
    from scripts.formula_extractor import LatexFormulaExtractor, FormulaMatch

    # 非流式模式
    extractor = LatexFormulaExtractor()
    formulas, replaced = extractor.extract_from_text(text)

    # 流式模式
    extractor = LatexFormulaExtractor()
    for chunk in stream:
        new_formulas = extractor.feed(chunk)
        for formula in new_formulas:
            print(f"发现公式: {formula.formula}")

    # 获取替换后的文本
    final_text = extractor.get_text_with_placeholders()
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FormulaMatch:
    """匹配到的公式信息"""
    formula: str           # 原始 LaTeX 公式
    start_pos: int         # 在文本中的起始位置
    end_pos: int           # 在文本中的结束位置
    formula_type: str      # 'inline' 或 'display'
    delimiter: str         # 使用的定界符类型，如 '$', '$$', '\(', '\['

    def __repr__(self) -> str:
        preview = self.formula[:30] + "..." if len(self.formula) > 30 else self.formula
        return f"FormulaMatch(type={self.formula_type}, formula='{preview}')"


class LatexFormulaExtractor:
    """
    从流式文本中提取 LaTeX 公式的类。

    支持:
    - $...$ (行内)
    - \(...\) (行内)
    - $$...$$ (行间)
    - \[...\] (行间)

    特点:
    - 状态机模式，支持流式处理
    - 返回公式位置信息用于替换
    - 支持嵌套检查

    状态转换:
        NORMAL --($/\\(/\\[/\\$)--> INLINE/DISPLAY
        INLINE/DISPLAY --(匹配结束符)--> NORMAL
    """

    # 定界符模式: (开始标记, (类型, 结束标记))
    PATTERNS = {
        '$$': ('display', '$$'),
        '$': ('inline', '$'),
        '\\[': ('display', '\\]'),
        '\\(': ('inline', '\\)'),
    }

    def __init__(self):
        self.buffer = ""                      # 累积的文本
        self.formulas: List[FormulaMatch] = []  # 已提取的公式列表
        self.state = "NORMAL"                 # 当前状态: NORMAL, INLINE, DISPLAY
        self.current_delimiter = None         # 当前定界符
        self.current_start = 0                # 当前公式在buffer中的起始位置
        self.current_formula = ""             # 当前公式内容（不包含定界符）

    def feed(self, chunk: str) -> List[FormulaMatch]:
        """
        喂入新的文本块，返回新发现的完整公式。

        Args:
            chunk: 新的文本块

        Returns:
            本轮提取到的完整公式列表
        """
        new_formulas = []
        i = 0

        while i < len(chunk):
            if self.state == "NORMAL":
                # 检查是否进入公式状态
                found = False
                # 优先检查更长的定界符（$$ 和 \[ 在 $ 和 \( 之前）
                for delim in sorted(self.PATTERNS.keys(), key=len, reverse=True):
                    ftype, end_delim = self.PATTERNS[delim]
                    if self._check_delimiter(chunk, i, delim):
                        self.state = f"{ftype.upper()}"
                        self.current_delimiter = delim
                        self.current_start = len(self.buffer)
                        self.current_formula = ""
                        i += len(delim)
                        found = True
                        break

                if not found:
                    self.buffer += chunk[i]
                    i += 1

            elif self.state in ("INLINE", "DISPLAY"):
                # 检查是否退出公式状态
                end_delim = self.PATTERNS[self.current_delimiter][1]

                if self._check_delimiter(chunk, i, end_delim):
                    # 公式结束
                    formula = FormulaMatch(
                        formula=self.current_formula,
                        start_pos=self.current_start,
                        end_pos=len(self.buffer),
                        formula_type=self.state.lower(),
                        delimiter=self.current_delimiter
                    )
                    new_formulas.append(formula)
                    self.formulas.append(formula)

                    # 重置状态
                    self.state = "NORMAL"
                    self.current_delimiter = None
                    i += len(end_delim)
                else:
                    self.current_formula += chunk[i]
                    self.buffer += chunk[i]
                    i += 1

        return new_formulas

    def _check_delimiter(self, text: str, pos: int, delim: str) -> bool:
        """
        检查指定位置是否有指定的定界符。

        Args:
            text: 文本
            pos: 位置
            delim: 定界符

        Returns:
            是否匹配
        """
        if pos + len(delim) > len(text):
            return False
        return text[pos:pos + len(delim)] == delim

    def get_text_with_placeholders(self, placeholder_format: str = "__FORMULA_{}__") -> str:
        """
        获取用占位符替换公式后的文本。

        Args:
            placeholder_format: 占位符格式，默认 "__FORMULA_{}__"

        Returns:
            替换后的文本
        """
        result = self.buffer
        # 从后往前替换，避免位置偏移
        for idx, formula in enumerate(reversed(self.formulas)):
            placeholder = placeholder_format.format(len(self.formulas) - 1 - idx)
            start = formula.start_pos
            end = formula.end_pos
            result = result[:start] + placeholder + result[end:]
        return result

    def replace_formulas(self, replacements: dict, placeholder_format: str = "__FORMULA_{}__") -> str:
        """
        用给定的替换文本替换公式。

        Args:
            replacements: {index: replacement_text} 字典
            placeholder_format: 占位符格式

        Returns:
            替换后的文本
        """
        result = self.get_text_with_placeholders(placeholder_format)

        for idx, replacement in replacements.items():
            placeholder = placeholder_format.format(idx)
            result = result.replace(placeholder, replacement)

        return result

    def reset(self):
        """重置提取器状态"""
        self.buffer = ""
        self.formulas = []
        self.state = "NORMAL"
        self.current_delimiter = None
        self.current_start = 0
        self.current_formula = ""

    def extract_from_text(self, text: str) -> Tuple[List[FormulaMatch], str]:
        """
        从完整文本中提取公式（非流式模式）。

        Args:
            text: 完整文本

        Returns:
            (公式列表, 用占位符替换后的文本)
        """
        self.reset()
        self.feed(text)
        replaced_text = self.get_text_with_placeholders()
        return self.formulas, replaced_text

    def get_formula_count(self) -> int:
        """获取已提取的公式数量"""
        return len(self.formulas)

    def get_formulas_by_type(self, formula_type: str) -> List[FormulaMatch]:
        """
        按类型获取公式。

        Args:
            formula_type: 'inline' 或 'display'

        Returns:
            指定类型的公式列表
        """
        return [f for f in self.formulas if f.formula_type == formula_type]

    @property
    def inline_formulas(self) -> List[FormulaMatch]:
        """获取所有行内公式"""
        return self.get_formulas_by_type('inline')

    @property
    def display_formulas(self) -> List[FormulaMatch]:
        """获取所有行间公式"""
        return self.get_formulas_by_type('display')
