#!/usr/bin/env python3
"""
数学公式转换器 - LaTeX 公式转口语描述

使用大语言模型将 LaTeX 数学公式转换为适合语音合成的自然语言描述。

用法:
    from scripts.math_formula_converter import MathFormulaConverter

    converter = MathFormulaConverter(
        api_key="your_api_key",
        base_url="https://api.siliconflow.cn/v1"
    )

    result = converter.convert_inline(r"$\frac{a}{b}$")
    print(result)  # 输出: a 除以 b
"""


import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class ConversionResult:
    """公式转换结果"""
    original: str        # 原始 LaTeX
    spoken: str          # 口语描述
    type: str            # "inline" 或 "interline"
    index: int           # 公式序号


class MathFormulaConverter:
    """
    LaTeX 数学公式转口语描述转换器

    使用大语言模型将 LaTeX 数学公式转换为适合语音合成的自然语言描述。
    """

    # 系统提示词 - 单个公式转换
    SYSTEM_PROMPT = r'''
你是一个专业的数学讲解助手，专门将LaTeX数学公式转换为流畅、准确、符合中文口语的自然语言描述，用于语音合成(TTS)。

转译规则：
- 完全忠于数学原意，不可更改或简化公式。
- 口语化表达：使用"除以"、"开方"、"求和"等口语词；用"的"连接修饰关系（如"x的平方"）。
- 明确结构：清晰说明"分子"、"分母"、"下标"、"上标"、"积分限"、"求和范围"。
- 流畅断句：输出为完整的短句，适合TTS平稳朗读。

重要：直接输出最终答案，不要输出任何推理过程、思考步骤或中间分析。

示例对照（LaTeX -> 描述）：
- $E = mc^2$ -> E 等于 m 乘以 c 的平方
- $\frac{a}{b}$ -> a 除以 b
- $\sqrt{x^2 + y^2}$ -> 根号下 x 平方 加 y 平方
- $\sum_{i=1}^{n} i^2$ -> 对 i 从 1 到 n 求和，i 的平方
- $\int_{0}^{1} f(x) dx$ -> 从 0 到 1，对函数 f(x) 积分
- $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ -> 一个矩阵，第一行为 a 和 b，第二行为 c 和 d

请严格遵循以上风格，直接输出转换结果，不要有任何其他说明。
'''

    def __init__(
        self,
        api_key: Optional[str] = "no-key",  # Ollama 不需要真实 API key，任意值即可
        base_url: str = "http://192.168.8.233:11434",
        model: str = "qwen3:14b",
        timeout: int = 30,  # API 请求超时时间（秒）
        max_formula_length: int = 500  # 最大公式长度限制
    ):
        """
        初始化公式转换器

        Args:
            api_key: API 密钥（Ollama 可以使用任意值，默认 "ollama"）
            base_url: API 基础 URL（Ollama 默认 http://192.168.8.233:11434/v1）
            model: 使用的模型名称
            timeout: API 请求超时时间（秒）
            max_formula_length: 最大公式长度限制，超过则跳过转换
        """
        print(f"api_key is {api_key}")
        print(f"base_url is {base_url}")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        self.model = model
        self.max_formula_length = max_formula_length

    def _call_api(self, latex: str) -> str:
        """
        调用 API 进行公式转换

        Args:
            latex: LaTeX 公式

        Returns:
            转换后的口语描述
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": latex}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # 超时或其他错误，返回原始 LaTeX
            print(f"  [警告] 公式转换失败 ({type(e).__name__}): {latex[:40]}...")
            return latex

    def convert_inline(self, latex: str) -> str:
        """
        转换单个行内公式

        Args:
            latex: LaTeX 公式，如 "$x^2$" 或 "x^2"

        Returns:
            口语描述
        """
        # 清理 $ 符号
        clean_latex = latex.strip()
        if clean_latex.startswith("$") and clean_latex.endswith("$"):
            clean_latex = clean_latex[1:-1]

        # 检查公式长度
        if len(clean_latex) > self.max_formula_length:
            print(f"  [跳过] 公式过长 ({len(clean_latex)}字符): {clean_latex[:40]}...")
            return latex

        # 重新包装以确保格式正确
        formula = f"${clean_latex}$"
        return self._call_api(formula)

    def convert_interline(self, latex: str) -> str:
        """
        转换单个行间公式

        Args:
            latex: LaTeX 公式，如 "$$\int_0^1 f(x) dx$$" 或 "\int_0^1 f(x) dx"

        Returns:
            口语描述
        """
        # 清理 $$ 符号
        clean_latex = latex.strip()
        if clean_latex.startswith("$$") and clean_latex.endswith("$$"):
            clean_latex = clean_latex[2:-2]
        elif clean_latex.startswith("$") and clean_latex.endswith("$"):
            clean_latex = clean_latex[1:-1]

        # 检查公式长度
        if len(clean_latex) > self.max_formula_length:
            print(f"  [跳过] 公式过长 ({len(clean_latex)}字符): {clean_latex[:40]}...")
            return latex

        # 重新包装
        formula = f"$${clean_latex}$$"
        return self._call_api(formula)

    def convert_batch(self, formulas: List[str], formula_type: str = "inline") -> List[str]:
        """
        批量转换公式

        Args:
            formulas: LaTeX 公式列表
            formula_type: 公式类型，"inline" 或 "interline"

        Returns:
            口语描述列表
        """
        results = []
        total = len(formulas)

        for i, formula in enumerate(formulas, 1):
            try:
                if formula_type == "inline":
                    result = self.convert_inline(formula)
                else:
                    result = self.convert_interline(formula)
                results.append(result)
                print(f"进度: {i}/{total} - {formula[:30]}... -> {result[:50]}...")
            except Exception as e:
                print(f"错误: {i}/{total} - {formula[:30]}... -> {e}")
                results.append(f"[转换失败: {e}]")

        return results

    

    def extract_and_convert(
        self,
        text: str,
        convert_inline: bool = True,
        convert_interline: bool = True
    ) -> Tuple[List[ConversionResult], str]:
        """
        从文本中提取公式并转换

        Args:
            text: 包含 LaTeX 公式的文本
            convert_inline: 是否转换行内公式
            convert_interline: 是否转换行间公式

        Returns:
            (转换结果列表, 替换后的文本)
        """
        results = []
        replaced_text = text
        result_index = 0

        # 先处理行间公式（因为它们更长）
        if convert_interline:
            interline_pattern = r'\$\$([^$]+)\$\$'
            matches = list(re.finditer(interline_pattern, text, flags=re.DOTALL))

            # 从后往前替换，避免位置偏移
            for match in reversed(matches):
                latex = match.group(0)
                spoken = self.convert_interline(latex)
                result_index += 1

                result = ConversionResult(
                    original=latex,
                    spoken=spoken,
                    type="interline",
                    index=result_index
                )
                results.insert(0, result)  # 插入到开头保持顺序

        # 再处理行内公式
        if convert_inline:
            inline_pattern = r'\$([^$\n]+)\$'
            matches = list(re.finditer(inline_pattern, text))

            for match in reversed(matches):
                latex = match.group(0)
                # 排除已经处理过的行间公式
                if re.match(r'\$\$.*?\$\$', latex):
                    continue

                spoken = self.convert_inline(latex)
                result_index += 1

                result = ConversionResult(
                    original=latex,
                    spoken=spoken,
                    type="inline",
                    index=result_index
                )
                results.insert(0, result)

        return results, replaced_text


if __name__ == "__main__":
    # 测试代码
    
    converter = MathFormulaConverter()

    # 测试行内公式
    print("=== 行内公式测试 ===")
    test_inline = [
        r"{x∈R|x²-2=0}",
        r"$\frac{a}{b}$",
        r"$\sqrt{x^2 + y^2}$",
    ]

    for formula in test_inline:
        result = converter.convert_inline(formula)
        print(f"{formula} -> {result}")

    # 测试行间公式
    print("\n=== 行间公式测试 ===")
    test_interline = [
        r"$$\int_{0}^{1} f(x) dx$$",
        r"$$\sum_{i=1}^{n} i^2$$",
    ]

    for formula in test_interline:
        result = converter.convert_interline(formula)
        print(f"{formula}")
        print(f"-> {result}\n")
