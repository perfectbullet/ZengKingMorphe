"""通用工具函数"""
import re


# =============================================================================
# 语言主导识别（基于 Unicode 块）
# 设计目标：
# - **可维护**：所有语言判定走同一处实现，增删语言只动这里；
# - **可扩展**：新增语种只需追加一个字符类正则；
# - **不依赖关键字**：纯字符级判定，不维护任何"中文词/英文词"列表；
# - **下游通用**：被 sanitize_resolved_query / 消息构造 / 日历直出 多处复用。
# =============================================================================

#: 语言标签：占位符 "" 表示"无法判定"（空串 / 仅数字标点 / 多语种势均力敌）。
DominantLanguage = str  # 取值：'zh' | 'en' | ''

_RE_CJK = re.compile(r"[\u4e00-\u9fff]")
_RE_LATIN = re.compile(r"[A-Za-z]")


def detect_dominant_language(text: str) -> DominantLanguage:
    """
    判定文本的主导语言。

    返回值：
    - ``'zh'``：含中文字符且不含拉丁字母（或中文显著多于英文）
    - ``'en'``：含拉丁字母且不含中文（纯英文场景）
    - ``''``：无法判定（空串 / 中英势均力敌 / 仅数字符号）

    判定规则使用 Unicode 块计数，不依赖任何关键字 / 词典，
    支持任何包含中文 + 拉丁字母的混合输入。
    """
    s = text or ""
    if not s.strip():
        return ""
    cjk_count = len(_RE_CJK.findall(s))
    latin_count = len(_RE_LATIN.findall(s))
    if cjk_count == 0 and latin_count == 0:
        return ""
    if cjk_count > 0 and latin_count == 0:
        return "zh"
    if latin_count > 0 and cjk_count == 0:
        return "en"
    # 混合：以"是否含中文"作为强信号（中文字符密度通常远高于英文单词字母）。
    # 含中文即视为中文主导，避免少量英文术语（如 "API" / "RAG"）翻转判定。
    return "zh" if cjk_count >= 1 else "en"


def has_language_drift(source: str, candidate: str) -> bool:
    """
    判断 ``candidate`` 是否相对 ``source`` 发生主导语言漂移。

    用于：
    - 上下文消歧器（rewriter）输出与原问句的语言一致性校验；
    - 任何"中介查询 vs 原始查询"的语言守门场景。

    若任一侧无法判定主导语言，返回 ``False``（不阻断），保守优先。
    """
    src = detect_dominant_language(source)
    cand = detect_dominant_language(candidate)
    if not src or not cand:
        return False
    return src != cand


def sanitize_filename(filename: str, max_length: int = 50) -> str:
    """
    将文本转换为安全的文件名

    移除或替换文件系统中的非法字符：< > : " / \\ | ? * 和控制字符

    Args:
        filename: 原始文件名
        max_length: 最大文件名长度

    Returns:
        安全的文件名字符串
    """
    # 移除 Windows/Linux 文件系统非法字符
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # 移除首尾空格和点
    safe_name = safe_name.strip('. ')
    # 限制长度
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    # 如果处理后为空，使用默认名称
    return safe_name if safe_name else "query"