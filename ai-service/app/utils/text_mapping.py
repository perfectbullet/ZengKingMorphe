"""
英文回复到中文的映射

处理 RAGAnything LLM 可能返回的英文拒绝/错误消息，映射为中文。
"""
from app.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# 英文回复到中文的映射字典
# =============================================================================
ENGLISH_TO_CHINESE_MAPPING = {
    # 拒绝回答类
    "Sorry, I'm not able to provide an answer to that question.": "抱歉，我无法回答这个问题。",
    "I'm sorry, but I cannot provide an answer to that question.": "抱歉，我无法回答这个问题。",
    "Sorry, I cannot answer that question.": "抱歉，我无法回答这个问题。",
    "I'm unable to provide an answer to that question.": "我无法回答这个问题。",
    "I cannot answer that.": "我无法回答这个问题。",
    "I'm sorry, I can't help with that.": "抱歉，我无法协助处理这个问题。",

    # 信息不足类
    "I don't have enough information to answer that question.": "我没有足够的信息来回答这个问题。",
    "I don't know.": "我不知道。",
    "I'm not sure.": "我不确定。",

    # 无法协助类
    "I'm sorry, but I can't assist with that request.": "抱歉，我无法协助处理该请求。",
    "I cannot provide that information.": "我无法提供该信息。",
    "I'm not able to help with that.": "我无法协助处理。",
}


def map_english_to_chinese(text: str) -> str:
    """
    将英文回复映射为中文

    Args:
        text: 待处理的文本

    Returns:
        映射后的文本（如果匹配则返回中文，否则返回原文）
    """
    # 精确匹配
    if text in ENGLISH_TO_CHINESE_MAPPING:
        chinese = ENGLISH_TO_CHINESE_MAPPING[text]
        logger.info(f"[英文映射] 精确匹配 | 原文={repr(text[:50])} | 映射={repr(chinese)}")
        return chinese

    # 前缀匹配（处理部分句子）
    for eng, chi in ENGLISH_TO_CHINESE_MAPPING.items():
        if text.startswith(eng):
            mapped = chi + text[len(eng):]
            logger.info(f"[英文映射] 前缀匹配 | 原文={repr(text[:50])} | 映射={repr(mapped[:50])}")
            return mapped
        if text.endswith(eng):
            mapped = text[:-len(eng)] + chi
            logger.info(f"[英文映射] 后缀匹配 | 原文={repr(text[:50])} | 映射={repr(mapped[:50])}")
            return mapped

    return text
