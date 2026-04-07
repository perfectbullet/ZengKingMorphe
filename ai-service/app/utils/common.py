"""通用工具函数"""
import re


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