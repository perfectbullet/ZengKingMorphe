#!/usr/bin/env python3
"""
过滤 info.log 中的指定 API 日志行
"""
import re

LOG_FILE = "info.log"

# 需要过滤的 API 路径模式
API_PATTERNS = [
    r"/api/knowledge-base/update",
    r"/api/knowledge-base/documents/create_with_segment",
    r"/api/knowledge-base/documents/tasks/",
]

def should_filter(line: str) -> bool:
    """判断行是否需要过滤"""
    for pattern in API_PATTERNS:
        if pattern in line:
            return True
    return False

def main():
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 过滤
    filtered_lines = [line for line in lines if not should_filter(line)]

    # 写回
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)

    print(f"过滤完成: {len(lines) - len(filtered_lines)} 行已删除")

if __name__ == "__main__":
    main()
