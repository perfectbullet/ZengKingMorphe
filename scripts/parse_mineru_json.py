#!/usr/bin/env python3
"""
MinerU JSON 解析脚本

解析 MinerU 本地部署生成的 JSON 结果文件，支持递归解析嵌套的 JSON 字符串。
"""
import json
import argparse
from typing import Any


def recursively_parse_json(obj: Any) -> Any:
    """
    递归遍历并解析对象中所有的 JSON 字符串

    Args:
        obj: 任意 Python 对象

    Returns:
        完全解析后的 Python 对象
    """
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return recursively_parse_json(parsed)
        except (json.JSONDecodeError, TypeError):
            return obj
    elif isinstance(obj, dict):
        return {k: recursively_parse_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursively_parse_json(item) for item in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="MinerU JSON 解析工具")
    parser.add_argument("input", help="MinerU JSON 文件路径")
    parser.add_argument("-o", "--output", help="输出解析后的 JSON 文件路径（默认：输入文件名_parsed.json）")

    args = parser.parse_args()

    # 读取原始 JSON
    print(f"读取文件: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 递归解析所有嵌套的 JSON 字符串
    print("解析嵌套 JSON 字符串...")
    parsed_data = recursively_parse_json(raw_data)

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = args.input.replace(".json", "_parsed.json")

    # 保存解析后的 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)

    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
