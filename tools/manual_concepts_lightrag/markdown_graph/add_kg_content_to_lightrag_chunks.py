#!/usr/bin/env python3
"""只为 LightRAG chunks JSONL 增加 kg_content 字段。

处理规则：
- 保留每条 JSON 记录的全部原字段和值；
- 从 content 中第一个“【正文】”标记之后提取正文；
- 仅删除标记之后紧邻的换行符，不对正文做 strip、替换或规范化；
- 默认写入新文件，不覆盖原文件；
- 如果记录已经存在 kg_content，默认报错，避免误处理后续教材。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


BODY_MARKER = "【正文】"


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_name(
            f"{input_path.stem}.with_kg_content{input_path.suffix}"
        )
    return input_path.with_name(f"{input_path.name}.with_kg_content")


def extract_kg_content(content: str, *, line_no: int, chunk_id: str) -> str:
    if BODY_MARKER not in content:
        raise ValueError(
            f"第 {line_no} 行缺少 {BODY_MARKER!r}: chunk_id={chunk_id}"
        )

    _, body = content.split(BODY_MARKER, 1)
    # 与后续教材的 kg_content 形式保持一致：去掉标记后紧邻的 CR/LF，
    # 不删除正文末尾空白，也不做其他内容修改。
    return body.lstrip("\r\n")


def process_file(input_path: Path, output_path: Path, *, overwrite: bool) -> dict:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if input_path == output_path:
        raise ValueError("输出路径不能与输入路径相同；本脚本不直接覆盖原文件")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"输出文件已存在: {output_path}；确认后使用 --overwrite"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    record_count = 0
    added_count = 0

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with input_path.open("r", encoding="utf-8") as src, temp_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as dst:
            for line_no, raw_line in enumerate(src, start=1):
                if not raw_line.strip():
                    # 空行原样保留。
                    dst.write(raw_line)
                    continue

                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise ValueError(f"第 {line_no} 行不是 JSON 对象")

                chunk_id = str(record.get("chunk_id") or "")
                if "kg_content" in record:
                    raise ValueError(
                        f"第 {line_no} 行已存在 kg_content，停止处理: "
                        f"chunk_id={chunk_id}"
                    )

                content = record.get("content")
                if not isinstance(content, str) or not content:
                    raise ValueError(
                        f"第 {line_no} 行 content 不是非空字符串: "
                        f"chunk_id={chunk_id}"
                    )

                record["kg_content"] = extract_kg_content(
                    content,
                    line_no=line_no,
                    chunk_id=chunk_id,
                )

                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_count += 1
                added_count += 1

        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    return {
        "input": str(input_path),
        "output": str(output_path),
        "record_count": record_count,
        "added_kg_content_count": added_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只为预切分 LightRAG JSONL 增加 kg_content 字段"
    )
    parser.add_argument("input", type=Path, help="输入 JSONL")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出 JSONL；默认在文件名后增加 .with_kg_content",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的输出文件；仍不会覆盖输入文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or default_output_path(args.input)
    result = process_file(args.input, output, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
