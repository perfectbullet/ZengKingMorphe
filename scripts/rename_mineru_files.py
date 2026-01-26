#!/usr/bin/env python3
"""
MinerU文件重命名脚本

将MinerU导出的文件重命名为简洁格式：
MinerU_01珐琅工艺__20260124090252.json -> 01珐琅工艺.json
MinerU_markdown_01珐琅工艺_20260124170709.md -> 01珐琅工艺.md

使用方式:
    python scripts/rename_mineru_files.py
"""
import os
import re
from pathlib import Path
from typing import Tuple, Optional


def parse_mineru_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    解析MinerU导出的文件名

    Args:
        filename: 原文件名

    Returns:
        (新文件名, 原始中文名, 扩展名) 或 None
    """
    # 匹配 MinerU_前缀的JSON文件
    # MinerU_01珐琅工艺__20260124090252.json
    json_pattern = r'MinerU_(.+?)__(\d{8}\d{6})\.json$'
    match = re.match(json_pattern, filename)
    if match:
        chinese_name = match.group(1)
        return f"{chinese_name}.json", chinese_name, ".json"

    # 匹配 MinerU_markdown_ 前缀的MD文件（两种格式）
    # 格式1: MinerU_markdown_01珐琅工艺_20260124170709.md
    md_pattern1 = r'MinerU_markdown_(.+?)_(\d{8}\d{6})\.md$'
    match = re.match(md_pattern1, filename)
    if match:
        chinese_name = match.group(1)
        return f"{chinese_name}.md", chinese_name, ".md"

    # 格式2: MinerU_markdown_05首饰雕蜡工艺_20260124172707_2014993350298587136.md
    md_pattern2 = r'MinerU_markdown_(.+?)_\d{8}\d{8}_\d+\.md$'
    match = re.match(md_pattern2, filename)
    if match:
        chinese_name = match.group(1)
        return f"{chinese_name}.md", chinese_name, ".md"

    return None


def rename_files_in_directory(directory: str, dry_run: bool = True) -> None:
    """
    重命名目录中的MinerU文件

    Args:
        directory: 目录路径
        dry_run: 是否只显示不执行重命名
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"错误: 目录不存在: {directory}")
        return

    print(f"处理目录: {directory}")
    print("=" * 60)

    # 获取所有文件
    files = list(dir_path.iterdir())
    mineru_files = [f for f in files if f.is_file()]

    if not mineru_files:
        print("目录中没有文件")
        return

    # 先解析所有文件，找出重名
    parsed_files = []
    for file_path in sorted(mineru_files):
        result = parse_mineru_filename(file_path.name)
        if result:
            new_name, chinese_name, ext = result
            parsed_files.append({
                "path": file_path,
                "new_name": new_name,
                "chinese_name": chinese_name,
                "ext": ext
            })

    # 统计每个新名称出现的次数
    name_counts = {}
    for pf in parsed_files:
        name = pf["new_name"]
        name_counts[name] = name_counts.get(name, 0) + 1

    renamed_count = 0
    skipped_count = 0

    for pf in parsed_files:
        file_path = pf["path"]
        new_name = pf["new_name"]

        # 如果有重名，检查是否需要添加序号
        if name_counts[new_name] > 1:
            # 检查是否是JSON或MD文件，分别编号
            base_name = pf["chinese_name"]
            ext = pf["ext"]

            # 找出所有同名文件，按扩展名分组
            same_name_files = [p for p in parsed_files if p["chinese_name"] == base_name]

            # 按扩展名和原始文件名排序，保证同名JSON和MD都能正确编号
            if ext == ".json":
                json_files = sorted([p for p in same_name_files if p["ext"] == ".json"],
                                   key=lambda x: x["path"].name)
                for idx, f in enumerate(json_files, 1):
                    if f["path"] == file_path:
                        if len(json_files) > 1:
                            new_name = f"{base_name}_{idx}{ext}"
                        break
            elif ext == ".md":
                md_files = sorted([p for p in same_name_files if p["ext"] == ".md"],
                                  key=lambda x: x["path"].name)
                for idx, f in enumerate(md_files, 1):
                    if f["path"] == file_path:
                        if len(md_files) > 1:
                            new_name = f"{base_name}_{idx}{ext}"
                        break

        new_path = dir_path / new_name

        # 执行重命名
        if dry_run:
            print(f"[预览] {file_path.name} -> {new_name}")
        else:
            file_path.rename(new_path)
            print(f"[重命名] {file_path.name} -> {new_name}")

        renamed_count += 1

    skipped_count = len(mineru_files) - renamed_count

    print("=" * 60)
    print(f"统计: 重命名={renamed_count}, 跳过={skipped_count}")

    if dry_run:
        print("\n这是预览模式，没有实际重命名文件")
        print("使用 --execute 参数执行实际重命名")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MinerU文件重命名工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览模式（默认）
  python scripts/rename_mineru_files.py

  # 执行重命名
  python scripts/rename_mineru_files.py --execute

  # 指定目录
  python scripts/rename_mineru_files.py --directory /path/to/files --execute
        """
    )

    parser.add_argument(
        "--directory", "-d",
        default="Digital-Human-Disciplinary-Dataset/Jewelry_Crafts_Dataset",
        help="文件所在目录（默认: Jewelry_Crafts_Dataset）"
    )
    parser.add_argument(
        "--execute", "-e",
        action="store_true",
        help="执行实际重命名（默认只预览）"
    )

    args = parser.parse_args()

    # 转换为绝对路径
    directory = os.path.abspath(args.directory)

    # 检查是否为相对路径
    if not os.path.isabs(args.directory):
        # 尝试从项目根目录找
        project_root = Path(__file__).parent.parent
        directory = project_root / args.directory

    rename_files_in_directory(str(directory), dry_run=not args.execute)


if __name__ == "__main__":
    main()
