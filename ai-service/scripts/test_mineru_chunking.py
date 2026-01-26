#!/usr/bin/env python3
"""
测试 MinerU JSON 分块，输出为 Markdown

使用示例:
    # 基础用法
    python scripts/test_mineru_chunking.py input.json

    # 指定分块策略
    python scripts/test_mineru_chunking.py input.json --strategy by_title

    # 自定义分块大小
    python scripts/test_mineru_chunking.py input.json --max-size 500

    # 指定输出文件
    python scripts/test_mineru_chunking.py input.json --output chunks.md
"""
import argparse
import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.mineru_json_parser import MinerUJsonParser
from app.services.mineru_aware_chunking import MinerUAwareChunker, ChunkingStrategy


def format_chunk_as_markdown(chunk, index: int) -> str:
    """将单个 chunk 格式化为 Markdown"""
    lines = [
        "---",
        f"## Chunk #{index}",
        "",
        f"**ID**: `{chunk.chunk_id}`",
        f"**Doc ID**: `{chunk.doc_id}`",
        f"**KB ID**: `{chunk.kb_id}`",
        f"**Chunk Index**: {chunk.chunk_index}",
        "",
        "### 元数据",
        f"- **页码**: {chunk.page_idx}",
        f"- **包含页数**: {chunk.page_indices}",
        f"- **块类型**: {', '.join(chunk.block_types) if chunk.block_types else 'N/A'}",
        f"- **结构层级**: {chunk.structure_level}",
        f"- **图片数量**: {len(chunk.image_references)}",
        "",
    ]

    # 标题路径
    if chunk.title_path:
        lines.append("### 标题路径")
        for i, title in enumerate(chunk.title_path):
            indent = "  " * i
            lines.append(f"{indent}- {title}")
        lines.append("")

    # 图片引用
    if chunk.image_references:
        lines.append("### 图片引用")
        for img_url in chunk.image_references:
            lines.append(f"- `{img_url}`")
        lines.append("")

    # 图片描述
    if chunk.image_captions:
        lines.append("### 图片描述")
        for caption in chunk.image_captions:
            lines.append(f"- {caption}")
        lines.append("")

    # 内容
    lines.append("### 内容")
    lines.append("")
    lines.append(chunk.content)
    lines.append("")

    # 统计信息
    lines.append("")
    lines.append(f"<!-- 字符数: {len(chunk.content)} -->")
    lines.append("")

    return "\n".join(lines)


async def test_chunking(
    json_file: str,
    strategy: str = "hybrid",
    max_chunk_size: int = 1000,
    output_file: str = None
):
    """测试分块并输出 Markdown"""
    print(f"📄 解析文件: {json_file}")

    # 解析 JSON
    parser = MinerUJsonParser()
    doc = parser.parse_file(json_file)
    print(f"   ✅ 解析完成: {doc.get_total_pages()} 页")

    # 统计 blocks
    total_blocks = sum(len(page.blocks) for page in doc.pdf_info)
    print(f"   📊 总 blocks: {total_blocks}")

    # 统计各类型 blocks
    block_types = {}
    for page in doc.pdf_info:
        for block in page.blocks:
            bt = block.block_type.value
            block_types[bt] = block_types.get(bt, 0) + 1
    print(f"   📊 Blocks 类型: {block_types}")

    # 分块
    print(f"\n🔪 开始分块 (策略: {strategy}, 最大大小: {max_chunk_size})...")
    chunker = MinerUAwareChunker(
        max_chunk_size=max_chunk_size,
        strategy=strategy
    )
    chunks = await chunker.chunk_document(doc, "test_doc", "test_kb")
    print(f"   ✅ 分块完成: {len(chunks)} 个 chunks")

    # 统计信息
    total_chars = sum(len(c.content) for c in chunks)
    avg_chars = total_chars // len(chunks) if chunks else 0
    chunks_with_images = sum(1 for c in chunks if c.image_references)
    chunks_with_titles = sum(1 for c in chunks if c.title_path)

    print(f"\n📊 分块统计:")
    print(f"   - 总字符数: {total_chars}")
    print(f"   - 平均大小: {avg_chars} 字符")
    print(f"   - 最小: {min((len(c.content) for c in chunks), default=0)} 字符")
    print(f"   - 最大: {max((len(c.content) for c in chunks), default=0)} 字符")
    print(f"   - 包含图片: {chunks_with_images}/{len(chunks)}")
    print(f"   - 包含标题: {chunks_with_titles}/{len(chunks)}")

    # 大小分布
    size_ranges = {
        "0-100": 0,
        "100-300": 0,
        "300-500": 0,
        "500-800": 0,
        "800-1000": 0,
        "1000+": 0,
    }
    for c in chunks:
        size = len(c.content)
        if size < 100:
            size_ranges["0-100"] += 1
        elif size < 300:
            size_ranges["100-300"] += 1
        elif size < 500:
            size_ranges["300-500"] += 1
        elif size < 800:
            size_ranges["500-800"] += 1
        elif size < 1000:
            size_ranges["800-1000"] += 1
        else:
            size_ranges["1000+"] += 1

    print(f"   - 大小分布:")
    for range_name, count in size_ranges.items():
        if count > 0:
            print(f"      {range_name}: {count}")

    # 生成 Markdown
    print(f"\n📝 生成 Markdown...")

    md_lines = [
        f"# MinerU 分块测试结果",
        "",
        f"## 文档信息",
        f"- **源文件**: {json_file}",
        f"- **总页数**: {doc.get_total_pages()}",
        f"- **总 blocks**: {total_blocks}",
        f"- **分块策略**: {strategy}",
        f"- **最大分块大小**: {max_chunk_size}",
        "",
        f"## 分块统计",
        f"- **分块数量**: {len(chunks)}",
        f"- **总字符数**: {total_chars}",
        f"- **平均大小**: {avg_chars} 字符",
        f"- **大小范围**: {min((len(c.content) for c in chunks), default=0)} - {max((len(c.content) for c in chunks), default=0)} 字符",
        f"- **包含图片的分块**: {chunks_with_images}",
        f"- **包含标题的分块**: {chunks_with_titles}",
        "",
        f"## Blocks 类型分布",
    ]
    for bt, count in block_types.items():
        md_lines.append(f"- **{bt}**: {count}")
    md_lines.append("")
    md_lines.append(f"## 大小分布")
    for range_name, count in size_ranges.items():
        if count > 0:
            md_lines.append(f"- **{range_name} 字符**: {count}")
    md_lines.append("")

    # 每个分块的详细内容
    md_lines.append("## 分块详情")
    md_lines.append("")

    for i, chunk in enumerate(chunks):
        md_lines.append(format_chunk_as_markdown(chunk, i))

    md_content = "\n".join(md_lines)

    # 确定输出文件
    if output_file is None:
        input_path = Path(json_file)
        output_file = input_path.parent / f"{input_path.stem}_chunks.md"

    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"   ✅ 已保存到: {output_path}")
    print(f"\n✨ 完成！")


def main():
    parser = argparse.ArgumentParser(
        description="测试 MinerU JSON 分块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s input.json
  %(prog)s input.json --strategy by_title --max-size 500
  %(prog)s input.json --output my_chunks.md
        """
    )

    parser.add_argument("input", help="MinerU JSON 文件路径")
    parser.add_argument(
        "--strategy",
        choices=["by_title", "by_page", "hybrid"],
        default="hybrid",
        help="分块策略 (默认: hybrid)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1000,
        help="最大分块大小（字符数，默认: 1000）"
    )
    parser.add_argument(
        "--output", "-o",
        help="输出 Markdown 文件路径（默认: input_chunks.md）"
    )

    args = parser.parse_args()

    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 错误: 文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    # 运行测试
    asyncio.run(test_chunking(
        json_file=str(input_path),
        strategy=args.strategy,
        max_chunk_size=args.max_size,
        output_file=args.output
    ))


if __name__ == "__main__":
    main()
