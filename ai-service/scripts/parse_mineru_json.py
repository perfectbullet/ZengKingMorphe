#!/usr/bin/env python3
"""
MinerU JSON 解析 CLI 工具

用于独立运行MinerU JSON解析和分块，支持：
1. 解析JSON文件并输出结构化数据
2. 生成增强版Markdown
3. 智能分块
4. 可选：生成图片描述

使用示例:
    # 基础解析
    python scripts/parse_mineru_json.py input.json --output result.json

    # 生成增强Markdown
    python scripts/parse_mineru_json.py input.json --markdown --output enhanced.md

    # 智能分块
    python scripts/parse_mineru_json.py input.json --chunk --chunks-output chunks.json

    # 生成图片描述（需要API密钥）
    python scripts/parse_mineru_json.py input.json --caption-images --vlm-api-key YOUR_KEY

    # 完整流程
    python scripts/parse_mineru_json.py input.json \\
        --output result.json \\
        --markdown --output-md enhanced.md \\
        --chunk --chunks-output chunks.json \\
        --caption-images --vlm-api-key YOUR_KEY
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.mineru_json_parser import MinerUJsonParser, MinerUDocument
from app.services.mineru_aware_chunking import MinerUAwareChunker, ChunkingStrategy
from app.services.mineru_image_handler import (
    MinerUImageHandler,
    VLMBackend,
    ImageInfo
)
from app.core.logging import logger


class MinerUJsonCLI:
    """MinerU JSON 解析 CLI"""

    def __init__(self):
        self.parser = MinerUJsonParser()

    def parse_file(self, input_file: str) -> MinerUDocument:
        """解析JSON文件"""
        logger.info(f"开始解析文件: {input_file}")
        doc = self.parser.parse_file(input_file)
        logger.info(f"解析完成: {doc.get_total_pages()} 页")
        return doc

    def save_summary(self, doc: MinerUDocument, output_file: str):
        """保存解析摘要"""
        summary = doc.to_dict()
        summary["pages_detail"] = []

        for page in doc.pdf_info:
            page_detail = {
                "page_idx": page.page_idx,
                "page_size": page.page_size,
                "blocks_count": len(page.blocks),
                "blocks_by_type": {}
            }

            for block in page.blocks:
                block_type = block.block_type.value
                if block_type not in page_detail["blocks_by_type"]:
                    page_detail["blocks_by_type"][block_type] = 0
                page_detail["blocks_by_type"][block_type] += 1

            summary["pages_detail"].append(page_detail)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"摘要已保存到: {output_file}")

    def save_markdown(self, doc: MinerUDocument, output_file: str, include_images: bool = True):
        """保存增强Markdown"""
        md_content = doc.to_enhanced_markdown(include_images=include_images)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Markdown已保存到: {output_file}")

    async def generate_captions(
        self,
        doc: MinerUDocument,
        vlm_backend: str,
        vlm_api_key: Optional[str],
        vlm_base_url: Optional[str],
        vlm_model: str
    ) -> dict:
        """生成图片描述"""
        if not vlm_api_key:
            logger.warning("未提供VLM API密钥，跳过图片描述生成")
            return {}

        # 确定后端类型
        try:
            backend = VLMBackend(vlm_backend)
        except ValueError:
            backend = VLMBackend.QWEN_VL
            logger.warning(f"未知的VLM后端: {vlm_backend}, 使用默认: qwen_vl")

        handler = MinerUImageHandler(
            vlm_backend=backend,
            vlm_api_key=vlm_api_key,
            vlm_base_url=vlm_base_url,
            vlm_model=vlm_model
        )

        # 提取图片信息
        images_data = doc.to_dict()
        images = []

        for img_info in images_data.get("images", []):
            images.append(ImageInfo(
                url=img_info["url"],
                page_idx=img_info["page_idx"],
                bbox=tuple(img_info["bbox"]),
                context_before="",
                context_after=""
            ))

        if not images:
            logger.info("文档中没有图片")
            return {}

        logger.info(f"开始生成 {len(images)} 张图片的描述...")

        captions = await handler.generate_captions_batch(
            images=images,
            use_cache=True,
            concurrent_limit=3
        )

        logger.info(f"图片描述生成完成: {len(captions)}/{len(images)}")
        return captions

    def save_chunks(
        self,
        doc: MinerUDocument,
        output_file: str,
        doc_id: str,
        kb_id: str,
        strategy: str,
        max_chunk_size: int,
        image_captions: Optional[dict] = None
    ):
        """保存分块结果"""
        chunker = MinerUAwareChunker(
            max_chunk_size=max_chunk_size,
            strategy=strategy
        )

        chunks = asyncio.run(chunker.chunk_document(
            doc=doc,
            doc_id=doc_id,
            kb_id=kb_id,
            image_captions=image_captions
        ))

        chunks_data = [chunk.to_dict() for chunk in chunks]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.info(f"分块结果已保存到: {output_file} ({len(chunks)} 个分块)")

        # 打印统计信息
        total_chars = sum(len(c["content"]) for c in chunks_data)
        chunks_with_images = sum(1 for c in chunks_data if c["image_references"])
        logger.info(f"  - 总字符数: {total_chars}")
        logger.info(f"  - 平均分块大小: {total_chars // len(chunks)} 字符")
        logger.info(f"  - 包含图片的分块: {chunks_with_images}")


def main():
    parser = argparse.ArgumentParser(
        description="MinerU JSON 解析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础解析
  %(prog)s input.json --output result.json

  # 生成增强Markdown
  %(prog)s input.json --markdown --output enhanced.md

  # 智能分块
  %(prog)s input.json --chunk --chunks-output chunks.json

  # 生成图片描述
  %(prog)s input.json --caption-images --vlm-api-key YOUR_KEY
        """
    )

    # 输入输出
    parser.add_argument("input", help="输入的MinerU JSON文件路径")
    parser.add_argument("--output", "-o", help="输出的摘要JSON文件路径")
    parser.add_argument("--markdown", action="store_true", help="生成增强Markdown")
    parser.add_argument("--output-md", help="Markdown输出文件路径")

    # 分块选项
    parser.add_argument("--chunk", action="store_true", help="启用智能分块")
    parser.add_argument("--chunks-output", help="分块结果输出文件路径")
    parser.add_argument("--strategy", choices=["by_title", "by_page", "hybrid"],
                       default="hybrid", help="分块策略 (默认: hybrid)")
    parser.add_argument("--max-chunk-size", type=int, default=1000,
                       help="最大分块大小（字符数，默认: 1000）")
    parser.add_argument("--doc-id", default="doc_001", help="文档ID（用于分块）")
    parser.add_argument("--kb-id", default="kb_001", help="知识库ID（用于分块）")

    # 图片描述选项
    parser.add_argument("--caption-images", action="store_true",
                       help="为图片生成描述（需要VLM API）")
    parser.add_argument("--vlm-backend", choices=["openai", "qwen_vl", "custom"],
                       default="qwen_vl", help="VLM后端类型")
    parser.add_argument("--vlm-api-key", help="VLM API密钥")
    parser.add_argument("--vlm-base-url", help="VLM API地址")
    parser.add_argument("--vlm-model", default="qwen-vl-max", help="VLM模型名称")

    # 其他选项
    parser.add_argument("--exclude-images", action="store_true",
                       help="在Markdown中排除图片")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    # 设置默认输出路径
    if args.output is None:
        args.output = str(input_path.with_suffix(".parsed.json"))

    if args.markdown and args.output_md is None:
        args.output_md = str(input_path.with_suffix(".enhanced.md"))

    if args.chunk and args.chunks_output is None:
        args.chunks_output = str(input_path.with_suffix(".chunks.json"))

    # 执行解析
    cli = MinerUJsonCLI()
    doc = cli.parse_file(args.input)

    # 保存摘要
    cli.save_summary(doc, args.output)

    # 生成Markdown
    if args.markdown:
        cli.save_markdown(doc, args.output_md, include_images=not args.exclude_images)

    # 生成图片描述
    image_captions = None
    if args.caption_images:
        image_captions = asyncio.run(cli.generate_captions(
            doc=doc,
            vlm_backend=args.vlm_backend,
            vlm_api_key=args.vlm_api_key,
            vlm_base_url=args.vlm_base_url,
            vlm_model=args.vlm_model
        ))

        # 保存图片描述
        captions_file = str(input_path.with_suffix(".captions.json"))
        with open(captions_file, "w", encoding="utf-8") as f:
            json.dump(image_captions, f, ensure_ascii=False, indent=2)
        logger.info(f"图片描述已保存到: {captions_file}")

    # 分块
    if args.chunk:
        cli.save_chunks(
            doc=doc,
            output_file=args.chunks_output,
            doc_id=args.doc_id,
            kb_id=args.kb_id,
            strategy=args.strategy,
            max_chunk_size=args.max_chunk_size,
            image_captions=image_captions
        )

    logger.info("处理完成!")


if __name__ == "__main__":
    main()
