"""
文档解析测试示例
"""

import asyncio
from pathlib import Path

from src.config import settings
from src.document_parser.mineru_client import MinerUParser
from src.document_parser.image_processor import ImageDescriptor
from src.utils import setup_logger


async def main():
    """主函数"""
    # 配置日志
    logger = setup_logger(
        log_level=settings.log_level,
        log_file=settings.log_file
    )
    logger.info("文档解析测试示例")

    # 确保必要的目录存在
    settings.ensure_directories()

    # 1. 解析单个文档
    print("\n=== 测试 1: 解析单个文档 ===")
    async with MinerUParser() as parser:
        async with ImageDescriptor() as descriptor:
            pdf_path = "data/sample_pdf/example.pdf"

            if Path(pdf_path).exists():
                document = await parser.parse(pdf_path)

                print(f"✓ 文档解析成功")
                print(f"  标题: {document.title}")
                print(f"  文本块数量: {len(document.chunks)}")
                print(f"  图片数量: {len(document.images)}")

                # 显示前几个文本块
                print("\n前 3 个文本块:")
                for i, chunk in enumerate(document.chunks[:3], 1):
                    print(f"  块 {i} (长度: {len(chunk.text)}):")
                    print(f"    {chunk.text[:100]}...")

                # 显示图片信息
                if document.images:
                    print(f"\n图片列表:")
                    for i, img in enumerate(document.images, 1):
                        print(f"  {i}. {Path(img.path).name}")
                        if img.description:
                            print(f"     描述: {img.description[:80]}...")
            else:
                print(f"✗ PDF 文件不存在: {pdf_path}")

    # 2. 批量解析
    print("\n=== 测试 2: 批量解析 ===")
    async with MinerUParser() as parser:
        pdf_dir = Path("data/sample_pdf")
        pdf_files = list(pdf_dir.glob("*.pdf"))

        if pdf_files:
            pdf_paths = [str(f) for f in pdf_files[:3]]  # 最多解析 3 个
            print(f"找到 {len(pdf_paths)} 个 PDF 文件")

            documents = await parser.parse_batch(pdf_paths)

            print(f"✓ 批量解析完成")
            for i, doc in enumerate(documents, 1):
                print(f"  {i}. {doc.title} - {len(doc.chunks)} 块, {len(doc.images)} 图片")
        else:
            print(f"✗ 未找到 PDF 文件: {pdf_dir}")

    # 3. 图片描述测试
    print("\n=== 测试 3: 图片描述生成 ===")
    async with ImageDescriptor() as descriptor:
        # 测试图片
        test_image = "data/output/test_image.png"

        if Path(test_image).exists():
            description = await descriptor.describe_image(test_image)
            print(f"✓ 图片描述生成成功")
            print(f"  图片: {test_image}")
            print(f"  描述: {description}")
        else:
            print(f"✗ 测试图片不存在: {test_image}")


if __name__ == "__main__":
    asyncio.run(main())
