"""
文档解析测试示例

演示 MinerUParser 的各种功能：
- 基本解析
- 使用自定义 ParseOptions 和 ReturnOptions
- 批量解析
- 进度回调
- 图片描述生成
"""

import asyncio
from pathlib import Path

from src.config import settings
from src.document_parser import (
    MinerUParser,
    ParseOptions,
    ReturnOptions,
    STAGE_NAMES,
)
from src.document_parser.image_processor import ImageDescriptor
from src.utils import setup_logger


async def example_basic_parse():
    """示例 1: 基本解析"""
    print("\n=== 示例 1: 基本解析 ===")
    async with MinerUParser() as parser:
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
        else:
            print(f"✗ PDF 文件不存在: {pdf_path}")


async def example_with_options():
    """示例 2: 使用自定义解析选项"""
    print("\n=== 示例 2: 使用自定义解析选项 ===")
    async with MinerUParser() as parser:
        pdf_path = "data/sample_pdf/example.pdf"

        if Path(pdf_path).exists():
            # 自定义解析选项
            parse_options = ParseOptions(
                backend="pipeline",  # 使用传统后端（显存占用低）
                parse_method="auto",  # 自动判断解析方法
                lang="ch",  # 中文
                start_page=0,  # 从第一页开始
                end_page=5,  # 只解析前 5 页（None 表示全部）
                formula_enable=True,  # 启用公式解析
                table_enable=True,  # 启用表格解析
            )

            # 自定义返回选项
            return_options = ReturnOptions(
                return_md=True,  # 返回 Markdown
                return_content_list=True,  # 返回内容列表（用于结构化分块）
                return_middle_json=False,  # 不返回中间 JSON
                return_images=True,  # 返回图片
            )

            document = await parser.parse(
                pdf_path,
                parse_options=parse_options,
                return_options=return_options,
            )

            print(f"✓ 文档解析成功（使用自定义选项）")
            print(f"  标题: {document.title}")
            print(f"  文本块数量: {len(document.chunks)}")
            print(f"  图片数量: {len(document.images)}")
        else:
            print(f"✗ PDF 文件不存在: {pdf_path}")


async def example_with_progress():
    """示例 3: 带进度回调"""
    print("\n=== 示例 3: 带进度回调 ===")

    def on_progress(stage: str, percent: float):
        name = STAGE_NAMES.get(stage, stage)
        print(f"  [{percent:5.1f}%] {name}")

    async with MinerUParser() as parser:
        pdf_path = "data/sample_pdf/example.pdf"

        if Path(pdf_path).exists():
            print("开始解析，进度:")
            document = await parser.parse(pdf_path, progress_callback=on_progress)
            print(f"✓ 文档解析完成")
            print(f"  标题: {document.title}")
            print(f"  文本块数量: {len(document.chunks)}")
        else:
            print(f"✗ PDF 文件不存在: {pdf_path}")


async def example_batch_parse():
    """示例 4: 批量解析"""
    print("\n=== 示例 4: 批量解析 ===")
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


async def example_image_description():
    """示例 5: 图片描述生成"""
    print("\n=== 示例 5: 图片描述生成 ===")

    # 首先解析一个包含图片的文档
    async with MinerUParser() as parser:
        pdf_path = "data/sample_pdf/example.pdf"

        if Path(pdf_path).exists():
            document = await parser.parse(pdf_path)

            if document.images:
                async with ImageDescriptor() as descriptor:
                    # 描述第一个图片
                    image_path = document.images[0].path
                    if Path(image_path).exists():
                        description = await descriptor.describe_image(image_path)
                        print(f"✓ 图片描述生成成功")
                        print(f"  图片: {Path(image_path).name}")
                        print(f"  描述: {description}")
                    else:
                        print(f"✗ 图片文件不存在: {image_path}")
            else:
                print("✗ 文档中没有图片")
        else:
            print(f"✗ PDF 文件不存在: {pdf_path}")


async def example_to_memory():
    """示例 6: 直接返回 JSON（不下载 ZIP）"""
    print("\n=== 示例 6: 直接返回 JSON ===")
    async with MinerUParser() as parser:
        pdf_path = "data/sample_pdf/example.pdf"

        if Path(pdf_path).exists():
            try:
                result = await parser.parse_to_memory(pdf_path)

                print(f"✓ JSON 解析成功")
                print(f"  返回的键: {list(result.keys())}")

                # 显示内容列表
                if "content_list" in result:
                    content_list = result["content_list"]
                    print(f"  内容列表项数: {len(content_list)}")

                    # 统计各类型数量
                    type_counts = {}
                    for item in content_list:
                        t = item.get("type", "unknown")
                        type_counts[t] = type_counts.get(t, 0) + 1
                    print(f"  内容类型统计: {type_counts}")
            except Exception as e:
                print(f"✗ JSON 解析失败: {e}")
        else:
            print(f"✗ PDF 文件不存在: {pdf_path}")


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

    # 健康检查
    print("=== 健康检查 ===")
    async with MinerUParser() as parser:
        is_healthy = await parser.health_check()
        print(f"MinerU API 服务状态: {'✓ 正常' if is_healthy else '✗ 不可用'}")
        if not is_healthy:
            print(f"  API 地址: {parser.api_url}")

    # 运行示例
    await example_basic_parse()
    await example_with_options()
    await example_with_progress()
    await example_batch_parse()
    await example_image_description()
    await example_to_memory()


if __name__ == "__main__":
    asyncio.run(main())
