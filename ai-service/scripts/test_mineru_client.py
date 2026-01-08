"""
MinerU 客户端测试脚本

用于测试 MinerU PDF 解析功能。

使用方法：
    cd ai-service
    python scripts/test_mineru_client.py --file path/to/document.pdf
    python scripts/test_mineru_client.py --file path/to/document.pdf --use-cache
    python scripts/test_mineru_client.py --job-id job_id_here
    python scripts/test_mineru_client.py --file ../test_files/首饰雕蜡工艺-全本.pdf
"""
import asyncio
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_mineru_processing(file_path: str, use_cache: bool = True):
    """测试 MinerU PDF 处理"""
    from app.services.mineru_client import get_mineru_client
    from app.core.logging import get_logger
    from app.core.database import mongodb

    logger = get_logger(__name__)

    if not Path(file_path).exists():
        logger.error(f"文件不存在: {file_path}")
        return

    logger.info(f"开始测试 MinerU 处理: {file_path}")

    try:
        # 初始化数据库连接
        logger.info("连接 MongoDB...")
        await mongodb.connect()

        client = get_mineru_client()

        async with client:
            # 处理 PDF
            result = await client.process_pdf(
                file_path=file_path,
                use_cache=use_cache
            )

            # 显示结果
            print("\n" + "="*60)
            print("处理结果")
            print("="*60)
            print(f"状态: {result.get('status')}")
            print(f"小文档数量: {result.get('chunks')}")
            print(f"总页数: {result['metadata'].get('total_pages')}")
            print(f"合并时间: {result['metadata'].get('merged_at')}")
            print(f"内容项数量: {len(result.get('content', []))}")

            # 显示前 3 个内容项的预览
            content_items = result.get('content', [])
            if content_items:
                print("\n内容预览（前 3 项）:")
                for i, item in enumerate(content_items[:3], 1):
                    if isinstance(item, dict):
                        text = item.get('text', '') or item.get('content', '')
                    else:
                        text = str(item)

                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"\n[{i}] {preview}\n")

            print("="*60 + "\n")

            logger.info("测试完成")

            # 关闭数据库连接
            await mongodb.disconnect()

    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}")

        # 确保关闭数据库连接
        try:
            await mongodb.disconnect()
        except Exception:
            pass


async def test_job_status(job_id: str):
    """测试查询任务状态"""
    from app.services.mineru_client import get_mineru_client
    from app.core.logging import get_logger
    from app.core.database import mongodb

    logger = get_logger(__name__)

    logger.info(f"查询任务状态: {job_id}")

    try:
        # 初始化数据库连接
        await mongodb.connect()

        client = get_mineru_client()
        job_status = await client.get_job_status(job_id)

        if job_status:
            print("\n" + "="*60)
            print("任务状态")
            print("="*60)
            print(f"任务 ID: {job_id}")
            print(f"文件名: {job_status.get('file_name')}")
            print(f"状态: {job_status.get('status')}")
            print(f"总小文档数: {job_status.get('total_chunks')}")
            print(f"已处理: {job_status.get('processed_chunks')}")
            print(f"失败: {job_status.get('failed_chunks')}")
            print(f"创建时间: {job_status.get('created_at')}")
            print(f"更新时间: {job_status.get('updated_at')}")

            if job_status.get('status') == 'completed':
                print(f"完成时间: {job_status.get('completed_at')}")
            elif job_status.get('status') == 'failed':
                print(f"错误信息: {job_status.get('error_message')}")

            print("="*60 + "\n")

            logger.info("查询完成")
        else:
            print(f"\n任务不存在: {job_id}\n")

        # 关闭数据库连接
        await mongodb.disconnect()

    except Exception as e:
        logger.error(f"查询失败: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}\n")

        # 确保关闭数据库连接
        try:
            await mongodb.disconnect()
        except Exception:
            pass


async def test_clear_cache(days: int = 30):
    """测试清理缓存"""
    from app.services.mineru_client import get_mineru_client
    from app.core.logging import get_logger
    from app.core.database import mongodb

    logger = get_logger(__name__)

    logger.info(f"清理 {days} 天以前的缓存")

    try:
        # 初始化数据库连接
        await mongodb.connect()

        client = get_mineru_client()
        deleted_count = await client.clear_cache(older_than_days=days)

        print(f"\n已清理 {deleted_count} 条缓存记录\n")
        logger.info(f"清理完成: {deleted_count} 条")

        # 关闭数据库连接
        await mongodb.disconnect()

    except Exception as e:
        logger.error(f"清理失败: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}\n")

        # 确保关闭数据库连接
        try:
            await mongodb.disconnect()
        except Exception:
            pass


async def test_with_document_processor(file_path: str, kb_id: str):
    """测试通过 DocumentProcessor 使用 MinerU"""
    from app.services.document_service import document_processor
    from app.core.logging import get_logger
    from app.core.database import mongodb

    logger = get_logger(__name__)

    if not Path(file_path).exists():
        logger.error(f"文件不存在: {file_path}")
        return

    logger.info(f"通过 DocumentProcessor 测试: {file_path}")

    try:
        # 初始化数据库连接
        await mongodb.connect()

        # 初始化其他数据库连接
        from app.core.chroma import chroma_db
        from app.core.elasticsearch import es_db

        chroma_db.connect()
        await es_db.connect()
        # 启用 MinerU 处理
        doc_id = await document_processor.process_document(
            file_path=file_path,
            filename=Path(file_path).name,
            kb_id=kb_id,
            metadata={"use_mineru": True}
        )

        print(f"\n文档已处理: {doc_id}\n")
        logger.info(f"文档处理完成: {doc_id}")

        # 关闭所有数据库连接
        await mongodb.disconnect()

    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}\n")

        # 确保关闭数据库连接
        try:
            await mongodb.disconnect()
        except Exception:
            pass


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MinerU 客户端测试工具")

    subparsers = parser.add_subparsers(dest="command", help="测试命令")

    # 处理 PDF 命令
    process_parser = subparsers.add_parser("process", help="处理 PDF 文件")
    process_parser.add_argument("--file", required=True, help="PDF 文件路径")
    process_parser.add_argument("--no-cache", action="store_true", help="禁用缓存")

    # 查询任务命令
    status_parser = subparsers.add_parser("status", help="查询任务状态")
    status_parser.add_argument("--job-id", required=True, help="任务 ID")

    # 清理缓存命令
    cache_parser = subparsers.add_parser("clear-cache", help="清理缓存")
    cache_parser.add_argument("--days", type=int, default=30, help="清理多少天以前的缓存（默认 30）")

    # 通过 DocumentProcessor 测试
    doc_parser = subparsers.add_parser("upload", help="通过 DocumentProcessor 上传文档")
    doc_parser.add_argument("--file", required=True, help="PDF 文件路径")
    doc_parser.add_argument("--kb-id", default="test_kb", help="知识库 ID")

    args = parser.parse_args()

    # 执行对应命令
    if args.command == "process":
        asyncio.run(test_mineru_processing(args.file, use_cache=not args.no_cache))
    elif args.command == "status":
        asyncio.run(test_job_status(args.job_id))
    elif args.command == "clear-cache":
        asyncio.run(test_clear_cache(args.days))
    elif args.command == "upload":
        asyncio.run(test_with_document_processor(args.file, args.kb_id))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
