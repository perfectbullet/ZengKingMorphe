"""
测试 MinerU 缓存功能的简单脚本
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_cache():
    """测试缓存功能"""
    from app.services.mineru_client import get_mineru_client
    from app.core.logging import get_logger
    from app.core.database import mongodb

    logger = get_logger(__name__)

    # 使用绝对路径
    file_path = Path(__file__).parent.parent.parent / "test_files" / "首饰雕蜡工艺-全本.pdf"
    file_path = str(file_path.resolve())

    print(f"\n测试文件: {file_path}")
    print("=" * 80)

    try:
        # 初始化数据库连接
        await mongodb.connect()

        client = get_mineru_client()

        # 第一次运行（应该调用 API）
        print("\n[第一次运行 - 应该调用 MinerU API]")
        print("-" * 80)
        async with client:
            result1 = await client.process_pdf(file_path, use_cache=True)

        print(f"\n状态: {result1['status']}")
        print(f"小文档数: {result1['chunks']}")

        # 第二次运行（应该使用缓存）
        print("\n[第二次运行 - 应该使用缓存]")
        print("-" * 80)
        async with client:
            result2 = await client.process_pdf(file_path, use_cache=True)

        print(f"\n状态: {result2['status']}")
        print(f"小文档数: {result2['chunks']}")

        # 关闭数据库连接
        await mongodb.disconnect()

        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}\n")

        # 确保关闭数据库连接
        try:
            await mongodb.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(test_cache())
