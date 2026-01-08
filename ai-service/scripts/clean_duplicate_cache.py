"""
清理 mineru_cache 集合中的重复数据

基于 start_page 和 end_page 组合去重，保留最新的记录
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python �径径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def clean_duplicates():
    """清理重复的缓存记录"""
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.core.config import settings
    from app.core.logging import get_logger

    logger = get_logger(__name__)

    try:
        # 连接 MongoDB
        client = AsyncIOMotorClient(settings.mongodb_uri)
        db = client[settings.mongodb_db_name]
        collection = db["mineru_cache"]

        logger.info("开始清理重复的缓存记录...")

        # 获取所有记录
        cursor = collection.find({})
        all_records = await cursor.to_list(length=None)

        logger.info(f"总共有 {len(all_records)} 条记录")

        # 按 start_page 和 end_page 分组
        groups = {}
        for record in all_records:
            key = (record.get("start_page"), record.get("end_page"))
            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        logger.info(f"发现 {len(groups)} 个不同的页码范围组")

        # 找出重复的并删除旧的
        duplicates_to_delete = []
        kept_count = 0

        for (start_page, end_page), records in groups.items():
            if len(records) > 1:
                # 按创建时间排序，保留最新的
                sorted_records = sorted(
                    records,
                    key=lambda x: x.get("created_at"),
                    reverse=True
                )

                # 保留第一条（最新的），删除其余的
                keep = sorted_records[0]
                delete = sorted_records[1:]

                duplicates_to_delete.extend([r["_id"] for r in delete])
                kept_count += 1

                logger.info(
                    f"页码范围 {start_page}-{end_page}: 保留 {keep['_id']}, "
                    f"删除 {[r['_id'] for r in delete]}"
                )

        # 删除重复记录
        if duplicates_to_delete:
            result = await collection.delete_many({
                "_id": {"$in": duplicates_to_delete}
            })

            logger.info(f"已删除 {result.deleted_count} 条重复记录")
            print(f"\n[OK] 成功清理 {result.deleted_count} 条重复记录")
            print(f"[OK] 保留了 {kept_count} 个页码范围的最佳记录")
        else:
            logger.info("没有发现重复记录")
            print("\n[OK] 没有发现重复记录")

        # 显示清理后的统计
        remaining_count = await collection.count_documents({})
        print(f"[OK] 清理后剩余 {remaining_count} 条记录\n")

        # 关闭连接
        client.close()

    except Exception as e:
        logger.error(f"清理失败: {str(e)}", exc_info=True)
        print(f"\n[ERROR] 错误: {str(e)}\n")


if __name__ == "__main__":
    asyncio.run(clean_duplicates())
