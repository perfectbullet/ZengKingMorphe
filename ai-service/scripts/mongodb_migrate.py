#!/usr/bin/env python3
"""
MongoDB 数据迁移脚本

支持将数据从一个 MongoDB 数据库迁移到另一个数据库。

功能：
- 迁移单个或多个集合
- 显示迁移进度
- 批量写入优化性能
- 支持增量迁移（跳过已存在的文档）
- 详细的日志输出

使用示例：
    # 迁移单个集合
    python scripts/mongodb_migrate.py --collection digital_employee_configs

    # 迁移多个集合
    python scripts/mongodb_migrate.py --collection digital_employee_configs --collection transcripts

    # 迁移所有集合
    python scripts/mongodb_migrate.py --all

    # 查看源数据库所有集合
    python scripts/mongodb_migrate.py --list

环境变量配置（优先级高于默认值）：
    SOURCE_URI: 源数据库连接 URI
    TARGET_URI: 目标数据库连接 URI
    BATCH_SIZE: 批量写入大小（默认 1000）
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.logging import get_logger

logger = get_logger(__name__)


# 默认配置
DEFAULT_SOURCE_URI = "mongodb://funasr:funasr2026@192.168.8.233:27017/funasr?authSource=admin"
DEFAULT_TARGET_URI = "mongodb://morphe_user:morphe_secure_2026@192.168.8.234:27017/morphe_db?authSource=morphe_db"
DEFAULT_BATCH_SIZE = 1000


def get_source_db_name(uri: str) -> str:
    """从 URI 中提取数据库名"""
    # 移除协议前缀
    uri_without_protocol = uri.replace("mongodb://", "").replace("mongodb+srv://", "")
    # 查找最后一个 / 后的数据库名
    if "/" in uri_without_protocol:
        db_part = uri_without_protocol.split("/")[-1]
        # 移除查询参数
        db_name = db_part.split("?")[0]
        return db_name
    return "unknown"


class MongoDBMigrator:
    """MongoDB 数据迁移器"""

    def __init__(
        self,
        source_uri: str,
        target_uri: str,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """
        初始化迁移器

        Args:
            source_uri: 源数据库 URI
            target_uri: 目标数据库 URI
            batch_size: 批量写入大小
        """
        self.source_uri = source_uri
        self.target_uri = target_uri
        self.batch_size = batch_size

        self.source_db_name = get_source_db_name(source_uri)
        self.target_db_name = get_source_db_name(target_uri)

        self.source_client: Optional['AsyncIOMotorClient'] = None
        self.target_client: Optional['AsyncIOMotorClient'] = None

    async def connect(self):
        """连接到源数据库和目标数据库"""
        logger.info(f"连接到源数据库: {self.source_db_name}")
        self.source_client = AsyncIOMotorClient(self.source_uri)
        # 测试连接
        await self.source_client.admin.command('ping')

        logger.info(f"连接到目标数据库: {self.target_db_name}")
        self.target_client = AsyncIOMotorClient(self.target_uri)
        await self.target_client.admin.command('ping')

        logger.info("数据库连接成功")

    async def close(self):
        """关闭数据库连接"""
        if self.source_client:
            self.source_client.close()
        if self.target_client:
            self.target_client.close()
        logger.info("数据库连接已关闭")

    async def list_collections(self) -> List[str]:
        """列出源数据库的所有集合"""
        if not self.source_client:
            raise RuntimeError("未连接到源数据库")

        source_db = self.source_client[self.source_db_name]
        collections = await source_db.list_collection_names()

        logger.info(f"源数据库 '{self.source_db_name}' 共有 {len(collections)} 个集合")
        for i, coll in enumerate(collections, 1):
            # 获取集合文档数量
            count = await source_db[coll].count_documents({})
            logger.info(f"  {i}. {coll}: {count} 个文档")

        return collections

    async def migrate_collection(
        self,
        collection_name: str,
        skip_existing: bool = True,
        drop_target: bool = False
    ) -> Dict[str, Any]:
        """
        迁移单个集合

        Args:
            collection_name: 集合名称
            skip_existing: 是否跳过已存在的文档（基于 _id）
            drop_target: 是否在迁移前清空目标集合

        Returns:
            迁移统计信息
        """
        if not self.source_client or not self.target_client:
            raise RuntimeError("数据库未连接")

        source_db = self.source_client[self.source_db_name]
        target_db = self.target_client[self.target_db_name]

        source_coll = source_db[collection_name]
        target_coll = target_db[collection_name]

        stats = {
            "collection": collection_name,
            "source_count": 0,
            "migrated_count": 0,
            "skipped_count": 0,
            "error_count": 0,
            "start_time": datetime.now(),
            "end_time": None,
        }

        logger.info(f"{'='*60}")
        logger.info(f"开始迁移集合: {collection_name}")
        logger.info(f"{'='*60}")

        # 获取源集合文档总数
        stats["source_count"] = await source_coll.count_documents({})
        logger.info(f"源集合文档总数: {stats['source_count']}")

        # 可选：清空目标集合
        if drop_target:
            logger.warning(f"清空目标集合: {collection_name}")
            await target_coll.delete_many({})
            logger.info(f"目标集合已清空")

        # 检查目标集合当前文档数
        target_count = await target_coll.count_documents({})
        if target_count > 0:
            logger.info(f"目标集合当前文档数: {target_count}")

        # 批量迁移数据
        skip = 0
        batch_num = 0

        while True:
            # 从源数据库读取一批数据
            cursor = source_coll.find().skip(skip).limit(self.batch_size)
            batch = await cursor.to_list(length=self.batch_size)

            if not batch:
                break

            batch_num += 1
            logger.info(f"处理批次 {batch_num}: {len(batch)} 个文档 (进度: {skip + len(batch)}/{stats['source_count']})")

            # 准备批量写入操作
            to_insert = []
            to_update = []

            for doc in batch:
                doc_id = doc.get("_id")
                if skip_existing and doc_id:
                    # 检查文档是否已存在
                    existing = await target_coll.find_one({"_id": doc_id})
                    if existing:
                        to_update.append(doc)
                        continue

                to_insert.append(doc)

            # 执行批量插入
            if to_insert:
                try:
                    await target_coll.insert_many(to_insert, ordered=False)
                    stats["migrated_count"] += len(to_insert)
                    logger.info(f"  插入 {len(to_insert)} 个新文档")
                except Exception as e:
                    error_str = str(e)
                    # 检查是否是重复键错误（可能是唯一索引冲突）
                    is_dup_error = "11000" in error_str or "duplicate key" in error_str

                    if is_dup_error:
                        logger.warning(f"  批量插入遇到重复键，尝试逐个插入以识别冲突文档")
                    else:
                        logger.error(f"  批量插入失败: {error_str}")

                    # 如果批量失败，尝试逐个插入
                    for doc in to_insert:
                        try:
                            await target_coll.insert_one(doc)
                            stats["migrated_count"] += 1
                        except Exception as doc_err:
                            doc_error_str = str(doc_err)
                            # 重复键错误视为已存在，而不是错误
                            if "11000" in doc_error_str or "duplicate key" in doc_error_str:
                                # 提取冲突的字段信息
                                stats["skipped_count"] += 1
                                logger.debug(f"    跳过重复文档: {doc_error_str[:100]}")
                            else:
                                stats["error_count"] += 1
                                logger.error(f"    文档插入失败: {doc_error_str[:200]}")

            # 更新已存在的文档（可选）
            if to_update:
                stats["skipped_count"] += len(to_update)
                logger.info(f"  跳过 {len(to_update)} 个已存在文档")

            skip += len(batch)

        stats["end_time"] = datetime.now()
        duration = (stats["end_time"] - stats["start_time"]).total_seconds()

        logger.info(f"{'='*60}")
        logger.info(f"集合 '{collection_name}' 迁移完成:")
        logger.info(f"  源文档总数: {stats['source_count']}")
        logger.info(f"  已迁移: {stats['migrated_count']}")
        logger.info(f"  跳过: {stats['skipped_count']}")
        logger.info(f"  错误: {stats['error_count']}")
        logger.info(f"  耗时: {duration:.2f} 秒")
        logger.info(f"{'='*60}")

        return stats

    async def migrate_all_collections(
        self,
        exclude: Optional[List[str]] = None,
        skip_existing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        迁移所有集合

        Args:
            exclude: 要排除的集合名称列表
            skip_existing: 是否跳过已存在的文档

        Returns:
            所有集合的迁移统计信息
        """
        exclude = exclude or []

        # 获取所有集合
        collections = await self.list_collections()

        # 过滤排除的集合
        collections_to_migrate = [c for c in collections if c not in exclude]

        logger.info(f"准备迁移 {len(collections_to_migrate)} 个集合")
        if exclude:
            logger.info(f"排除集合: {', '.join(exclude)}")

        # 迁移每个集合
        all_stats = []
        for collection_name in collections_to_migrate:
            try:
                stats = await self.migrate_collection(
                    collection_name,
                    skip_existing=skip_existing
                )
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"迁移集合 '{collection_name}' 失败: {str(e)}", exc_info=True)
                all_stats.append({
                    "collection": collection_name,
                    "error": str(e),
                    "migrated_count": 0
                })

        # 打印汇总
        self._print_summary(all_stats)

        return all_stats

    def _print_summary(self, stats_list: List[Dict[str, Any]]):
        """打印迁移汇总信息"""
        logger.info(f"\n{'='*60}")
        logger.info("迁移汇总:")
        logger.info(f"{'='*60}")

        total_source = sum(s.get("source_count", 0) for s in stats_list)
        total_migrated = sum(s.get("migrated_count", 0) for s in stats_list)
        total_skipped = sum(s.get("skipped_count", 0) for s in stats_list)
        total_errors = sum(s.get("error_count", 0) for s in stats_list)

        for stats in stats_list:
            collection = stats.get("collection", "unknown")
            migrated = stats.get("migrated_count", 0)
            source = stats.get("source_count", 0)
            error = stats.get("error", None)

            if error:
                logger.info(f"  ❌ {collection}: 失败 - {error}")
            else:
                logger.info(f"  ✅ {collection}: {migrated}/{source} 个文档")

        logger.info(f"\n总计:")
        logger.info(f"  源文档总数: {total_source}")
        logger.info(f"  已迁移: {total_migrated}")
        logger.info(f"  跳过: {total_skipped}")
        logger.info(f"  错误: {total_errors}")
        logger.info(f"{'='*60}\n")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MongoDB 数据迁移脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--source-uri",
        default=os.getenv("SOURCE_URI", DEFAULT_SOURCE_URI),
        help="源数据库 URI（默认从环境变量 SOURCE_URI 读取）"
    )

    parser.add_argument(
        "--target-uri",
        default=os.getenv("TARGET_URI", DEFAULT_TARGET_URI),
        help="目标数据库 URI（默认从环境变量 TARGET_URI 读取）"
    )

    parser.add_argument(
        "--collection", "-c",
        action="append",
        dest="collections",
        help="要迁移的集合名称（可多次使用）"
    )

    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="迁移所有集合"
    )

    parser.add_argument(
        "--exclude", "-e",
        action="append",
        dest="exclude",
        help="要排除的集合名称（可多次使用，仅用于 --all）"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出源数据库的所有集合"
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=int(os.getenv("BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        help=f"批量写入大小（默认: {DEFAULT_BATCH_SIZE}）"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制迁移前清空目标集合（危险操作！）"
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="不跳过已存在的文档（会覆盖或报错）"
    )

    args = parser.parse_args()

    # 验证参数
    if not args.all and not args.collections and not args.list:
        parser.print_help()
        print("\n错误: 请指定 --collection、--all 或 --list")
        sys.exit(1)

    # 创建迁移器
    migrator = MongoDBMigrator(
        source_uri=args.source_uri,
        target_uri=args.target_uri,
        batch_size=args.batch_size
    )

    try:
        # 连接数据库
        await migrator.connect()

        # 执行操作
        if args.list:
            # 仅列出集合
            await migrator.list_collections()
        elif args.all:
            # 迁移所有集合
            skip_existing = not args.no_skip_existing
            await migrator.migrate_all_collections(
                exclude=args.exclude,
                skip_existing=skip_existing
            )
        else:
            # 迁移指定集合
            skip_existing = not args.no_skip_existing
            all_stats = []
            for collection_name in args.collections:
                stats = await migrator.migrate_collection(
                    collection_name,
                    skip_existing=skip_existing,
                    drop_target=args.force
                )
                all_stats.append(stats)

            if len(args.collections) > 1:
                migrator._print_summary(all_stats)

    except Exception as e:
        logger.error(f"迁移失败: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        await migrator.close()


if __name__ == "__main__":
    asyncio.run(main())
