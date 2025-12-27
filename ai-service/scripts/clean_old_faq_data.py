"""
清理 FAQ 旧数据脚本

由于字段类型从 int 改为 str，需要清理以下数据库中的历史数据：
1. ElasticSearch - FAQ 索引（如果 external_faq_id 为 integer 类型）
2. MongoDB - faqs 集合
3. ChromaDB - faqs 集合

使用方法：
    python ai-service/scripts/clean_old_faq_data.py [--dry-run] [--confirm]
    
参数：
    --dry-run: 只显示将要删除的数据，不实际执行删除
    --confirm: 确认执行删除操作（需要显式确认）
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import mongodb
from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


async def clean_mongodb_faqs(dry_run: bool = True):
    """清理 MongoDB 中的 FAQ 数据"""
    logger.info("=" * 80)
    logger.info("清理 MongoDB faqs 集合")
    logger.info("=" * 80)
    
    try:
        db = mongodb.get_client()[mongodb.db_name]
        
        # 统计记录数
        total_count = await db.faqs.count_documents({})
        logger.info(f"当前 faqs 集合中有 {total_count} 条记录")
        
        if total_count == 0:
            logger.info("✅ MongoDB faqs 集合为空，无需清理")
            return
        
        if dry_run:
            logger.warning(f"⚠️  [DRY RUN] 将删除 {total_count} 条 FAQ 记录")
            
            # 显示前 5 条示例
            sample_faqs = await db.faqs.find().limit(5).to_list(length=5)
            logger.info(f"前 5 条示例记录：")
            for faq in sample_faqs:
                logger.info(f"  - faq_id: {faq.get('faq_id')}, external_faq_id: {faq.get('external_faq_id')} (类型: {type(faq.get('external_faq_id')).__name__})")
        else:
            result = await db.faqs.delete_many({})
            logger.info(f"✅ 成功删除 {result.deleted_count} 条 FAQ 记录")
    
    except Exception as e:
        logger.error(f"❌ MongoDB 清理失败: {e}", exc_info=True)
        raise


def clean_chroma_faqs(dry_run: bool = True):
    """清理 ChromaDB 中的 FAQ 向量数据"""
    logger.info("=" * 80)
    logger.info("清理 ChromaDB faqs 集合")
    logger.info("=" * 80)
    
    try:
        # 尝试获取 faqs 集合
        try:
            collection = chroma_db.get_collection("faqs")
            total_count = collection.count()
            logger.info(f"当前 faqs 集合中有 {total_count} 个向量")
            
            if total_count == 0:
                logger.info("✅ ChromaDB faqs 集合为空，无需清理")
                return
            
            if dry_run:
                logger.warning(f"⚠️  [DRY RUN] 将删除整个 faqs 集合（{total_count} 个向量）")
                
                # 显示前 5 个示例
                results = collection.peek(limit=5)
                logger.info(f"前 5 个示例记录：")
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    logger.info(f"  - id: {doc_id}, metadata: {metadata}")
            else:
                # 删除整个集合
                chroma_db.client.delete_collection(name="faqs")
                logger.info(f"✅ 成功删除 faqs 集合（{total_count} 个向量）")
                
                # 重新创建空集合
                chroma_db.client.create_collection(
                    name="faqs",
                    metadata={"description": "FAQ vectors for semantic search"}
                )
                logger.info("✅ 重新创建空的 faqs 集合")
        
        except Exception as e:
            if "does not exist" in str(e).lower():
                logger.info("✅ ChromaDB faqs 集合不存在，无需清理")
            else:
                raise
    
    except Exception as e:
        logger.error(f"❌ ChromaDB 清理失败: {e}", exc_info=True)
        raise


async def clean_elasticsearch_faqs(dry_run: bool = True):
    """清理 ElasticSearch 中的 FAQ 索引"""
    logger.info("=" * 80)
    logger.info("清理 ElasticSearch FAQ 索引")
    logger.info("=" * 80)
    
    try:
        index_pattern = f"{settings.es_index_prefix}faqs*"
        
        # 获取所有匹配的索引
        indices = await es_db.client.cat.indices(index=index_pattern, format="json")
        
        if not indices:
            logger.info(f"✅ 没有找到匹配 {index_pattern} 的索引，无需清理")
            return
        
        logger.info(f"找到 {len(indices)} 个 FAQ 相关索引：")
        for idx in indices:
            index_name = idx['index']
            doc_count = idx['docs.count']
            logger.info(f"  - {index_name}: {doc_count} 条文档")
        
        if dry_run:
            logger.warning(f"⚠️  [DRY RUN] 将删除 {len(indices)} 个索引")
        else:
            for idx in indices:
                index_name = idx['index']
                await es_db.client.indices.delete(index=index_name)
                logger.info(f"✅ 成功删除索引: {index_name}")
            
            logger.info(f"✅ 总共删除 {len(indices)} 个 FAQ 索引")
    
    except Exception as e:
        if "index_not_found_exception" in str(e).lower():
            logger.info(f"✅ 没有找到 FAQ 索引，无需清理")
        else:
            logger.error(f"❌ ElasticSearch 清理失败: {e}", exc_info=True)
            raise


async def main():
    parser = argparse.ArgumentParser(description="清理 FAQ 旧数据")
    parser.add_argument("--dry-run", action="store_true", default=False, 
                        help="只显示将要删除的数据，不实际执行")
    parser.add_argument("--confirm", action="store_true", default=False,
                        help="确认执行删除操作")
    
    args = parser.parse_args()
    
    # 如果不是 dry-run 且未确认，拒绝执行
    if not args.dry_run and not args.confirm:
        logger.error("❌ 实际删除操作需要显式确认！请使用 --confirm 参数")
        logger.error("或者使用 --dry-run 查看将要删除的数据")
        sys.exit(1)
    
    logger.info("🔧 FAQ 数据清理工具")
    logger.info("=" * 80)
    if args.dry_run:
        logger.info("模式: DRY RUN（仅查看，不删除）")
    else:
        logger.info("模式: 实际删除")
        logger.warning("⚠️⚠️⚠️  警告：将删除所有 FAQ 数据，操作不可恢复！⚠️⚠️⚠️")
    logger.info("=" * 80)
    
    try:
        # 连接数据库
        logger.info("连接数据库...")
        await mongodb.connect()
        chroma_db.connect()
        await es_db.connect()
        logger.info("✅ 数据库连接成功")
        logger.info("")
        
        # 清理 MongoDB
        await clean_mongodb_faqs(dry_run=args.dry_run)
        logger.info("")
        
        # 清理 ChromaDB
        clean_chroma_faqs(dry_run=args.dry_run)
        logger.info("")
        
        # 清理 ElasticSearch
        await clean_elasticsearch_faqs(dry_run=args.dry_run)
        logger.info("")
        
        logger.info("=" * 80)
        if args.dry_run:
            logger.info("✅ DRY RUN 完成！使用 --confirm 参数执行实际删除")
        else:
            logger.info("✅✅✅ 所有 FAQ 数据清理完成！✅✅✅")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"❌❌❌ 清理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # 断开数据库连接
        await mongodb.disconnect()
        await es_db.disconnect()
        logger.info("数据库连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
