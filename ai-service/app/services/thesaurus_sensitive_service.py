"""
ThesaurusSensitive processing service.
"""

from datetime import datetime
from typing import Any

from app.core.logging import get_logger
from app.core.database import get_database
from app.utils.embeddings import get_embedding
from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db

logger = get_logger(__name__)


class ThesaurusSensitiveProcessor:

    async def thesaurus_sensitive_vectorization(self, task_id: str, thesaurus_id: str):
        """
            处理流程：
            1. 从MongoDB的敏感词库集合按employee_id查询词库的词条数据
            2. 比较update_time和synced_at判断是否需要更新
            3. 使用combined_text生成向量（已在sync时生成）
            4. 调用Embedding API生成向量
            5. 写入ChromaDB（向量存储）
            6. 写入ElasticSearch（关键词索引）
            7. 更新MongoDB的synced_at时间戳
        """
        db = await get_database()
        try:
            thesaurus = await db.thesaurus_sensitive.find_one({"thesaurus_id": thesaurus_id})
            if not thesaurus:
                logger.warning(f"thesaurus_sensitive_vectorization not found: task_id={task_id}")
                return

            await self._thesaurus_vectorization(thesaurus)

            return thesaurus_id
        except Exception as e:
            logger.error(f"thesaurus_sensitive_vectorization exception: task_id={task_id} error={str(e)}", exc_info=True)
            raise

    async def _thesaurus_vectorization(self, thesaurus: Any):
        db = await get_database()
        thesaurus_id = thesaurus["thesaurus_id"]
        logger.info(f"_thesaurus_vectorization request: thesaurus_id={thesaurus_id}")
        try:
            # 1、词库不启用则删除向量数据库记录和ES记录
            if thesaurus.get("is_enable", 0) == 0:
                await self.delete_thesaurus_vectorization_data(thesaurus_id)

                # 更新MongoDB数据的记录状态和更新时间
                await db.thesaurus_sensitive.update_one(
                    {"thesaurus_id": thesaurus_id},
                    {"$set": {
                        "vector_id": None,
                        "es_indexed": False,
                        "synced_at": datetime.now()
                    }}
                )

            # 2、比较update_time和synced_at的大小，判断是否需要向量化数据
            update_time = thesaurus.get("update_time", "")
            synced_at = thesaurus.get("synced_at")

            # 是否需要向量化标志
            needs_update = False

            # 没有synced_at则表示从未同步过，必须更新
            if not synced_at:
                needs_update = True

            # Convert synced_at to string for comparison (ISO format)
            synced_at_str = ""
            if synced_at:
                if isinstance(synced_at, datetime):
                    synced_at_str = synced_at.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    synced_at_str = str(synced_at)

            # Handle empty update_time - treat as latest and set current timestamp
            if not update_time or update_time.strip() == "":
                # Empty update_time means it's a new/latest FAQ, process it
                update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"_thesaurus_vectorization thesaurus_id={thesaurus_id} has empty update_time, "
                            f"setting to current time: {update_time}")
                needs_update = True
            elif not needs_update and synced_at_str and update_time <= synced_at_str:
                # Skip if already synced and not updated
                logger.info(
                    f"_thesaurus_vectorization thesaurus_id={thesaurus_id} already synced (update_time: {update_time}, "
                    f"synced_at: {synced_at_str}), synced_at: {synced_at}")
            else:
                needs_update = True

            # Mark as update if synced_at exists
            if synced_at and needs_update:
                logger.info(
                    f"_thesaurus_vectorization thesaurus_id={thesaurus_id} needs update (update_time: {update_time}, "
                    f"synced_at: {synced_at_str})")

            # 数据向量化并保存
            await self.create_thesaurus_vectorization_data(thesaurus, update_time)

            # 7、更新MongoDB数据的记录状态并更新synced_at和update_time的时间值
            await db.thesaurus_sensitive.update_one(
                {"thesaurus_id": thesaurus_id},
                {"$set": {
                    "vector_id": thesaurus_id,
                    "es_indexed": True,
                    "update_time": update_time,
                    "synced_at": datetime.now()
                }}
            )
        except Exception as e:
            logger.error(f"_thesaurus_vectorization exception: thesaurus_id={thesaurus_id} error={str(e)}",
                         exc_info=True)
            raise

    async def delete_thesaurus_vectorization_data(self, thesaurus_id: str):
        logger.info(
            f"delete_thesaurus_vectorization_data is disabled and delete chroma_db: thesaurus_id={thesaurus_id}",
            exc_info=True)

        # 删除ChromaDB记录
        try:
            chroma_collection = chroma_db._get_collection("thesaurus_sensitive")
            chroma_collection.delete(ids=[thesaurus_id])
            logger.debug(f"delete_thesaurus_vectorization_data Deleted FAQ thesaurus_id={thesaurus_id} from ChromaDB")
        except Exception as e:
            logger.warning(f"delete_thesaurus_vectorization_data ChromaDB delete failed: thesaurus_id={thesaurus_id} "
                           f"error={str(e)}", exc_info=True)
            raise

        # 删除ElasticSearch记录
        try:
            await es_db.delete_document(
                index="thesaurus_sensitive",
                doc_id=thesaurus_id
            )
            logger.debug(
                f"delete_thesaurus_vectorization_data Deleted FAQ thesaurus_id={thesaurus_id} from ElasticSearch")
        except Exception as e:
            logger.warning(
                f"delete_thesaurus_vectorization_data ElasticSearch delete failed: thesaurus_id={thesaurus_id} "
                f"error={str(e)}", exc_info=True)
            raise

    async def create_thesaurus_vectorization_data(self, thesaurus: Any, update_time: str):
        thesaurus_id = thesaurus["thesaurus_id"]
        logger.info(
            f"create_thesaurus_vectorization_data is disabled and delete chroma_db: thesaurus_id={thesaurus_id}",
            exc_info=True)

        # 3、提取要进行向量化的文本数据单元
        combined_text = thesaurus.get("combined_text", "").strip()

        if not combined_text:
            logger.info(
                f"create_thesaurus_vectorization_data thesaurus_id={thesaurus_id} has empty combined_text, skipping",
                exc_info=True)
            return

        # 4、根据配置创建 Embedding model 实例的工厂函数
        embedding_model = get_embedding()

        # 5、生成向量并保存到ChromaDB向量库中
        try:
            chroma_collection = chroma_db._get_collection("thesaurus_sensitive")

            # 先删除已存在的向量数据
            if thesaurus.get("vector_id"):
                try:
                    chroma_collection.delete(ids=[thesaurus_id])
                except Exception as e:
                    logger.warning(
                        f"create_thesaurus_vectorization_data Failed to delete existing FAQ vector: thesaurus_id= {thesaurus_id} "
                        f"error={str(e)}", exc_info=True)

            # 再添加向量数据
            chroma_collection.add(
                ids=[thesaurus_id],
                embeddings=[embedding_model.embed_query(combined_text)],
                metadatas=[{
                    "employee_id": thesaurus["employee_id"],
                    "thesaurus_id": thesaurus_id,
                    "thesaurus_name": thesaurus.get("thesaurus_name", ""),
                    "word_name": thesaurus.get("word_name", ""),
                    "combined_text": combined_text[:500]
                }],
                documents=[combined_text]
            )
            logger.info(f"create_thesaurus_vectorization_data vectorized in ChromaDB: thesaurus_id={thesaurus_id}")
        except Exception as e:
            logger.error(
                f"create_thesaurus_vectorization_data Failed to write FAQ to ChromaDB: thesaurus_id={thesaurus_id} "
                f"error={str(e)}", exc_info=True)
            raise

        # 6: 生成ElasticSearch的查询关键字索引
        try:
            es_doc = {
                "thesaurus_id": thesaurus_id,
                "employee_id": thesaurus["employee_id"],
                "thesaurus_name": thesaurus.get("thesaurus_name", ""),
                "word_name": thesaurus.get("word_name", ""),
                "combined_text": combined_text,
                "is_enable": thesaurus.get("is_enable", 1),
                "update_time": update_time
            }

            await es_db.index_document(
                index="thesaurus_sensitive",
                doc_id=thesaurus_id,
                document=es_doc
            )

            logger.info(f"create_thesaurus_vectorization_data indexed in ElasticSearch: thesaurus_id={thesaurus_id}")
        except Exception as e:
            logger.error(f"create_thesaurus_vectorization_data Failed to write FAQ to ElasticSearch: thesaurus_id={thesaurus_id} "
                         f"error={str(e)}", exc_info=True)
            raise


# Global thesaurus_sensitive processor instance
thesaurus_sensitive_processor = ThesaurusSensitiveProcessor()
