"""
FAQ processing service.
"""

from datetime import datetime
from typing import Any, List

from app.core.logging import get_logger
from app.core.database import get_database
from app.utils.embeddings import get_embedding
# from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db

logger = get_logger(__name__)


class FaqProcessor:

    async def faq_vectorization(self, task_id: str, faq_id: str):
        """
            处理流程：
            1. 从MongoDB的faqs集合按employee_id查询FAQ数据
            2. 比较update_time和synced_at判断是否需要更新
            3. 使用combined_text生成向量（已在sync时生成）
            4. 调用Embedding API生成向量
            5. 写入ChromaDB（向量存储）
            6. 写入ElasticSearch（关键词索引）
            7. 更新MongoDB的synced_at时间戳
        """
        db = await get_database()
        try:
            faq = await db.faqs.find_one({"faq_id": faq_id})
            if not faq:
                logger.warning(f"vectorization_faq not found: task_id={task_id}")
                return

            await self._faq_vectorization(faq)

            return faq_id
        except Exception as e:
            logger.error(f"vectorization_faq exception: task_id={task_id} error={str(e)}", exc_info=True)
            raise

    async def _faq_vectorization(self, faq: Any):
        db = await get_database()
        faq_id = faq["faq_id"]
        logger.info(f"_faq_vectorization request: faq_id={faq_id}")
        try:
            # 1、faq不启用则删除向量数据库记录和ES记录
            if faq.get("is_enable", 0) == 0:
                await self.delete_faq_vectorization_data(faq_id)

                # 更新MongoDB数据的记录状态和更新时间
                await db.faqs.update_one(
                    {"faq_id": faq_id},
                    {"$set": {
                        "vector_id": None,
                        "es_indexed": False,
                        "synced_at": datetime.now()
                    }}
                )

            # 2、比较update_time和synced_at的大小，判断是否需要向量化数据
            update_time = faq.get("update_time", "")
            synced_at = faq.get("synced_at")

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
                logger.info(f"vectorization_faq_one faq_id={faq_id} has empty update_time, "
                            f"setting to current time: {update_time}")
                needs_update = True
            elif not needs_update and synced_at_str and update_time <= synced_at_str:
                # Skip if already synced and not updated
                logger.info(f"_faq_vectorization faq_id={faq_id} already synced (update_time: {update_time}, "
                            f"synced_at: {synced_at_str}), synced_at: {synced_at}")
            else:
                needs_update = True

            # Mark as update if synced_at exists
            if synced_at and needs_update:
                logger.info(f"_faq_vectorization faq_id={faq_id} needs update (update_time: {update_time}, "
                            f"synced_at: {synced_at_str})")

            # 数据向量化并保存
            await self.create_faq_vectorization_data(faq, update_time)

            # 7、更新MongoDB数据的记录状态并更新synced_at和update_time的时间值
            await db.faqs.update_one(
                {"faq_id": faq_id},
                {"$set": {
                    "vector_id": faq_id,
                    "es_indexed": True,
                    "update_time": update_time,
                    "synced_at": datetime.now()
                }}
            )
        except Exception as e:
            logger.error(f"_faq_vectorization exception: faq_id={faq_id} error={str(e)}", exc_info=True)
            raise

    async def delete_faq_vectorization_data(self, faq_id: str):
        logger.info(f"delete_faq_vectorization_data is disabled and delete chroma_db: faq_id={faq_id}", exc_info=True)

        # 删除ChromaDB记录
        try:
            chroma_db.faq_collection.delete(ids=[faq_id])
            logger.debug(f"delete_faq_vectorization_data Deleted FAQ faq_id={faq_id} from ChromaDB")
        except Exception as e:
            logger.warning(f"delete_faq_vectorization_data ChromaDB delete failed: faq_id={faq_id} "
                           f"error={str(e)}", exc_info=True)
            raise

        # 删除ElasticSearch记录
        try:
            await es_db.delete_document(
                index="faq",
                doc_id=faq_id
            )
            logger.debug(f"delete_faq_vectorization_data Deleted FAQ faq_id={faq_id} from ElasticSearch")
        except Exception as e:
            logger.warning(f"delete_faq_vectorization_data ElasticSearch delete failed: faq_id={faq_id} "
                           f"error={str(e)}", exc_info=True)
            raise

    async def create_faq_vectorization_data(self, faq: Any, update_time: str):
        faq_id = faq["faq_id"]
        logger.info(f"create_faq_vectorization_data is disabled and delete chroma_db: faq_id={faq_id}", exc_info=True)

        # 3、提取要进行向量化的文本数据单元
        combined_text = faq.get("combined_text", "").strip()

        if not combined_text:
            logger.info(f"create_faq_vectorization_data faq_id={faq_id} has empty combined_text, skipping", exc_info=True)
            return

        # 4、根据配置创建 Embedding model 实例的工厂函数
        embedding_model = get_embedding()

        # 5、生成向量并保存到ChromaDB向量库中
        try:
            chroma_collection = chroma_db.faq_collection

            # 先删除已存在的向量数据
            if faq.get("vector_id"):
                try:
                    chroma_collection.delete(ids=[faq_id])
                except Exception as e:
                    logger.warning(f"create_faq_vectorization_data Failed to delete existing FAQ vector: faq_id= {faq_id} "
                                   f"error={str(e)}", exc_info=True)

            # 再添加向量数据
            chroma_collection.add(
                ids=[faq_id],
                embeddings=[embedding_model.embed_query(combined_text)],
                metadatas=[{
                    "employee_id": faq["employee_id"],
                    "faq_id": faq_id,
                    "question_name": faq.get("question_name", ""),
                    "combined_text": combined_text[:500]
                }],
                documents=[combined_text]
            )
            logger.info(f"create_faq_vectorization_data vectorized in ChromaDB: faq_id={faq_id}")
        except Exception as e:
            logger.error(f"create_faq_vectorization_data Failed to write FAQ to ChromaDB: faq_id={faq_id} "
                         f"error={str(e)}", exc_info=True)
            raise

        # 6: 生成ElasticSearch的查询关键字索引
        try:
            es_doc = {
                "faq_id": faq_id,
                "employee_id": faq["employee_id"],
                "question_name": faq.get("question_name", ""),
                "similar_questions": faq.get("similar_questions", []),
                "combined_text": combined_text,
                "answers": faq.get("answers", []),
                "is_enable": faq.get("is_enable", 1),
                "update_time": update_time
            }

            await es_db.index_document(
                index="faq",
                doc_id=faq_id,
                document=es_doc
            )

            logger.info(f"create_faq_vectorization_data indexed in ElasticSearch: faq_id={faq_id}")
        except Exception as e:
            logger.error(f"create_faq_vectorization_data Failed to write FAQ to ElasticSearch: faq_id={faq_id} "
                         f"error={str(e)}", exc_info=True)
            raise


# Global faq processor instance
faq_processor = FaqProcessor()
