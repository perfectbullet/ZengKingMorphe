"""
FAQ processing service.
"""

from datetime import datetime
from app.core.logging import get_logger
from app.core.database import get_database
from app.utils.embeddings import get_embedding
from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db

logger = get_logger(__name__)


class FaqProcessor:

    async def vectorization_faq(self, task_id: str, faq_id: str) -> str:
        db = await get_database()

        try:
            faq = db.faqs.find({"faq_id": faq_id})
            if not faq:
                logger.warning(f"vectorization_faq not found task_id={task_id} faq_id={faq_id}")
                return faq_id

            try:
                # 1、faq不启用则删除向量数据库记录和ES记录
                if faq.get("is_enable", 0) == 0:
                    logger.info(f"vectorization_faq is disabled and delete chroma_db faq_id={faq_id}")

                    # 删除ChromaDB记录
                    try:
                        chroma_collection = chroma_db._get_collection("faq")
                        chroma_collection.delete(ids=[faq_id])
                        logger.debug(f"vectorization_faq Deleted FAQ faq_id={faq_id} from ChromaDB")
                    except Exception as e:
                        logger.warning(f"vectorization_faq ChromaDB delete failed faq_id={faq_id} error={str(e)}")

                    # 删除ElasticSearch记录
                    try:
                        await es_db.delete_document(
                            index="faq",
                            doc_id=faq_id
                        )
                        logger.debug(f"vectorization_faq Deleted FAQ faq_id={faq_id} from ElasticSearch")
                    except Exception as e:
                        logger.warning(f"vectorization_faq ElasticSearch delete failed faq_id={faq_id} error={str(e)}")

                    # 更新MongoDB数据的记录状态和更新时间
                    await db.faqs.update_one(
                        {"faq_id": faq_id},
                        {"$set": {
                            "vector_id": None,
                            "es_indexed": False,
                            "synced_at": datetime.utcnow()
                        }}
                    )
                    return faq_id

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
                        synced_at_str = synced_at.isoformat()
                    else:
                        synced_at_str = str(synced_at)

                # Handle empty update_time - treat as latest and set current timestamp
                if not update_time or update_time.strip() == "":
                    # Empty update_time means it's a new/latest FAQ, process it
                    update_time = datetime.utcnow().isoformat() + "Z"
                    logger.info(f"vectorization_faq faq_id={faq_id} has empty update_time, setting to current time: {update_time}")
                    needs_update = True
                elif not needs_update and synced_at_str and update_time <= synced_at_str:
                    # Skip if already synced and not updated
                    logger.info(f"vectorization_faq faq_id={faq_id} already synced (update_time: {update_time}, synced_at: {synced_at_str}), synced_at: {synced_at}")
                else:
                    needs_update = True

                # Mark as update if synced_at exists
                if synced_at and needs_update:
                    logger.info(f"vectorization_faq faq_id={faq_id} needs update (update_time: {update_time}, synced_at: {synced_at_str})")

                # 3、提取要进行向量化的文本数据单元
                combined_text = faq.get("combined_text", "").strip()

                if not combined_text:
                    logger.warning(f"vectorization_faq faq_id={faq_id} has empty combined_text, skipping")
                    return faq_id

                # 4、根据配置创建 Embedding model 实例的工厂函数
                embedding_model = get_embedding()

                # 5、生成向量并保存到ChromaDB向量库中
                try:
                    chroma_collection = chroma_db._get_collection("faq")

                    # 先删除已存在的向量数据
                    if faq.get("vector_id"):
                        try:
                            chroma_collection.delete(ids=[faq_id])
                        except Exception as e:
                            logger.warning(f"vectorization_faq Failed to delete existing FAQ vector faq_id= {faq_id} error={str(e)}")

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
                    logger.info(f"vectorization_faq vectorized in ChromaDB faq_id={faq_id}")
                except Exception as e:
                    logger.error(f"vectorization_faq Failed to write FAQ to ChromaDB faq_id={faq_id} error={str(e)}", exc_info=True)
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

                    logger.info(f"vectorization_faq indexed in ElasticSearch faq_id={faq_id}")
                except Exception as e:
                    logger.error(f"vectorization_faq Failed to write FAQ to ElasticSearch faq_id={faq_id} error={str(e)}", exc_info=True)

                # 7、更新MongoDB数据的记录状态并更新synced_at和update_time的时间值
                await db.faqs.update_one(
                    {"faq_id": faq_id},
                    {"$set": {
                        "vector_id": faq_id,
                        "es_indexed": True,
                        "update_time": update_time,
                        "synced_at": datetime.utcnow()
                    }}
                )

            except Exception as e:
                logger.error(f"vectorization_faq exception faq_id={faq_id} error={str(e)}", exc_info=True)

        except Exception as e:
            logger.error(f"vectorization_faq exception task_id={task_id} faq_id={faq_id} error={str(e)}", exc_info=True)
            raise

    async def vectorization_faq_by_employee_id(self, task_id: str, employee_id: str) -> str:
        """
            执行FAQ向量化任务（包含增量更新逻辑）。

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
            # Step 1: Fetch FAQs from MongoDB by employee_id
            faqs_cursor = db.faqs.find({"employee_id": employee_id})
            faqs = await faqs_cursor.to_list(length=None)

            if not faqs:
                logger.warning(f"No FAQs found for employee_id={employee_id}, task_id={task_id}")
                return ""

            logger.info(f"Starting FAQ vectorization: task_id={task_id}, employee_id={employee_id}, total_faqs={len(faqs)}")

            vectorized_count = 0
            skipped_count = 0
            updated_count = 0
            disabled_count = 0

            for idx, faq_doc in enumerate(faqs):
                try:
                    faq_id = faq_doc["faq_id"]

                    # Handle disabled FAQs - delete from vector stores
                    if faq_doc.get("is_enable", 0) != 1:
                        logger.info(f"FAQ {faq_id} is disabled (is_enable=0), cleaning up vector stores")
                        deleted_count = 0

                        # Delete from ChromaDB
                        try:
                            chroma_collection = chroma_db._get_collection("faq")
                            chroma_collection.delete(ids=[faq_id])
                            deleted_count += 1
                            logger.debug(f"Deleted FAQ {faq_id} from ChromaDB")
                        except Exception as e:
                            logger.warning(f"ChromaDB delete failed for {faq_id}: {e}")

                        # Delete from ElasticSearch
                        try:
                            await es_db.delete_document(
                                index="faq",
                                doc_id=faq_id
                            )
                            deleted_count += 1
                            logger.debug(f"Deleted FAQ {faq_id} from ElasticSearch")
                        except Exception as e:
                            logger.warning(f"ES delete failed for {faq_id}: {e}")

                        # Update MongoDB status (mark as cleaned)
                        await db.faqs.update_one(
                            {"faq_id": faq_id},
                            {"$set": {
                                "es_indexed": False,
                                "vector_id": None,
                                "synced_at": datetime.utcnow()
                            }}
                        )

                        disabled_count += 1
                        logger.info(f"FAQ {faq_id} cleanup completed: {deleted_count}/2 vector stores cleaned")
                        continue

                    # Step 2: Check if vectorization is needed (compare update_time vs synced_at)
                    update_time = faq_doc.get("update_time", "")
                    synced_at = faq_doc.get("synced_at")

                    # Initialize needs_update flag
                    needs_update = False

                    # 没有synced_at则表示从未同步过，必须更新
                    if not synced_at:
                        needs_update = True

                    # Convert synced_at to string for comparison (ISO format)
                    synced_at_str = ""
                    if synced_at:
                        if isinstance(synced_at, datetime):
                            synced_at_str = synced_at.isoformat()
                        else:
                            synced_at_str = str(synced_at)

                    # Handle empty update_time - treat as latest and set current timestamp
                    if not update_time or update_time.strip() == "":
                        # Empty update_time means it's a new/latest FAQ, process it
                        update_time = datetime.utcnow().isoformat() + "Z"
                        logger.info(f"FAQ {faq_id} has empty update_time, setting to current time: {update_time}")
                        needs_update = True
                    elif not needs_update and synced_at_str and update_time <= synced_at_str:
                        # Skip if already synced and not updated
                        # skipped_count += 1
                        logger.info(f"FAQ {faq_id} already synced (update_time: {update_time}, synced_at: {synced_at_str}), synced_at: {synced_at}")
                        # continue
                    else:
                        needs_update = True

                    # Mark as update if synced_at exists
                    if synced_at and needs_update:
                        updated_count += 1
                        logger.info(f"FAQ {faq_id} needs update (update_time: {update_time}, synced_at: {synced_at_str})")

                    # Step 3: Extract fields from MongoDB document
                    combined_text = faq_doc.get("combined_text", "").strip()

                    if not combined_text:
                        skipped_count += 1
                        logger.warning(f"FAQ {faq_id} has empty combined_text, skipping")
                        continue

                    # Step 4: 根据配置创建 Embedding model 实例的工厂函数。
                    embedding_model = get_embedding()

                    # Step 5: Write to ChromaDB (vector storage)
                    try:
                        chroma_collection = chroma_db._get_collection("faq")

                        # Delete existing vector if updating
                        if faq_doc.get("vector_id"):
                            try:
                                chroma_collection.delete(ids=[faq_id])
                            except Exception as e:
                                logger.warning(f"Failed to delete existing FAQ vector {faq_id}: {e}")

                        # Add new vector
                        chroma_collection.add(
                            ids=[faq_id],
                            embeddings=[embedding_model.embed_query(combined_text)],
                            metadatas=[{
                                "employee_id": employee_id,
                                "faq_id": faq_id,
                                "question_name": faq_doc.get("question_name", ""),
                                "combined_text": combined_text[:500]
                            }],
                            documents=[combined_text]
                        )
                        logger.info(f"FAQ {faq_id} vectorized in ChromaDB")
                    except Exception as e:
                        logger.error(f"Failed to write FAQ {faq_id} to ChromaDB: {e}", exc_info=True)
                        raise

                    # Step 6: Write to ElasticSearch (keyword index)
                    try:
                        es_doc = {
                            "faq_id": faq_id,
                            "employee_id": employee_id,
                            "question_name": faq_doc.get("question_name", ""),
                            "similar_questions": faq_doc.get("similar_questions", []),
                            "combined_text": combined_text,
                            "answers": faq_doc.get("answers", []),
                            "is_enable": faq_doc.get("is_enable", 1),
                            "update_time": update_time
                        }

                        await es_db.index_document(
                            index="faq",
                            doc_id=faq_id,
                            document=es_doc
                        )

                        logger.info(f"FAQ {faq_id} indexed in ElasticSearch")
                    except Exception as e:
                        logger.error(f"Failed to write FAQ {faq_id} to ElasticSearch: {e}", exc_info=True)
                        # Continue even if ES fails

                    # Step 7: Update MongoDB synced_at timestamp and update_time
                    await db.faqs.update_one(
                        {"faq_id": faq_id},
                        {"$set": {
                            "vector_id": faq_id,
                            "es_indexed": True,
                            "update_time": update_time,
                            "synced_at": datetime.utcnow()
                        }}
                    )

                    vectorized_count += 1
                    logger.info(f"FAQ {faq_id} synced successfully ({vectorized_count}/{len(faqs)})")

                except Exception as e:
                    logger.error(f"Failed to process FAQ {idx} (faq_id={faq_doc.get('faq_id', 'unknown')}): {e}", exc_info=True)
                    # Continue processing other FAQs

            logger.info(
                f"FAQ vectorization completed: task_id={task_id}, employee_id={employee_id}, "
                f"total={len(faqs)}, vectorized={vectorized_count}, updated={updated_count}, "
                f"disabled={disabled_count}, skipped={skipped_count}"
            )

            return employee_id

        except Exception as e:
            logger.error(f"FAQ vectorization task failed: task_id={task_id}, error={str(e)}", exc_info=True)
            raise


# Global faq processor instance
faq_processor = FaqProcessor()
