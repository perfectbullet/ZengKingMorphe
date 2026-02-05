from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.database import get_database
from app.models.schemas import ThesaurusRequest, ResponseResult
from app.api.middleware.auth import verify_api_key, get_api_key
from app.core.logging import get_logger
from app.services.thesaurus_major_service import thesaurus_major_processor
from app.services.task_processor import task_processor

logger = get_logger(__name__)

router = APIRouter()


@router.post("/update")
async def update_thesaurus_major(
    request: ThesaurusRequest,
    api_key: str = Depends(verify_api_key),
    db=Depends(get_database)
):
    """
        专业词库更新
        一个词条保存一条记录

        Args:
            - request: 词库请求参数对象
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
    """
    thesaurus_id = request.thesaurus_id
    try:
        logger.info(f"update_thesaurus_major request: thesaurus_id={thesaurus_id} employee_ids={request.employee_ids}")

        for employee_id in request.employee_ids:
            for thesaurus_word in request.thesaurus_words:
                try:
                    # Build combined_text for embedding (thesaurus_name + word_name + similar_word_name)
                    combined_text = request.thesaurus_name + " " + thesaurus_word.word_name
                    if thesaurus_word.similar_words:
                        combined_text += " " + " ".join(thesaurus_word.similar_words)

                    update_thesaurus_id = f"major_{employee_id}_{thesaurus_id}_{thesaurus_word.word_id}"
                    update_data = {
                        "thesaurus_id": update_thesaurus_id,
                        "employee_id": employee_id,
                        "external_thesaurus_id": thesaurus_id,
                        "external_word_id": thesaurus_word.word_id,
                        "thesaurus_name": request.thesaurus_name,
                        "is_enable": request.is_enable,
                        "update_time": request.update_time,
                        "word_name": thesaurus_word.word_name,
                        "similar_words": thesaurus_word.similar_words,
                        "combined_text": combined_text,
                        "keywords": [],  # Will be populated by vectorization task
                        "vector_id": None,  # Will be set after vectorization
                        "es_indexed": False,  # Will be set after ElasticSearch indexing
                        "created_at": datetime.now(),
                        "synced_at": datetime.now()
                    }
                    await db.thesaurus_major.update_one(
                        {"thesaurus_id": update_thesaurus_id},
                        {"$set": update_data},
                        upsert=True
                    )

                    await task_processor.submit_thesaurus_major_vectorization_task(thesaurus_id=thesaurus_id, kb_id=f"major_{thesaurus_id}")
                except Exception as e:
                    logger.error(f"update_thesaurus_major failed: thesaurus_id={thesaurus_id} error={str(e)}", exc_info=True)

        return ResponseResult.success(None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_thesaurus_major exception: thesaurus_id={thesaurus_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="update_thesaurus_major error"
        )


@router.delete("/delete/{thesaurus_id}")
async def delete_thesaurus_major(
    thesaurus_id: int,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        删除专业词库

        Args:
            - thesaurus_id: 专业词库id
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
    """
    try:
        logger.info(f"delete_thesaurus_major request: thesaurus_id={thesaurus_id}")

        thesaurus_major_cursor = db.thesaurus_major.find({"external_thesaurus_id": thesaurus_id}, {"thesaurus_id": 1})
        thesaurus_majors = await thesaurus_major_cursor.to_list(length=None)
        thesaurus_ids = [c["thesaurus_id"] for c in thesaurus_majors]

        # 刪除ChromaDB记录和ElasticSearch记录
        for thesaurus_id in thesaurus_ids:
            await thesaurus_major_processor.delete_thesaurus_vectorization_data(thesaurus_id)

        # 刪除专业词库任务记录
        await db.document_tasks.delete_many({"kb_id": f"major_{thesaurus_id}"})

        # 删除专业词库记录
        result = await db.thesaurus_major.delete_many({"external_thesaurus_id": thesaurus_id})

        if result:
            logger.info(f"delete_thesaurus_major success: thesaurus_id={thesaurus_id}")

            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"delete_thesaurus_major not found: thesaurus_id={thesaurus_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_thesaurus_major exception: thesaurus_id={thesaurus_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="delete_thesaurus_major error"
        )
