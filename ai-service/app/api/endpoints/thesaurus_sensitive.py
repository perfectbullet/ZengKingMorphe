"""
    敏感词库接口服务
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.database import get_database
from app.models.database import ThesaurusSensitiveModel
from app.models.schemas import ResponseResult, ThesaurusRequest
from app.api.middleware.auth import verify_api_key, get_api_key
from app.core.logging import get_logger
from app.services.task_processor import task_processor
from app.services.thesaurus_sensitive_service import thesaurus_sensitive_processor

logger = get_logger(__name__)

router = APIRouter()


@router.put("/update")
async def update_thesaurus_sensitive(
    request: ThesaurusRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        敏感词库更新
        一个词条保存一条记录

        \nArgs:
            \n- request: 词库请求参数对象
            \n- api_key: API key from auth
            \n- db: Database instance

        \nReturns:
            \n- ResponseResult
    """
    thesaurus_id = request.thesaurus_id
    try:
        logger.info(f"update_thesaurus_sensitive request: thesaurus_id={thesaurus_id} employee_ids={request.employee_ids}")

        for employee_id in request.employee_ids:
            for thesaurus_word in request.thesaurus_words:
                try:
                    # Build combined_text for embedding (thesaurus_name + word_name)
                    combined_text = request.thesaurus_name + " " + thesaurus_word.word_name

                    update_thesaurus_id = f"sensitive_{employee_id}_{thesaurus_id}_{thesaurus_word.word_id}"
                    update_data = ThesaurusSensitiveModel(
                        thesaurus_id=update_thesaurus_id,
                        employee_id=str(employee_id),
                        external_thesaurus_id=thesaurus_id,
                        external_word_id=thesaurus_word.word_id,
                        thesaurus_name=request.thesaurus_name,
                        is_enable=request.is_enable,
                        update_time=request.update_time,
                        word_name=thesaurus_word.word_name,
                        combined_text=combined_text,
                        keywords=thesaurus_word.word_name,
                        vector_id=update_thesaurus_id,  # Will be set after vectorization
                        es_indexed=False,  # Will be set after ElasticSearch indexing
                    ).model_dump()
                    await db.thesaurus_sensitive.update_one(
                        {"thesaurus_id": update_thesaurus_id},
                        {"$set": update_data},
                        upsert=True
                    )

                    await task_processor.submit_thesaurus_sensitive_vectorization_task(thesaurus_id=update_thesaurus_id, kb_id=f"sensitive_{thesaurus_id}")
                except Exception as e:
                    logger.error(f"update_thesaurus_sensitive failed: thesaurus_id={thesaurus_id} error={str(e)}", exc_info=True)

        return ResponseResult.success(None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_thesaurus_sensitive exception: thesaurus_id={thesaurus_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="update_thesaurus_sensitive error"
        )


@router.delete("/delete/{thesaurus_id}")
async def delete_thesaurus_sensitive(
    thesaurus_id: int,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        删除敏感词库

        \nArgs:
            \n- thesaurus_id: 敏感词库id
            \n- api_key: API key from auth
            \n- db: Database instance

        \nReturns:
            - ResponseResult
    """
    try:
        logger.info(f"delete_thesaurus_sensitive request: thesaurus_id={thesaurus_id}")

        thesaurus_sensitive_cursor = db.thesaurus_sensitive.find({"external_thesaurus_id": thesaurus_id}, {"thesaurus_id": 1})
        thesaurus_sensitives = await thesaurus_sensitive_cursor.to_list(length=None)
        thesaurus_ids = [c["thesaurus_id"] for c in thesaurus_sensitives]

        # 刪除ChromaDB记录和ElasticSearch记录
        for thesaurus_id in thesaurus_ids:
            await thesaurus_sensitive_processor.delete_thesaurus_vectorization_data(thesaurus_id)

        # 刪除专业词库任务记录
        await db.document_tasks.delete_many({"kb_id": f"sensitive_{thesaurus_id}"})

        # 删除专业词库记录
        result = await db.thesaurus_sensitive.delete_many({"external_thesaurus_id": thesaurus_id})

        if result:
            logger.info(f"delete_thesaurus_sensitive success: thesaurus_id={thesaurus_id}")

            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"delete_thesaurus_sensitive not found: thesaurus_id={thesaurus_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_thesaurus_sensitive exception: thesaurus_id={thesaurus_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="delete_thesaurus_sensitive error"
        )
