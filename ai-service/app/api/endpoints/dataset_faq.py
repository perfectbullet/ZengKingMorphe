"""
    FAQ问答接口服务
"""
from fastapi import (
    APIRouter, Depends, HTTPException
)
from starlette import status

from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger
from app.models.database import FAQModel
from app.models.schemas import ResponseResult, DatasetFaqRequest
from app.services.dataset_faq_service import faq_processor
from app.services.task_processor import task_processor

logger = get_logger(__name__)

router = APIRouter()


@router.put("/update")
async def update_faq(
    request: DatasetFaqRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        FAQ更新

        \nArgs:
            \n- request: FAQ请求参数对象
            \n- api_key: API key from auth
            \n- db: Database instance

        Returns:
            \n- ResponseResult
    """
    faq_id = request.faq_id
    try:
        logger.info(f"update_faq request: faq_id={faq_id} employee_ids={request.employee_ids}")

        # Build combined_text for embedding (question + similar questions)
        combined_text = request.question_name
        if request.similar_questions:
            combined_text += " " + " ".join(request.similar_questions)

        for employee_id in request.employee_ids:
            try:
                update_faq_id = f"faq_{employee_id}_{faq_id}"
                update_data = FAQModel(
                    faq_id=update_faq_id,
                    employee_id=str(employee_id),
                    external_faq_id=faq_id,
                    question_name=request.question_name,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    is_enable=request.is_enable,
                    is_clear=request.is_clear,
                    similar_questions=request.similar_questions,
                    answers=request.answers,
                    update_time=request.update_time,
                    combined_text=combined_text,
                    keywords=request.similar_questions,  # Use similar questions as keywords
                    vector_id=update_faq_id,  # Use faq_id as vector_id
                    es_indexed=False,  # Will be set to True after ES indexing
                )
                await db.faqs.update_one(
                    {"faq_id": update_faq_id},
                    {"$set": update_data.model_dump()},
                    upsert=True
                )

                await task_processor.submit_faq_vectorization_task(faq_id=update_faq_id, kb_id=f"faq_{faq_id}")
            except Exception as e:
                logger.error(f"update_faq failed: faq_id={faq_id} error={str(e)}", exc_info=True)

        return ResponseResult.success(None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_faq exception: faq_id={faq_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="update_faq error"
        )


@router.delete("/delete/{faq_id}")
async def delete_faq(
    faq_id: int,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        FAQ删除

        \nArgs:
            \n- faq_id: FAQ问答id
            \n- api_key: API key from auth
            \n- db: Database instance

        \nReturns:
            - ResponseResult
    """
    try:
        logger.info(f"delete_faq request: faq_id={faq_id}")

        faq_cursor = db.faqs.find({"external_faq_id": faq_id}, {"faq_id": 1})
        faqs = await faq_cursor.to_list(length=None)
        faq_ids = [c["faq_id"] for c in faqs]

        # 刪除ChromaDB记录和ElasticSearch记录
        for faq_id in faq_ids:
            await faq_processor.delete_faq_vectorization_data(faq_id)

        # 刪除faq任务记录
        await db.document_tasks.delete_many({"kb_id": f"faq_{faq_id}"})

        # 删除faq记录
        result = await db.faqs.delete_many({"external_faq_id": faq_id})

        if result and result.deleted_count == 1:
            logger.info(f"delete_faq success: faq_id={faq_id}")

            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"delete_faq failed: faq_id={faq_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_faq exception: faq_id={faq_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="delete_faq error"
        )
