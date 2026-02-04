from datetime import datetime

from fastapi import (
    APIRouter, Depends, HTTPException
)
from starlette import status

from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger
from app.models.schemas import ResponseResult, DatasetFaqRequest
from app.services.task_processor import task_processor

logger = get_logger(__name__)

router = APIRouter()


@router.post("/update")
async def update_faq(
    request: DatasetFaqRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        FAQ更新

        Args:
            - request: FAQ请求参数对象
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
    """
    faq_id = request.faq_id
    try:
        logger.info(f"update_faq request faq_id={faq_id} employee_ids={request.employee_ids}")

        # Build combined_text for embedding (question + similar questions)
        combined_text = request.question_name
        if request.similar_questions:
            combined_text += " " + " ".join(request.similar_questions)

        answer_texts = [ans for ans in request.answers]

        for employee_id in request.employee_ids:
            try:
                update_faq_id = f"faq_{employee_id}_{faq_id}"
                update_data = {
                    "faq_id": update_faq_id,
                    "employee_id": employee_id,
                    "external_faq_id": faq_id,
                    "question_name": request.question_name,
                    "start_time": request.start_time,
                    "end_time": request.end_time,
                    "is_enable": request.is_enable,
                    "is_clear": request.is_clear,
                    "similar_questions": request.similar_questions,
                    "answers": answer_texts,
                    "update_time": request.update_time,
                    "combined_text": combined_text,
                    "keywords": [],  # Will be populated by vectorization task
                    "vector_id": None,  # Will be set after vectorization
                    "es_indexed": False,  # Will be set after ElasticSearch indexing
                    "created_at": datetime.now(),
                    "synced_at": datetime.now(),
                }
                await db.faqs.update_one(
                    {"faq_id": update_faq_id},
                    {"$set": update_data},
                    upsert=True
                )

                await task_processor.submit_faq_vectorization_task(faq_id=update_faq_id)
            except Exception as e:
                logger.error(f"update_faq faq_id={faq_id} error={str(e)}", exc_info=True)

        return ResponseResult.success(None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_faq exception faq_id={faq_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="update_faq error"
        )


@router.delete("/delete")
async def delete_faq(
    faq_id: int,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        删除FAQ

        Args:
            - faq_id: FAQ问答id
            - employee_id: 数字员工id
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
    """
    try:
        logger.info(f"delete_faq request faq_id={faq_id}")

        # 删除数据
        result = await db.faqs.delete_one({'external_faq_id': {faq_id}})

        if result and result.deleted_count == 1:
            logger.info(f"delete_faq success faq_id={faq_id}")

            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"delete_faq not found faq_id={faq_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_faq exception faq_id={faq_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="delete_faq error"
        )
