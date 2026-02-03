from fastapi import APIRouter, Depends, HTTPException, status

from app.core.database import get_database
from app.models.schemas import ThesaurusMajorRequest, ResponseResult
from app.api.middleware.auth import verify_api_key
from app.core.logging import get_logger
from app.services.task_processor import task_processor

logger = get_logger(__name__)

router = APIRouter()


@router.post("/update")
async def thesaurus_major_update(
    request: ThesaurusMajorRequest,
    api_key: str = Depends(verify_api_key),
    db=Depends(get_database)
):
    """
        接收通知：专业词库更新，因为数字员工新建时下载相关专业词库记录，所以只需要更新

        nArgs:
            - request: 专业词库请求参数对象
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
    """
    thesaurus_major_id = request.thesaurus_major_id
    try:
        logger.info(f"thesaurus_major_update request thesaurus_major_id={thesaurus_major_id}")

        thesaurus_major_list = await db.thesaurus_major.find({"external_thesaurus_major_id": thesaurus_major_id})
        thesaurus_majors = await thesaurus_major_list.to_list(length=None)

        if not thesaurus_majors:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"thesaurus_major_update not found thesaurus_major_id={thesaurus_major_id}")
        for idx, thesaurus_major in enumerate(thesaurus_majors):
            try:
                update_data = {
                    "thesaurus_name": request.thesaurus_name,
                    "update_time": request.update_time,
                    "thesaurus_major_words": request.thesaurus_major_words
                }
                await db.thesaurus_major.update_one(
                    {"thesaurus_major_id": thesaurus_major["thesaurus_major_id"]},
                    {"$set": update_data}
                )

                # await task_processor.submit_thesaurus_major_task(employee_id=faq["employee_id"])
            except Exception as e:
                logger.error(f"thesaurus_major_update thesaurus_major_id={thesaurus_major_id} idx={idx} error={str(e)}", exc_info=True)

        return ResponseResult.success(None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"thesaurus_major_update exception thesaurus_major_id={thesaurus_major_id} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="thesaurus_major_update error"
        )
