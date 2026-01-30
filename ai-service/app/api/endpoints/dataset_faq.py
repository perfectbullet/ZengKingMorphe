from fastapi import (
    APIRouter, Depends, HTTPException
)
from starlette import status

from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger
from app.models.schemas import ResponseResult, SyncNotifyRequest

logger = get_logger(__name__)

router = APIRouter()


@router.post("/notice")
async def faq_notice(
    request: SyncNotifyRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        接收通知：FAQ创建、更新、删除

        nArgs:
            - event_type: create/update/delete
            - id: 主键id
            - api_key: API key from auth
            - db: Database instance

        Returns:
            - result
        """
    try:
        logger.info(f"faq_notice request event_type={request.event_type}")

        if request.event_type == "insert":
            await db.faqs.insert_one({"faq_id": request.id})
        elif request.event_type == "update":
            await db.faqs.update_one({"faq_id": request.id})
        elif request.event_type == "delete":
            await db.faqs.delete_one({"faq_id": request.id})
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error",
                                        f"faq_notice failed")

        return ResponseResult.success(None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"faq_notice exception event_type={request.event_type} error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="faq_notice error"
        )
