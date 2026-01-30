from fastapi import APIRouter, Depends, HTTPException, status
from app.models.schemas import SyncNotifyRequest, ResponseResult
from app.api.middleware.auth import verify_api_key
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/notice")
async def sensitive_word_notice(
        request: SyncNotifyRequest,
        api_key: str = Depends(verify_api_key)
):
    """
        敏感词同步回调接口。

        \nArgs:
            \n- request: Sync notify request
            \n- api_key: Validated API key

        \nReturns:
            \n- Success message
    """
    try:
        logger.info(f"sensitive_word_notice event_type={request.event_type}, word_count={len(request.word_ids)}")

        # TODO: Implement sensitive words sync logic in Phase 5
        # 1. Fetch updated words from Java API
        # 2. Update local sensitive words index
        # 3. Rebuild AC automaton if needed

        return ResponseResult.success(None)

    except Exception as e:
        logger.error(f"sensitive_word_notice exception error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="notice_sensitive_word error"
        )
