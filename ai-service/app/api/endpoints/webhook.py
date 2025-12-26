"""
Webhook API endpoints for Java platform integration.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from app.models.schemas import SyncNotifyRequest
from app.api.middleware.auth import verify_api_key
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/sensitive-words/sync-notify")
async def sync_sensitive_words(
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
        logger.info(f"Sensitive words sync notification: event_type={request.event_type}, word_count={len(request.word_ids)}")
        
        # TODO: Implement sensitive words sync logic in Phase 5
        # 1. Fetch updated words from Java API
        # 2. Update local sensitive words index
        # 3. Rebuild AC automaton if needed
        
        return {
            "code": 200,
            "message": "Sensitive words synced successfully"
        }
        
    except Exception as e:
        logger.error(f"Sync sensitive words error: error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sync sensitive words"
        )


@router.post("/professional-words/sync-notify")
async def sync_professional_words(
    request: SyncNotifyRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    专业词库同步回调接口。
    
    \nArgs:
        \n- request: Sync notify request
        \n- api_key: Validated API key

    \nReturns:
        \n- Success message
    """
    try:
        logger.info(f"Professional words sync notification: event_type={request.event_type}, word_count={len(request.word_ids)}")
        
        # TODO: Implement professional words sync logic in Phase 5
        # 1. Fetch updated words from Java API
        # 2. Update local professional words index
        # 3. Update Chroma dictionary collection if needed
        
        return {
            "code": 200,
            "message": "Professional words synced successfully"
        }
        
    except Exception as e:
        logger.error(f"Sync professional words error: error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sync professional words"
        )


@router.post("/faq/sync-notify")
async def sync_faq(
    request: SyncNotifyRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    FAQ 同步回调接口。
    
    \nArgs:
        \n- request: Sync notify request
        \n- api_key: Validated API key

    \nReturns:
        \n- Success message
    """
    try:
        logger.info(f"FAQ sync notification: event_type={request.event_type}, faq_count={len(request.word_ids)}")
        
        # TODO: Implement FAQ sync logic in Phase 2
        # 1. Fetch updated FAQs from Java API
        # 2. Update Chroma FAQ collection
        # 3. Update ElasticSearch FAQ index
        
        return {
            "code": 200,
            "message": "FAQ synced successfully"
        }
        
    except Exception as e:
        logger.error(f"Sync FAQ error: error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sync FAQ"
        )
