"""
Authentication and authorization middleware.
"""
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify API key.
    
    Args:
        api_key: API key from header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing",
        )
    
    # Validate against configured API keys
    if api_key not in settings.api_keys:
        logger.warning("Invalid API key attempt", api_key_prefix=api_key[:8] if len(api_key) >= 8 else "***")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    logger.debug("API key validated successfully")
    return api_key


async def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    Get API key optionally (for endpoints that don't require authentication).
    
    Args:
        api_key: API key from header
        
    Returns:
        API key if provided and valid, None otherwise
    """
    if not api_key:
        return None
    
    try:
        return await verify_api_key(api_key)
    except HTTPException:
        return None
