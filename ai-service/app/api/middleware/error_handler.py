"""
Error handling middleware and utilities.
"""
from typing import Any, Dict
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


def create_error_response(
    code: int,
    message: str,
    error_type: str = "Error",
    details: Any = None
) -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        code: HTTP status code
        message: Error message
        error_type: Error type
        details: Additional error details
        
    Returns:
        Error response dict
    """
    error_response = {
        "code": code,
        "message": message,
        "error": {
            "type": error_type,
        }
    }
    
    if details:
        error_response["error"]["details"] = details
    
    return error_response


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions.
    
    Args:
        request: Request object
        exc: HTTP exception
        
    Returns:
        JSON response
    """
    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            code=exc.status_code,
            message=str(exc.detail),
            error_type="HTTPException"
        )
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle request validation errors.
    
    Args:
        request: Request object
        exc: Validation error
        
    Returns:
        JSON response
    """
    logger.warning(
        "Validation error",
        errors=exc.errors(),
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=create_error_response(
            code=400,
            message="Invalid request parameters",
            error_type="ValidationError",
            details=exc.errors()
        )
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle general exceptions.
    
    Args:
        request: Request object
        exc: Exception
        
    Returns:
        JSON response
    """
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            code=500,
            message="Internal server error",
            error_type=type(exc).__name__,
            details=str(exc) if settings.debug else None
        )
    )
