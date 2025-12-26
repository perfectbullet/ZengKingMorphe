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
    logger.error(f"http_exception_handler request: {request}")
    logger.error(f"http_exception_handler exc: {exc}")
    logger.exception(exc)
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
    # Get request body for logging (be careful with sensitive data)
    try:
        body = await request.body()
        body_str = body.decode('utf-8') if body else "No body"
    except Exception:
        body_str = "Could not decode body"
    
    # Format validation errors for better readability
    error_details = exc.errors()
    formatted_errors = []
    for error in error_details:
        formatted_errors.append({
            "field": " -> ".join(str(loc) for loc in error.get("loc", [])),
            "message": error.get("msg", ""),
            "type": error.get("type", ""),
            "input": error.get("input", "")
        })
    
    logger.warning(
        f"""Validation error on {request.method} {request.url.path}",
        errors={formatted_errors},
        error_count=len({error_details}),
        request_body={body_str[:500]},  # Truncate to prevent PII leakage
        query_params={request.query_params}"""
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
        """Unhandled exception",
        error={exc},
        error_type={type(exc).__name__},
        path={request.url.path},
        exc_info=True"""
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
