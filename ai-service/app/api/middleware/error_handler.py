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
    
    # 友好的错误提示
    friendly_message = f"哎呀～服务遇到了点小问题呢😅 {str(exc.detail)}"
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            code=exc.status_code,
            message=friendly_message,
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
    error_fields = []
    
    for error in error_details:
        field_name = " -> ".join(str(loc) for loc in error.get("loc", []))
        error_fields.append(field_name)
        formatted_errors.append({
            "field": field_name,
            "message": error.get("msg", ""),
            "type": error.get("type", ""),
            "input": error.get("input", "")
        })
    
    # 构建友好的错误提示信息
    fields_str = "、".join(error_fields[:3])  # 最多显示3个字段
    if len(error_fields) > 3:
        fields_str += "等"
    
    friendly_message = (
        f"哎呀～您传的参数好像跟我们接口八字不合呢😜， 麻烦检查下 [{fields_str}] 的值是不是填错啦？\n\n"
        f"详细错误信息：\n"
    )
    
    # 添加详细的错误信息
    for i, err in enumerate(formatted_errors, 1):
        friendly_message += f"{i}. 字段 '{err['field']}': {err['message']}\n"

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=create_error_response(
            code=400,
            message=friendly_message,
            error_type="ValidationError",
            details=formatted_errors
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
    
    # 友好的错误提示
    error_type_name = type(exc).__name__
    error_details = str(exc) if settings.debug else None
    
    friendly_message = f"哎呀～服务器开小差了呢🙈 我们的工程师已经收到通知啦！错误类型：{error_type_name}"
    
    if settings.debug and error_details:
        friendly_message += f"\n\n调试信息：{error_details}"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            code=500,
            message=friendly_message,
            error_type=error_type_name,
            details=error_details
        )
    )
