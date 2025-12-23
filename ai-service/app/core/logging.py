"""
Logging configuration for the Digital Employee AI Service.
Using loguru with structlog-style keyword argument support.
"""

import sys
from loguru import logger as _loguru_logger
from app.core.config import settings


class LoguruAdapter:
    """
    Adapter to support structlog-style keyword arguments with loguru.
    
    Allows seamless migration from structlog syntax like:
        logger.info("Message", key1=value1, key2=value2)
    
    To loguru's structured logging via extra dict:
        logger.bind(key1=value1, key2=value2).info("Message")
    """
    
    def __init__(self, name: str):
        """Initialize adapter with module name."""
        self.name = name
        self._logger = _loguru_logger.bind(module=name)
    
    def _bind_and_log(
        self, 
        level: str, 
        message: str, 
        exc_info: bool = False,
        **kwargs
    ) -> None:
        """
        Internal method to bind context and log message.
        
        Args:
            level: Log level (debug/info/warning/error)
            message: Log message
            exc_info: Whether to include exception info (for error logs)
            **kwargs: Structured context fields
        """
        # Filter out None values to keep logs clean
        context = {k: v for k, v in kwargs.items() if v is not None}
        
        # Bind context to logger
        bound_logger = self._logger.bind(**context) if context else self._logger
        
        # Handle exception info
        if exc_info:
            bound_logger.opt(exception=True).log(level.upper(), message)
        else:
            bound_logger.log(level.upper(), message)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional structured fields."""
        self._bind_and_log("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional structured fields."""
        self._bind_and_log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional structured fields."""
        self._bind_and_log("warning", message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """
        Log error message with optional structured fields and exception info.
        
        Args:
            message: Error message
            exc_info: If True, includes exception traceback
            **kwargs: Structured context fields
        """
        self._bind_and_log("error", message, exc_info=exc_info, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback (alias for error with exc_info=True)."""
        self._bind_and_log("error", message, exc_info=True, **kwargs)


def setup_logging() -> None:
    """Configure loguru logging with structured output support."""
    
    # Remove default handler
    _loguru_logger.remove()
    
    # Parse log level from settings
    log_level = settings.log_level.upper()
    
    # Development mode: colorized console output with detailed format
    if settings.debug:
        _loguru_logger.add(
            sys.stdout,
            level=log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[module]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level> | "
                "{extra}"
            ),
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    else:
        # Production mode: JSON structured logging
        _loguru_logger.add(
            sys.stdout,
            level=log_level,
            format="{message}",
            serialize=True,  # Output as JSON
            backtrace=False,
            diagnose=False,
        )
    
    # Optional: Add file logging with rotation
    if hasattr(settings, 'log_file') and settings.log_file:
        _loguru_logger.add(
            settings.log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[module]}:{function}:{line} | {message} | {extra}",
            rotation="100 MB",  # Rotate when file reaches 100MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
            serialize=True,  # JSON format for production
            enqueue=True,  # Async logging for better performance
        )
    
    _loguru_logger.info("Loguru logging configured", level=log_level, debug=settings.debug)


def get_logger(name: str) -> LoguruAdapter:
    """
    Get a logger adapter instance with structlog-compatible API.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        LoguruAdapter instance supporting keyword argument logging
    
    Example:
        logger = get_logger(__name__)
        logger.info("User login", user_id=123, ip="192.168.1.1")
    """
    return LoguruAdapter(name)
