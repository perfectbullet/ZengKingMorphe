"""
Logging configuration for the Digital Employee AI Service.
Using loguru for structured logging.
"""

import sys
from pathlib import Path
from loguru import logger
from app.core.config import settings


def setup_logging() -> None:
    """Configure loguru logging with structured output support."""

    # Clear log file on startup
    if hasattr(settings, 'log_file') and settings.log_file:
        log_path = Path(settings.log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("")  # Truncate the file
        except Exception:
            pass  # Ignore if we can't clear the file

    # Remove default handler
    logger.remove()
    
    # Parse log level from settings
    log_level = settings.log_level.upper()
    
    # Unified simple text format for all modes (use {name} instead of {extra[module]})
    logger.add(
        sys.stdout,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        colorize=False,
        backtrace=False,
        diagnose=False,
    )
    
    # Optional: Add file logging with rotation
    if hasattr(settings, 'log_file') and settings.log_file:
        logger.add(
            settings.log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="100 MB",  # Rotate when file reaches 100MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
            enqueue=True,  # Async logging for better performance
        )
    
    logger.info(f"Loguru logging configured (level={log_level}, debug={settings.debug})")


def get_logger(name: str):
    """
    Get a logger instance bound with module name.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Loguru logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("User login", user_id=user_id, ip=ip)
    """
    return logger.bind(module=name)
