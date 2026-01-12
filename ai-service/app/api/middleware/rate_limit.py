"""
Rate limiting middleware.
"""
from typing import Dict
from datetime import datetime, timedelta
from collections import defaultdict
from fastapi import HTTPException, Request, status
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TokenBucket:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens refilled per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = datetime.utcnow()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        # Refill tokens
        now = datetime.utcnow()
        time_passed = (now - self.last_refill).total_seconds()
        self.tokens = min(
            self.capacity,
            self.tokens + time_passed * self.refill_rate
        )
        self.last_refill = now
        
        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """Rate limiter using token bucket algorithm."""
    
    def __init__(self):
        # User-level rate limiting
        self.user_buckets: Dict[str, TokenBucket] = {}
        # Session-level rate limiting
        self.session_buckets: Dict[str, TokenBucket] = {}
        # Cleanup timestamp
        self.last_cleanup = datetime.utcnow()
    
    def _cleanup_old_buckets(self) -> None:
        """Clean up old buckets to prevent memory leak."""
        now = datetime.utcnow()
        if (now - self.last_cleanup).total_seconds() < 3600:  # Cleanup every hour
            return
        
        # Clean user buckets
        expired_users = [
            user_id for user_id, bucket in self.user_buckets.items()
            if (now - bucket.last_refill).total_seconds() > 7200  # 2 hours
        ]
        for user_id in expired_users:
            del self.user_buckets[user_id]
        
        # Clean session buckets
        expired_sessions = [
            session_id for session_id, bucket in self.session_buckets.items()
            if (now - bucket.last_refill).total_seconds() > 7200
        ]
        for session_id in expired_sessions:
            del self.session_buckets[session_id]
        
        self.last_cleanup = now
        logger.info(f"Cleaned up old rate limit buckets: expired_users={len(expired_users)}, expired_sessions={len(expired_sessions)}")
    
    async def check_user_rate_limit(self, user_id: str) -> None:
        """
        Check user-level rate limit.
        
        Args:
            user_id: User ID
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        self._cleanup_old_buckets()
        
        if user_id not in self.user_buckets:
            # Create new bucket: 60 requests per minute
            self.user_buckets[user_id] = TokenBucket(
                capacity=settings.rate_limit_per_minute,
                refill_rate=settings.rate_limit_per_minute / 60.0
            )
        
        bucket = self.user_buckets[user_id]
        if not bucket.consume():
            logger.warning(f"User rate limit exceeded: user_id={user_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
            )
    
    async def check_session_rate_limit(self, session_id: str) -> None:
        """
        Check session-level rate limit.
        
        Args:
            session_id: Session ID
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        if session_id not in self.session_buckets:
            # Create new bucket: 10 requests per session per minute
            self.session_buckets[session_id] = TokenBucket(
                capacity=settings.rate_limit_per_session,
                refill_rate=settings.rate_limit_per_session / 60.0
            )
        
        bucket = self.session_buckets[session_id]
        if not bucket.consume():
            logger.warning(f"Session rate limit exceeded: session_id={session_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests in this session. Please slow down.",
            )


# Global rate limiter instance
rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, user_id: str, session_id: str = None) -> None:
    """
    Rate limiting middleware.
    
    Args:
        request: Request object
        user_id: User ID
        session_id: Session ID (optional)
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # Check user-level rate limit
    await rate_limiter.check_user_rate_limit(user_id)
    
    # Check session-level rate limit if session_id provided
    if session_id:
        await rate_limiter.check_session_rate_limit(session_id)
