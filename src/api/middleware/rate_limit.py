"""
Rate limiting middleware.

Provides configurable rate limiting for API endpoints
with Redis-based storage and sliding window algorithm.
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import get_settings
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.

    Limits requests per IP address or authenticated user
    with configurable limits and time windows.
    """

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.cache = None
        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize Redis cache for rate limiting."""
        try:
            self.cache = get_provider(ProviderType.CACHE, "redis")
        except Exception as e:
            logger.warning(f"Redis cache not available for rate limiting: {e}")
            self.cache = None

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""

        # Skip rate limiting if disabled or cache unavailable
        if not self.settings.api.enable_rate_limiting or not self.cache:
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Check rate limits
        rate_limit_result = await self._check_rate_limit(request, client_id)

        if rate_limit_result["exceeded"]:
            return self._create_rate_limit_response(rate_limit_result)

        # Process request
        start_time = time.time()
        response = await call_next(request)

        # Record successful request
        await self._record_request(client_id, request.url.path)

        # Add rate limit headers to response
        self._add_rate_limit_headers(response, rate_limit_result)

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""

        # Use authenticated user ID if available
        auth_info = getattr(request.state, "auth", None)
        if auth_info and auth_info.get("user_id"):
            return f"user:{auth_info['user_id']}"

        # Use IP address as fallback
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""

        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct connection
        return request.client.host if request.client else "unknown"

    async def _check_rate_limit(
        self, request: Request, client_id: str
    ) -> Dict[str, Any]:
        """Check if request exceeds rate limits."""

        # Get rate limit configuration for endpoint
        limits = self._get_rate_limits_for_endpoint(request.url.path)

        current_time = time.time()
        results = []

        for limit_config in limits:
            window_seconds = limit_config["window_seconds"]
            max_requests = limit_config["max_requests"]

            # Check this rate limit
            result = await self._check_sliding_window(
                client_id, window_seconds, max_requests, current_time
            )

            results.append(
                {
                    "window_seconds": window_seconds,
                    "max_requests": max_requests,
                    "current_requests": result["current_requests"],
                    "exceeded": result["exceeded"],
                    "retry_after": result["retry_after"],
                }
            )

            # If any limit is exceeded, request is blocked
            if result["exceeded"]:
                return {
                    "exceeded": True,
                    "limits": results,
                    "retry_after": result["retry_after"],
                }

        return {"exceeded": False, "limits": results, "retry_after": 0}

    def _get_rate_limits_for_endpoint(self, path: str) -> list:
        """Get rate limit configuration for an endpoint."""

        # Default rate limits
        default_limits = [
            {"window_seconds": 60, "max_requests": 100},  # 100 per minute
            {"window_seconds": 3600, "max_requests": 1000},  # 1000 per hour
        ]

        # Endpoint-specific limits
        endpoint_limits = {
            "/v1/search/": [
                {"window_seconds": 60, "max_requests": 30},
                {"window_seconds": 3600, "max_requests": 300},
            ],
            "/v1/rag/": [
                {"window_seconds": 60, "max_requests": 20},
                {"window_seconds": 3600, "max_requests": 200},
            ],
            "/v1/chat/": [
                {"window_seconds": 60, "max_requests": 10},
                {"window_seconds": 3600, "max_requests": 100},
            ],
            "/v1/admin/": [
                {"window_seconds": 60, "max_requests": 5},
                {"window_seconds": 3600, "max_requests": 50},
            ],
        }

        # Find matching endpoint
        for endpoint, limits in endpoint_limits.items():
            if path.startswith(endpoint):
                return limits

        return default_limits

    async def _check_sliding_window(
        self,
        client_id: str,
        window_seconds: int,
        max_requests: int,
        current_time: float,
    ) -> Dict[str, Any]:
        """Check sliding window rate limit."""

        key = f"rate_limit:{client_id}:{window_seconds}"

        try:
            # Get current window data
            window_data = await self.cache.get(key)

            if window_data:
                window_data = json.loads(window_data)
                requests = window_data.get("requests", [])
            else:
                requests = []

            # Remove requests outside the window
            window_start = current_time - window_seconds
            requests = [req_time for req_time in requests if req_time > window_start]

            # Check if limit exceeded
            current_requests = len(requests)
            exceeded = current_requests >= max_requests

            # Calculate retry after
            retry_after = 0
            if exceeded and requests:
                oldest_request = min(requests)
                retry_after = int(oldest_request + window_seconds - current_time) + 1

            return {
                "current_requests": current_requests,
                "exceeded": exceeded,
                "retry_after": max(0, retry_after),
            }

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if check fails
            return {"current_requests": 0, "exceeded": False, "retry_after": 0}

    async def _record_request(self, client_id: str, path: str):
        """Record a successful request for rate limiting."""

        if not self.cache:
            return

        current_time = time.time()

        # Record for each window size
        window_sizes = [60, 3600]  # 1 minute, 1 hour

        for window_seconds in window_sizes:
            key = f"rate_limit:{client_id}:{window_seconds}"

            try:
                # Get current window data
                window_data = await self.cache.get(key)

                if window_data:
                    window_data = json.loads(window_data)
                    requests = window_data.get("requests", [])
                else:
                    requests = []

                # Add current request
                requests.append(current_time)

                # Remove requests outside the window
                window_start = current_time - window_seconds
                requests = [
                    req_time for req_time in requests if req_time > window_start
                ]

                # Store updated window data
                window_data = {"requests": requests, "last_request": current_time}

                await self.cache.set(
                    key,
                    json.dumps(window_data),
                    ttl=window_seconds + 60,  # Extra TTL buffer
                )

            except Exception as e:
                logger.error(f"Failed to record request: {e}")

    def _add_rate_limit_headers(
        self, response: Response, rate_limit_result: Dict[str, Any]
    ):
        """Add rate limit headers to response."""

        if not rate_limit_result.get("limits"):
            return

        # Use the most restrictive limit for headers
        primary_limit = rate_limit_result["limits"][0]

        response.headers["X-RateLimit-Limit"] = str(primary_limit["max_requests"])
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, primary_limit["max_requests"] - primary_limit["current_requests"])
        )
        response.headers["X-RateLimit-Window"] = str(primary_limit["window_seconds"])

        if rate_limit_result["exceeded"]:
            response.headers["Retry-After"] = str(rate_limit_result["retry_after"])

    def _create_rate_limit_response(
        self, rate_limit_result: Dict[str, Any]
    ) -> Response:
        """Create rate limit exceeded response."""

        primary_limit = rate_limit_result["limits"][0]

        error_response = {
            "error": "Rate Limit Exceeded",
            "message": f"Too many requests. Limit: {primary_limit['max_requests']} per {primary_limit['window_seconds']} seconds",
            "type": "rate_limit_error",
            "retry_after": rate_limit_result["retry_after"],
            "limits": rate_limit_result["limits"],
        }

        headers = {
            "X-RateLimit-Limit": str(primary_limit["max_requests"]),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Window": str(primary_limit["window_seconds"]),
            "Retry-After": str(rate_limit_result["retry_after"]),
        }

        return Response(
            content=json.dumps(error_response),
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers=headers,
            media_type="application/json",
        )


class RateLimitBypass:
    """
    Utility to bypass rate limiting for specific clients.

    Can be used for internal services or premium users.
    """

    def __init__(self, cache: Optional[RedisCache] = None):
        self.cache = cache or get_provider(ProviderType.CACHE, "redis")

    async def add_bypass(self, client_id: str, duration_seconds: int = 3600):
        """Add rate limit bypass for a client."""

        if not self.cache:
            logger.warning("Cannot add rate limit bypass: cache unavailable")
            return

        key = f"rate_limit_bypass:{client_id}"

        await self.cache.set(
            key,
            json.dumps({"bypassed": True, "created_at": time.time()}),
            ttl=duration_seconds,
        )

        logger.info(f"Rate limit bypass added for {client_id} for {duration_seconds}s")

    async def remove_bypass(self, client_id: str):
        """Remove rate limit bypass for a client."""

        if not self.cache:
            return

        key = f"rate_limit_bypass:{client_id}"
        await self.cache.delete(key)

        logger.info(f"Rate limit bypass removed for {client_id}")

    async def is_bypassed(self, client_id: str) -> bool:
        """Check if client has rate limit bypass."""

        if not self.cache:
            return False

        key = f"rate_limit_bypass:{client_id}"
        bypass_data = await self.cache.get(key)

        return bypass_data is not None


# Utility functions
async def get_rate_limit_status(
    client_id: str, cache: Optional[RedisCache] = None
) -> Dict[str, Any]:
    """Get current rate limit status for a client."""

    if not cache:
        cache = get_provider(ProviderType.CACHE, "redis")

    if not cache:
        return {"error": "Cache unavailable"}

    # Check different window sizes
    windows = [60, 3600]  # 1 minute, 1 hour
    status = {}

    for window_seconds in windows:
        key = f"rate_limit:{client_id}:{window_seconds}"

        try:
            window_data = await cache.get(key)

            if window_data:
                window_data = json.loads(window_data)
                requests = window_data.get("requests", [])

                # Remove old requests
                current_time = time.time()
                window_start = current_time - window_seconds
                recent_requests = [
                    req_time for req_time in requests if req_time > window_start
                ]

                status[f"{window_seconds}s"] = {
                    "requests": len(recent_requests),
                    "window_seconds": window_seconds,
                }
            else:
                status[f"{window_seconds}s"] = {
                    "requests": 0,
                    "window_seconds": window_seconds,
                }

        except Exception as e:
            logger.error(f"Failed to get rate limit status: {e}")
            status[f"{window_seconds}s"] = {"error": str(e)}

    return status


async def clear_rate_limits(client_id: str, cache: Optional[RedisCache] = None):
    """Clear rate limits for a client."""

    if not cache:
        cache = get_provider(ProviderType.CACHE, "redis")

    if not cache:
        logger.warning("Cannot clear rate limits: cache unavailable")
        return

    # Clear all window sizes
    windows = [60, 3600]

    for window_seconds in windows:
        key = f"rate_limit:{client_id}:{window_seconds}"
        await cache.delete(key)

    logger.info(f"Rate limits cleared for {client_id}")
