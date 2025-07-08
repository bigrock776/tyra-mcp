"""
Authentication middleware.

Provides optional authentication for API endpoints,
including API key validation and session management.
"""

import hashlib
import hmac
import time
from typing import Optional, Set

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ...core.utils.config import get_settings
from ...core.utils.logger import get_logger

logger = get_logger(__name__)

# Security scheme for API key authentication
bearer_scheme = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for API endpoints.

    Provides optional API key authentication with configurable
    exempt endpoints and flexible validation.
    """

    def __init__(self, app, exempt_paths: Optional[Set[str]] = None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or {
            "/health",
            "/live",
            "/ready",
            "/startup",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next):
        """Process request with authentication."""

        # Skip authentication for exempt paths
        if self._is_exempt_path(request.url.path):
            return await call_next(request)

        # Skip if authentication is disabled
        if not self.settings.api.enable_auth:
            return await call_next(request)

        # Validate authentication
        auth_result = await self._validate_auth(request)

        if not auth_result["valid"]:
            return self._create_auth_error_response(auth_result["error"])

        # Add authentication info to request state
        request.state.auth = auth_result["auth_info"]

        # Process request
        start_time = time.time()
        response = await call_next(request)

        # Log authenticated request
        duration = time.time() - start_time
        logger.info(
            f"Authenticated request: {request.method} {request.url.path} "
            f"- {response.status_code} - {duration:.3f}s - "
            f"user: {auth_result['auth_info'].get('user_id', 'unknown')}"
        )

        return response

    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from authentication."""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)

    async def _validate_auth(self, request: Request) -> dict:
        """Validate request authentication."""

        # Try different authentication methods
        auth_methods = [
            self._validate_api_key,
            self._validate_bearer_token,
            self._validate_basic_auth,
        ]

        for auth_method in auth_methods:
            result = await auth_method(request)
            if result["valid"]:
                return result

        return {
            "valid": False,
            "error": "No valid authentication provided",
            "auth_info": None,
        }

    async def _validate_api_key(self, request: Request) -> dict:
        """Validate API key authentication."""

        # Check header
        api_key = request.headers.get("X-API-Key")

        # Check query parameter
        if not api_key:
            api_key = request.query_params.get("api_key")

        if not api_key:
            return {"valid": False, "error": "No API key provided", "auth_info": None}

        # Validate API key
        valid_keys = self.settings.api.api_keys or []

        if api_key in valid_keys:
            return {
                "valid": True,
                "error": None,
                "auth_info": {
                    "method": "api_key",
                    "user_id": f"api_key_{hashlib.md5(api_key.encode()).hexdigest()[:8]}",
                    "permissions": ["read", "write"],  # Default permissions
                },
            }

        return {"valid": False, "error": "Invalid API key", "auth_info": None}

    async def _validate_bearer_token(self, request: Request) -> dict:
        """Validate Bearer token authentication."""

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return {
                "valid": False,
                "error": "No bearer token provided",
                "auth_info": None,
            }

        token = auth_header.split(" ", 1)[1]

        # Validate token (placeholder - would integrate with JWT or session store)
        if self._is_valid_token(token):
            return {
                "valid": True,
                "error": None,
                "auth_info": {
                    "method": "bearer_token",
                    "user_id": self._extract_user_from_token(token),
                    "permissions": self._extract_permissions_from_token(token),
                },
            }

        return {"valid": False, "error": "Invalid bearer token", "auth_info": None}

    async def _validate_basic_auth(self, request: Request) -> dict:
        """Validate Basic authentication."""

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return {
                "valid": False,
                "error": "No basic auth provided",
                "auth_info": None,
            }

        try:
            import base64

            encoded_credentials = auth_header.split(" ", 1)[1]
            decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
            username, password = decoded_credentials.split(":", 1)

            # Validate credentials (placeholder)
            if self._validate_credentials(username, password):
                return {
                    "valid": True,
                    "error": None,
                    "auth_info": {
                        "method": "basic_auth",
                        "user_id": username,
                        "permissions": ["read", "write"],
                    },
                }

        except Exception as e:
            logger.warning(f"Basic auth validation error: {e}")

        return {
            "valid": False,
            "error": "Invalid basic auth credentials",
            "auth_info": None,
        }

    def _is_valid_token(self, token: str) -> bool:
        """Validate a bearer token."""
        # Placeholder - would validate JWT or session token
        return len(token) >= 20  # Simple validation

    def _extract_user_from_token(self, token: str) -> str:
        """Extract user ID from token."""
        # Placeholder - would decode JWT or lookup session
        return f"user_{hashlib.md5(token.encode()).hexdigest()[:8]}"

    def _extract_permissions_from_token(self, token: str) -> list:
        """Extract permissions from token."""
        # Placeholder - would decode JWT claims or lookup permissions
        return ["read", "write"]

    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate username/password credentials."""
        # Placeholder - would check against user database
        return (
            username == "admin" and password == "password"
        )  # Obviously not for production!

    def _create_auth_error_response(self, error_message: str) -> Response:
        """Create authentication error response."""

        error_response = {
            "error": "Authentication Required",
            "message": error_message,
            "type": "authentication_error",
        }

        return Response(
            content=error_response,
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer, API-Key"},
            media_type="application/json",
        )


class APIKeyDependency:
    """
    Dependency for API key authentication.

    Can be used with FastAPI's Depends() for endpoint-specific authentication.
    """

    def __init__(self, required: bool = True):
        self.required = required
        self.settings = get_settings()

    async def __call__(self, request: Request) -> Optional[dict]:
        """Validate API key and return auth info."""

        # Skip if authentication is disabled globally
        if not self.settings.api.enable_auth and not self.required:
            return None

        # Get API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            api_key = request.query_params.get("api_key")

        if not api_key:
            if self.required:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required",
                    headers={"WWW-Authenticate": "API-Key"},
                )
            return None

        # Validate API key
        valid_keys = self.settings.api.api_keys or []

        if api_key not in valid_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "API-Key"},
            )

        return {
            "method": "api_key",
            "user_id": f"api_key_{hashlib.md5(api_key.encode()).hexdigest()[:8]}",
            "permissions": ["read", "write"],
        }


class PermissionChecker:
    """
    Permission checking utility.

    Validates user permissions for specific operations.
    """

    def __init__(self, required_permissions: list):
        self.required_permissions = required_permissions

    async def __call__(self, request: Request) -> bool:
        """Check if user has required permissions."""

        # Get auth info from request state
        auth_info = getattr(request.state, "auth", None)

        if not auth_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        user_permissions = auth_info.get("permissions", [])

        # Check if user has all required permissions
        for permission in self.required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission} required",
                )

        return True


# Convenience instances
require_api_key = APIKeyDependency(required=True)
optional_api_key = APIKeyDependency(required=False)
require_read_permission = PermissionChecker(["read"])
require_write_permission = PermissionChecker(["write"])
require_admin_permission = PermissionChecker(["admin"])
