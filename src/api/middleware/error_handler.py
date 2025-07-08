"""
Error handling middleware.

Provides comprehensive error handling with structured logging,
error tracking, and user-friendly error responses.
"""

import time
import traceback
import uuid
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ...core.utils.config import get_settings
from ...core.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.

    Catches all unhandled exceptions and converts them to
    structured JSON responses with appropriate logging.
    """

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next):
        """Process request with error handling."""

        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        try:
            response = await call_next(request)

            # Log successful requests
            duration = time.time() - start_time
            self._log_request(request, response.status_code, duration, request_id)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except HTTPException as exc:
            # Handle FastAPI HTTP exceptions
            duration = time.time() - start_time

            self._log_http_exception(request, exc, duration, request_id)

            return self._create_error_response(
                status_code=exc.status_code,
                error_type="http_error",
                message=exc.detail,
                request_id=request_id,
                headers=exc.headers,
            )

        except Exception as exc:
            # Handle unexpected exceptions
            duration = time.time() - start_time

            self._log_unhandled_exception(request, exc, duration, request_id)

            # Determine if we should expose the error details
            if self.settings.api.debug_mode:
                error_message = str(exc)
                error_details = {
                    "exception_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
            else:
                error_message = "An internal error occurred"
                error_details = None

            return self._create_error_response(
                status_code=500,
                error_type="internal_error",
                message=error_message,
                request_id=request_id,
                details=error_details,
            )

    def _log_request(
        self, request: Request, status_code: int, duration: float, request_id: str
    ):
        """Log successful request."""

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "status_code": status_code,
            "duration_ms": duration * 1000,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", "unknown"),
        }

        # Add auth info if available
        auth_info = getattr(request.state, "auth", None)
        if auth_info:
            log_data["user_id"] = auth_info.get("user_id")
            log_data["auth_method"] = auth_info.get("method")

        logger.info(f"Request completed", extra=log_data)

    def _log_http_exception(
        self, request: Request, exc: HTTPException, duration: float, request_id: str
    ):
        """Log HTTP exception."""

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "status_code": exc.status_code,
            "error_detail": exc.detail,
            "duration_ms": duration * 1000,
            "client_ip": self._get_client_ip(request),
        }

        # Log as warning for client errors, error for server errors
        if exc.status_code < 500:
            logger.warning(f"HTTP exception: {exc.status_code}", extra=log_data)
        else:
            logger.error(f"HTTP server error: {exc.status_code}", extra=log_data)

    def _log_unhandled_exception(
        self, request: Request, exc: Exception, duration: float, request_id: str
    ):
        """Log unhandled exception."""

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "duration_ms": duration * 1000,
            "client_ip": self._get_client_ip(request),
            "traceback": traceback.format_exc(),
        }

        logger.error(f"Unhandled exception: {type(exc).__name__}", extra=log_data)

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

    def _create_error_response(
        self,
        status_code: int,
        error_type: str,
        message: str,
        request_id: str,
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> JSONResponse:
        """Create structured error response."""

        error_response = {
            "error": self._get_error_title(status_code),
            "message": message,
            "type": error_type,
            "request_id": request_id,
            "timestamp": time.time(),
        }

        # Add details if provided and debug mode is enabled
        if details and self.settings.api.debug_mode:
            error_response["details"] = details

        # Add help information for common errors
        if status_code == 401:
            error_response["help"] = (
                "Ensure you have provided valid authentication credentials"
            )
        elif status_code == 403:
            error_response["help"] = "You don't have permission to access this resource"
        elif status_code == 404:
            error_response["help"] = "The requested resource was not found"
        elif status_code == 429:
            error_response["help"] = (
                "You have exceeded the rate limit. Please try again later"
            )
        elif status_code >= 500:
            error_response["help"] = (
                "An internal server error occurred. Please try again later"
            )

        # Prepare headers
        response_headers = {"X-Request-ID": request_id}
        if headers:
            response_headers.update(headers)

        return JSONResponse(
            content=error_response, status_code=status_code, headers=response_headers
        )

    def _get_error_title(self, status_code: int) -> str:
        """Get error title for status code."""

        error_titles = {
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            422: "Validation Error",
            429: "Too Many Requests",
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Gateway Timeout",
        }

        return error_titles.get(status_code, "Error")


class ValidationErrorHandler:
    """
    Custom handler for Pydantic validation errors.

    Provides detailed validation error messages with field-specific information.
    """

    @staticmethod
    def format_validation_error(exc) -> Dict[str, Any]:
        """Format Pydantic validation error."""

        errors = []

        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])

            errors.append(
                {
                    "field": field_path,
                    "message": error["msg"],
                    "type": error["type"],
                    "input": error.get("input"),
                }
            )

        return {
            "error": "Validation Error",
            "message": "The request contains invalid data",
            "type": "validation_error",
            "errors": errors,
        }


class DatabaseErrorHandler:
    """
    Handler for database-related errors.

    Provides appropriate error messages for common database issues.
    """

    @staticmethod
    def handle_database_error(exc: Exception) -> Dict[str, Any]:
        """Handle database error and return appropriate response."""

        error_message = str(exc).lower()

        if "connection" in error_message:
            return {
                "error": "Database Connection Error",
                "message": "Unable to connect to the database",
                "type": "database_connection_error",
                "retry_after": 5,
            }
        elif "timeout" in error_message:
            return {
                "error": "Database Timeout",
                "message": "The database query timed out",
                "type": "database_timeout_error",
                "retry_after": 1,
            }
        elif "unique" in error_message or "duplicate" in error_message:
            return {
                "error": "Duplicate Data",
                "message": "A record with this data already exists",
                "type": "database_duplicate_error",
            }
        elif "foreign key" in error_message:
            return {
                "error": "Invalid Reference",
                "message": "Referenced record does not exist",
                "type": "database_foreign_key_error",
            }
        else:
            return {
                "error": "Database Error",
                "message": "A database error occurred",
                "type": "database_error",
            }


class ExternalServiceErrorHandler:
    """
    Handler for external service errors.

    Provides appropriate error messages for external API failures.
    """

    @staticmethod
    def handle_service_error(service_name: str, exc: Exception) -> Dict[str, Any]:
        """Handle external service error."""

        error_message = str(exc).lower()

        if "timeout" in error_message:
            return {
                "error": f"{service_name} Timeout",
                "message": f"The {service_name} service timed out",
                "type": "external_service_timeout",
                "service": service_name,
                "retry_after": 5,
            }
        elif "connection" in error_message:
            return {
                "error": f"{service_name} Unavailable",
                "message": f"The {service_name} service is currently unavailable",
                "type": "external_service_unavailable",
                "service": service_name,
                "retry_after": 10,
            }
        else:
            return {
                "error": f"{service_name} Error",
                "message": f"An error occurred with the {service_name} service",
                "type": "external_service_error",
                "service": service_name,
            }


# Error reporting utilities
class ErrorReporter:
    """
    Utility for reporting errors to external monitoring services.

    Can be integrated with services like Sentry, Rollbar, etc.
    """

    def __init__(self):
        self.settings = get_settings()

    async def report_error(
        self,
        exc: Exception,
        request: Request,
        additional_context: Optional[Dict[str, Any]] = None,
    ):
        """Report error to monitoring service."""

        # Skip reporting for expected errors
        if isinstance(exc, HTTPException) and exc.status_code < 500:
            return

        try:
            error_data = {
                "exception": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "request": {
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "headers": dict(request.headers),
                    "client_ip": self._get_client_ip(request),
                },
                "context": additional_context or {},
            }

            # Add auth info if available
            auth_info = getattr(request.state, "auth", None)
            if auth_info:
                error_data["user"] = {
                    "id": auth_info.get("user_id"),
                    "auth_method": auth_info.get("method"),
                }

            # In a real implementation, this would send to an error tracking service
            logger.error("Error reported to monitoring service", extra=error_data)

        except Exception as e:
            logger.error(f"Failed to report error: {e}")

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# Global error reporter instance
error_reporter = ErrorReporter()
