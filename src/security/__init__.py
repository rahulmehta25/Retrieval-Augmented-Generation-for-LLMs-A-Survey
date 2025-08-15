"""
Security Module for RAG System

Provides comprehensive security features including:
- Input validation and sanitization
- Rate limiting and DoS protection
- Authentication and authorization
- Security headers and CORS configuration
- Audit logging and monitoring
- SQL injection and XSS prevention
"""

from .input_validator import InputValidator, ValidationError
from .rate_limiter import RateLimiter, RateLimitExceeded
from .authentication import (
    AuthenticationMiddleware,
    JWTHandler,
    APIKeyValidator,
    create_access_token,
    verify_token
)
from .security_headers import SecurityHeadersMiddleware
from .audit_logger import AuditLogger, SecurityEvent
from .config_manager import SecureConfigManager
from .cors_handler import CORSConfigManager

__all__ = [
    'InputValidator',
    'ValidationError',
    'RateLimiter',
    'RateLimitExceeded',
    'AuthenticationMiddleware',
    'JWTHandler',
    'APIKeyValidator',
    'create_access_token',
    'verify_token',
    'SecurityHeadersMiddleware',
    'AuditLogger',
    'SecurityEvent',
    'SecureConfigManager',
    'CORSConfigManager'
]