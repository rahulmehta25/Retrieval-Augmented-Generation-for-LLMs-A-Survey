"""
CORS Configuration Handler

Implements secure Cross-Origin Resource Sharing (CORS) configuration to prevent:
- Cross-Origin attacks (OWASP A05:2021)
- Security Misconfiguration (OWASP A05:2021)
- Broken Access Control (OWASP A01:2021)
"""

from typing import List, Optional, Dict, Set
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import re
import logging
from urllib.parse import urlparse
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class CORSConfigManager:
    """
    Secure CORS configuration manager with whitelist-based approach
    
    Implements OWASP CORS security best practices
    """
    
    def __init__(self,
                 allowed_origins: Optional[List[str]] = None,
                 allowed_methods: Optional[List[str]] = None,
                 allowed_headers: Optional[List[str]] = None,
                 expose_headers: Optional[List[str]] = None,
                 allow_credentials: bool = False,
                 max_age: int = 600,
                 strict_origin_check: bool = True,
                 enable_logging: bool = True):
        """
        Initialize CORS configuration manager
        
        Args:
            allowed_origins: List of allowed origins (whitelist)
            allowed_methods: Allowed HTTP methods
            allowed_headers: Allowed request headers
            expose_headers: Headers to expose to the client
            allow_credentials: Allow credentials in CORS requests
            max_age: Max age for preflight cache (seconds)
            strict_origin_check: Enable strict origin validation
            enable_logging: Enable CORS violation logging
        """
        # Default to restrictive settings
        self.allowed_origins = set(allowed_origins or [])
        self.allowed_methods = set(allowed_methods or ['GET', 'POST'])
        self.allowed_headers = set(allowed_headers or ['Content-Type', 'Authorization'])
        self.expose_headers = set(expose_headers or [])
        self.allow_credentials = allow_credentials
        self.max_age = max_age
        self.strict_origin_check = strict_origin_check
        self.enable_logging = enable_logging
        
        # Origin patterns for more flexible matching
        self.origin_patterns = []
        
        # Blocked origins (blacklist)
        self.blocked_origins = set()
        
        # CORS violation tracking
        self.violation_log = []
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate CORS configuration for security issues"""
        issues = []
        
        # Check for wildcard origin with credentials
        if '*' in self.allowed_origins and self.allow_credentials:
            issues.append("Cannot use wildcard origin (*) with credentials")
        
        # Check for overly permissive methods
        dangerous_methods = {'DELETE', 'PUT', 'PATCH'}
        if dangerous_methods.issubset(self.allowed_methods) and '*' in self.allowed_origins:
            issues.append("Dangerous HTTP methods allowed with wildcard origin")
        
        # Warn about wildcard headers
        if '*' in self.allowed_headers:
            logger.warning("Using wildcard (*) for allowed headers is not recommended")
        
        if issues:
            raise ValueError(f"CORS configuration security issues: {', '.join(issues)}")
    
    def add_allowed_origin(self, origin: str):
        """
        Add an allowed origin with validation
        
        Args:
            origin: Origin to allow (e.g., https://example.com)
        """
        # Validate origin format
        if not self._is_valid_origin(origin):
            raise ValueError(f"Invalid origin format: {origin}")
        
        # Check if origin is blocked
        if origin in self.blocked_origins:
            raise ValueError(f"Origin is in blocklist: {origin}")
        
        self.allowed_origins.add(origin)
        logger.info(f"Added allowed origin: {origin}")
    
    def add_origin_pattern(self, pattern: str):
        """
        Add origin pattern for flexible matching
        
        Args:
            pattern: Regex pattern for origin matching
        """
        try:
            compiled_pattern = re.compile(pattern)
            self.origin_patterns.append(compiled_pattern)
            logger.info(f"Added origin pattern: {pattern}")
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    
    def block_origin(self, origin: str):
        """
        Block an origin
        
        Args:
            origin: Origin to block
        """
        self.blocked_origins.add(origin)
        # Remove from allowed if present
        self.allowed_origins.discard(origin)
        logger.info(f"Blocked origin: {origin}")
    
    def is_origin_allowed(self, origin: str) -> bool:
        """
        Check if origin is allowed
        
        Args:
            origin: Origin to check
            
        Returns:
            True if origin is allowed
        """
        # Check if origin is blocked
        if origin in self.blocked_origins:
            self._log_violation(origin, "Blocked origin")
            return False
        
        # Check if origin validation is disabled (not recommended)
        if not self.strict_origin_check:
            return True
        
        # Check exact match
        if origin in self.allowed_origins:
            return True
        
        # Check wildcard (if configured)
        if '*' in self.allowed_origins:
            return True
        
        # Check pattern match
        for pattern in self.origin_patterns:
            if pattern.match(origin):
                return True
        
        # Check subdomain wildcard (e.g., *.example.com)
        for allowed in self.allowed_origins:
            if allowed.startswith('*.'):
                domain = allowed[2:]  # Remove *.
                origin_domain = urlparse(origin).netloc
                if origin_domain.endswith(domain):
                    return True
        
        self._log_violation(origin, "Origin not in whitelist")
        return False
    
    def _is_valid_origin(self, origin: str) -> bool:
        """
        Validate origin format
        
        Args:
            origin: Origin to validate
            
        Returns:
            True if origin format is valid
        """
        if origin == '*':
            return True
        
        # Check for subdomain wildcard
        if origin.startswith('*.'):
            return True
        
        # Parse and validate URL
        try:
            parsed = urlparse(origin)
            return parsed.scheme in ['http', 'https'] and parsed.netloc
        except:
            return False
    
    def _log_violation(self, origin: str, reason: str):
        """Log CORS violation"""
        if not self.enable_logging:
            return
        
        violation = {
            'timestamp': datetime.now().isoformat(),
            'origin': origin,
            'reason': reason
        }
        
        self.violation_log.append(violation)
        logger.warning(f"CORS violation: {reason} for origin {origin}")
        
        # Keep only last 1000 violations
        if len(self.violation_log) > 1000:
            self.violation_log = self.violation_log[-1000:]
    
    def get_cors_middleware(self, app):
        """
        Get configured CORS middleware for FastAPI
        
        Args:
            app: FastAPI application instance
            
        Returns:
            Configured CORSMiddleware
        """
        # Convert sets to lists for middleware
        origins = list(self.allowed_origins) if self.allowed_origins else []
        
        # If using patterns, we need custom origin validation
        if self.origin_patterns or self.strict_origin_check:
            # Use a callback for origin validation
            return CORSMiddleware(
                app,
                allow_origins=origins,
                allow_origin_regex=self._build_origin_regex(),
                allow_credentials=self.allow_credentials,
                allow_methods=list(self.allowed_methods),
                allow_headers=list(self.allowed_headers),
                expose_headers=list(self.expose_headers),
                max_age=self.max_age
            )
        else:
            # Standard CORS middleware
            return CORSMiddleware(
                app,
                allow_origins=origins,
                allow_credentials=self.allow_credentials,
                allow_methods=list(self.allowed_methods),
                allow_headers=list(self.allowed_headers),
                expose_headers=list(self.expose_headers),
                max_age=self.max_age
            )
    
    def _build_origin_regex(self) -> Optional[str]:
        """Build regex pattern for origin validation"""
        if not self.origin_patterns:
            return None
        
        # Combine all patterns
        patterns = [p.pattern for p in self.origin_patterns]
        combined = '|'.join(f'({p})' for p in patterns)
        return combined
    
    def get_cors_headers(self, origin: str) -> Dict[str, str]:
        """
        Get CORS headers for a specific origin
        
        Args:
            origin: Request origin
            
        Returns:
            Dictionary of CORS headers
        """
        headers = {}
        
        if self.is_origin_allowed(origin):
            headers['Access-Control-Allow-Origin'] = origin
            
            if self.allow_credentials:
                headers['Access-Control-Allow-Credentials'] = 'true'
            
            if self.allowed_methods:
                headers['Access-Control-Allow-Methods'] = ', '.join(self.allowed_methods)
            
            if self.allowed_headers:
                headers['Access-Control-Allow-Headers'] = ', '.join(self.allowed_headers)
            
            if self.expose_headers:
                headers['Access-Control-Expose-Headers'] = ', '.join(self.expose_headers)
            
            if self.max_age:
                headers['Access-Control-Max-Age'] = str(self.max_age)
        
        return headers
    
    def handle_preflight(self, request: Request) -> Dict[str, str]:
        """
        Handle CORS preflight request
        
        Args:
            request: Preflight request
            
        Returns:
            CORS headers for preflight response
        """
        origin = request.headers.get('Origin', '')
        
        if not self.is_origin_allowed(origin):
            return {}
        
        headers = self.get_cors_headers(origin)
        
        # Add additional preflight headers
        request_method = request.headers.get('Access-Control-Request-Method')
        if request_method and request_method in self.allowed_methods:
            headers['Access-Control-Allow-Methods'] = request_method
        
        request_headers = request.headers.get('Access-Control-Request-Headers')
        if request_headers:
            # Validate requested headers
            requested = set(h.strip() for h in request_headers.split(','))
            if requested.issubset(self.allowed_headers) or '*' in self.allowed_headers:
                headers['Access-Control-Allow-Headers'] = request_headers
        
        return headers
    
    def get_violation_stats(self) -> Dict[str, any]:
        """
        Get CORS violation statistics
        
        Returns:
            Dictionary of violation statistics
        """
        if not self.violation_log:
            return {'total': 0, 'violations': []}
        
        # Group violations by origin
        by_origin = {}
        for violation in self.violation_log:
            origin = violation['origin']
            if origin not in by_origin:
                by_origin[origin] = 0
            by_origin[origin] += 1
        
        # Sort by frequency
        top_violators = sorted(by_origin.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total': len(self.violation_log),
            'unique_origins': len(by_origin),
            'top_violators': top_violators,
            'recent': self.violation_log[-10:]
        }
    
    def export_config(self) -> Dict[str, any]:
        """
        Export CORS configuration
        
        Returns:
            CORS configuration dictionary
        """
        return {
            'allowed_origins': list(self.allowed_origins),
            'allowed_methods': list(self.allowed_methods),
            'allowed_headers': list(self.allowed_headers),
            'expose_headers': list(self.expose_headers),
            'blocked_origins': list(self.blocked_origins),
            'allow_credentials': self.allow_credentials,
            'max_age': self.max_age,
            'strict_origin_check': self.strict_origin_check,
            'origin_patterns': [p.pattern for p in self.origin_patterns]
        }
    
    def load_config(self, config: Dict[str, any]):
        """
        Load CORS configuration from dictionary
        
        Args:
            config: Configuration dictionary
        """
        self.allowed_origins = set(config.get('allowed_origins', []))
        self.allowed_methods = set(config.get('allowed_methods', ['GET', 'POST']))
        self.allowed_headers = set(config.get('allowed_headers', ['Content-Type']))
        self.expose_headers = set(config.get('expose_headers', []))
        self.blocked_origins = set(config.get('blocked_origins', []))
        self.allow_credentials = config.get('allow_credentials', False)
        self.max_age = config.get('max_age', 600)
        self.strict_origin_check = config.get('strict_origin_check', True)
        
        # Load origin patterns
        patterns = config.get('origin_patterns', [])
        self.origin_patterns = []
        for pattern in patterns:
            self.add_origin_pattern(pattern)
        
        # Validate loaded configuration
        self._validate_config()


def create_secure_cors_config(
    production: bool = False,
    api_domain: Optional[str] = None
) -> CORSConfigManager:
    """
    Create a secure CORS configuration based on environment
    
    Args:
        production: Whether running in production
        api_domain: API domain for production
        
    Returns:
        Configured CORSConfigManager
    """
    if production:
        # Production configuration - very restrictive
        config = CORSConfigManager(
            allowed_origins=[
                f"https://{api_domain}",
                f"https://www.{api_domain}"
            ] if api_domain else [],
            allowed_methods=['GET', 'POST'],
            allowed_headers=['Content-Type', 'Authorization'],
            allow_credentials=True,
            strict_origin_check=True,
            max_age=3600
        )
    else:
        # Development configuration - slightly more permissive
        config = CORSConfigManager(
            allowed_origins=[
                'http://localhost:3000',
                'http://localhost:5173',
                'http://localhost:5174',
                'http://127.0.0.1:3000',
                'http://127.0.0.1:5173'
            ],
            allowed_methods=['GET', 'POST', 'PUT', 'DELETE'],
            allowed_headers=['Content-Type', 'Authorization', 'X-API-Key'],
            allow_credentials=True,
            strict_origin_check=True,
            max_age=600
        )
    
    return config