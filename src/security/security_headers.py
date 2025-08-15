"""
Security Headers Middleware

Implements HTTP security headers to prevent:
- Cross-Site Scripting (XSS) (OWASP A03:2021)
- Clickjacking (OWASP A05:2021)
- MIME type sniffing
- Protocol downgrade attacks
"""

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from typing import Dict, Optional, List
import logging
import hashlib
import secrets
import json

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add comprehensive security headers to all responses
    
    Implements OWASP security headers best practices
    """
    
    def __init__(self,
                 app,
                 enable_hsts: bool = True,
                 enable_csp: bool = True,
                 csp_policy: Optional[str] = None,
                 enable_permissions_policy: bool = True,
                 custom_headers: Optional[Dict[str, str]] = None):
        """
        Initialize security headers middleware
        
        Args:
            app: FastAPI application
            enable_hsts: Enable HTTP Strict Transport Security
            enable_csp: Enable Content Security Policy
            csp_policy: Custom CSP policy (uses default if None)
            enable_permissions_policy: Enable Permissions Policy
            custom_headers: Additional custom headers
        """
        super().__init__(app)
        self.enable_hsts = enable_hsts
        self.enable_csp = enable_csp
        self.csp_policy = csp_policy or self._get_default_csp()
        self.enable_permissions_policy = enable_permissions_policy
        self.custom_headers = custom_headers or {}
        
        # Nonce generator for CSP
        self.nonce_generator = lambda: secrets.token_urlsafe(16)
    
    def _get_default_csp(self) -> str:
        """
        Get default Content Security Policy
        
        Returns:
            CSP policy string
        """
        # Strict CSP policy
        policies = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",  # Allow for legitimate JS
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",  # Allow inline styles and Google Fonts
            "img-src 'self' data: https:",  # Allow images from HTTPS sources
            "font-src 'self' https://fonts.gstatic.com",  # Allow fonts
            "connect-src 'self' https://api.openai.com",  # Allow API connections
            "media-src 'none'",  # No media by default
            "object-src 'none'",  # No plugins
            "frame-src 'none'",  # No iframes
            "base-uri 'self'",  # Restrict base URL
            "form-action 'self'",  # Restrict form submissions
            "frame-ancestors 'none'",  # Prevent clickjacking
            "block-all-mixed-content",  # Block HTTP content on HTTPS pages
            "upgrade-insecure-requests"  # Upgrade HTTP to HTTPS
        ]
        
        return "; ".join(policies)
    
    def _get_permissions_policy(self) -> str:
        """
        Get Permissions Policy (formerly Feature Policy)
        
        Returns:
            Permissions policy string
        """
        # Restrictive permissions policy
        policies = [
            "accelerometer=()",  # Disable accelerometer
            "camera=()",  # Disable camera
            "geolocation=()",  # Disable geolocation
            "gyroscope=()",  # Disable gyroscope
            "magnetometer=()",  # Disable magnetometer
            "microphone=()",  # Disable microphone
            "payment=()",  # Disable payment
            "usb=()",  # Disable USB
            "interest-cohort=()",  # Disable FLoC
            "fullscreen=(self)",  # Allow fullscreen only for same origin
            "display-capture=()"  # Disable screen capture
        ]
        
        return ", ".join(policies)
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and add security headers to response
        
        Args:
            request: Incoming request
            call_next: Next middleware in chain
            
        Returns:
            Response with security headers
        """
        # Generate CSP nonce for this request
        csp_nonce = self.nonce_generator()
        request.state.csp_nonce = csp_nonce
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response, csp_nonce)
        
        return response
    
    def _add_security_headers(self, response: Response, csp_nonce: str):
        """
        Add security headers to response
        
        Args:
            response: Response object
            csp_nonce: CSP nonce for this request
        """
        # X-Content-Type-Options - Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-Frame-Options - Prevent clickjacking (legacy, CSP frame-ancestors is better)
        response.headers["X-Frame-Options"] = "DENY"
        
        # X-XSS-Protection - Enable XSS filter (legacy, CSP is better)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy - Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # HTTP Strict Transport Security (HSTS)
        if self.enable_hsts:
            # max-age=31536000 (1 year), includeSubDomains, preload
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # Content Security Policy (CSP)
        if self.enable_csp:
            # Add nonce to CSP for inline scripts
            csp_with_nonce = self.csp_policy.replace(
                "script-src 'self'",
                f"script-src 'self' 'nonce-{csp_nonce}'"
            )
            response.headers["Content-Security-Policy"] = csp_with_nonce
            
            # Report-only CSP for testing
            # response.headers["Content-Security-Policy-Report-Only"] = csp_with_nonce
        
        # Permissions Policy
        if self.enable_permissions_policy:
            response.headers["Permissions-Policy"] = self._get_permissions_policy()
        
        # Clear-Site-Data - Clear browsing data on logout
        if "logout" in str(response.url) if hasattr(response, 'url') else False:
            response.headers["Clear-Site-Data"] = '"cache", "cookies", "storage"'
        
        # Custom headers
        for header, value in self.custom_headers.items():
            response.headers[header] = value


class CSPReportHandler:
    """
    Handler for Content Security Policy violation reports
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize CSP report handler
        
        Args:
            log_file: Path to log file for CSP violations
        """
        self.log_file = log_file
        self.violations = []
    
    async def handle_csp_report(self, request: Request) -> Dict:
        """
        Handle CSP violation report
        
        Args:
            request: Request containing CSP report
            
        Returns:
            Response acknowledging report
        """
        try:
            # Parse CSP report
            report_data = await request.json()
            csp_report = report_data.get('csp-report', {})
            
            # Extract violation details
            violation = {
                'document_uri': csp_report.get('document-uri'),
                'violated_directive': csp_report.get('violated-directive'),
                'blocked_uri': csp_report.get('blocked-uri'),
                'line_number': csp_report.get('line-number'),
                'column_number': csp_report.get('column-number'),
                'source_file': csp_report.get('source-file'),
                'timestamp': csp_report.get('timestamp'),
                'referrer': csp_report.get('referrer')
            }
            
            # Log violation
            logger.warning(f"CSP Violation: {json.dumps(violation, indent=2)}")
            
            # Store violation
            self.violations.append(violation)
            
            # Write to file if configured
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(violation) + '\n')
            
            return {"status": "received"}
            
        except Exception as e:
            logger.error(f"Error processing CSP report: {e}")
            return {"status": "error"}
    
    def get_violation_summary(self) -> Dict:
        """
        Get summary of CSP violations
        
        Returns:
            Dictionary with violation statistics
        """
        if not self.violations:
            return {"total": 0, "violations": []}
        
        # Group violations by directive
        by_directive = {}
        for v in self.violations:
            directive = v.get('violated_directive', 'unknown')
            if directive not in by_directive:
                by_directive[directive] = []
            by_directive[directive].append(v)
        
        return {
            "total": len(self.violations),
            "by_directive": {
                k: len(v) for k, v in by_directive.items()
            },
            "recent": self.violations[-10:]  # Last 10 violations
        }


class SecurityHeadersConfig:
    """
    Configuration builder for security headers
    """
    
    def __init__(self):
        """Initialize configuration"""
        self.headers = {}
        self.csp_directives = {}
        self.permissions = {}
    
    def set_hsts(self, 
                max_age: int = 31536000,
                include_subdomains: bool = True,
                preload: bool = False) -> 'SecurityHeadersConfig':
        """
        Configure HSTS header
        
        Args:
            max_age: Max age in seconds
            include_subdomains: Include all subdomains
            preload: Request inclusion in HSTS preload list
            
        Returns:
            Self for chaining
        """
        hsts = f"max-age={max_age}"
        if include_subdomains:
            hsts += "; includeSubDomains"
        if preload:
            hsts += "; preload"
        
        self.headers["Strict-Transport-Security"] = hsts
        return self
    
    def add_csp_directive(self, directive: str, *values: str) -> 'SecurityHeadersConfig':
        """
        Add CSP directive
        
        Args:
            directive: CSP directive name
            values: Directive values
            
        Returns:
            Self for chaining
        """
        self.csp_directives[directive] = " ".join(values)
        return self
    
    def set_frame_options(self, option: str = "DENY") -> 'SecurityHeadersConfig':
        """
        Set X-Frame-Options
        
        Args:
            option: DENY, SAMEORIGIN, or ALLOW-FROM uri
            
        Returns:
            Self for chaining
        """
        self.headers["X-Frame-Options"] = option
        return self
    
    def set_referrer_policy(self, policy: str = "strict-origin-when-cross-origin") -> 'SecurityHeadersConfig':
        """
        Set Referrer-Policy
        
        Args:
            policy: Referrer policy value
            
        Returns:
            Self for chaining
        """
        self.headers["Referrer-Policy"] = policy
        return self
    
    def add_permission(self, feature: str, allowlist: str = "()") -> 'SecurityHeadersConfig':
        """
        Add Permissions Policy feature
        
        Args:
            feature: Feature name
            allowlist: Allowlist for feature
            
        Returns:
            Self for chaining
        """
        self.permissions[feature] = allowlist
        return self
    
    def build(self) -> Dict[str, str]:
        """
        Build final headers dictionary
        
        Returns:
            Dictionary of headers
        """
        headers = dict(self.headers)
        
        # Add default security headers
        headers["X-Content-Type-Options"] = "nosniff"
        headers["X-XSS-Protection"] = "1; mode=block"
        
        # Build CSP if directives present
        if self.csp_directives:
            csp_parts = [f"{k} {v}" for k, v in self.csp_directives.items()]
            headers["Content-Security-Policy"] = "; ".join(csp_parts)
        
        # Build Permissions Policy
        if self.permissions:
            perm_parts = [f"{k}={v}" for k, v in self.permissions.items()]
            headers["Permissions-Policy"] = ", ".join(perm_parts)
        
        return headers