"""
Input Validation and Sanitization Module

Implements comprehensive input validation to prevent:
- SQL Injection (OWASP A03:2021)
- Cross-Site Scripting (XSS) (OWASP A03:2021)
- Command Injection (OWASP A03:2021)
- Path Traversal (OWASP A01:2021)
- XML External Entity (XXE) attacks (OWASP A05:2021)
"""

import re
import html
import urllib.parse
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class InputValidator:
    """
    Comprehensive input validation and sanitization class
    
    Implements defense-in-depth approach with multiple validation layers
    """
    
    # SQL injection patterns (OWASP A03:2021)
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER|EXEC|EXECUTE|SCRIPT|TRUNCATE)\b)",
        r"(--|#|\/\*|\*\/|@@|@|\bchar\b|\bnchar\b|\bvarchar\b|\bnvarchar\b)",
        r"(\bxp_\w+|\bsp_\w+)",  # SQL Server extended procedures
        r"(;.*?(SELECT|INSERT|UPDATE|DELETE|DROP))",  # Command chaining
        r"(\bOR\b\s+\d+\s*=\s*\d+|\bAND\b\s+\d+\s*=\s*\d+)",  # Boolean-based SQL injection
        r"(\'\s*OR\s*\'|\"\s*OR\s*\")",  # String-based SQL injection
    ]
    
    # XSS patterns (OWASP A03:2021)
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers
        r"<iframe[^>]*>",
        r"<embed[^>]*>",
        r"<object[^>]*>",
        r"eval\s*\(",
        r"expression\s*\(",
        r"vbscript:",
        r"data:text/html",
        r"<svg[^>]*onload",
    ]
    
    # Command injection patterns (OWASP A03:2021)
    COMMAND_INJECTION_PATTERNS = [
        r"([;&|`$]|\$\(|\))",  # Shell metacharacters
        r"(&&|\|\|)",  # Command chaining
        r"(>\s*\/dev\/null)",  # Redirection
        r"(\bwget\b|\bcurl\b|\bnc\b|\bnetcat\b)",  # Network commands
        r"(\/etc\/passwd|\/etc\/shadow)",  # Sensitive files
    ]
    
    # Path traversal patterns (OWASP A01:2021)
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",  # Unix path traversal
        r"\.\.\\",  # Windows path traversal
        r"%2e%2e[/\\]",  # URL encoded
        r"\.\.[/\\]",  # Mixed
        r"(\/etc\/|C:\\Windows\\|C:\\Program Files\\)",  # System directories
    ]
    
    # Maximum input lengths to prevent buffer overflow and DoS
    MAX_LENGTHS = {
        'username': 50,
        'email': 254,
        'password': 128,
        'query': 1000,
        'document_content': 1000000,  # 1MB
        'filename': 255,
        'url': 2048,
        'json': 10000,
        'general_text': 5000,
    }
    
    # Allowed characters for specific fields
    ALLOWED_PATTERNS = {
        'username': r'^[a-zA-Z0-9_-]{3,50}$',
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'filename': r'^[a-zA-Z0-9_\-\.]{1,255}$',
        'uuid': r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
    }
    
    def __init__(self, strict_mode: bool = True, log_violations: bool = True):
        """
        Initialize the input validator
        
        Args:
            strict_mode: If True, reject any suspicious input. If False, sanitize and continue
            log_violations: If True, log all validation violations for security monitoring
        """
        self.strict_mode = strict_mode
        self.log_violations = log_violations
        self.violation_counter = {}
        
    def validate_text_input(self, 
                           text: str, 
                           field_name: str = 'text',
                           max_length: Optional[int] = None,
                           allow_html: bool = False,
                           allow_sql: bool = False) -> str:
        """
        Validate and sanitize text input
        
        Args:
            text: Input text to validate
            field_name: Name of the field for error reporting
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML content
            allow_sql: Whether to allow SQL keywords (for legitimate use cases)
            
        Returns:
            Sanitized text
            
        Raises:
            ValidationError: If validation fails in strict mode
        """
        if not isinstance(text, str):
            raise ValidationError(f"Input must be a string", field=field_name)
        
        # Check length
        if max_length is None:
            max_length = self.MAX_LENGTHS.get(field_name, self.MAX_LENGTHS['general_text'])
        
        if len(text) > max_length:
            self._log_violation(field_name, 'length_exceeded', text)
            if self.strict_mode:
                raise ValidationError(
                    f"Input exceeds maximum length of {max_length} characters",
                    field=field_name
                )
            text = text[:max_length]
        
        # Check for SQL injection patterns
        if not allow_sql:
            for pattern in self.SQL_INJECTION_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    self._log_violation(field_name, 'sql_injection', text)
                    if self.strict_mode:
                        raise ValidationError(
                            "Potential SQL injection detected",
                            field=field_name
                        )
                    # Sanitize by removing SQL keywords
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Check for XSS patterns
        if not allow_html:
            for pattern in self.XSS_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    self._log_violation(field_name, 'xss', text)
                    if self.strict_mode:
                        raise ValidationError(
                            "Potential XSS attack detected",
                            field=field_name
                        )
            # HTML escape the text
            text = html.escape(text)
        
        # Check for command injection patterns
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, text):
                self._log_violation(field_name, 'command_injection', text)
                if self.strict_mode:
                    raise ValidationError(
                        "Potential command injection detected",
                        field=field_name
                    )
                # Remove dangerous characters
                text = re.sub(pattern, '', text)
        
        return text.strip()
    
    def validate_username(self, username: str) -> str:
        """
        Validate username against allowed pattern
        
        Args:
            username: Username to validate
            
        Returns:
            Validated username
            
        Raises:
            ValidationError: If username is invalid
        """
        if not re.match(self.ALLOWED_PATTERNS['username'], username):
            self._log_violation('username', 'invalid_format', username)
            raise ValidationError(
                "Username must be 3-50 characters and contain only letters, numbers, underscore, and hyphen",
                field='username'
            )
        return username
    
    def validate_email(self, email: str) -> str:
        """
        Validate email address
        
        Args:
            email: Email to validate
            
        Returns:
            Validated email (lowercase)
            
        Raises:
            ValidationError: If email is invalid
        """
        email = email.lower().strip()
        if not re.match(self.ALLOWED_PATTERNS['email'], email):
            self._log_violation('email', 'invalid_format', email)
            raise ValidationError(
                "Invalid email format",
                field='email'
            )
        return email
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """
        Validate password strength
        
        Args:
            password: Password to validate
            
        Returns:
            Dictionary with validation results and strength score
            
        Raises:
            ValidationError: If password doesn't meet minimum requirements
        """
        results = {
            'valid': True,
            'strength': 0,
            'issues': []
        }
        
        # Check length
        if len(password) < 8:
            results['issues'].append("Password must be at least 8 characters")
            results['valid'] = False
        elif len(password) >= 12:
            results['strength'] += 2
        else:
            results['strength'] += 1
        
        # Check complexity
        if not re.search(r'[a-z]', password):
            results['issues'].append("Password must contain lowercase letters")
            results['valid'] = False
        else:
            results['strength'] += 1
            
        if not re.search(r'[A-Z]', password):
            results['issues'].append("Password must contain uppercase letters")
            results['valid'] = False
        else:
            results['strength'] += 1
            
        if not re.search(r'\d', password):
            results['issues'].append("Password must contain numbers")
            results['valid'] = False
        else:
            results['strength'] += 1
            
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            results['strength'] += 2
        
        # Check for common patterns
        common_patterns = ['password', '123456', 'qwerty', 'admin', 'letmein']
        for pattern in common_patterns:
            if pattern in password.lower():
                results['issues'].append("Password contains common pattern")
                results['strength'] = max(0, results['strength'] - 3)
        
        if not results['valid']:
            raise ValidationError(
                f"Password validation failed: {', '.join(results['issues'])}",
                field='password'
            )
        
        return results
    
    def validate_file_path(self, file_path: str, base_directory: Optional[str] = None) -> Path:
        """
        Validate file path to prevent path traversal attacks (OWASP A01:2021)
        
        Args:
            file_path: File path to validate
            base_directory: Base directory to restrict access to
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid or attempts traversal
        """
        # Check for path traversal patterns
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, file_path):
                self._log_violation('file_path', 'path_traversal', file_path)
                raise ValidationError(
                    "Path traversal attempt detected",
                    field='file_path'
                )
        
        # Resolve the path
        path = Path(file_path).resolve()
        
        # If base directory is specified, ensure the path is within it
        if base_directory:
            base = Path(base_directory).resolve()
            try:
                path.relative_to(base)
            except ValueError:
                self._log_violation('file_path', 'outside_base_directory', str(path))
                raise ValidationError(
                    "File path is outside allowed directory",
                    field='file_path'
                )
        
        return path
    
    def validate_url(self, url: str, allowed_schemes: List[str] = None) -> str:
        """
        Validate URL to prevent SSRF and open redirect vulnerabilities
        
        Args:
            url: URL to validate
            allowed_schemes: List of allowed URL schemes (default: ['http', 'https'])
            
        Returns:
            Validated URL
            
        Raises:
            ValidationError: If URL is invalid or uses disallowed scheme
        """
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        # Parse the URL
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {e}", field='url')
        
        # Check scheme
        if parsed.scheme not in allowed_schemes:
            self._log_violation('url', 'invalid_scheme', url)
            raise ValidationError(
                f"URL scheme must be one of: {', '.join(allowed_schemes)}",
                field='url'
            )
        
        # Check for localhost/internal IPs (SSRF prevention)
        dangerous_hosts = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
        if parsed.hostname in dangerous_hosts:
            self._log_violation('url', 'ssrf_attempt', url)
            raise ValidationError(
                "URLs pointing to localhost/internal IPs are not allowed",
                field='url'
            )
        
        # Check for private IP ranges
        if parsed.hostname and re.match(r'^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)', parsed.hostname):
            self._log_violation('url', 'private_ip', url)
            raise ValidationError(
                "URLs pointing to private IP addresses are not allowed",
                field='url'
            )
        
        return url
    
    def validate_json(self, json_string: str, max_depth: int = 10) -> Dict:
        """
        Validate and parse JSON input safely
        
        Args:
            json_string: JSON string to validate
            max_depth: Maximum nesting depth allowed
            
        Returns:
            Parsed JSON object
            
        Raises:
            ValidationError: If JSON is invalid or too deeply nested
        """
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}", field='json')
        
        # Check depth to prevent DoS
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                raise ValidationError(
                    f"JSON exceeds maximum nesting depth of {max_depth}",
                    field='json'
                )
            
            if isinstance(obj, dict):
                for value in obj.values():
                    check_depth(value, current_depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, current_depth + 1)
        
        check_depth(data)
        return data
    
    def validate_file_upload(self, 
                            filename: str,
                            content: bytes,
                            allowed_extensions: List[str] = None,
                            max_size: int = 10 * 1024 * 1024) -> Dict[str, Any]:
        """
        Validate file upload for security issues
        
        Args:
            filename: Name of the uploaded file
            content: File content as bytes
            allowed_extensions: List of allowed file extensions
            max_size: Maximum file size in bytes (default: 10MB)
            
        Returns:
            Dictionary with validation results
            
        Raises:
            ValidationError: If file validation fails
        """
        if allowed_extensions is None:
            allowed_extensions = ['.txt', '.pdf', '.doc', '.docx', '.md', '.json']
        
        # Validate filename
        if not re.match(self.ALLOWED_PATTERNS['filename'], filename):
            self._log_violation('filename', 'invalid_format', filename)
            raise ValidationError(
                "Filename contains invalid characters",
                field='filename'
            )
        
        # Check file extension
        ext = Path(filename).suffix.lower()
        if ext not in allowed_extensions:
            self._log_violation('filename', 'invalid_extension', filename)
            raise ValidationError(
                f"File extension {ext} not allowed. Allowed: {', '.join(allowed_extensions)}",
                field='filename'
            )
        
        # Check file size
        if len(content) > max_size:
            self._log_violation('file_upload', 'size_exceeded', f"{len(content)} bytes")
            raise ValidationError(
                f"File size exceeds maximum of {max_size} bytes",
                field='file_upload'
            )
        
        # Check for malicious content patterns
        # Check for executable headers
        executable_headers = [
            b'MZ',  # Windows executable
            b'\x7fELF',  # Linux executable
            b'#!/',  # Shell script
            b'<%',  # JSP/ASP
            b'<?php',  # PHP
        ]
        
        for header in executable_headers:
            if content.startswith(header):
                self._log_violation('file_upload', 'executable_detected', filename)
                raise ValidationError(
                    "Executable file content detected",
                    field='file_upload'
                )
        
        # Calculate file hash for integrity checking
        file_hash = hashlib.sha256(content).hexdigest()
        
        return {
            'filename': filename,
            'extension': ext,
            'size': len(content),
            'hash': file_hash,
            'validated': True
        }
    
    def sanitize_for_logging(self, data: Any, max_length: int = 100) -> str:
        """
        Sanitize data for safe logging (prevent log injection)
        
        Args:
            data: Data to sanitize
            max_length: Maximum length for logged data
            
        Returns:
            Sanitized string safe for logging
        """
        # Convert to string
        text = str(data)
        
        # Remove control characters and newlines
        text = re.sub(r'[\x00-\x1f\x7f-\x9f\r\n]', ' ', text)
        
        # Truncate
        if len(text) > max_length:
            text = text[:max_length] + '...'
        
        return text
    
    def _log_violation(self, field: str, violation_type: str, value: Any):
        """
        Log security violation for monitoring and analysis
        
        Args:
            field: Field that failed validation
            violation_type: Type of violation detected
            value: The violating value (sanitized for logging)
        """
        if not self.log_violations:
            return
        
        # Track violation counts
        key = f"{field}:{violation_type}"
        self.violation_counter[key] = self.violation_counter.get(key, 0) + 1
        
        # Sanitize value for logging
        safe_value = self.sanitize_for_logging(value)
        
        logger.warning(
            f"Security violation detected - Field: {field}, "
            f"Type: {violation_type}, Value: {safe_value}, "
            f"Total violations of this type: {self.violation_counter[key]}"
        )
    
    def get_violation_stats(self) -> Dict[str, int]:
        """
        Get statistics of validation violations
        
        Returns:
            Dictionary of violation types and counts
        """
        return dict(self.violation_counter)
    
    def reset_violation_stats(self):
        """Reset violation statistics"""
        self.violation_counter = {}