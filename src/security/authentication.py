"""
Authentication and Authorization Module

Implements secure authentication mechanisms to prevent:
- Broken Authentication (OWASP A07:2021)
- Broken Access Control (OWASP A01:2021)
- Security Misconfiguration (OWASP A05:2021)
"""

import jwt
import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Any, Tuple
from enum import Enum
import logging
import json
from pathlib import Path
import bcrypt
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
import re

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    pass


class UserRole(Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_CLIENT = "api_client"


class PermissionLevel(Enum):
    """Permission levels for resources"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class PasswordHasher:
    """
    Secure password hashing using bcrypt
    
    Implements OWASP password storage best practices
    """
    
    def __init__(self, rounds: int = 12):
        """
        Initialize password hasher
        
        Args:
            rounds: Number of bcrypt rounds (12-15 recommended)
        """
        self.rounds = rounds
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using bcrypt
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        # Generate salt and hash
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify password against hash
        
        Args:
            password: Plain text password
            hashed: Hashed password
            
        Returns:
            True if password matches
        """
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False


class JWTHandler:
    """
    JWT token handler for stateless authentication
    
    Implements secure JWT practices
    """
    
    def __init__(self,
                 secret_key: str,
                 algorithm: str = "HS256",
                 access_token_expire_minutes: int = 30,
                 refresh_token_expire_days: int = 7,
                 issuer: str = "rag-system"):
        """
        Initialize JWT handler
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT signing algorithm
            access_token_expire_minutes: Access token expiration time
            refresh_token_expire_days: Refresh token expiration time
            issuer: Token issuer identifier
        """
        if len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters")
        
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.issuer = issuer
        
        # Token blacklist for revocation
        self.blacklisted_tokens = set()
    
    def create_access_token(self,
                          user_id: str,
                          role: UserRole,
                          additional_claims: Optional[Dict] = None) -> str:
        """
        Create JWT access token
        
        Args:
            user_id: User identifier
            role: User role
            additional_claims: Additional JWT claims
            
        Returns:
            JWT access token
        """
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=self.access_token_expire_minutes)
        
        claims = {
            "sub": user_id,
            "role": role.value,
            "iat": now,
            "exp": expire,
            "iss": self.issuer,
            "type": "access",
            "jti": secrets.token_urlsafe(16)  # Unique token ID
        }
        
        if additional_claims:
            claims.update(additional_claims)
        
        token = jwt.encode(claims, self.secret_key, algorithm=self.algorithm)
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """
        Create JWT refresh token
        
        Args:
            user_id: User identifier
            
        Returns:
            JWT refresh token
        """
        now = datetime.now(timezone.utc)
        expire = now + timedelta(days=self.refresh_token_expire_days)
        
        claims = {
            "sub": user_id,
            "iat": now,
            "exp": expire,
            "iss": self.issuer,
            "type": "refresh",
            "jti": secrets.token_urlsafe(16)
        }
        
        token = jwt.encode(claims, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token
            token_type: Expected token type
            
        Returns:
            Token claims
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise AuthenticationError("Token has been revoked")
            
            # Decode token
            claims = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer
            )
            
            # Verify token type
            if claims.get("type") != token_type:
                raise AuthenticationError(f"Invalid token type: expected {token_type}")
            
            return claims
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    def revoke_token(self, token: str):
        """Revoke a token by adding to blacklist"""
        self.blacklisted_tokens.add(token)
        logger.info(f"Token revoked: {token[:20]}...")
    
    def refresh_access_token(self, refresh_token: str) -> Tuple[str, str]:
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Tuple of (new_access_token, new_refresh_token)
        """
        # Verify refresh token
        claims = self.verify_token(refresh_token, token_type="refresh")
        
        # Create new tokens
        user_id = claims["sub"]
        # Note: You would fetch the actual role from database
        role = UserRole.USER  # Default role
        
        new_access = self.create_access_token(user_id, role)
        new_refresh = self.create_refresh_token(user_id)
        
        # Revoke old refresh token
        self.revoke_token(refresh_token)
        
        return new_access, new_refresh


class APIKeyValidator:
    """
    API key validation for service-to-service authentication
    """
    
    def __init__(self, api_keys_file: Optional[str] = None):
        """
        Initialize API key validator
        
        Args:
            api_keys_file: Path to file containing valid API keys
        """
        self.api_keys = {}
        self.key_permissions = {}
        
        if api_keys_file:
            self._load_api_keys(api_keys_file)
    
    def _load_api_keys(self, file_path: str):
        """Load API keys from file"""
        try:
            path = Path(file_path)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    for key_data in data.get('keys', []):
                        key_hash = self._hash_api_key(key_data['key'])
                        self.api_keys[key_hash] = {
                            'name': key_data.get('name'),
                            'created': key_data.get('created'),
                            'permissions': key_data.get('permissions', [])
                        }
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def generate_api_key(self, name: str, permissions: List[str]) -> str:
        """
        Generate new API key
        
        Args:
            name: Name/description for the API key
            permissions: List of permissions for the key
            
        Returns:
            Generated API key
        """
        # Generate secure random key
        api_key = f"rag_{secrets.token_urlsafe(32)}"
        
        # Store hashed version
        key_hash = self._hash_api_key(api_key)
        self.api_keys[key_hash] = {
            'name': name,
            'created': datetime.now().isoformat(),
            'permissions': permissions
        }
        
        logger.info(f"Generated API key for {name}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Validate API key
        
        Args:
            api_key: API key to validate
            
        Returns:
            Key information if valid
            
        Raises:
            AuthenticationError: If key is invalid
        """
        key_hash = self._hash_api_key(api_key)
        
        if key_hash not in self.api_keys:
            raise AuthenticationError("Invalid API key")
        
        return self.api_keys[key_hash]
    
    def revoke_api_key(self, api_key: str):
        """Revoke an API key"""
        key_hash = self._hash_api_key(api_key)
        if key_hash in self.api_keys:
            del self.api_keys[key_hash]
            logger.info(f"Revoked API key")


class AuthenticationMiddleware:
    """
    FastAPI authentication middleware
    
    Provides dependency injection for route protection
    """
    
    def __init__(self,
                 jwt_handler: JWTHandler,
                 api_key_validator: Optional[APIKeyValidator] = None,
                 enable_jwt: bool = True,
                 enable_api_key: bool = True):
        """
        Initialize authentication middleware
        
        Args:
            jwt_handler: JWT handler instance
            api_key_validator: API key validator instance
            enable_jwt: Enable JWT authentication
            enable_api_key: Enable API key authentication
        """
        self.jwt_handler = jwt_handler
        self.api_key_validator = api_key_validator
        self.enable_jwt = enable_jwt
        self.enable_api_key = enable_api_key
        
        # FastAPI security schemes
        self.bearer_scheme = HTTPBearer(auto_error=False)
        self.api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
    
    async def authenticate_request(self,
                                  credentials: Optional[HTTPAuthorizationCredentials] = Security(HTTPBearer(auto_error=False)),
                                  api_key: Optional[str] = Security(APIKeyHeader(name="X-API-Key", auto_error=False)),
                                  request: Request = None) -> Dict[str, Any]:
        """
        Authenticate incoming request
        
        Args:
            credentials: Bearer token credentials
            api_key: API key from header
            request: FastAPI request object
            
        Returns:
            Authentication context
            
        Raises:
            HTTPException: If authentication fails
        """
        # Try JWT authentication
        if self.enable_jwt and credentials:
            try:
                claims = self.jwt_handler.verify_token(credentials.credentials)
                return {
                    'authenticated': True,
                    'method': 'jwt',
                    'user_id': claims['sub'],
                    'role': claims.get('role'),
                    'claims': claims
                }
            except AuthenticationError as e:
                logger.warning(f"JWT authentication failed: {e}")
        
        # Try API key authentication
        if self.enable_api_key and api_key and self.api_key_validator:
            try:
                key_info = self.api_key_validator.validate_api_key(api_key)
                return {
                    'authenticated': True,
                    'method': 'api_key',
                    'key_name': key_info['name'],
                    'permissions': key_info['permissions']
                }
            except AuthenticationError as e:
                logger.warning(f"API key authentication failed: {e}")
        
        # Authentication failed
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    def require_role(self, required_role: UserRole):
        """
        Dependency to require specific role
        
        Args:
            required_role: Required user role
            
        Returns:
            FastAPI dependency
        """
        async def role_checker(auth_context: Dict = Depends(self.authenticate_request)):
            user_role = auth_context.get('role')
            
            if not user_role:
                raise HTTPException(
                    status_code=403,
                    detail="Role information not available"
                )
            
            # Check role hierarchy
            role_hierarchy = {
                UserRole.VIEWER.value: 0,
                UserRole.USER.value: 1,
                UserRole.API_CLIENT.value: 1,
                UserRole.ADMIN.value: 2
            }
            
            if role_hierarchy.get(user_role, -1) < role_hierarchy.get(required_role.value, 999):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required role: {required_role.value}"
                )
            
            return auth_context
        
        return role_checker
    
    def require_permission(self, resource: str, permission: PermissionLevel):
        """
        Dependency to require specific permission
        
        Args:
            resource: Resource identifier
            permission: Required permission level
            
        Returns:
            FastAPI dependency
        """
        async def permission_checker(auth_context: Dict = Depends(self.authenticate_request)):
            # For JWT auth, check role-based permissions
            if auth_context.get('method') == 'jwt':
                role = auth_context.get('role')
                
                # Admin has all permissions
                if role == UserRole.ADMIN.value:
                    return auth_context
                
                # Check specific role permissions
                role_permissions = {
                    UserRole.USER.value: [PermissionLevel.READ, PermissionLevel.WRITE],
                    UserRole.VIEWER.value: [PermissionLevel.READ]
                }
                
                if permission not in role_permissions.get(role, []):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Insufficient permissions for {resource}"
                    )
            
            # For API key auth, check key permissions
            elif auth_context.get('method') == 'api_key':
                key_permissions = auth_context.get('permissions', [])
                required = f"{resource}:{permission.value}"
                
                if required not in key_permissions and "*" not in key_permissions:
                    raise HTTPException(
                        status_code=403,
                        detail=f"API key lacks permission: {required}"
                    )
            
            return auth_context
        
        return permission_checker


# Helper functions for easy integration
def create_access_token(user_id: str, role: UserRole, secret_key: str) -> str:
    """
    Create JWT access token
    
    Args:
        user_id: User identifier
        role: User role
        secret_key: Secret key for signing
        
    Returns:
        JWT token
    """
    handler = JWTHandler(secret_key)
    return handler.create_access_token(user_id, role)


def verify_token(token: str, secret_key: str) -> Dict[str, Any]:
    """
    Verify JWT token
    
    Args:
        token: JWT token
        secret_key: Secret key for verification
        
    Returns:
        Token claims
    """
    handler = JWTHandler(secret_key)
    return handler.verify_token(token)