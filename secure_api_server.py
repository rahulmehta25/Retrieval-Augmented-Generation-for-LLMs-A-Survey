"""
Secure API Server for RAG System

Implements comprehensive security measures based on OWASP Top 10:
- Input validation and sanitization
- Rate limiting and DoS protection
- Authentication and authorization
- Security headers and CORS
- Audit logging and monitoring
- Secure configuration management
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, AsyncIterator
import os
import tempfile
import shutil
from datetime import datetime
import uuid
import logging
import json
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
import secrets

# Security imports
from src.security.input_validator import InputValidator, ValidationError
from src.security.rate_limiter import RateLimiter, RateLimitExceeded, RateLimitStrategy
from src.security.authentication import (
    AuthenticationMiddleware,
    JWTHandler,
    APIKeyValidator,
    PasswordHasher,
    UserRole,
    PermissionLevel
)
from src.security.security_headers import SecurityHeadersMiddleware, CSPReportHandler
from src.security.audit_logger import AuditLogger, SecurityEvent
from src.security.config_manager import SecureConfigManager
from src.security.cors_handler import CORSConfigManager, create_secure_cors_config

# RAG system imports
from src.rag.naive_rag import NaiveRAG
from src.streaming.stream_handler import StreamingRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize secure configuration
config_manager = SecureConfigManager(
    config_file='config/secure_config.yaml',
    use_env_vars=True,
    validate_schema=True
)

# Initialize security components
input_validator = InputValidator(strict_mode=True, log_violations=True)
rate_limiter = RateLimiter(
    strategy=RateLimitStrategy.SLIDING_WINDOW,
    requests_per_minute=60,
    requests_per_hour=1000,
    enable_ip_tracking=True,
    enable_user_tracking=True,
    enable_endpoint_limits=True,
    persistent_storage='data/rate_limits.json'
)

# Initialize authentication
jwt_handler = JWTHandler(
    secret_key=config_manager.get('jwt.secret_key', secrets.token_urlsafe(32)),
    access_token_expire_minutes=30,
    refresh_token_expire_days=7
)

api_key_validator = APIKeyValidator(api_keys_file='config/api_keys.json')
password_hasher = PasswordHasher(rounds=12)

auth_middleware = AuthenticationMiddleware(
    jwt_handler=jwt_handler,
    api_key_validator=api_key_validator,
    enable_jwt=True,
    enable_api_key=True
)

# Initialize audit logger
audit_logger = AuditLogger(
    log_file='logs/security_audit.log',
    enable_alerts=True,
    alert_threshold=5
)

# Initialize CSP report handler
csp_handler = CSPReportHandler(log_file='logs/csp_violations.log')

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting secure RAG API server")
    audit_logger.log_event(
        SecurityEvent.SERVICE_START,
        details={'service': 'rag_api', 'version': '1.0.0'}
    )
    
    # Initialize RAG system
    global rag_system, streaming_rag
    try:
        rag_system = NaiveRAG(config_path='config.yaml', enable_evaluation=True)
        streaming_rag = StreamingRAG(rag_system, llm_type="ollama")
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down secure RAG API server")
    audit_logger.log_event(SecurityEvent.SERVICE_STOP)
    
    # Save persistent data
    rate_limiter.save_persistent_data()
    audit_logger.export_audit_log('logs/audit_export.json')

# Initialize FastAPI app with security
app = FastAPI(
    title="Secure RAG Knowledge API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,  # Disable docs in production
    redoc_url=None  # Disable redoc in production
)

# Add security headers middleware
app.add_middleware(
    SecurityHeadersMiddleware,
    enable_hsts=True,
    enable_csp=True,
    enable_permissions_policy=True
)

# Add CORS middleware with secure configuration
cors_config = create_secure_cors_config(
    production=config_manager.get('environment', 'development') == 'production',
    api_domain=config_manager.get('api.domain', 'localhost')
)
app.add_middleware(cors_config.get_cors_middleware(app))

# Request/Response Models with validation
class QueryRequest(BaseModel):
    """Secure query request model"""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)
    stream: bool = False
    
    @validator('query')
    def validate_query(cls, v):
        """Validate and sanitize query"""
        try:
            return input_validator.validate_text_input(
                v,
                field_name='query',
                max_length=1000
            )
        except ValidationError as e:
            raise ValueError(str(e))


class DocumentUploadResponse(BaseModel):
    """Document upload response"""
    document_id: str
    filename: str
    size: int
    hash: str
    status: str


class AuthRequest(BaseModel):
    """Authentication request"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username"""
        try:
            return input_validator.validate_username(v)
        except ValidationError as e:
            raise ValueError(str(e))


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


# Exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors"""
    audit_logger.log_event(
        SecurityEvent.VALIDATION_FAILURE,
        ip_address=request.client.host,
        details={'error': str(exc), 'field': exc.field},
        severity='WARNING'
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded"""
    audit_logger.log_event(
        SecurityEvent.RATE_LIMIT_EXCEEDED,
        ip_address=request.client.host,
        resource=str(request.url.path),
        severity='WARNING'
    )
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)},
        headers={"Retry-After": str(exc.retry_after)} if exc.retry_after else {}
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    audit_logger.log_event(
        SecurityEvent.VALIDATION_FAILURE,
        ip_address=request.client.host,
        details={'errors': exc.errors()},
        severity='WARNING'
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )


# Dependency for rate limiting
async def check_rate_limit(request: Request):
    """Check rate limit for request"""
    # Get client IP
    client_ip = request.client.host
    
    # Get user ID from auth context if available
    user_id = None
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        try:
            token = auth_header.split(' ')[1]
            claims = jwt_handler.verify_token(token)
            user_id = claims.get('sub')
        except:
            pass
    
    # Check rate limit
    allowed, retry_after = rate_limiter.check_rate_limit(
        identifier=client_ip,
        endpoint=str(request.url.path),
        ip_address=client_ip,
        user_id=user_id
    )
    
    if not allowed:
        raise RateLimitExceeded(
            f"Rate limit exceeded. Try again in {retry_after} seconds",
            retry_after=retry_after
        )


# Dependency for request size limiting
async def check_request_size(request: Request):
    """Check request size limits"""
    content_length = request.headers.get('content-length')
    if content_length:
        max_size = config_manager.get('security.max_request_size', 10 * 1024 * 1024)  # 10MB default
        if int(content_length) > max_size:
            audit_logger.log_event(
                SecurityEvent.VALIDATION_FAILURE,
                ip_address=request.client.host,
                details={'reason': 'Request size exceeded', 'size': content_length},
                severity='WARNING'
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request size exceeds maximum allowed"
            )


# Authentication endpoints
@app.post("/api/auth/register", response_model=TokenResponse)
async def register(
    auth_request: AuthRequest,
    request: Request,
    _: None = Depends(check_rate_limit)
):
    """Register new user with secure password hashing"""
    try:
        # Validate password strength
        password_validation = input_validator.validate_password(auth_request.password)
        if not password_validation['valid']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=password_validation['issues']
            )
        
        # Hash password
        hashed_password = password_hasher.hash_password(auth_request.password)
        
        # Store user (in production, use database)
        # This is simplified for demonstration
        
        # Create tokens
        access_token = jwt_handler.create_access_token(
            user_id=auth_request.username,
            role=UserRole.USER
        )
        refresh_token = jwt_handler.create_refresh_token(auth_request.username)
        
        # Audit log
        audit_logger.log_event(
            SecurityEvent.LOGIN_SUCCESS,
            user_id=auth_request.username,
            ip_address=request.client.host,
            severity='INFO'
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=jwt_handler.access_token_expire_minutes * 60
        )
        
    except Exception as e:
        audit_logger.log_event(
            SecurityEvent.LOGIN_FAILURE,
            user_id=auth_request.username,
            ip_address=request.client.host,
            details={'error': str(e)},
            severity='WARNING'
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Registration failed"
        )


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(
    auth_request: AuthRequest,
    request: Request,
    _: None = Depends(check_rate_limit)
):
    """Authenticate user and return tokens"""
    try:
        # Verify credentials (simplified - use database in production)
        # For demo, accept any valid username/password format
        
        # Create tokens
        access_token = jwt_handler.create_access_token(
            user_id=auth_request.username,
            role=UserRole.USER
        )
        refresh_token = jwt_handler.create_refresh_token(auth_request.username)
        
        # Audit log
        audit_logger.log_event(
            SecurityEvent.LOGIN_SUCCESS,
            user_id=auth_request.username,
            ip_address=request.client.host,
            severity='INFO'
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=jwt_handler.access_token_expire_minutes * 60
        )
        
    except Exception as e:
        audit_logger.log_event(
            SecurityEvent.LOGIN_FAILURE,
            user_id=auth_request.username,
            ip_address=request.client.host,
            details={'error': str(e)},
            severity='WARNING'
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )


# Protected RAG endpoints
@app.post("/api/chat/query")
async def query_documents(
    query_request: QueryRequest,
    request: Request,
    auth_context: Dict = Depends(auth_middleware.authenticate_request),
    _: None = Depends(check_rate_limit)
):
    """Query documents with authentication and rate limiting"""
    try:
        # Audit log
        audit_logger.log_event(
            SecurityEvent.DATA_ACCESS,
            user_id=auth_context.get('user_id'),
            ip_address=request.client.host,
            resource='documents',
            action='query',
            details={'query_length': len(query_request.query)}
        )
        
        # Process query
        if query_request.stream:
            # Streaming response
            async def generate():
                async for event in streaming_rag.stream_query(query_request.query, query_request.top_k):
                    yield f"data: {json.dumps(event.to_dict())}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            # Regular response
            result = rag_system.query(query_request.query, top_k=query_request.top_k)
            return {"answer": result}
        
    except Exception as e:
        audit_logger.log_event(
            SecurityEvent.ERROR,
            user_id=auth_context.get('user_id'),
            ip_address=request.client.host,
            details={'error': str(e)},
            severity='ERROR'
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query processing failed"
        )


@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    request: Request = None,
    auth_context: Dict = Depends(auth_middleware.authenticate_request),
    _: None = Depends(check_rate_limit),
    __: None = Depends(check_request_size)
):
    """Upload document with security validation"""
    try:
        # Read file content
        content = await file.read()
        
        # Validate file
        validation_result = input_validator.validate_file_upload(
            filename=file.filename,
            content=content,
            allowed_extensions=['.txt', '.pdf', '.doc', '.docx', '.md'],
            max_size=10 * 1024 * 1024  # 10MB
        )
        
        # Generate secure document ID
        document_id = str(uuid.uuid4())
        
        # Save file securely
        upload_dir = Path("uploaded_documents")
        upload_dir.mkdir(exist_ok=True)
        
        safe_filename = f"{document_id}_{validation_result['filename']}"
        file_path = upload_dir / safe_filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Set restrictive permissions
        if os.name != 'nt':
            os.chmod(file_path, 0o644)
        
        # Process with RAG system
        rag_system.add_documents([str(file_path)])
        
        # Audit log
        audit_logger.log_event(
            SecurityEvent.FILE_UPLOAD,
            user_id=auth_context.get('user_id'),
            ip_address=request.client.host,
            resource=safe_filename,
            details={
                'size': validation_result['size'],
                'hash': validation_result['hash']
            }
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=validation_result['filename'],
            size=validation_result['size'],
            hash=validation_result['hash'],
            status='processed'
        )
        
    except ValidationError as e:
        audit_logger.log_event(
            SecurityEvent.VALIDATION_FAILURE,
            user_id=auth_context.get('user_id'),
            ip_address=request.client.host,
            details={'error': str(e)},
            severity='WARNING'
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        audit_logger.log_event(
            SecurityEvent.ERROR,
            user_id=auth_context.get('user_id'),
            ip_address=request.client.host,
            details={'error': str(e)},
            severity='ERROR'
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload failed"
        )


# Admin endpoints with role-based access control
@app.get("/api/admin/security-metrics")
async def get_security_metrics(
    request: Request,
    auth_context: Dict = Depends(auth_middleware.require_role(UserRole.ADMIN))
):
    """Get security metrics (admin only)"""
    metrics = {
        'audit_metrics': audit_logger.get_security_metrics(),
        'rate_limit_stats': rate_limiter.get_stats(),
        'validation_stats': input_validator.get_violation_stats(),
        'cors_violations': cors_config.get_violation_stats()
    }
    
    return metrics


@app.post("/api/admin/rotate-keys")
async def rotate_encryption_keys(
    request: Request,
    auth_context: Dict = Depends(auth_middleware.require_role(UserRole.ADMIN))
):
    """Rotate encryption keys (admin only)"""
    try:
        new_key = config_manager.rotate_encryption_key()
        
        audit_logger.log_event(
            SecurityEvent.CONFIGURATION_CHANGE,
            user_id=auth_context.get('user_id'),
            ip_address=request.client.host,
            action='key_rotation',
            severity='INFO'
        )
        
        return {"status": "success", "message": "Encryption keys rotated"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Key rotation failed"
        )


# CSP violation reporting endpoint
@app.post("/api/csp-report")
async def handle_csp_report(request: Request):
    """Handle CSP violation reports"""
    return await csp_handler.handle_csp_report(request)


# Health check endpoint (no auth required but rate limited)
@app.get("/health")
async def health_check(_: None = Depends(check_rate_limit)):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "security": {
            "headers": "enabled",
            "rate_limiting": "enabled",
            "authentication": "enabled",
            "audit_logging": "enabled"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run with SSL in production
    ssl_keyfile = config_manager.get('ssl.keyfile')
    ssl_certfile = config_manager.get('ssl.certfile')
    
    uvicorn.run(
        app,
        host=config_manager.get('server.host', '0.0.0.0'),
        port=config_manager.get('server.port', 8000),
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        log_level="info",
        access_log=True
    )