"""
Rate Limiting Module

Implements rate limiting to prevent:
- Denial of Service (DoS) attacks (OWASP A05:2021)
- Brute force attacks (OWASP A07:2021)
- Resource exhaustion
"""

import time
import asyncio
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
import hashlib
import logging
from enum import Enum
import json
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        self.message = message
        self.retry_after = retry_after  # Seconds until next request allowed
        super().__init__(self.message)


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimiter:
    """
    Advanced rate limiter with multiple strategies
    
    Implements defense against DoS and brute force attacks
    """
    
    def __init__(self,
                 strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
                 requests_per_minute: int = 60,
                 requests_per_hour: int = 1000,
                 burst_size: int = 10,
                 enable_ip_tracking: bool = True,
                 enable_user_tracking: bool = True,
                 enable_endpoint_limits: bool = True,
                 persistent_storage: Optional[str] = None):
        """
        Initialize rate limiter
        
        Args:
            strategy: Rate limiting strategy to use
            requests_per_minute: Maximum requests per minute
            requests_per_hour: Maximum requests per hour
            burst_size: Maximum burst size for token bucket
            enable_ip_tracking: Track and limit by IP address
            enable_user_tracking: Track and limit by user ID
            enable_endpoint_limits: Apply different limits per endpoint
            persistent_storage: Path to store persistent rate limit data
        """
        self.strategy = strategy
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.enable_ip_tracking = enable_ip_tracking
        self.enable_user_tracking = enable_user_tracking
        self.enable_endpoint_limits = enable_endpoint_limits
        self.persistent_storage = persistent_storage
        
        # Storage for different tracking methods
        self.ip_limiter = defaultdict(lambda: self._create_limiter())
        self.user_limiter = defaultdict(lambda: self._create_limiter())
        self.endpoint_limiter = defaultdict(lambda: self._create_limiter())
        
        # Blacklist for repeat offenders
        self.blacklist = set()
        self.offense_counter = defaultdict(int)
        
        # Endpoint-specific limits
        self.endpoint_configs = {
            '/api/auth/login': {
                'requests_per_minute': 5,  # Strict limit for login attempts
                'requests_per_hour': 20,
                'burst_size': 2
            },
            '/api/auth/register': {
                'requests_per_minute': 3,
                'requests_per_hour': 10,
                'burst_size': 1
            },
            '/api/documents/upload': {
                'requests_per_minute': 10,
                'requests_per_hour': 100,
                'burst_size': 3
            },
            '/api/chat/query': {
                'requests_per_minute': 30,
                'requests_per_hour': 500,
                'burst_size': 5
            }
        }
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Load persistent data if available
        if persistent_storage:
            self._load_persistent_data()
    
    def _create_limiter(self) -> Dict[str, Any]:
        """Create a new limiter instance based on strategy"""
        if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return {
                'requests': deque(),
                'tokens': self.burst_size
            }
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return {
                'tokens': self.burst_size,
                'last_refill': time.time()
            }
        elif self.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return {
                'queue': deque(maxlen=self.burst_size),
                'last_leak': time.time()
            }
        else:  # FIXED_WINDOW
            return {
                'count': 0,
                'window_start': time.time()
            }
    
    def check_rate_limit(self,
                        identifier: str,
                        endpoint: Optional[str] = None,
                        ip_address: Optional[str] = None,
                        user_id: Optional[str] = None) -> Tuple[bool, Optional[int]]:
        """
        Check if request should be rate limited
        
        Args:
            identifier: Primary identifier for the request
            endpoint: API endpoint being accessed
            ip_address: Client IP address
            user_id: User ID if authenticated
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        with self._lock:
            # Check blacklist first
            if self._is_blacklisted(identifier, ip_address, user_id):
                logger.warning(f"Blacklisted entity attempted access: {identifier}")
                return False, 3600  # Retry after 1 hour
            
            # Get endpoint-specific limits
            limits = self._get_endpoint_limits(endpoint)
            
            # Check IP-based rate limit
            if self.enable_ip_tracking and ip_address:
                allowed, retry_after = self._check_limit(
                    self.ip_limiter[ip_address],
                    limits
                )
                if not allowed:
                    self._record_offense(ip_address)
                    return False, retry_after
            
            # Check user-based rate limit
            if self.enable_user_tracking and user_id:
                allowed, retry_after = self._check_limit(
                    self.user_limiter[user_id],
                    limits
                )
                if not allowed:
                    self._record_offense(user_id)
                    return False, retry_after
            
            # Check endpoint-based rate limit
            if self.enable_endpoint_limits and endpoint:
                allowed, retry_after = self._check_limit(
                    self.endpoint_limiter[endpoint],
                    limits
                )
                if not allowed:
                    return False, retry_after
            
            return True, None
    
    def _check_limit(self, limiter: Dict[str, Any], limits: Dict[str, int]) -> Tuple[bool, Optional[int]]:
        """
        Check rate limit based on strategy
        
        Args:
            limiter: Limiter instance
            limits: Rate limit configuration
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        current_time = time.time()
        
        if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(limiter, limits, current_time)
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(limiter, limits, current_time)
        elif self.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return self._check_leaky_bucket(limiter, limits, current_time)
        else:  # FIXED_WINDOW
            return self._check_fixed_window(limiter, limits, current_time)
    
    def _check_sliding_window(self, limiter: Dict, limits: Dict, current_time: float) -> Tuple[bool, Optional[int]]:
        """Sliding window rate limiting algorithm"""
        # Clean old requests
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        
        # Remove requests older than 1 hour
        while limiter['requests'] and limiter['requests'][0] < hour_ago:
            limiter['requests'].popleft()
        
        # Count requests in different windows
        minute_count = sum(1 for t in limiter['requests'] if t > minute_ago)
        hour_count = len(limiter['requests'])
        
        # Check limits
        if minute_count >= limits['requests_per_minute']:
            retry_after = 60 - (current_time - minute_ago)
            return False, int(retry_after)
        
        if hour_count >= limits['requests_per_hour']:
            retry_after = 3600 - (current_time - hour_ago)
            return False, int(retry_after)
        
        # Add current request
        limiter['requests'].append(current_time)
        return True, None
    
    def _check_token_bucket(self, limiter: Dict, limits: Dict, current_time: float) -> Tuple[bool, Optional[int]]:
        """Token bucket rate limiting algorithm"""
        # Refill tokens based on time passed
        time_passed = current_time - limiter['last_refill']
        tokens_to_add = time_passed * (limits['requests_per_minute'] / 60.0)
        
        limiter['tokens'] = min(
            limits['burst_size'],
            limiter['tokens'] + tokens_to_add
        )
        limiter['last_refill'] = current_time
        
        # Check if token available
        if limiter['tokens'] >= 1:
            limiter['tokens'] -= 1
            return True, None
        
        # Calculate retry time
        tokens_needed = 1 - limiter['tokens']
        retry_after = tokens_needed / (limits['requests_per_minute'] / 60.0)
        return False, int(retry_after)
    
    def _check_leaky_bucket(self, limiter: Dict, limits: Dict, current_time: float) -> Tuple[bool, Optional[int]]:
        """Leaky bucket rate limiting algorithm"""
        # Leak requests based on time passed
        leak_rate = limits['requests_per_minute'] / 60.0
        time_passed = current_time - limiter['last_leak']
        requests_to_leak = int(time_passed * leak_rate)
        
        for _ in range(min(requests_to_leak, len(limiter['queue']))):
            if limiter['queue']:
                limiter['queue'].popleft()
        
        limiter['last_leak'] = current_time
        
        # Check if bucket has space
        if len(limiter['queue']) < limits['burst_size']:
            limiter['queue'].append(current_time)
            return True, None
        
        # Calculate retry time
        retry_after = 1.0 / leak_rate
        return False, int(retry_after)
    
    def _check_fixed_window(self, limiter: Dict, limits: Dict, current_time: float) -> Tuple[bool, Optional[int]]:
        """Fixed window rate limiting algorithm"""
        window_duration = 60  # 1 minute window
        
        # Check if we're in a new window
        if current_time - limiter['window_start'] >= window_duration:
            limiter['count'] = 0
            limiter['window_start'] = current_time
        
        # Check limit
        if limiter['count'] >= limits['requests_per_minute']:
            retry_after = window_duration - (current_time - limiter['window_start'])
            return False, int(retry_after)
        
        limiter['count'] += 1
        return True, None
    
    def _get_endpoint_limits(self, endpoint: Optional[str]) -> Dict[str, int]:
        """Get rate limits for specific endpoint"""
        if endpoint and endpoint in self.endpoint_configs:
            return self.endpoint_configs[endpoint]
        
        return {
            'requests_per_minute': self.requests_per_minute,
            'requests_per_hour': self.requests_per_hour,
            'burst_size': self.burst_size
        }
    
    def _is_blacklisted(self, identifier: str, ip_address: Optional[str], user_id: Optional[str]) -> bool:
        """Check if entity is blacklisted"""
        return (identifier in self.blacklist or
                (ip_address and ip_address in self.blacklist) or
                (user_id and user_id in self.blacklist))
    
    def _record_offense(self, identifier: str):
        """Record rate limit violation"""
        self.offense_counter[identifier] += 1
        
        # Auto-blacklist after repeated offenses
        if self.offense_counter[identifier] >= 10:
            logger.warning(f"Auto-blacklisting {identifier} after {self.offense_counter[identifier]} offenses")
            self.blacklist.add(identifier)
    
    def add_to_blacklist(self, identifier: str):
        """Manually add identifier to blacklist"""
        self.blacklist.add(identifier)
        logger.info(f"Added {identifier} to blacklist")
    
    def remove_from_blacklist(self, identifier: str):
        """Remove identifier from blacklist"""
        self.blacklist.discard(identifier)
        if identifier in self.offense_counter:
            del self.offense_counter[identifier]
        logger.info(f"Removed {identifier} from blacklist")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            'strategy': self.strategy.value,
            'blacklist_size': len(self.blacklist),
            'tracked_ips': len(self.ip_limiter),
            'tracked_users': len(self.user_limiter),
            'tracked_endpoints': len(self.endpoint_limiter),
            'total_offenses': sum(self.offense_counter.values()),
            'top_offenders': sorted(
                self.offense_counter.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def reset_limits(self, identifier: Optional[str] = None):
        """Reset rate limits for specific identifier or all"""
        with self._lock:
            if identifier:
                if identifier in self.ip_limiter:
                    del self.ip_limiter[identifier]
                if identifier in self.user_limiter:
                    del self.user_limiter[identifier]
                if identifier in self.offense_counter:
                    del self.offense_counter[identifier]
            else:
                self.ip_limiter.clear()
                self.user_limiter.clear()
                self.endpoint_limiter.clear()
                self.offense_counter.clear()
    
    def _load_persistent_data(self):
        """Load persistent rate limit data from disk"""
        if not self.persistent_storage:
            return
        
        path = Path(self.persistent_storage)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.blacklist = set(data.get('blacklist', []))
                    self.offense_counter = defaultdict(int, data.get('offense_counter', {}))
                logger.info(f"Loaded persistent rate limit data from {path}")
            except Exception as e:
                logger.error(f"Failed to load persistent data: {e}")
    
    def save_persistent_data(self):
        """Save persistent rate limit data to disk"""
        if not self.persistent_storage:
            return
        
        path = Path(self.persistent_storage)
        try:
            data = {
                'blacklist': list(self.blacklist),
                'offense_counter': dict(self.offense_counter),
                'timestamp': datetime.now().isoformat()
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved persistent rate limit data to {path}")
        except Exception as e:
            logger.error(f"Failed to save persistent data: {e}")


class DistributedRateLimiter(RateLimiter):
    """
    Distributed rate limiter for multi-instance deployments
    
    Uses Redis or similar for shared state
    """
    
    def __init__(self, redis_client=None, **kwargs):
        """
        Initialize distributed rate limiter
        
        Args:
            redis_client: Redis client for distributed state
            **kwargs: Arguments for parent RateLimiter
        """
        super().__init__(**kwargs)
        self.redis_client = redis_client
    
    def check_rate_limit(self, identifier: str, **kwargs) -> Tuple[bool, Optional[int]]:
        """Check rate limit using distributed state"""
        if not self.redis_client:
            # Fallback to local rate limiting
            return super().check_rate_limit(identifier, **kwargs)
        
        # Implement Redis-based rate limiting
        # This is a placeholder for actual Redis implementation
        # You would use Redis commands like INCR, EXPIRE, etc.
        try:
            key = f"rate_limit:{identifier}"
            # Simplified example - actual implementation would be more complex
            count = self.redis_client.incr(key)
            if count == 1:
                self.redis_client.expire(key, 60)  # 1 minute window
            
            if count > self.requests_per_minute:
                ttl = self.redis_client.ttl(key)
                return False, ttl
            
            return True, None
        except Exception as e:
            logger.error(f"Redis error, falling back to local: {e}")
            return super().check_rate_limit(identifier, **kwargs)