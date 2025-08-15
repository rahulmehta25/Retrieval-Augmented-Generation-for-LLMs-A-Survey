"""
Fallback Mechanism Implementation

Provides fallback strategies when primary services fail.
"""

import logging
from typing import Callable, Any, Optional, List, Dict, Union
from enum import Enum
from dataclasses import dataclass
import asyncio
import time

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Fallback strategy types"""
    STATIC_VALUE = "static_value"
    CACHED_VALUE = "cached_value"
    ALTERNATIVE_SERVICE = "alternative_service"
    DEGRADED_SERVICE = "degraded_service"
    CUSTOM_HANDLER = "custom_handler"


@dataclass
class FallbackContext:
    """Context information for fallback execution"""
    strategy: FallbackStrategy
    original_exception: Exception
    attempt_count: int
    elapsed_time: float
    metadata: Dict[str, Any]


class FallbackHandler:
    """
    Handles fallback logic when primary operations fail
    
    Provides multiple fallback strategies including static values,
    cached results, alternative services, and custom handlers.
    """
    
    def __init__(
        self,
        strategy: FallbackStrategy = FallbackStrategy.STATIC_VALUE,
        fallback_value: Any = None,
        fallback_function: Optional[Callable] = None,
        cache_ttl: float = 300.0,
        log_fallbacks: bool = True
    ):
        """
        Initialize fallback handler
        
        Args:
            strategy: Fallback strategy to use
            fallback_value: Static value to return for STATIC_VALUE strategy
            fallback_function: Function to call for fallback
            cache_ttl: Cache time-to-live in seconds
            log_fallbacks: Whether to log fallback events
        """
        self.strategy = strategy
        self.fallback_value = fallback_value
        self.fallback_function = fallback_function
        self.cache_ttl = cache_ttl
        self.log_fallbacks = log_fallbacks
        
        # Cache for CACHED_VALUE strategy
        self._cache = {}
        self._cache_timestamps = {}
        
        # Fallback statistics
        self._fallback_count = 0
        self._success_count = 0
    
    def execute(
        self,
        primary_func: Callable,
        *args,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute primary function with fallback
        
        Args:
            primary_func: Primary function to execute
            *args: Function arguments
            cache_key: Key for caching results
            **kwargs: Function keyword arguments
            
        Returns:
            Primary function result or fallback value
        """
        start_time = time.time()
        
        try:
            # Try primary function
            result = primary_func(*args, **kwargs)
            
            # Cache successful result if using cached strategy
            if self.strategy == FallbackStrategy.CACHED_VALUE and cache_key:
                self._update_cache(cache_key, result)
            
            self._success_count += 1
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self._fallback_count += 1
            
            if self.log_fallbacks:
                logger.warning(
                    f"Primary function failed, using fallback strategy: {self.strategy.value}. "
                    f"Error: {e}"
                )
            
            # Create fallback context
            context = FallbackContext(
                strategy=self.strategy,
                original_exception=e,
                attempt_count=1,
                elapsed_time=elapsed_time,
                metadata={'cache_key': cache_key} if cache_key else {}
            )
            
            # Execute fallback based on strategy
            return self._execute_fallback(context, *args, **kwargs)
    
    async def execute_async(
        self,
        primary_func: Callable,
        *args,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute async primary function with fallback
        
        Args:
            primary_func: Async primary function to execute
            *args: Function arguments
            cache_key: Key for caching results
            **kwargs: Function keyword arguments
            
        Returns:
            Primary function result or fallback value
        """
        start_time = time.time()
        
        try:
            # Try primary function
            result = await primary_func(*args, **kwargs)
            
            # Cache successful result if using cached strategy
            if self.strategy == FallbackStrategy.CACHED_VALUE and cache_key:
                self._update_cache(cache_key, result)
            
            self._success_count += 1
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self._fallback_count += 1
            
            if self.log_fallbacks:
                logger.warning(
                    f"Primary async function failed, using fallback strategy: {self.strategy.value}. "
                    f"Error: {e}"
                )
            
            # Create fallback context
            context = FallbackContext(
                strategy=self.strategy,
                original_exception=e,
                attempt_count=1,
                elapsed_time=elapsed_time,
                metadata={'cache_key': cache_key} if cache_key else {}
            )
            
            # Execute fallback based on strategy
            return await self._execute_fallback_async(context, *args, **kwargs)
    
    def _execute_fallback(
        self,
        context: FallbackContext,
        *args,
        **kwargs
    ) -> Any:
        """Execute fallback based on strategy"""
        
        if self.strategy == FallbackStrategy.STATIC_VALUE:
            return self.fallback_value
        
        elif self.strategy == FallbackStrategy.CACHED_VALUE:
            cache_key = context.metadata.get('cache_key')
            if cache_key:
                cached_value = self._get_cached_value(cache_key)
                if cached_value is not None:
                    if self.log_fallbacks:
                        logger.info(f"Using cached value for key: {cache_key}")
                    return cached_value
            
            # No cached value available, use static fallback
            return self.fallback_value
        
        elif self.strategy == FallbackStrategy.ALTERNATIVE_SERVICE:
            if self.fallback_function:
                try:
                    return self.fallback_function(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Alternative service also failed: {e}")
                    return self.fallback_value
            return self.fallback_value
        
        elif self.strategy == FallbackStrategy.DEGRADED_SERVICE:
            # Implement degraded service logic
            # For example, return partial data or simplified response
            if self.fallback_function:
                try:
                    # Pass context to allow degraded service to make decisions
                    return self.fallback_function(context, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Degraded service failed: {e}")
                    return self.fallback_value
            return self.fallback_value
        
        elif self.strategy == FallbackStrategy.CUSTOM_HANDLER:
            if self.fallback_function:
                try:
                    return self.fallback_function(context, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Custom fallback handler failed: {e}")
                    return self.fallback_value
            return self.fallback_value
        
        # Default fallback
        return self.fallback_value
    
    async def _execute_fallback_async(
        self,
        context: FallbackContext,
        *args,
        **kwargs
    ) -> Any:
        """Execute async fallback based on strategy"""
        
        if self.strategy == FallbackStrategy.STATIC_VALUE:
            return self.fallback_value
        
        elif self.strategy == FallbackStrategy.CACHED_VALUE:
            cache_key = context.metadata.get('cache_key')
            if cache_key:
                cached_value = self._get_cached_value(cache_key)
                if cached_value is not None:
                    if self.log_fallbacks:
                        logger.info(f"Using cached value for key: {cache_key}")
                    return cached_value
            
            # No cached value available, use static fallback
            return self.fallback_value
        
        elif self.strategy == FallbackStrategy.ALTERNATIVE_SERVICE:
            if self.fallback_function:
                try:
                    if asyncio.iscoroutinefunction(self.fallback_function):
                        return await self.fallback_function(*args, **kwargs)
                    else:
                        return self.fallback_function(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Alternative service also failed: {e}")
                    return self.fallback_value
            return self.fallback_value
        
        elif self.strategy == FallbackStrategy.DEGRADED_SERVICE:
            if self.fallback_function:
                try:
                    if asyncio.iscoroutinefunction(self.fallback_function):
                        return await self.fallback_function(context, *args, **kwargs)
                    else:
                        return self.fallback_function(context, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Degraded service failed: {e}")
                    return self.fallback_value
            return self.fallback_value
        
        elif self.strategy == FallbackStrategy.CUSTOM_HANDLER:
            if self.fallback_function:
                try:
                    if asyncio.iscoroutinefunction(self.fallback_function):
                        return await self.fallback_function(context, *args, **kwargs)
                    else:
                        return self.fallback_function(context, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Custom fallback handler failed: {e}")
                    return self.fallback_value
            return self.fallback_value
        
        # Default fallback
        return self.fallback_value
    
    def _update_cache(self, key: str, value: Any):
        """Update cache with new value"""
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()
    
    def _get_cached_value(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self._cache:
            return None
        
        # Check if cache entry has expired
        timestamp = self._cache_timestamps.get(key, 0)
        if time.time() - timestamp > self.cache_ttl:
            # Cache expired, remove entry
            del self._cache[key]
            del self._cache_timestamps[key]
            return None
        
        return self._cache[key]
    
    def clear_cache(self):
        """Clear all cached values"""
        self._cache.clear()
        self._cache_timestamps.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        total_calls = self._success_count + self._fallback_count
        
        return {
            'total_calls': total_calls,
            'success_count': self._success_count,
            'fallback_count': self._fallback_count,
            'success_rate': self._success_count / total_calls if total_calls > 0 else 0.0,
            'fallback_rate': self._fallback_count / total_calls if total_calls > 0 else 0.0,
            'cache_size': len(self._cache),
            'strategy': self.strategy.value
        }


class ChainedFallbackHandler:
    """
    Chains multiple fallback handlers for multi-level fallback
    """
    
    def __init__(self, handlers: List[FallbackHandler]):
        """
        Initialize chained fallback handler
        
        Args:
            handlers: List of fallback handlers to chain
        """
        self.handlers = handlers
    
    def execute(self, primary_func: Callable, *args, **kwargs) -> Any:
        """
        Execute with chained fallbacks
        
        Tries each handler in sequence until one succeeds
        """
        last_exception = None
        
        # Try primary function first
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
        
        # Try each fallback handler
        for handler in self.handlers:
            try:
                if handler.fallback_function:
                    return handler.fallback_function(*args, **kwargs)
                else:
                    return handler.fallback_value
            except Exception as e:
                last_exception = e
                continue
        
        # All fallbacks failed
        if last_exception:
            raise last_exception
        
        raise RuntimeError("All fallback handlers failed")
    
    async def execute_async(self, primary_func: Callable, *args, **kwargs) -> Any:
        """
        Execute async with chained fallbacks
        
        Tries each handler in sequence until one succeeds
        """
        last_exception = None
        
        # Try primary function first
        try:
            return await primary_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
        
        # Try each fallback handler
        for handler in self.handlers:
            try:
                if handler.fallback_function:
                    if asyncio.iscoroutinefunction(handler.fallback_function):
                        return await handler.fallback_function(*args, **kwargs)
                    else:
                        return handler.fallback_function(*args, **kwargs)
                else:
                    return handler.fallback_value
            except Exception as e:
                last_exception = e
                continue
        
        # All fallbacks failed
        if last_exception:
            raise last_exception
        
        raise RuntimeError("All fallback handlers failed")