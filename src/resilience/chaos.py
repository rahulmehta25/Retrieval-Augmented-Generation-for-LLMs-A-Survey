"""
Chaos Engineering Module

Provides controlled failure injection for testing resilience.
"""

import random
import time
import threading
import asyncio
from typing import Callable, Any, Optional, Dict, List
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ChaosStrategy(Enum):
    """Types of chaos injection strategies"""
    LATENCY = "latency"
    ERROR = "error"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PARTIAL_FAILURE = "partial_failure"
    DATA_CORRUPTION = "data_corruption"
    NETWORK_PARTITION = "network_partition"


@dataclass
class ChaosConfig:
    """Configuration for chaos injection"""
    strategy: ChaosStrategy
    probability: float = 0.1  # Probability of chaos occurring
    latency_ms: Optional[float] = None
    error_type: Optional[type] = None
    error_message: Optional[str] = None
    timeout_duration: Optional[float] = None
    corruption_function: Optional[Callable] = None
    enabled: bool = True


class ChaosEngine:
    """
    Chaos engineering implementation for testing resilience
    
    Injects controlled failures to test system resilience.
    WARNING: Only use in testing/staging environments!
    """
    
    def __init__(
        self,
        enabled: bool = False,
        safe_mode: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize chaos engine
        
        Args:
            enabled: Whether chaos injection is enabled
            safe_mode: Require explicit enablement (safety feature)
            seed: Random seed for reproducible chaos
        """
        self.enabled = enabled and not safe_mode
        self.safe_mode = safe_mode
        self._configs: Dict[str, ChaosConfig] = {}
        self._statistics = {
            'total_calls': 0,
            'chaos_injected': 0,
            'by_strategy': {}
        }
        self._lock = threading.Lock()
        
        if seed is not None:
            random.seed(seed)
        
        if self.enabled:
            logger.warning("CHAOS ENGINE ENABLED - Only use in testing!")
    
    def register_chaos(self, name: str, config: ChaosConfig):
        """
        Register a chaos configuration
        
        Args:
            name: Name for the chaos configuration
            config: Chaos configuration
        """
        with self._lock:
            self._configs[name] = config
            if config.strategy.value not in self._statistics['by_strategy']:
                self._statistics['by_strategy'][config.strategy.value] = 0
    
    def unregister_chaos(self, name: str):
        """Unregister a chaos configuration"""
        with self._lock:
            if name in self._configs:
                del self._configs[name]
    
    def inject(
        self,
        func: Callable,
        *args,
        chaos_config: Optional[ChaosConfig] = None,
        chaos_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with chaos injection
        
        Args:
            func: Function to execute
            *args: Function arguments
            chaos_config: Specific chaos config to use
            chaos_name: Name of registered chaos config
            **kwargs: Function keyword arguments
            
        Returns:
            Function result (potentially affected by chaos)
        """
        if not self.enabled:
            return func(*args, **kwargs)
        
        # Get chaos configuration
        config = chaos_config
        if not config and chaos_name:
            with self._lock:
                config = self._configs.get(chaos_name)
        
        if not config or not config.enabled:
            return func(*args, **kwargs)
        
        # Record call
        with self._lock:
            self._statistics['total_calls'] += 1
        
        # Determine if chaos should occur
        if random.random() > config.probability:
            # No chaos this time
            return func(*args, **kwargs)
        
        # Inject chaos based on strategy
        with self._lock:
            self._statistics['chaos_injected'] += 1
            self._statistics['by_strategy'][config.strategy.value] += 1
        
        logger.info(f"Injecting chaos: {config.strategy.value}")
        
        if config.strategy == ChaosStrategy.LATENCY:
            return self._inject_latency(func, args, kwargs, config)
        
        elif config.strategy == ChaosStrategy.ERROR:
            return self._inject_error(func, args, kwargs, config)
        
        elif config.strategy == ChaosStrategy.TIMEOUT:
            return self._inject_timeout(func, args, kwargs, config)
        
        elif config.strategy == ChaosStrategy.RESOURCE_EXHAUSTION:
            return self._inject_resource_exhaustion(func, args, kwargs, config)
        
        elif config.strategy == ChaosStrategy.PARTIAL_FAILURE:
            return self._inject_partial_failure(func, args, kwargs, config)
        
        elif config.strategy == ChaosStrategy.DATA_CORRUPTION:
            return self._inject_data_corruption(func, args, kwargs, config)
        
        elif config.strategy == ChaosStrategy.NETWORK_PARTITION:
            return self._inject_network_partition(func, args, kwargs, config)
        
        else:
            return func(*args, **kwargs)
    
    async def inject_async(
        self,
        func: Callable,
        *args,
        chaos_config: Optional[ChaosConfig] = None,
        chaos_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute async function with chaos injection
        
        Args:
            func: Async function to execute
            *args: Function arguments
            chaos_config: Specific chaos config to use
            chaos_name: Name of registered chaos config
            **kwargs: Function keyword arguments
            
        Returns:
            Function result (potentially affected by chaos)
        """
        if not self.enabled:
            return await func(*args, **kwargs)
        
        # Get chaos configuration
        config = chaos_config
        if not config and chaos_name:
            with self._lock:
                config = self._configs.get(chaos_name)
        
        if not config or not config.enabled:
            return await func(*args, **kwargs)
        
        # Record call
        with self._lock:
            self._statistics['total_calls'] += 1
        
        # Determine if chaos should occur
        if random.random() > config.probability:
            # No chaos this time
            return await func(*args, **kwargs)
        
        # Inject chaos based on strategy
        with self._lock:
            self._statistics['chaos_injected'] += 1
            self._statistics['by_strategy'][config.strategy.value] += 1
        
        logger.info(f"Injecting async chaos: {config.strategy.value}")
        
        if config.strategy == ChaosStrategy.LATENCY:
            return await self._inject_latency_async(func, args, kwargs, config)
        
        elif config.strategy == ChaosStrategy.ERROR:
            return await self._inject_error_async(func, args, kwargs, config)
        
        elif config.strategy == ChaosStrategy.TIMEOUT:
            return await self._inject_timeout_async(func, args, kwargs, config)
        
        else:
            return await func(*args, **kwargs)
    
    def _inject_latency(self, func, args, kwargs, config):
        """Inject latency before executing function"""
        latency = config.latency_ms or random.uniform(100, 5000)
        time.sleep(latency / 1000.0)
        return func(*args, **kwargs)
    
    async def _inject_latency_async(self, func, args, kwargs, config):
        """Inject latency before executing async function"""
        latency = config.latency_ms or random.uniform(100, 5000)
        await asyncio.sleep(latency / 1000.0)
        return await func(*args, **kwargs)
    
    def _inject_error(self, func, args, kwargs, config):
        """Inject error instead of executing function"""
        error_type = config.error_type or Exception
        error_message = config.error_message or "Chaos: Injected error"
        raise error_type(error_message)
    
    async def _inject_error_async(self, func, args, kwargs, config):
        """Inject error instead of executing async function"""
        error_type = config.error_type or Exception
        error_message = config.error_message or "Chaos: Injected error"
        raise error_type(error_message)
    
    def _inject_timeout(self, func, args, kwargs, config):
        """Simulate timeout by sleeping longer than expected"""
        timeout = config.timeout_duration or 30.0
        time.sleep(timeout)
        raise TimeoutError("Chaos: Simulated timeout")
    
    async def _inject_timeout_async(self, func, args, kwargs, config):
        """Simulate async timeout"""
        timeout = config.timeout_duration or 30.0
        await asyncio.sleep(timeout)
        raise TimeoutError("Chaos: Simulated timeout")
    
    def _inject_resource_exhaustion(self, func, args, kwargs, config):
        """Simulate resource exhaustion"""
        # Allocate memory to simulate memory pressure
        memory_hog = [0] * (10 * 1024 * 1024)  # ~80MB
        
        try:
            result = func(*args, **kwargs)
        finally:
            del memory_hog
        
        return result
    
    def _inject_partial_failure(self, func, args, kwargs, config):
        """Execute function but return partial/degraded result"""
        try:
            result = func(*args, **kwargs)
            
            # Modify result to simulate partial failure
            if isinstance(result, list):
                # Return only half of list items
                return result[:len(result)//2]
            elif isinstance(result, dict):
                # Remove some keys
                keys = list(result.keys())
                for key in keys[:len(keys)//2]:
                    del result[key]
                return result
            elif isinstance(result, str):
                # Truncate string
                return result[:len(result)//2] + "... [TRUNCATED BY CHAOS]"
            else:
                # Return None for other types
                return None
        except:
            # If function fails, let it fail
            raise
    
    def _inject_data_corruption(self, func, args, kwargs, config):
        """Execute function but corrupt the result"""
        result = func(*args, **kwargs)
        
        if config.corruption_function:
            return config.corruption_function(result)
        
        # Default corruption strategies
        if isinstance(result, str):
            # Flip random characters
            chars = list(result)
            if chars:
                idx = random.randint(0, len(chars) - 1)
                chars[idx] = chr(ord(chars[idx]) ^ 1)
            return ''.join(chars)
        
        elif isinstance(result, (int, float)):
            # Add random noise
            return result * random.uniform(0.9, 1.1)
        
        elif isinstance(result, list):
            # Shuffle list
            if result:
                random.shuffle(result)
            return result
        
        elif isinstance(result, dict):
            # Swap some values
            keys = list(result.keys())
            if len(keys) >= 2:
                k1, k2 = random.sample(keys, 2)
                result[k1], result[k2] = result[k2], result[k1]
            return result
        
        return result
    
    def _inject_network_partition(self, func, args, kwargs, config):
        """Simulate network partition by raising connection error"""
        raise ConnectionError("Chaos: Simulated network partition")
    
    def enable(self, confirmation: str = ""):
        """
        Enable chaos engine (requires confirmation)
        
        Args:
            confirmation: Must be "ENABLE_CHAOS" to enable
        """
        if confirmation == "ENABLE_CHAOS":
            self.enabled = True
            logger.warning("CHAOS ENGINE ENABLED - System may behave unpredictably!")
        else:
            logger.error("Chaos engine enable failed - incorrect confirmation")
    
    def disable(self):
        """Disable chaos engine"""
        self.enabled = False
        logger.info("Chaos engine disabled")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chaos injection statistics"""
        with self._lock:
            total = self._statistics['total_calls']
            injected = self._statistics['chaos_injected']
            
            return {
                'enabled': self.enabled,
                'total_calls': total,
                'chaos_injected': injected,
                'injection_rate': injected / total if total > 0 else 0.0,
                'by_strategy': dict(self._statistics['by_strategy'])
            }
    
    def reset_statistics(self):
        """Reset statistics"""
        with self._lock:
            self._statistics = {
                'total_calls': 0,
                'chaos_injected': 0,
                'by_strategy': {}
            }


def chaos_test(
    strategy: ChaosStrategy,
    probability: float = 0.1,
    **kwargs
):
    """
    Decorator for chaos testing
    
    Usage:
        @chaos_test(ChaosStrategy.LATENCY, probability=0.5, latency_ms=1000)
        def my_function():
            pass
    """
    def decorator(func):
        config = ChaosConfig(
            strategy=strategy,
            probability=probability,
            **kwargs
        )
        
        def wrapper(*args, **kwargs):
            engine = ChaosEngine(enabled=True, safe_mode=False)
            return engine.inject(func, *args, chaos_config=config, **kwargs)
        
        async def async_wrapper(*args, **kwargs):
            engine = ChaosEngine(enabled=True, safe_mode=False)
            return await engine.inject_async(func, *args, chaos_config=config, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator