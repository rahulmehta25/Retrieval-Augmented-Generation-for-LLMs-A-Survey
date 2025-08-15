"""
Bulkhead Pattern Implementation for Resource Isolation

Prevents cascading failures by isolating resources and limiting concurrent operations.
"""

import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
from typing import Callable, Any, Optional, Dict
from contextlib import contextmanager
import time
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class BulkheadType(Enum):
    """Types of bulkhead implementations"""
    SEMAPHORE = "semaphore"
    THREAD_POOL = "thread_pool"


class BulkheadFullException(Exception):
    """Exception raised when bulkhead is at capacity"""
    pass


@dataclass
class BulkheadMetrics:
    """Metrics for bulkhead monitoring"""
    name: str
    type: BulkheadType
    max_concurrent: int
    active_count: int
    waiting_count: int
    completed_count: int
    rejected_count: int
    total_execution_time: float
    average_execution_time: float


class Bulkhead:
    """Base class for bulkhead implementations"""
    
    def __init__(self, name: str, max_concurrent: int):
        """
        Initialize bulkhead
        
        Args:
            name: Bulkhead name for identification
            max_concurrent: Maximum concurrent operations
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self._completed_count = 0
        self._rejected_count = 0
        self._total_execution_time = 0.0
        self._lock = threading.Lock()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead protection"""
        raise NotImplementedError
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with bulkhead protection"""
        raise NotImplementedError
    
    def get_metrics(self) -> BulkheadMetrics:
        """Get bulkhead metrics"""
        raise NotImplementedError
    
    def _record_execution(self, execution_time: float):
        """Record execution metrics"""
        with self._lock:
            self._completed_count += 1
            self._total_execution_time += execution_time
    
    def _record_rejection(self):
        """Record rejection"""
        with self._lock:
            self._rejected_count += 1


class SemaphoreBulkhead(Bulkhead):
    """
    Semaphore-based bulkhead implementation
    
    Uses a semaphore to limit concurrent operations. Lightweight and suitable
    for I/O-bound operations.
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent: int,
        max_wait_time: Optional[float] = None,
        queue_size: int = 0
    ):
        """
        Initialize semaphore bulkhead
        
        Args:
            name: Bulkhead name
            max_concurrent: Maximum concurrent operations
            max_wait_time: Maximum time to wait for semaphore (seconds)
            queue_size: Maximum queue size (0 for unlimited)
        """
        super().__init__(name, max_concurrent)
        self._semaphore = threading.Semaphore(max_concurrent)
        self._async_semaphore = asyncio.Semaphore(max_concurrent)
        self.max_wait_time = max_wait_time
        self.queue_size = queue_size
        self._waiting_count = 0
        self._active_count = 0
        self._queue_lock = threading.Lock()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with semaphore protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            BulkheadFullException: If bulkhead is at capacity
        """
        # Check queue size
        with self._queue_lock:
            if self.queue_size > 0 and self._waiting_count >= self.queue_size:
                self._record_rejection()
                raise BulkheadFullException(
                    f"Bulkhead '{self.name}' queue is full: {self._waiting_count}/{self.queue_size}"
                )
            self._waiting_count += 1
        
        acquired = False
        start_wait = time.time()
        
        try:
            # Try to acquire semaphore
            if self.max_wait_time:
                acquired = self._semaphore.acquire(timeout=self.max_wait_time)
                if not acquired:
                    self._record_rejection()
                    raise BulkheadFullException(
                        f"Bulkhead '{self.name}' timeout waiting for slot"
                    )
            else:
                acquired = self._semaphore.acquire(blocking=True)
            
            with self._queue_lock:
                self._waiting_count -= 1
                self._active_count += 1
            
            # Execute function
            start_exec = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_exec
                self._record_execution(execution_time)
                return result
            finally:
                with self._queue_lock:
                    self._active_count -= 1
        
        finally:
            # Release semaphore if acquired
            if acquired:
                self._semaphore.release()
            else:
                with self._queue_lock:
                    self._waiting_count -= 1
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with semaphore protection
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            BulkheadFullException: If bulkhead is at capacity
        """
        # Check queue size
        with self._queue_lock:
            if self.queue_size > 0 and self._waiting_count >= self.queue_size:
                self._record_rejection()
                raise BulkheadFullException(
                    f"Bulkhead '{self.name}' queue is full: {self._waiting_count}/{self.queue_size}"
                )
            self._waiting_count += 1
        
        try:
            # Try to acquire semaphore
            if self.max_wait_time:
                try:
                    await asyncio.wait_for(
                        self._async_semaphore.acquire(),
                        timeout=self.max_wait_time
                    )
                except asyncio.TimeoutError:
                    self._record_rejection()
                    raise BulkheadFullException(
                        f"Bulkhead '{self.name}' timeout waiting for slot"
                    )
            else:
                await self._async_semaphore.acquire()
            
            with self._queue_lock:
                self._waiting_count -= 1
                self._active_count += 1
            
            # Execute function
            start_exec = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_exec
                self._record_execution(execution_time)
                return result
            finally:
                with self._queue_lock:
                    self._active_count -= 1
                self._async_semaphore.release()
        
        except Exception as e:
            with self._queue_lock:
                self._waiting_count -= 1
            raise
    
    def get_metrics(self) -> BulkheadMetrics:
        """Get bulkhead metrics"""
        with self._lock:
            avg_exec_time = (
                self._total_execution_time / self._completed_count
                if self._completed_count > 0 else 0.0
            )
            
            return BulkheadMetrics(
                name=self.name,
                type=BulkheadType.SEMAPHORE,
                max_concurrent=self.max_concurrent,
                active_count=self._active_count,
                waiting_count=self._waiting_count,
                completed_count=self._completed_count,
                rejected_count=self._rejected_count,
                total_execution_time=self._total_execution_time,
                average_execution_time=avg_exec_time
            )


class ThreadPoolBulkhead(Bulkhead):
    """
    Thread pool-based bulkhead implementation
    
    Uses a thread pool to isolate and limit concurrent operations.
    Suitable for CPU-bound operations or when complete isolation is needed.
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent: int,
        max_queue_size: Optional[int] = None,
        thread_name_prefix: Optional[str] = None
    ):
        """
        Initialize thread pool bulkhead
        
        Args:
            name: Bulkhead name
            max_concurrent: Maximum concurrent threads
            max_queue_size: Maximum queue size (None for 2 * max_concurrent)
            thread_name_prefix: Prefix for thread names
        """
        super().__init__(name, max_concurrent)
        self.max_queue_size = max_queue_size or (2 * max_concurrent)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix=thread_name_prefix or f"bulkhead-{name}"
        )
        self._pending_futures = set()
        self._futures_lock = threading.Lock()
    
    def execute(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute function in thread pool
        
        Args:
            func: Function to execute
            *args: Function arguments
            timeout: Execution timeout (seconds)
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            BulkheadFullException: If thread pool queue is full
        """
        # Check queue size
        with self._futures_lock:
            if len(self._pending_futures) >= self.max_queue_size:
                self._record_rejection()
                raise BulkheadFullException(
                    f"Bulkhead '{self.name}' queue is full: {len(self._pending_futures)}/{self.max_queue_size}"
                )
        
        # Submit to thread pool
        start_time = time.time()
        future = self._executor.submit(func, *args, **kwargs)
        
        with self._futures_lock:
            self._pending_futures.add(future)
        
        try:
            # Wait for result
            if timeout:
                result = future.result(timeout=timeout)
            else:
                result = future.result()
            
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            return result
            
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(f"Bulkhead '{self.name}' execution timeout")
        
        finally:
            with self._futures_lock:
                self._pending_futures.discard(future)
    
    async def execute_async(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute function in thread pool (async wrapper)
        
        Args:
            func: Function to execute
            *args: Function arguments
            timeout: Execution timeout (seconds)
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            BulkheadFullException: If thread pool queue is full
        """
        loop = asyncio.get_event_loop()
        
        # Check queue size
        with self._futures_lock:
            if len(self._pending_futures) >= self.max_queue_size:
                self._record_rejection()
                raise BulkheadFullException(
                    f"Bulkhead '{self.name}' queue is full: {len(self._pending_futures)}/{self.max_queue_size}"
                )
        
        # Submit to thread pool
        start_time = time.time()
        future = self._executor.submit(func, *args, **kwargs)
        
        with self._futures_lock:
            self._pending_futures.add(future)
        
        try:
            # Convert to asyncio future
            async_future = asyncio.wrap_future(future, loop=loop)
            
            # Wait for result
            if timeout:
                result = await asyncio.wait_for(async_future, timeout=timeout)
            else:
                result = await async_future
            
            execution_time = time.time() - start_time
            self._record_execution(execution_time)
            return result
            
        except asyncio.TimeoutError:
            future.cancel()
            raise TimeoutError(f"Bulkhead '{self.name}' execution timeout")
        
        finally:
            with self._futures_lock:
                self._pending_futures.discard(future)
    
    def shutdown(self, wait: bool = True):
        """Shutdown thread pool"""
        self._executor.shutdown(wait=wait)
    
    def get_metrics(self) -> BulkheadMetrics:
        """Get bulkhead metrics"""
        with self._lock:
            avg_exec_time = (
                self._total_execution_time / self._completed_count
                if self._completed_count > 0 else 0.0
            )
            
            with self._futures_lock:
                active_count = len([f for f in self._pending_futures if f.running()])
                waiting_count = len([f for f in self._pending_futures if not f.running()])
            
            return BulkheadMetrics(
                name=self.name,
                type=BulkheadType.THREAD_POOL,
                max_concurrent=self.max_concurrent,
                active_count=active_count,
                waiting_count=waiting_count,
                completed_count=self._completed_count,
                rejected_count=self._rejected_count,
                total_execution_time=self._total_execution_time,
                average_execution_time=avg_exec_time
            )


@contextmanager
def bulkhead_context(bulkhead: Bulkhead):
    """
    Context manager for bulkhead operations
    
    Usage:
        with bulkhead_context(my_bulkhead):
            # Protected code here
            pass
    """
    semaphore = getattr(bulkhead, '_semaphore', None)
    if semaphore:
        acquired = semaphore.acquire(blocking=True)
        try:
            yield
        finally:
            if acquired:
                semaphore.release()
    else:
        yield