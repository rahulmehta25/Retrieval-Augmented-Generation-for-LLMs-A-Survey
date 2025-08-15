"""
Health Checking and Recovery Mechanisms

Provides health checks for services with automatic recovery capabilities.
"""

import time
import threading
import asyncio
import logging
from typing import Callable, Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_function: Callable[[], bool]
    interval: float = 30.0
    timeout: float = 5.0
    failure_threshold: int = 3
    success_threshold: int = 2
    recovery_action: Optional[Callable] = None
    critical: bool = False


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    timestamp: datetime
    response_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class HealthChecker:
    """
    Manages health checks for services with recovery mechanisms
    
    Periodically checks service health and triggers recovery actions
    when services become unhealthy.
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        enable_auto_recovery: bool = True,
        max_recovery_attempts: int = 3
    ):
        """
        Initialize health checker
        
        Args:
            check_interval: Default interval between health checks (seconds)
            enable_auto_recovery: Whether to automatically trigger recovery actions
            max_recovery_attempts: Maximum recovery attempts before giving up
        """
        self.check_interval = check_interval
        self.enable_auto_recovery = enable_auto_recovery
        self.max_recovery_attempts = max_recovery_attempts
        
        self._health_checks: Dict[str, HealthCheck] = {}
        self._check_results: Dict[str, List[HealthCheckResult]] = {}
        self._failure_counts: Dict[str, int] = {}
        self._success_counts: Dict[str, int] = {}
        self._recovery_attempts: Dict[str, int] = {}
        self._check_threads: Dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Overall health status
        self._overall_status = HealthStatus.UNKNOWN
        self._status_change_listeners = []
    
    def register_check(self, health_check: HealthCheck):
        """
        Register a health check
        
        Args:
            health_check: Health check configuration
        """
        with self._lock:
            self._health_checks[health_check.name] = health_check
            self._check_results[health_check.name] = []
            self._failure_counts[health_check.name] = 0
            self._success_counts[health_check.name] = 0
            self._recovery_attempts[health_check.name] = 0
        
        logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_check(self, name: str):
        """
        Unregister a health check
        
        Args:
            name: Name of health check to unregister
        """
        with self._lock:
            if name in self._health_checks:
                del self._health_checks[name]
                del self._check_results[name]
                del self._failure_counts[name]
                del self._success_counts[name]
                del self._recovery_attempts[name]
                
                # Stop check thread if running
                if name in self._check_threads:
                    thread = self._check_threads[name]
                    if thread.is_alive():
                        thread.join(timeout=1.0)
                    del self._check_threads[name]
        
        logger.info(f"Unregistered health check: {name}")
    
    def start(self):
        """Start health checking"""
        self._stop_event.clear()
        
        with self._lock:
            for name, check in self._health_checks.items():
                if name not in self._check_threads or not self._check_threads[name].is_alive():
                    thread = threading.Thread(
                        target=self._run_check_loop,
                        args=(check,),
                        name=f"health-check-{name}"
                    )
                    thread.daemon = True
                    thread.start()
                    self._check_threads[name] = thread
        
        logger.info("Health checker started")
    
    def stop(self):
        """Stop health checking"""
        self._stop_event.set()
        
        with self._lock:
            for thread in self._check_threads.values():
                if thread.is_alive():
                    thread.join(timeout=1.0)
            self._check_threads.clear()
        
        logger.info("Health checker stopped")
    
    def _run_check_loop(self, check: HealthCheck):
        """Run health check loop for a specific check"""
        while not self._stop_event.is_set():
            try:
                # Perform health check
                result = self._perform_check(check)
                
                # Process result
                self._process_check_result(check, result)
                
                # Wait for next check
                self._stop_event.wait(check.interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop for {check.name}: {e}")
                self._stop_event.wait(check.interval)
    
    def _perform_check(self, check: HealthCheck) -> HealthCheckResult:
        """Perform a single health check"""
        start_time = time.time()
        
        try:
            # Execute check function with timeout
            if asyncio.iscoroutinefunction(check.check_function):
                # Handle async check function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    future = asyncio.wait_for(
                        check.check_function(),
                        timeout=check.timeout
                    )
                    is_healthy = loop.run_until_complete(future)
                finally:
                    loop.close()
            else:
                # Handle sync check function
                # Note: Simple timeout implementation, consider using threading
                is_healthy = check.check_function()
            
            response_time = time.time() - start_time
            
            if is_healthy:
                return HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.HEALTHY,
                    timestamp=datetime.now(),
                    response_time=response_time
                )
            else:
                return HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(),
                    response_time=response_time,
                    error="Check returned False"
                )
        
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return HealthCheckResult(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time=response_time,
                error=f"Check timed out after {check.timeout}s"
            )
        
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time=response_time,
                error=str(e)
            )
    
    def _process_check_result(self, check: HealthCheck, result: HealthCheckResult):
        """Process health check result and trigger recovery if needed"""
        with self._lock:
            # Store result
            self._check_results[check.name].append(result)
            
            # Maintain result history (keep last 100)
            if len(self._check_results[check.name]) > 100:
                self._check_results[check.name].pop(0)
            
            # Update counters based on status
            if result.status == HealthStatus.HEALTHY:
                self._success_counts[check.name] += 1
                self._failure_counts[check.name] = 0
                
                # Reset recovery attempts on success
                if self._success_counts[check.name] >= check.success_threshold:
                    self._recovery_attempts[check.name] = 0
            
            else:
                self._failure_counts[check.name] += 1
                self._success_counts[check.name] = 0
                
                # Check if we should trigger recovery
                if (self._failure_counts[check.name] >= check.failure_threshold and
                    self.enable_auto_recovery and
                    check.recovery_action and
                    self._recovery_attempts[check.name] < self.max_recovery_attempts):
                    
                    self._trigger_recovery(check)
            
            # Update overall status
            self._update_overall_status()
    
    def _trigger_recovery(self, check: HealthCheck):
        """Trigger recovery action for failed health check"""
        with self._lock:
            self._recovery_attempts[check.name] += 1
        
        logger.warning(
            f"Triggering recovery for {check.name} "
            f"(attempt {self._recovery_attempts[check.name]}/{self.max_recovery_attempts})"
        )
        
        try:
            if asyncio.iscoroutinefunction(check.recovery_action):
                # Handle async recovery action
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(check.recovery_action())
                finally:
                    loop.close()
            else:
                # Handle sync recovery action
                check.recovery_action()
            
            logger.info(f"Recovery action completed for {check.name}")
            
        except Exception as e:
            logger.error(f"Recovery action failed for {check.name}: {e}")
    
    def _update_overall_status(self):
        """Update overall health status based on individual checks"""
        with self._lock:
            if not self._health_checks:
                new_status = HealthStatus.UNKNOWN
            else:
                all_healthy = True
                any_unhealthy = False
                any_critical_unhealthy = False
                
                for name, check in self._health_checks.items():
                    if self._check_results[name]:
                        latest_result = self._check_results[name][-1]
                        
                        if latest_result.status != HealthStatus.HEALTHY:
                            all_healthy = False
                            any_unhealthy = True
                            
                            if check.critical:
                                any_critical_unhealthy = True
                
                if any_critical_unhealthy:
                    new_status = HealthStatus.UNHEALTHY
                elif any_unhealthy:
                    new_status = HealthStatus.DEGRADED
                elif all_healthy:
                    new_status = HealthStatus.HEALTHY
                else:
                    new_status = HealthStatus.UNKNOWN
            
            # Notify listeners if status changed
            if new_status != self._overall_status:
                old_status = self._overall_status
                self._overall_status = new_status
                self._notify_status_change(old_status, new_status)
    
    def _notify_status_change(self, old_status: HealthStatus, new_status: HealthStatus):
        """Notify listeners of status change"""
        for listener in self._status_change_listeners:
            try:
                listener(old_status, new_status)
            except Exception as e:
                logger.error(f"Error notifying status change listener: {e}")
    
    def add_status_change_listener(self, listener: Callable):
        """Add listener for status changes"""
        self._status_change_listeners.append(listener)
    
    def get_status(self, name: Optional[str] = None) -> HealthStatus:
        """
        Get health status
        
        Args:
            name: Name of specific check (None for overall status)
            
        Returns:
            Health status
        """
        with self._lock:
            if name is None:
                return self._overall_status
            
            if name not in self._check_results:
                return HealthStatus.UNKNOWN
            
            if not self._check_results[name]:
                return HealthStatus.UNKNOWN
            
            return self._check_results[name][-1].status
    
    def get_latest_result(self, name: str) -> Optional[HealthCheckResult]:
        """Get latest result for a specific health check"""
        with self._lock:
            if name in self._check_results and self._check_results[name]:
                return self._check_results[name][-1]
            return None
    
    def get_all_results(self) -> Dict[str, HealthCheckResult]:
        """Get latest results for all health checks"""
        with self._lock:
            results = {}
            for name in self._health_checks:
                if self._check_results[name]:
                    results[name] = self._check_results[name][-1]
            return results
    
    def perform_check_now(self, name: str) -> HealthCheckResult:
        """
        Perform a health check immediately
        
        Args:
            name: Name of health check
            
        Returns:
            Health check result
        """
        with self._lock:
            if name not in self._health_checks:
                raise ValueError(f"Health check '{name}' not registered")
            
            check = self._health_checks[name]
        
        result = self._perform_check(check)
        self._process_check_result(check, result)
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get health checker metrics"""
        with self._lock:
            metrics = {
                'overall_status': self._overall_status.value,
                'total_checks': len(self._health_checks),
                'checks': {}
            }
            
            for name, check in self._health_checks.items():
                check_metrics = {
                    'status': self.get_status(name).value,
                    'failure_count': self._failure_counts[name],
                    'success_count': self._success_counts[name],
                    'recovery_attempts': self._recovery_attempts[name],
                    'critical': check.critical
                }
                
                if self._check_results[name]:
                    latest = self._check_results[name][-1]
                    check_metrics['last_check_time'] = latest.timestamp.isoformat()
                    check_metrics['response_time'] = latest.response_time
                    if latest.error:
                        check_metrics['last_error'] = latest.error
                
                metrics['checks'][name] = check_metrics
            
            return metrics