"""
Resilience Monitoring and Metrics Collection

Provides monitoring for circuit breakers, retry policies, and other resilience patterns.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResilienceMetrics:
    """Container for resilience metrics"""
    timestamp: datetime
    circuit_breakers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    retry_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bulkheads: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timeouts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fallbacks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    health_checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects metrics from resilience components
    
    Aggregates metrics from circuit breakers, retry policies, bulkheads, etc.
    """
    
    def __init__(
        self,
        collection_interval: float = 60.0,
        retention_period: timedelta = timedelta(hours=24),
        max_metrics_size: int = 10000
    ):
        """
        Initialize metrics collector
        
        Args:
            collection_interval: Interval between metric collections (seconds)
            retention_period: How long to retain metrics
            max_metrics_size: Maximum number of metrics to retain
        """
        self.collection_interval = collection_interval
        self.retention_period = retention_period
        self.max_metrics_size = max_metrics_size
        
        self._metrics_history: deque = deque(maxlen=max_metrics_size)
        self._components = {
            'circuit_breakers': {},
            'retry_policies': {},
            'bulkheads': {},
            'timeout_managers': {},
            'fallback_handlers': {},
            'health_checkers': {}
        }
        
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Metrics listeners
        self._metrics_listeners: List[Callable] = []
    
    def register_circuit_breaker(self, name: str, circuit_breaker):
        """Register a circuit breaker for monitoring"""
        with self._lock:
            self._components['circuit_breakers'][name] = circuit_breaker
    
    def register_retry_policy(self, name: str, retry_policy):
        """Register a retry policy for monitoring"""
        with self._lock:
            self._components['retry_policies'][name] = retry_policy
    
    def register_bulkhead(self, name: str, bulkhead):
        """Register a bulkhead for monitoring"""
        with self._lock:
            self._components['bulkheads'][name] = bulkhead
    
    def register_timeout_manager(self, name: str, timeout_manager):
        """Register a timeout manager for monitoring"""
        with self._lock:
            self._components['timeout_managers'][name] = timeout_manager
    
    def register_fallback_handler(self, name: str, fallback_handler):
        """Register a fallback handler for monitoring"""
        with self._lock:
            self._components['fallback_handlers'][name] = fallback_handler
    
    def register_health_checker(self, name: str, health_checker):
        """Register a health checker for monitoring"""
        with self._lock:
            self._components['health_checkers'][name] = health_checker
    
    def start(self):
        """Start metrics collection"""
        if self._collection_thread and self._collection_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            name="metrics-collector"
        )
        self._collection_thread.daemon = True
        self._collection_thread.start()
        
        logger.info("Metrics collector started")
    
    def stop(self):
        """Stop metrics collection"""
        self._stop_event.set()
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        
        logger.info("Metrics collector stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while not self._stop_event.is_set():
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Store metrics
                self._store_metrics(metrics)
                
                # Notify listeners
                self._notify_listeners(metrics)
                
                # Clean old metrics
                self._clean_old_metrics()
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
            
            # Wait for next collection
            self._stop_event.wait(self.collection_interval)
    
    def _collect_metrics(self) -> ResilienceMetrics:
        """Collect metrics from all registered components"""
        metrics = ResilienceMetrics(timestamp=datetime.now())
        
        with self._lock:
            # Collect circuit breaker metrics
            for name, cb in self._components['circuit_breakers'].items():
                try:
                    metrics.circuit_breakers[name] = cb.get_metrics()
                except Exception as e:
                    logger.error(f"Error collecting circuit breaker metrics for {name}: {e}")
            
            # Collect bulkhead metrics
            for name, bulkhead in self._components['bulkheads'].items():
                try:
                    bulkhead_metrics = bulkhead.get_metrics()
                    metrics.bulkheads[name] = {
                        'type': bulkhead_metrics.type.value,
                        'max_concurrent': bulkhead_metrics.max_concurrent,
                        'active_count': bulkhead_metrics.active_count,
                        'waiting_count': bulkhead_metrics.waiting_count,
                        'completed_count': bulkhead_metrics.completed_count,
                        'rejected_count': bulkhead_metrics.rejected_count,
                        'average_execution_time': bulkhead_metrics.average_execution_time
                    }
                except Exception as e:
                    logger.error(f"Error collecting bulkhead metrics for {name}: {e}")
            
            # Collect timeout manager metrics
            for name, tm in self._components['timeout_managers'].items():
                try:
                    metrics.timeouts[name] = tm.get_stats()
                except Exception as e:
                    logger.error(f"Error collecting timeout metrics for {name}: {e}")
            
            # Collect fallback handler metrics
            for name, fh in self._components['fallback_handlers'].items():
                try:
                    metrics.fallbacks[name] = fh.get_statistics()
                except Exception as e:
                    logger.error(f"Error collecting fallback metrics for {name}: {e}")
            
            # Collect health checker metrics
            for name, hc in self._components['health_checkers'].items():
                try:
                    metrics.health_checks[name] = hc.get_metrics()
                except Exception as e:
                    logger.error(f"Error collecting health check metrics for {name}: {e}")
        
        return metrics
    
    def _store_metrics(self, metrics: ResilienceMetrics):
        """Store metrics in history"""
        self._metrics_history.append(metrics)
    
    def _clean_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.now() - self.retention_period
        
        # Remove old metrics
        while self._metrics_history and self._metrics_history[0].timestamp < cutoff_time:
            self._metrics_history.popleft()
    
    def _notify_listeners(self, metrics: ResilienceMetrics):
        """Notify metrics listeners"""
        for listener in self._metrics_listeners:
            try:
                listener(metrics)
            except Exception as e:
                logger.error(f"Error notifying metrics listener: {e}")
    
    def add_metrics_listener(self, listener: Callable):
        """Add a metrics listener"""
        self._metrics_listeners.append(listener)
    
    def get_latest_metrics(self) -> Optional[ResilienceMetrics]:
        """Get the most recent metrics"""
        if self._metrics_history:
            return self._metrics_history[-1]
        return None
    
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[ResilienceMetrics]:
        """
        Get metrics history within time range
        
        Args:
            start_time: Start of time range (None for beginning)
            end_time: End of time range (None for current)
            
        Returns:
            List of metrics within range
        """
        if not start_time:
            start_time = datetime.min
        if not end_time:
            end_time = datetime.now()
        
        return [
            m for m in self._metrics_history
            if start_time <= m.timestamp <= end_time
        ]
    
    def export_metrics(self, format: str = 'json') -> str:
        """
        Export metrics in specified format
        
        Args:
            format: Export format ('json', 'prometheus')
            
        Returns:
            Formatted metrics string
        """
        latest = self.get_latest_metrics()
        if not latest:
            return "{}" if format == 'json' else ""
        
        if format == 'json':
            return self._export_json(latest)
        elif format == 'prometheus':
            return self._export_prometheus(latest)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, metrics: ResilienceMetrics) -> str:
        """Export metrics as JSON"""
        data = {
            'timestamp': metrics.timestamp.isoformat(),
            'circuit_breakers': metrics.circuit_breakers,
            'retry_policies': metrics.retry_policies,
            'bulkheads': metrics.bulkheads,
            'timeouts': metrics.timeouts,
            'fallbacks': metrics.fallbacks,
            'health_checks': metrics.health_checks
        }
        return json.dumps(data, indent=2)
    
    def _export_prometheus(self, metrics: ResilienceMetrics) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Circuit breaker metrics
        for name, cb_metrics in metrics.circuit_breakers.items():
            lines.append(f'# HELP circuit_breaker_state Circuit breaker state (0=closed, 1=open, 2=half_open)')
            state_value = {'closed': 0, 'open': 1, 'half_open': 2}.get(cb_metrics.get('state', 'closed'), 0)
            lines.append(f'circuit_breaker_state{{name="{name}"}} {state_value}')
            lines.append(f'circuit_breaker_failure_count{{name="{name}"}} {cb_metrics.get("failure_count", 0)}')
            lines.append(f'circuit_breaker_failure_rate{{name="{name}"}} {cb_metrics.get("failure_rate", 0.0)}')
        
        # Bulkhead metrics
        for name, bh_metrics in metrics.bulkheads.items():
            lines.append(f'bulkhead_active_count{{name="{name}"}} {bh_metrics.get("active_count", 0)}')
            lines.append(f'bulkhead_waiting_count{{name="{name}"}} {bh_metrics.get("waiting_count", 0)}')
            lines.append(f'bulkhead_rejected_count{{name="{name}"}} {bh_metrics.get("rejected_count", 0)}')
        
        # Timeout metrics
        for name, tm_metrics in metrics.timeouts.items():
            lines.append(f'timeout_total_calls{{name="{name}"}} {tm_metrics.get("total_calls", 0)}')
            lines.append(f'timeout_timeout_count{{name="{name}"}} {tm_metrics.get("timeout_count", 0)}')
            lines.append(f'timeout_rate{{name="{name}"}} {tm_metrics.get("timeout_rate", 0.0)}')
        
        # Fallback metrics
        for name, fb_metrics in metrics.fallbacks.items():
            lines.append(f'fallback_total_calls{{name="{name}"}} {fb_metrics.get("total_calls", 0)}')
            lines.append(f'fallback_count{{name="{name}"}} {fb_metrics.get("fallback_count", 0)}')
            lines.append(f'fallback_rate{{name="{name}"}} {fb_metrics.get("fallback_rate", 0.0)}')
        
        return '\n'.join(lines)


class ResilienceMonitor:
    """
    High-level monitor for resilience patterns
    
    Provides aggregated view and alerts for resilience metrics.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_thresholds: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize resilience monitor
        
        Args:
            metrics_collector: Metrics collector instance
            alert_thresholds: Thresholds for alerts
        """
        self.metrics_collector = metrics_collector
        self.alert_thresholds = alert_thresholds or {
            'circuit_breaker_open_duration': 300.0,  # 5 minutes
            'retry_failure_rate': 0.5,
            'bulkhead_rejection_rate': 0.1,
            'timeout_rate': 0.2,
            'fallback_rate': 0.3
        }
        
        self._alerts: List[Dict[str, Any]] = []
        self._alert_listeners: List[Callable] = []
        
        # Register as metrics listener
        metrics_collector.add_metrics_listener(self._process_metrics)
    
    def _process_metrics(self, metrics: ResilienceMetrics):
        """Process metrics and generate alerts"""
        alerts = []
        
        # Check circuit breakers
        for name, cb_metrics in metrics.circuit_breakers.items():
            if cb_metrics.get('state') == 'open':
                alerts.append({
                    'type': 'circuit_breaker_open',
                    'component': name,
                    'message': f"Circuit breaker '{name}' is OPEN",
                    'severity': 'high',
                    'timestamp': metrics.timestamp
                })
        
        # Check bulkhead rejections
        for name, bh_metrics in metrics.bulkheads.items():
            total = bh_metrics.get('completed_count', 0) + bh_metrics.get('rejected_count', 0)
            if total > 0:
                rejection_rate = bh_metrics.get('rejected_count', 0) / total
                if rejection_rate > self.alert_thresholds['bulkhead_rejection_rate']:
                    alerts.append({
                        'type': 'bulkhead_high_rejection',
                        'component': name,
                        'message': f"Bulkhead '{name}' rejection rate: {rejection_rate:.2%}",
                        'severity': 'medium',
                        'timestamp': metrics.timestamp
                    })
        
        # Check timeout rates
        for name, tm_metrics in metrics.timeouts.items():
            timeout_rate = tm_metrics.get('timeout_rate', 0.0)
            if timeout_rate > self.alert_thresholds['timeout_rate']:
                alerts.append({
                    'type': 'high_timeout_rate',
                    'component': name,
                    'message': f"Timeout rate for '{name}': {timeout_rate:.2%}",
                    'severity': 'medium',
                    'timestamp': metrics.timestamp
                })
        
        # Check fallback rates
        for name, fb_metrics in metrics.fallbacks.items():
            fallback_rate = fb_metrics.get('fallback_rate', 0.0)
            if fallback_rate > self.alert_thresholds['fallback_rate']:
                alerts.append({
                    'type': 'high_fallback_rate',
                    'component': name,
                    'message': f"Fallback rate for '{name}': {fallback_rate:.2%}",
                    'severity': 'low',
                    'timestamp': metrics.timestamp
                })
        
        # Store and notify alerts
        if alerts:
            self._alerts.extend(alerts)
            self._notify_alert_listeners(alerts)
    
    def _notify_alert_listeners(self, alerts: List[Dict[str, Any]]):
        """Notify alert listeners"""
        for listener in self._alert_listeners:
            try:
                listener(alerts)
            except Exception as e:
                logger.error(f"Error notifying alert listener: {e}")
    
    def add_alert_listener(self, listener: Callable):
        """Add an alert listener"""
        self._alert_listeners.append(listener)
    
    def get_alerts(
        self,
        severity: Optional[str] = None,
        component: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get alerts filtered by criteria
        
        Args:
            severity: Filter by severity ('low', 'medium', 'high')
            component: Filter by component name
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        filtered_alerts = self._alerts
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.get('severity') == severity]
        
        if component:
            filtered_alerts = [a for a in filtered_alerts if a.get('component') == component]
        
        # Return most recent alerts
        return filtered_alerts[-limit:]
    
    def clear_alerts(self):
        """Clear all alerts"""
        self._alerts.clear()