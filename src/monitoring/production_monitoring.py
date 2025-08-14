"""
Production Monitoring and Observability System
"""

import time
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
from enum import Enum
import numpy as np
import threading
from pathlib import Path
import traceback
import psutil
import prometheus_client as prom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """System alert"""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class HealthCheck:
    """Component health check"""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: float
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class ProductionMonitoring:
    """
    Comprehensive monitoring system with:
    - Metrics collection and aggregation
    - Health checks
    - Alerting
    - Performance tracking
    - Error tracking
    - Resource monitoring
    - Distributed tracing
    """
    
    def __init__(
        self,
        service_name: str = "rag_system",
        prometheus_port: int = 8000,
        enable_prometheus: bool = True,
        persist_path: Optional[str] = "./monitoring"
    ):
        """Initialize monitoring system"""
        
        self.service_name = service_name
        self.persist_path = Path(persist_path) if persist_path else None
        
        # Metrics storage
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(lambda: deque(maxlen=10000))
        
        # Prometheus metrics
        self.prom_metrics = {}
        if enable_prometheus:
            self._setup_prometheus_metrics()
            prom.start_http_server(prometheus_port)
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_check_callbacks: Dict[str, Callable] = {}
        
        # Alerts
        self.alerts: deque = deque(maxlen=1000)
        self.alert_rules: List[Dict] = []
        
        # Performance tracking
        self.request_traces: deque = deque(maxlen=5000)
        self.slow_queries: deque = deque(maxlen=100)
        
        # Error tracking
        self.errors: deque = deque(maxlen=1000)
        self.error_rates: Dict[str, float] = defaultdict(float)
        
        # Resource monitoring
        self.resource_metrics = {
            'cpu_percent': deque(maxlen=1000),
            'memory_percent': deque(maxlen=1000),
            'disk_usage': deque(maxlen=1000),
            'network_io': deque(maxlen=1000)
        }
        
        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Monitoring system initialized for {service_name}")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        
        # Request metrics
        self.prom_metrics['request_count'] = prom.Counter(
            'rag_requests_total',
            'Total number of RAG requests',
            ['method', 'status']
        )
        
        self.prom_metrics['request_duration'] = prom.Histogram(
            'rag_request_duration_seconds',
            'Request duration in seconds',
            ['method'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Retrieval metrics
        self.prom_metrics['retrieval_count'] = prom.Counter(
            'rag_retrieval_total',
            'Total number of retrieval operations',
            ['retriever_type']
        )
        
        self.prom_metrics['retrieval_latency'] = prom.Histogram(
            'rag_retrieval_latency_seconds',
            'Retrieval latency in seconds',
            ['retriever_type']
        )
        
        self.prom_metrics['contexts_retrieved'] = prom.Histogram(
            'rag_contexts_retrieved',
            'Number of contexts retrieved',
            buckets=(1, 2, 3, 5, 10, 15, 20, 25, 30)
        )
        
        # Generation metrics
        self.prom_metrics['generation_tokens'] = prom.Histogram(
            'rag_generation_tokens',
            'Number of tokens generated',
            buckets=(10, 25, 50, 100, 250, 500, 1000, 2000)
        )
        
        # Cache metrics
        self.prom_metrics['cache_hits'] = prom.Counter(
            'rag_cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.prom_metrics['cache_misses'] = prom.Counter(
            'rag_cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
        
        # Error metrics
        self.prom_metrics['errors'] = prom.Counter(
            'rag_errors_total',
            'Total errors',
            ['error_type', 'component']
        )
        
        # System metrics
        self.prom_metrics['cpu_usage'] = prom.Gauge(
            'rag_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.prom_metrics['memory_usage'] = prom.Gauge(
            'rag_memory_usage_percent',
            'Memory usage percentage'
        )
        
        # Health metrics
        self.prom_metrics['health_status'] = prom.Gauge(
            'rag_health_status',
            'Health status (1=healthy, 0.5=degraded, 0=unhealthy)',
            ['component']
        )
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value"""
        
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        
        self.metrics[name].append(point)
        
        # Update Prometheus metrics if applicable
        if name in self.prom_metrics:
            metric = self.prom_metrics[name]
            
            if isinstance(metric, prom.Counter):
                metric.labels(**labels).inc(value)
            elif isinstance(metric, prom.Gauge):
                metric.labels(**labels) if labels else metric.set(value)
            elif isinstance(metric, prom.Histogram):
                metric.labels(**labels) if labels else metric.observe(value)
    
    def track_request(
        self,
        method: str,
        duration_ms: float,
        status: str = "success",
        metadata: Optional[Dict] = None
    ):
        """Track a request"""
        
        # Record metrics
        self.record_metric(
            'request_count',
            1,
            MetricType.COUNTER,
            {'method': method, 'status': status}
        )
        
        self.record_metric(
            'request_duration',
            duration_ms / 1000,
            MetricType.HISTOGRAM,
            {'method': method}
        )
        
        # Store trace
        trace = {
            'timestamp': datetime.now(),
            'method': method,
            'duration_ms': duration_ms,
            'status': status,
            'metadata': metadata or {}
        }
        
        self.request_traces.append(trace)
        
        # Check for slow queries
        if duration_ms > 1000:  # > 1 second
            self.slow_queries.append(trace)
            
            # Alert on slow query
            self.create_alert(
                name="slow_query",
                severity=AlertSeverity.WARNING,
                message=f"Slow query detected: {method} took {duration_ms:.2f}ms",
                metadata=trace
            )
    
    def track_retrieval(
        self,
        retriever_type: str,
        latency_ms: float,
        contexts_count: int,
        metadata: Optional[Dict] = None
    ):
        """Track retrieval operation"""
        
        self.record_metric(
            'retrieval_count',
            1,
            MetricType.COUNTER,
            {'retriever_type': retriever_type}
        )
        
        self.record_metric(
            'retrieval_latency',
            latency_ms / 1000,
            MetricType.HISTOGRAM,
            {'retriever_type': retriever_type}
        )
        
        self.record_metric(
            'contexts_retrieved',
            contexts_count,
            MetricType.HISTOGRAM
        )
    
    def track_generation(
        self,
        tokens: int,
        latency_ms: float,
        model: str,
        metadata: Optional[Dict] = None
    ):
        """Track generation operation"""
        
        self.record_metric(
            'generation_tokens',
            tokens,
            MetricType.HISTOGRAM
        )
        
        self.record_metric(
            'generation_latency',
            latency_ms / 1000,
            MetricType.HISTOGRAM,
            {'model': model}
        )
    
    def track_cache(
        self,
        cache_type: str,
        hit: bool
    ):
        """Track cache operation"""
        
        if hit:
            self.record_metric(
                'cache_hits',
                1,
                MetricType.COUNTER,
                {'cache_type': cache_type}
            )
        else:
            self.record_metric(
                'cache_misses',
                1,
                MetricType.COUNTER,
                {'cache_type': cache_type}
            )
    
    def track_error(
        self,
        error_type: str,
        component: str,
        message: str,
        stack_trace: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Track an error"""
        
        error = {
            'timestamp': datetime.now(),
            'error_type': error_type,
            'component': component,
            'message': message,
            'stack_trace': stack_trace or traceback.format_exc(),
            'metadata': metadata or {}
        }
        
        self.errors.append(error)
        
        # Update error rate
        self.error_rates[component] += 1
        
        # Record metric
        self.record_metric(
            'errors',
            1,
            MetricType.COUNTER,
            {'error_type': error_type, 'component': component}
        )
        
        # Create alert for critical errors
        if error_type in ['CRITICAL', 'FATAL']:
            self.create_alert(
                name=f"critical_error_{component}",
                severity=AlertSeverity.CRITICAL,
                message=f"Critical error in {component}: {message}",
                metadata=error
            )
    
    def register_health_check(
        self,
        component: str,
        check_func: Callable[[], Tuple[bool, str]]
    ):
        """Register a health check function"""
        
        self.health_check_callbacks[component] = check_func
    
    def perform_health_checks(self) -> Dict[str, HealthCheck]:
        """Perform all registered health checks"""
        
        results = {}
        
        for component, check_func in self.health_check_callbacks.items():
            start_time = time.time()
            
            try:
                is_healthy, message = check_func()
                status = "healthy" if is_healthy else "unhealthy"
            except Exception as e:
                status = "unhealthy"
                message = f"Health check failed: {str(e)}"
            
            latency_ms = (time.time() - start_time) * 1000
            
            health = HealthCheck(
                component=component,
                status=status,
                latency_ms=latency_ms,
                message=message,
                timestamp=datetime.now()
            )
            
            self.health_checks[component] = health
            results[component] = health
            
            # Update Prometheus metric
            if 'health_status' in self.prom_metrics:
                value = 1.0 if status == "healthy" else 0.5 if status == "degraded" else 0.0
                self.prom_metrics['health_status'].labels(component=component).set(value)
        
        return results
    
    def create_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        metadata: Optional[Dict] = None
    ):
        """Create an alert"""
        
        alert = Alert(
            alert_id=f"{name}_{datetime.now().timestamp()}",
            name=name,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Log alert
        if severity == AlertSeverity.CRITICAL:
            logger.critical(f"ALERT: {message}")
        elif severity == AlertSeverity.ERROR:
            logger.error(f"ALERT: {message}")
        elif severity == AlertSeverity.WARNING:
            logger.warning(f"ALERT: {message}")
        else:
            logger.info(f"ALERT: {message}")
        
        # TODO: Send to external alerting system (PagerDuty, Slack, etc.)
        
        return alert
    
    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[], bool],
        severity: AlertSeverity,
        message: str,
        check_interval: int = 60
    ):
        """Add an alert rule"""
        
        rule = {
            'name': name,
            'condition': condition,
            'severity': severity,
            'message': message,
            'check_interval': check_interval,
            'last_checked': datetime.now()
        }
        
        self.alert_rules.append(rule)
    
    def _check_alert_rules(self):
        """Check all alert rules"""
        
        now = datetime.now()
        
        for rule in self.alert_rules:
            # Check if it's time to evaluate
            if (now - rule['last_checked']).seconds < rule['check_interval']:
                continue
            
            rule['last_checked'] = now
            
            try:
                if rule['condition']():
                    self.create_alert(
                        name=rule['name'],
                        severity=rule['severity'],
                        message=rule['message']
                    )
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    def _collect_resource_metrics(self):
        """Collect system resource metrics"""
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.resource_metrics['cpu_percent'].append({
            'timestamp': datetime.now(),
            'value': cpu_percent
        })
        
        if 'cpu_usage' in self.prom_metrics:
            self.prom_metrics['cpu_usage'].set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.resource_metrics['memory_percent'].append({
            'timestamp': datetime.now(),
            'value': memory_percent
        })
        
        if 'memory_usage' in self.prom_metrics:
            self.prom_metrics['memory_usage'].set(memory_percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.resource_metrics['disk_usage'].append({
            'timestamp': datetime.now(),
            'value': disk.percent
        })
        
        # Network I/O
        net_io = psutil.net_io_counters()
        self.resource_metrics['network_io'].append({
            'timestamp': datetime.now(),
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv
        })
        
        # Check for resource alerts
        if cpu_percent > 90:
            self.create_alert(
                name="high_cpu",
                severity=AlertSeverity.WARNING,
                message=f"High CPU usage: {cpu_percent:.1f}%"
            )
        
        if memory_percent > 90:
            self.create_alert(
                name="high_memory",
                severity=AlertSeverity.WARNING,
                message=f"High memory usage: {memory_percent:.1f}%"
            )
    
    def _background_monitoring(self):
        """Background monitoring thread"""
        
        while True:
            try:
                # Collect resource metrics
                self._collect_resource_metrics()
                
                # Perform health checks
                self.perform_health_checks()
                
                # Check alert rules
                self._check_alert_rules()
                
                # Calculate error rates
                self._calculate_error_rates()
                
                # Clean old data
                self._cleanup_old_data()
                
                # Sleep
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                time.sleep(60)  # Back off on error
    
    def _calculate_error_rates(self):
        """Calculate error rates over time windows"""
        
        now = datetime.now()
        time_windows = [60, 300, 900]  # 1min, 5min, 15min
        
        for window in time_windows:
            cutoff = now - timedelta(seconds=window)
            
            # Count recent errors
            recent_errors = [e for e in self.errors 
                           if e['timestamp'] > cutoff]
            
            # Calculate rate per component
            component_errors = Counter(e['component'] for e in recent_errors)
            
            for component, count in component_errors.items():
                rate = count / window * 60  # Errors per minute
                
                # Check threshold
                if rate > 1.0:  # More than 1 error per minute
                    self.create_alert(
                        name=f"high_error_rate_{component}",
                        severity=AlertSeverity.ERROR,
                        message=f"High error rate in {component}: {rate:.2f} errors/min",
                        metadata={'window_seconds': window, 'error_count': count}
                    )
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        
        # Keep only recent data (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        
        # Clean metrics
        for metric_name, points in self.metrics.items():
            while points and points[0].timestamp < cutoff:
                points.popleft()
    
    def get_metrics_summary(self, time_window: int = 300) -> Dict[str, Any]:
        """Get summary of recent metrics"""
        
        now = datetime.now()
        cutoff = now - timedelta(seconds=time_window)
        
        summary = {
            'time_window_seconds': time_window,
            'timestamp': now,
            'metrics': {}
        }
        
        # Aggregate metrics
        for metric_name, points in self.metrics.items():
            recent_points = [p for p in points if p.timestamp > cutoff]
            
            if recent_points:
                values = [p.value for p in recent_points]
                summary['metrics'][metric_name] = {
                    'count': len(values),
                    'sum': sum(values),
                    'mean': np.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        return summary
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        
        return {
            'service': self.service_name,
            'timestamp': datetime.now(),
            'health_checks': {k: asdict(v) for k, v in self.health_checks.items()},
            'metrics_summary': self.get_metrics_summary(),
            'recent_errors': [e for e in self.errors][-10:],
            'active_alerts': [asdict(a) for a in self.alerts if not a.resolved],
            'slow_queries': list(self.slow_queries)[-10:],
            'resource_usage': {
                'cpu': self.resource_metrics['cpu_percent'][-1]['value'] 
                       if self.resource_metrics['cpu_percent'] else 0,
                'memory': self.resource_metrics['memory_percent'][-1]['value']
                         if self.resource_metrics['memory_percent'] else 0
            }
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to file"""
        
        data = {
            'service': self.service_name,
            'exported_at': datetime.now().isoformat(),
            'metrics': {},
            'health_checks': {k: asdict(v) for k, v in self.health_checks.items()},
            'alerts': [asdict(a) for a in self.alerts],
            'errors': list(self.errors)
        }
        
        # Convert metrics
        for metric_name, points in self.metrics.items():
            data['metrics'][metric_name] = [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'value': p.value,
                    'labels': p.labels
                }
                for p in points
            ]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")