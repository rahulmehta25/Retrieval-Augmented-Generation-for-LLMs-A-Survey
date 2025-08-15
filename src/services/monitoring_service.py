"""
Monitoring Service - Handles metrics, monitoring and health checks
"""

import logging
from typing import Dict, Any, Optional, Callable
from .interfaces import MonitoringServiceInterface
from ..monitoring.production_monitoring import ProductionMonitoring, AlertSeverity

logger = logging.getLogger(__name__)

class MonitoringService:
    """
    Service responsible for monitoring, metrics and health checks
    Implements single responsibility principle for monitoring operations
    """
    
    def __init__(
        self,
        service_name: str = "production_rag",
        prometheus_port: int = 8000,
        persist_path: str = "./monitoring",
        enable_prometheus: bool = True
    ):
        """Initialize monitoring service"""
        self.monitoring = ProductionMonitoring(
            service_name=service_name,
            prometheus_port=prometheus_port,
            enable_prometheus=enable_prometheus,
            persist_path=persist_path
        )
        self.service_name = service_name
        logger.info(f"MonitoringService initialized for service: {service_name}")
    
    def track_request(
        self,
        method: str,
        duration_ms: float,
        status: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Track request metrics
        
        Args:
            method: Request method/endpoint
            duration_ms: Request duration in milliseconds
            status: Request status (success, error, etc.)
            metadata: Additional request metadata
        """
        try:
            self.monitoring.track_request(
                method=method,
                duration_ms=duration_ms,
                status=status,
                metadata=metadata or {}
            )
            
            # Log slow requests
            if duration_ms > 5000:  # 5 seconds
                logger.warning(f"Slow request detected: {method} took {duration_ms:.0f}ms")
            
        except Exception as e:
            logger.error(f"Error tracking request: {e}")
    
    def track_retrieval(
        self,
        retriever_type: str,
        latency_ms: float,
        contexts_count: int
    ) -> None:
        """
        Track retrieval metrics
        
        Args:
            retriever_type: Type of retriever used
            latency_ms: Retrieval latency in milliseconds
            contexts_count: Number of contexts retrieved
        """
        try:
            self.monitoring.track_retrieval(
                retriever_type=retriever_type,
                latency_ms=latency_ms,
                contexts_count=contexts_count
            )
            
        except Exception as e:
            logger.error(f"Error tracking retrieval: {e}")
    
    def track_generation(
        self,
        tokens: int,
        latency_ms: float,
        model: str
    ) -> None:
        """
        Track generation metrics
        
        Args:
            tokens: Number of tokens generated
            latency_ms: Generation latency in milliseconds
            model: Model used for generation
        """
        try:
            self.monitoring.track_generation(
                tokens=tokens,
                latency_ms=latency_ms,
                model=model
            )
            
            # Track tokens per second
            if latency_ms > 0:
                tokens_per_second = (tokens / latency_ms) * 1000
                logger.debug(f"Generation rate: {tokens_per_second:.2f} tokens/sec")
            
        except Exception as e:
            logger.error(f"Error tracking generation: {e}")
    
    def track_error(
        self,
        error_type: str,
        component: str,
        message: str
    ) -> None:
        """
        Track error occurrence
        
        Args:
            error_type: Type/category of error
            component: Component where error occurred
            message: Error message
        """
        try:
            self.monitoring.track_error(
                error_type=error_type,
                component=component,
                message=message
            )
            
            logger.error(f"Tracked error in {component}: {error_type} - {message}")
            
        except Exception as e:
            logger.error(f"Error tracking error: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Get basic health checks
            health_status = {
                "service_name": self.service_name,
                "status": "healthy",
                "checks": {},
                "metrics_summary": {}
            }
            
            # Run health checks
            if hasattr(self.monitoring, 'health_checks'):
                for check_name, check_func in self.monitoring.health_checks.items():
                    try:
                        is_healthy, message = check_func()
                        health_status["checks"][check_name] = {
                            "healthy": is_healthy,
                            "message": message
                        }
                        
                        if not is_healthy:
                            health_status["status"] = "unhealthy"
                            
                    except Exception as e:
                        health_status["checks"][check_name] = {
                            "healthy": False,
                            "message": f"Health check failed: {str(e)}"
                        }
                        health_status["status"] = "unhealthy"
            
            # Get metrics summary
            try:
                metrics_summary = self.monitoring.get_metrics_summary(window_seconds=300)
                health_status["metrics_summary"] = metrics_summary
            except Exception as e:
                logger.warning(f"Error getting metrics summary: {e}")
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "service_name": self.service_name,
                "status": "error",
                "message": str(e)
            }
    
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """
        Register a health check function
        
        Args:
            name: Name of the health check
            check_func: Function that returns (is_healthy: bool, message: str)
        """
        try:
            self.monitoring.register_health_check(name, check_func)
            logger.info(f"Registered health check: {name}")
            
        except Exception as e:
            logger.error(f"Error registering health check {name}: {e}")
    
    def create_alert(
        self,
        name: str,
        severity: str,
        message: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Create monitoring alert
        
        Args:
            name: Alert name
            severity: Alert severity (critical, error, warning, info)
            message: Alert message
            metadata: Additional alert metadata
        """
        try:
            # Map severity string to enum
            severity_map = {
                "critical": AlertSeverity.CRITICAL,
                "error": AlertSeverity.ERROR,
                "warning": AlertSeverity.WARNING,
                "info": AlertSeverity.INFO
            }
            
            alert_severity = severity_map.get(severity.lower(), AlertSeverity.INFO)
            
            self.monitoring.create_alert(
                name=name,
                severity=alert_severity,
                message=message,
                metadata=metadata or {}
            )
            
            logger.info(f"Created {severity} alert: {name}")
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    def get_metrics_summary(self, window_seconds: int = 300) -> Dict[str, Any]:
        """
        Get metrics summary for the specified time window
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Dictionary with metrics summary
        """
        try:
            return self.monitoring.get_metrics_summary(window_seconds)
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}
    
    def export_metrics(self, output_path: str) -> None:
        """
        Export metrics to file
        
        Args:
            output_path: Path to export metrics to
        """
        try:
            self.monitoring.export_metrics(output_path)
            logger.info(f"Metrics exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def track_custom_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Track custom metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
        """
        try:
            # Use the underlying monitoring system to track custom metrics
            # This would need to be implemented based on the specific monitoring backend
            logger.info(f"Tracking custom metric: {metric_name} = {value}")
            
        except Exception as e:
            logger.error(f"Error tracking custom metric {metric_name}: {e}")
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring system statistics"""
        return {
            "service_name": self.service_name,
            "prometheus_enabled": hasattr(self.monitoring, 'prometheus_enabled'),
            "alerts_created": 0,  # Would track actual counts
            "health_checks_registered": len(getattr(self.monitoring, 'health_checks', {}))
        }