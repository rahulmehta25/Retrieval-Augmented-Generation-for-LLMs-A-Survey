"""
Distributed Tracing Module

Provides distributed tracing support for monitoring request flow across services.
"""

import time
import uuid
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TraceContext:
    """Context for distributed tracing"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, Any] = field(default_factory=dict)
    flags: int = 0


@dataclass
class Span:
    """Represents a span in a distributed trace"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "in_progress"
    error: Optional[str] = None


class DistributedTracer:
    """
    Distributed tracing implementation
    
    Tracks request flow across services for monitoring and debugging.
    Compatible with OpenTelemetry and Jaeger formats.
    """
    
    def __init__(
        self,
        service_name: str,
        enabled: bool = True,
        sampling_rate: float = 1.0,
        max_spans: int = 10000,
        export_endpoint: Optional[str] = None
    ):
        """
        Initialize distributed tracer
        
        Args:
            service_name: Name of the service
            enabled: Whether tracing is enabled
            sampling_rate: Sampling rate (0.0 to 1.0)
            max_spans: Maximum spans to keep in memory
            export_endpoint: Endpoint to export traces
        """
        self.service_name = service_name
        self.enabled = enabled
        self.sampling_rate = sampling_rate
        self.max_spans = max_spans
        self.export_endpoint = export_endpoint
        
        self._spans: Dict[str, Span] = {}
        self._completed_spans: List[Span] = []
        self._current_context: threading.local = threading.local()
        self._lock = threading.Lock()
        self._span_listeners: List[Callable] = []
    
    def start_span(
        self,
        operation_name: str,
        parent_context: Optional[TraceContext] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> TraceContext:
        """
        Start a new span
        
        Args:
            operation_name: Name of the operation
            parent_context: Parent trace context
            tags: Initial tags for the span
            
        Returns:
            Trace context for the new span
        """
        if not self.enabled:
            return self._create_noop_context()
        
        # Check sampling
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            trace_id = self._generate_trace_id()
            parent_span_id = None
            
            # Apply sampling decision
            import random
            if random.random() > self.sampling_rate:
                return self._create_noop_context()
        
        span_id = self._generate_span_id()
        
        # Create span
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        # Add service name tag
        span.tags['service'] = self.service_name
        
        # Store span
        with self._lock:
            self._spans[span_id] = span
        
        # Create context
        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )
        
        # Set as current context
        self._current_context.context = context
        
        return context
    
    def finish_span(
        self,
        context: TraceContext,
        status: str = "success",
        error: Optional[str] = None
    ):
        """
        Finish a span
        
        Args:
            context: Trace context of the span
            status: Final status of the span
            error: Error message if span failed
        """
        if not self.enabled or not context.span_id:
            return
        
        with self._lock:
            if context.span_id not in self._spans:
                return
            
            span = self._spans[context.span_id]
            span.end_time = time.time()
            span.status = status
            span.error = error
            
            # Move to completed spans
            del self._spans[context.span_id]
            self._completed_spans.append(span)
            
            # Maintain max spans limit
            if len(self._completed_spans) > self.max_spans:
                self._completed_spans.pop(0)
        
        # Notify listeners
        self._notify_span_complete(span)
        
        # Export if configured
        if self.export_endpoint:
            self._export_span(span)
    
    @contextmanager
    def span(
        self,
        operation_name: str,
        parent_context: Optional[TraceContext] = None,
        tags: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for creating spans
        
        Usage:
            with tracer.span("database_query") as context:
                # Code being traced
                pass
        """
        context = self.start_span(operation_name, parent_context, tags)
        
        try:
            yield context
            self.finish_span(context, status="success")
        except Exception as e:
            self.finish_span(context, status="error", error=str(e))
            raise
    
    def get_current_context(self) -> Optional[TraceContext]:
        """Get current trace context"""
        return getattr(self._current_context, 'context', None)
    
    def set_current_context(self, context: TraceContext):
        """Set current trace context"""
        self._current_context.context = context
    
    def add_tag(self, key: str, value: Any, context: Optional[TraceContext] = None):
        """
        Add tag to current or specified span
        
        Args:
            key: Tag key
            value: Tag value
            context: Trace context (uses current if None)
        """
        if not self.enabled:
            return
        
        context = context or self.get_current_context()
        if not context:
            return
        
        with self._lock:
            if context.span_id in self._spans:
                self._spans[context.span_id].tags[key] = value
    
    def add_log(
        self,
        message: str,
        level: str = "info",
        context: Optional[TraceContext] = None,
        **fields
    ):
        """
        Add log entry to current or specified span
        
        Args:
            message: Log message
            level: Log level
            context: Trace context (uses current if None)
            **fields: Additional log fields
        """
        if not self.enabled:
            return
        
        context = context or self.get_current_context()
        if not context:
            return
        
        log_entry = {
            'timestamp': time.time(),
            'message': message,
            'level': level,
            **fields
        }
        
        with self._lock:
            if context.span_id in self._spans:
                self._spans[context.span_id].logs.append(log_entry)
    
    def inject_context(
        self,
        context: TraceContext,
        carrier: Dict[str, Any],
        format: str = "http"
    ):
        """
        Inject trace context into carrier for propagation
        
        Args:
            context: Trace context to inject
            carrier: Carrier dictionary (e.g., HTTP headers)
            format: Injection format
        """
        if format == "http":
            # W3C Trace Context format
            carrier['traceparent'] = f"00-{context.trace_id}-{context.span_id}-01"
            if context.baggage:
                carrier['baggage'] = ','.join(
                    f"{k}={v}" for k, v in context.baggage.items()
                )
        elif format == "grpc":
            # gRPC metadata format
            carrier['trace-id'] = context.trace_id
            carrier['span-id'] = context.span_id
            if context.parent_span_id:
                carrier['parent-span-id'] = context.parent_span_id
        else:
            # Custom format
            carrier['X-Trace-Id'] = context.trace_id
            carrier['X-Span-Id'] = context.span_id
            if context.parent_span_id:
                carrier['X-Parent-Span-Id'] = context.parent_span_id
    
    def extract_context(
        self,
        carrier: Dict[str, Any],
        format: str = "http"
    ) -> Optional[TraceContext]:
        """
        Extract trace context from carrier
        
        Args:
            carrier: Carrier dictionary (e.g., HTTP headers)
            format: Extraction format
            
        Returns:
            Extracted trace context or None
        """
        try:
            if format == "http":
                # W3C Trace Context format
                traceparent = carrier.get('traceparent')
                if traceparent:
                    parts = traceparent.split('-')
                    if len(parts) >= 4:
                        trace_id = parts[1]
                        span_id = parts[2]
                        
                        # Parse baggage
                        baggage = {}
                        baggage_header = carrier.get('baggage', '')
                        if baggage_header:
                            for item in baggage_header.split(','):
                                if '=' in item:
                                    k, v = item.split('=', 1)
                                    baggage[k.strip()] = v.strip()
                        
                        return TraceContext(
                            trace_id=trace_id,
                            span_id=span_id,
                            baggage=baggage
                        )
            
            elif format == "grpc":
                # gRPC metadata format
                trace_id = carrier.get('trace-id')
                span_id = carrier.get('span-id')
                if trace_id and span_id:
                    return TraceContext(
                        trace_id=trace_id,
                        span_id=span_id,
                        parent_span_id=carrier.get('parent-span-id')
                    )
            
            else:
                # Custom format
                trace_id = carrier.get('X-Trace-Id')
                span_id = carrier.get('X-Span-Id')
                if trace_id and span_id:
                    return TraceContext(
                        trace_id=trace_id,
                        span_id=span_id,
                        parent_span_id=carrier.get('X-Parent-Span-Id')
                    )
        
        except Exception as e:
            logger.error(f"Error extracting trace context: {e}")
        
        return None
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID"""
        return uuid.uuid4().hex
    
    def _generate_span_id(self) -> str:
        """Generate unique span ID"""
        return uuid.uuid4().hex[:16]
    
    def _create_noop_context(self) -> TraceContext:
        """Create no-op context for disabled/unsampled traces"""
        return TraceContext(trace_id="", span_id="")
    
    def _notify_span_complete(self, span: Span):
        """Notify listeners of completed span"""
        for listener in self._span_listeners:
            try:
                listener(span)
            except Exception as e:
                logger.error(f"Error notifying span listener: {e}")
    
    def _export_span(self, span: Span):
        """Export span to configured endpoint"""
        # This would typically send to Jaeger, Zipkin, etc.
        # Simplified implementation for demonstration
        try:
            span_data = {
                'traceId': span.trace_id,
                'spanId': span.span_id,
                'parentSpanId': span.parent_span_id,
                'operationName': span.operation_name,
                'startTime': span.start_time * 1000000,  # Convert to microseconds
                'duration': (span.end_time - span.start_time) * 1000000 if span.end_time else 0,
                'tags': span.tags,
                'logs': span.logs,
                'process': {
                    'serviceName': self.service_name,
                    'tags': {}
                }
            }
            
            # In production, this would send to the export endpoint
            logger.debug(f"Would export span: {json.dumps(span_data)}")
            
        except Exception as e:
            logger.error(f"Error exporting span: {e}")
    
    def add_span_listener(self, listener: Callable):
        """Add listener for completed spans"""
        self._span_listeners.append(listener)
    
    def get_active_spans(self) -> List[Span]:
        """Get currently active spans"""
        with self._lock:
            return list(self._spans.values())
    
    def get_completed_spans(
        self,
        trace_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Span]:
        """
        Get completed spans
        
        Args:
            trace_id: Filter by trace ID
            limit: Maximum spans to return
            
        Returns:
            List of completed spans
        """
        with self._lock:
            spans = self._completed_spans
            
            if trace_id:
                spans = [s for s in spans if s.trace_id == trace_id]
            
            return spans[-limit:]
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """
        Get all spans for a trace
        
        Args:
            trace_id: Trace ID
            
        Returns:
            List of spans in the trace
        """
        with self._lock:
            active = [s for s in self._spans.values() if s.trace_id == trace_id]
            completed = [s for s in self._completed_spans if s.trace_id == trace_id]
            return active + completed
    
    def clear_completed_spans(self):
        """Clear completed spans from memory"""
        with self._lock:
            self._completed_spans.clear()


def trace(operation_name: str, **tags):
    """
    Decorator for tracing functions
    
    Usage:
        @trace("database_query", db="postgres")
        def query_database():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get tracer from somewhere (e.g., global, dependency injection)
            # This is simplified for demonstration
            tracer = getattr(func, '__tracer__', None)
            if not tracer:
                return func(*args, **kwargs)
            
            with tracer.span(operation_name, tags=tags):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator