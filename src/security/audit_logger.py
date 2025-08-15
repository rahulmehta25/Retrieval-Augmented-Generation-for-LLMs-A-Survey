"""
Audit Logging Module

Implements comprehensive security audit logging to detect:
- Unauthorized access attempts (OWASP A01:2021)
- Security misconfiguration (OWASP A05:2021)
- Identification and authentication failures (OWASP A07:2021)
- Security logging and monitoring failures (OWASP A09:2021)
"""

import logging
import json
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List
from enum import Enum
from pathlib import Path
import hashlib
import threading
from collections import deque
import asyncio

# Configure audit logger
audit_logger = logging.getLogger('security.audit')
audit_logger.setLevel(logging.INFO)


class SecurityEvent(Enum):
    """Types of security events to audit"""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_CREATED = "token_created"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_REVOKED = "token_revoked"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    
    # Input validation events
    VALIDATION_FAILURE = "validation_failure"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    
    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    IP_BLACKLISTED = "ip_blacklisted"
    
    # Data events
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    FILE_UPLOAD = "file_upload"
    FILE_DOWNLOAD = "file_download"
    
    # System events
    CONFIGURATION_CHANGE = "configuration_change"
    SERVICE_START = "service_start"
    SERVICE_STOP = "service_stop"
    ERROR = "error"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class AuditLogger:
    """
    Centralized audit logging system for security events
    
    Implements OWASP logging best practices
    """
    
    def __init__(self,
                 log_file: Optional[str] = None,
                 max_entries: int = 10000,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_alerts: bool = True,
                 alert_threshold: int = 5):
        """
        Initialize audit logger
        
        Args:
            log_file: Path to audit log file
            max_entries: Maximum entries to keep in memory
            enable_console: Log to console
            enable_file: Log to file
            enable_alerts: Enable security alerts
            alert_threshold: Number of failures before alert
        """
        self.log_file = log_file or "security_audit.log"
        self.max_entries = max_entries
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_alerts = enable_alerts
        self.alert_threshold = alert_threshold
        
        # In-memory audit trail
        self.audit_trail = deque(maxlen=max_entries)
        
        # Failure tracking for alerts
        self.failure_tracker = {}
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
        
        # Setup file handler if enabled
        if self.enable_file:
            self._setup_file_handler()
        
        # Setup console handler if enabled
        if self.enable_console:
            self._setup_console_handler()
    
    def _setup_file_handler(self):
        """Setup file handler for audit logs"""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # JSON formatter for structured logging
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        audit_logger.addHandler(file_handler)
    
    def _setup_console_handler(self):
        """Setup console handler for audit logs"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY AUDIT - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        audit_logger.addHandler(console_handler)
    
    def log_event(self,
                  event_type: SecurityEvent,
                  user_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  resource: Optional[str] = None,
                  action: Optional[str] = None,
                  result: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  severity: str = "INFO"):
        """
        Log a security event
        
        Args:
            event_type: Type of security event
            user_id: User identifier
            ip_address: Client IP address
            resource: Resource being accessed
            action: Action performed
            result: Result of the action
            details: Additional event details
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
        """
        with self._lock:
            # Create audit entry
            entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'event_type': event_type.value,
                'severity': severity,
                'user_id': user_id,
                'ip_address': ip_address,
                'resource': resource,
                'action': action,
                'result': result,
                'details': details or {},
                'session_id': self._get_session_id(user_id, ip_address)
            }
            
            # Add entry to in-memory trail
            self.audit_trail.append(entry)
            
            # Log to file/console
            self._write_log(entry, severity)
            
            # Check for security alerts
            if self.enable_alerts:
                self._check_alerts(event_type, user_id, ip_address)
            
            # Track failures
            if event_type in [
                SecurityEvent.LOGIN_FAILURE,
                SecurityEvent.ACCESS_DENIED,
                SecurityEvent.VALIDATION_FAILURE
            ]:
                self._track_failure(user_id or ip_address, event_type)
    
    def _write_log(self, entry: Dict[str, Any], severity: str):
        """Write log entry to configured handlers"""
        log_message = json.dumps(entry, default=str)
        
        if severity == "CRITICAL":
            audit_logger.critical(log_message)
        elif severity == "ERROR":
            audit_logger.error(log_message)
        elif severity == "WARNING":
            audit_logger.warning(log_message)
        else:
            audit_logger.info(log_message)
    
    def _get_session_id(self, user_id: Optional[str], ip_address: Optional[str]) -> str:
        """Generate session identifier for correlation"""
        data = f"{user_id or 'anonymous'}:{ip_address or 'unknown'}:{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _track_failure(self, identifier: str, event_type: SecurityEvent):
        """Track failures for alert generation"""
        if identifier not in self.failure_tracker:
            self.failure_tracker[identifier] = []
        
        self.failure_tracker[identifier].append({
            'timestamp': time.time(),
            'event_type': event_type.value
        })
        
        # Clean old failures (older than 1 hour)
        current_time = time.time()
        self.failure_tracker[identifier] = [
            f for f in self.failure_tracker[identifier]
            if current_time - f['timestamp'] < 3600
        ]
    
    def _check_alerts(self, event_type: SecurityEvent, user_id: Optional[str], ip_address: Optional[str]):
        """Check if security alert should be triggered"""
        identifier = user_id or ip_address
        
        if not identifier:
            return
        
        # Check for brute force attempts
        if event_type == SecurityEvent.LOGIN_FAILURE:
            failures = self.failure_tracker.get(identifier, [])
            recent_failures = [
                f for f in failures
                if time.time() - f['timestamp'] < 300  # Last 5 minutes
            ]
            
            if len(recent_failures) >= self.alert_threshold:
                self._trigger_alert(
                    "BRUTE_FORCE_ATTEMPT",
                    f"Multiple login failures detected for {identifier}",
                    {
                        'identifier': identifier,
                        'failure_count': len(recent_failures),
                        'event_type': event_type.value
                    }
                )
        
        # Check for suspicious patterns
        elif event_type in [
            SecurityEvent.SQL_INJECTION_ATTEMPT,
            SecurityEvent.XSS_ATTEMPT,
            SecurityEvent.PATH_TRAVERSAL_ATTEMPT
        ]:
            self._trigger_alert(
                "ATTACK_ATTEMPT",
                f"Potential attack detected: {event_type.value}",
                {
                    'identifier': identifier,
                    'event_type': event_type.value
                }
            )
    
    def _trigger_alert(self, alert_type: str, message: str, details: Dict[str, Any]):
        """Trigger security alert"""
        alert = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'alert_type': alert_type,
            'message': message,
            'details': details
        }
        
        # Log critical alert
        audit_logger.critical(f"SECURITY ALERT: {json.dumps(alert, default=str)}")
        
        # Here you would integrate with alerting systems like:
        # - Send email/SMS
        # - Post to Slack/Teams
        # - Create incident ticket
        # - Trigger automated response
    
    def get_audit_trail(self,
                       event_type: Optional[SecurityEvent] = None,
                       user_id: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query audit trail
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum entries to return
            
        Returns:
            List of audit entries
        """
        with self._lock:
            filtered_entries = []
            
            for entry in reversed(self.audit_trail):
                # Apply filters
                if event_type and entry['event_type'] != event_type.value:
                    continue
                
                if user_id and entry['user_id'] != user_id:
                    continue
                
                if start_time:
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if entry_time < start_time:
                        continue
                
                if end_time:
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if entry_time > end_time:
                        continue
                
                filtered_entries.append(entry)
                
                if len(filtered_entries) >= limit:
                    break
            
            return filtered_entries
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """
        Get security metrics and statistics
        
        Returns:
            Dictionary of security metrics
        """
        with self._lock:
            metrics = {
                'total_events': len(self.audit_trail),
                'event_types': {},
                'severity_counts': {},
                'recent_failures': {},
                'active_sessions': set()
            }
            
            # Count events by type and severity
            for entry in self.audit_trail:
                event_type = entry['event_type']
                severity = entry['severity']
                
                metrics['event_types'][event_type] = metrics['event_types'].get(event_type, 0) + 1
                metrics['severity_counts'][severity] = metrics['severity_counts'].get(severity, 0) + 1
                
                if entry['session_id']:
                    metrics['active_sessions'].add(entry['session_id'])
            
            # Count recent failures
            for identifier, failures in self.failure_tracker.items():
                recent = len([
                    f for f in failures
                    if time.time() - f['timestamp'] < 3600
                ])
                if recent > 0:
                    metrics['recent_failures'][identifier] = recent
            
            metrics['active_sessions'] = len(metrics['active_sessions'])
            
            return metrics
    
    def export_audit_log(self, output_file: str, format: str = 'json'):
        """
        Export audit log to file
        
        Args:
            output_file: Output file path
            format: Export format (json, csv)
        """
        with self._lock:
            if format == 'json':
                with open(output_file, 'w') as f:
                    json.dump(list(self.audit_trail), f, indent=2, default=str)
            
            elif format == 'csv':
                import csv
                
                if not self.audit_trail:
                    return
                
                with open(output_file, 'w', newline='') as f:
                    fieldnames = self.audit_trail[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for entry in self.audit_trail:
                        # Convert complex types to strings
                        row = {
                            k: json.dumps(v) if isinstance(v, (dict, list)) else v
                            for k, v in entry.items()
                        }
                        writer.writerow(row)
    
    def clear_old_entries(self, days: int = 30):
        """
        Clear audit entries older than specified days
        
        Args:
            days: Number of days to retain
        """
        with self._lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            
            new_trail = deque(maxlen=self.max_entries)
            for entry in self.audit_trail:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time > cutoff_time:
                    new_trail.append(entry)
            
            self.audit_trail = new_trail


class AsyncAuditLogger(AuditLogger):
    """
    Asynchronous version of audit logger for high-performance applications
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue = asyncio.Queue()
        self._processing = False
    
    async def log_event_async(self, *args, **kwargs):
        """Asynchronously log security event"""
        await self._queue.put((args, kwargs))
        
        if not self._processing:
            asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Process queued audit events"""
        self._processing = True
        
        while not self._queue.empty():
            try:
                args, kwargs = await self._queue.get()
                super().log_event(*args, **kwargs)
            except Exception as e:
                audit_logger.error(f"Error processing audit event: {e}")
        
        self._processing = False