"""
Logging configuration and utilities for LLM Router Platform
Production-grade logging with structured output, correlation IDs, and multiple outputs
"""

import logging
import logging.handlers
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextvars import ContextVar
import uuid

import structlog
from pythonjsonlogger import jsonlogger


# Context variables for request tracking
request_id_context: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
session_id_context: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional context"""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ):
        """Add custom fields to log record"""
        super().add_fields(log_record, record, message_dict)

        # Add timestamp
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add level
        log_record["level"] = record.levelname

        # Add logger name
        log_record["logger"] = record.name

        # Add context from context variables
        request_id = request_id_context.get()
        if request_id:
            log_record["request_id"] = request_id

        user_id = user_id_context.get()
        if user_id:
            log_record["user_id"] = user_id

        session_id = session_id_context.get()
        if session_id:
            log_record["session_id"] = session_id

        # Add file and line info
        log_record["file"] = f"{record.filename}:{record.lineno}"
        log_record["function"] = record.funcName

        # Add process and thread info
        log_record["process_id"] = record.process
        log_record["thread_id"] = record.thread

        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for development"""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        # Add colors
        level_color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{level_color}{self.BOLD}{record.levelname}{self.RESET}"

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

        # Format message
        formatted = (
            f"{timestamp} {record.levelname:<8} {record.name:<20} {record.getMessage()}"
        )

        # Add context if available
        request_id = request_id_context.get()
        if request_id:
            formatted += f" [req:{request_id[:8]}]"

        user_id = user_id_context.get()
        if user_id:
            formatted += f" [user:{user_id}]"

        # Add exception info
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


class MetricsHandler(logging.Handler):
    """Log handler that extracts metrics from log records"""

    def __init__(self):
        super().__init__()
        self.metrics = {
            "log_counts": {},
            "error_counts": {},
            "response_times": [],
            "last_errors": [],
        }

    def emit(self, record: logging.LogRecord):
        """Process log record for metrics"""
        try:
            # Count logs by level
            level = record.levelname
            self.metrics["log_counts"][level] = (
                self.metrics["log_counts"].get(level, 0) + 1
            )

            # Track errors
            if level in ["ERROR", "CRITICAL"]:
                logger_name = record.name
                self.metrics["error_counts"][logger_name] = (
                    self.metrics["error_counts"].get(logger_name, 0) + 1
                )

                # Store recent errors
                error_info = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "logger": logger_name,
                    "message": record.getMessage(),
                    "level": level,
                }

                if hasattr(record, "request_id"):
                    error_info["request_id"] = record.request_id

                self.metrics["last_errors"].append(error_info)

                # Keep only last 100 errors
                if len(self.metrics["last_errors"]) > 100:
                    self.metrics["last_errors"] = self.metrics["last_errors"][-100:]

            # Extract response times if present
            if hasattr(record, "response_time_ms"):
                self.metrics["response_times"].append(record.response_time_ms)

                # Keep only recent response times
                if len(self.metrics["response_times"]) > 1000:
                    self.metrics["response_times"] = self.metrics["response_times"][
                        -1000:
                    ]

        except Exception:
            # Don't let metrics handling break logging
            pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        metrics = self.metrics.copy()

        # Calculate response time statistics
        if self.metrics["response_times"]:
            response_times = self.metrics["response_times"]
            metrics["response_time_stats"] = {
                "count": len(response_times),
                "average": sum(response_times) / len(response_times),
                "min": min(response_times),
                "max": max(response_times),
            }
        else:
            metrics["response_time_stats"] = None

        return metrics


class RequestContextFilter(logging.Filter):
    """Filter to add request context to log records"""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context variables to record"""
        # Add context from context variables
        record.request_id = request_id_context.get()
        record.user_id = user_id_context.get()
        record.session_id = session_id_context.get()

        return True


class PerformanceLogger:
    """Context manager for performance logging"""

    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.context = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(
            f"Starting {self.operation}",
            extra={"operation": self.operation, "event": "start", **self.context},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000

        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                extra={
                    "operation": self.operation,
                    "event": "complete",
                    "duration_ms": duration_ms,
                    "response_time_ms": duration_ms,  # For metrics handler
                    **self.context,
                },
            )
        else:
            self.logger.error(
                f"Failed {self.operation}",
                extra={
                    "operation": self.operation,
                    "event": "error",
                    "duration_ms": duration_ms,
                    "error_type": exc_type.__name__ if exc_type else None,
                    "error_message": str(exc_val) if exc_val else None,
                    **self.context,
                },
                exc_info=True,
            )


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter with structured logging support"""

    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and kwargs"""
        # Merge extra context
        if "extra" in kwargs:
            kwargs["extra"].update(self.extra)
        else:
            kwargs["extra"] = self.extra.copy()

        return msg, kwargs

    def with_context(self, **context) -> "LoggerAdapter":
        """Create new adapter with additional context"""
        new_extra = self.extra.copy()
        new_extra.update(context)
        return LoggerAdapter(self.logger, new_extra)

    def time_operation(self, operation: str, **context) -> PerformanceLogger:
        """Create performance logger for operation"""
        full_context = self.extra.copy()
        full_context.update(context)
        return PerformanceLogger(self, operation, **full_context)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    structured_logs: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
) -> Dict[str, Any]:
    """
    Setup logging configuration for the application

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        structured_logs: Whether to use structured JSON logs
        max_bytes: Maximum bytes per log file
        backup_count: Number of backup files to keep

    Returns:
        Dictionary with logger configuration info
    """

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create request context filter
    context_filter = RequestContextFilter()

    # Setup console logging
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.addFilter(context_filter)

        if structured_logs:
            console_formatter = StructuredFormatter(
                fmt="%(timestamp)s %(level)s %(logger)s %(message)s"
            )
        else:
            console_formatter = ColoredConsoleFormatter()

        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Setup file logging
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.addFilter(context_filter)

        # Always use structured format for file logs
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Add metrics handler
    metrics_handler = MetricsHandler()
    metrics_handler.setLevel(logging.INFO)
    root_logger.addHandler(metrics_handler)

    # Configure third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("kafka").setLevel(logging.WARNING)
    logging.getLogger("clickhouse_connect").setLevel(logging.WARNING)

    # Configure structlog if using structured logging
    if structured_logs:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    config_info = {
        "level": log_level,
        "numeric_level": numeric_level,
        "file_logging": log_file is not None,
        "console_logging": console_output,
        "structured_logs": structured_logs,
        "handlers_count": len(root_logger.handlers),
        "metrics_handler": metrics_handler,
    }

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured", extra={"config": config_info, "event": "logging_setup"}
    )

    return config_info


def get_logger(name: str, **context) -> LoggerAdapter:
    """
    Get a logger with optional context

    Args:
        name: Logger name (usually __name__)
        **context: Additional context to include in all log messages

    Returns:
        LoggerAdapter with context
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)


def set_request_context(
    request_id: str = None, user_id: str = None, session_id: str = None
):
    """
    Set request context for logging

    Args:
        request_id: Request identifier
        user_id: User identifier
        session_id: Session identifier
    """
    if request_id:
        request_id_context.set(request_id)
    if user_id:
        user_id_context.set(user_id)
    if session_id:
        session_id_context.set(session_id)


def clear_request_context():
    """Clear all request context"""
    request_id_context.set(None)
    user_id_context.set(None)
    session_id_context.set(None)


def generate_request_id() -> str:
    """Generate a new request ID"""
    return str(uuid.uuid4())


class RequestContextManager:
    """Context manager for request-scoped logging context"""

    def __init__(
        self, request_id: str = None, user_id: str = None, session_id: str = None
    ):
        self.request_id = request_id or generate_request_id()
        self.user_id = user_id
        self.session_id = session_id
        self.previous_context = {}

    def __enter__(self):
        # Store previous context
        self.previous_context = {
            "request_id": request_id_context.get(),
            "user_id": user_id_context.get(),
            "session_id": session_id_context.get(),
        }

        # Set new context
        set_request_context(self.request_id, self.user_id, self.session_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        request_id_context.set(self.previous_context["request_id"])
        user_id_context.set(self.previous_context["user_id"])
        session_id_context.set(self.previous_context["session_id"])


class SecurityLogger:
    """Security-focused logger for sensitive operations"""

    def __init__(self, logger_name: str = "security"):
        self.logger = get_logger(logger_name)

    def log_authentication_attempt(
        self, user_id: str, success: bool, source_ip: str = None, **context
    ):
        """Log authentication attempt"""
        self.logger.info(
            "Authentication attempt",
            extra={
                "event_type": "authentication",
                "user_id": user_id,
                "success": success,
                "source_ip": source_ip,
                "security_event": True,
                **context,
            },
        )

    def log_authorization_failure(
        self, user_id: str, resource: str, action: str, **context
    ):
        """Log authorization failure"""
        self.logger.warning(
            "Authorization denied",
            extra={
                "event_type": "authorization_denied",
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "security_event": True,
                **context,
            },
        )

    def log_rate_limit_exceeded(self, user_id: str, endpoint: str, **context):
        """Log rate limit exceeded"""
        self.logger.warning(
            "Rate limit exceeded",
            extra={
                "event_type": "rate_limit_exceeded",
                "user_id": user_id,
                "endpoint": endpoint,
                "security_event": True,
                **context,
            },
        )

    def log_suspicious_activity(
        self, user_id: str, activity: str, severity: str = "medium", **context
    ):
        """Log suspicious activity"""
        log_method = getattr(self.logger, severity.lower(), self.logger.warning)
        log_method(
            "Suspicious activity detected",
            extra={
                "event_type": "suspicious_activity",
                "user_id": user_id,
                "activity": activity,
                "severity": severity,
                "security_event": True,
                **context,
            },
        )

    def log_data_access(self, user_id: str, data_type: str, operation: str, **context):
        """Log sensitive data access"""
        self.logger.info(
            "Sensitive data access",
            extra={
                "event_type": "data_access",
                "user_id": user_id,
                "data_type": data_type,
                "operation": operation,
                "security_event": True,
                **context,
            },
        )


class AuditLogger:
    """Audit logger for compliance and tracking"""

    def __init__(self, logger_name: str = "audit"):
        self.logger = get_logger(logger_name)

    def log_model_request(
        self,
        user_id: str,
        model_name: str,
        query_type: str,
        token_count: int,
        **context,
    ):
        """Log model inference request"""
        self.logger.info(
            "Model request",
            extra={
                "event_type": "model_request",
                "user_id": user_id,
                "model_name": model_name,
                "query_type": query_type,
                "token_count": token_count,
                "audit_event": True,
                **context,
            },
        )

    def log_configuration_change(
        self, user_id: str, component: str, changes: Dict[str, Any], **context
    ):
        """Log configuration changes"""
        self.logger.info(
            "Configuration changed",
            extra={
                "event_type": "configuration_change",
                "user_id": user_id,
                "component": component,
                "changes": changes,
                "audit_event": True,
                **context,
            },
        )

    def log_system_event(self, event_type: str, details: Dict[str, Any], **context):
        """Log system events"""
        self.logger.info(
            "System event",
            extra={
                "event_type": event_type,
                "details": details,
                "audit_event": True,
                **context,
            },
        )

    def log_cost_event(
        self, user_id: str, model_name: str, cost: float, tokens: int, **context
    ):
        """Log cost-related events"""
        self.logger.info(
            "Cost event",
            extra={
                "event_type": "cost_tracking",
                "user_id": user_id,
                "model_name": model_name,
                "cost_usd": cost,
                "token_count": tokens,
                "audit_event": True,
                **context,
            },
        )


class PerformanceTracker:
    """Performance tracking utilities"""

    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)

    def track_request_latency(self, endpoint: str, latency_ms: float, **context):
        """Track request latency"""
        self.logger.info(
            "Request latency",
            extra={
                "metric_type": "latency",
                "endpoint": endpoint,
                "latency_ms": latency_ms,
                "response_time_ms": latency_ms,  # For metrics handler
                **context,
            },
        )

    def track_model_performance(
        self, model_name: str, latency_ms: float, tokens_per_second: float, **context
    ):
        """Track model performance"""
        self.logger.info(
            "Model performance",
            extra={
                "metric_type": "model_performance",
                "model_name": model_name,
                "latency_ms": latency_ms,
                "tokens_per_second": tokens_per_second,
                "response_time_ms": latency_ms,  # For metrics handler
                **context,
            },
        )

    def track_resource_usage(self, resource_type: str, usage_percent: float, **context):
        """Track resource usage"""
        level = logging.WARNING if usage_percent > 80 else logging.INFO
        self.logger.log(
            level,
            "Resource usage",
            extra={
                "metric_type": "resource_usage",
                "resource_type": resource_type,
                "usage_percent": usage_percent,
                **context,
            },
        )


class ErrorTracker:
    """Enhanced error tracking and reporting"""

    def __init__(self, logger_name: str = "errors"):
        self.logger = get_logger(logger_name)

    def track_error(self, error: Exception, context: Dict[str, Any] = None, **kwargs):
        """Track error with enhanced context"""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_module": error.__class__.__module__,
            **(context or {}),
            **kwargs,
        }

        # Add traceback for non-user errors
        if not isinstance(error, (ValueError, TypeError)):
            error_context["traceback"] = traceback.format_exc()

        self.logger.error("Error tracked", extra=error_context, exc_info=True)

    def track_model_error(self, model_name: str, error: Exception, **context):
        """Track model-specific errors"""
        self.track_error(
            error,
            {"error_category": "model_error", "model_name": model_name, **context},
        )

    def track_api_error(
        self, endpoint: str, status_code: int, error: Exception, **context
    ):
        """Track API errors"""
        self.track_error(
            error,
            {
                "error_category": "api_error",
                "endpoint": endpoint,
                "status_code": status_code,
                **context,
            },
        )

    def track_pipeline_error(self, pipeline_stage: str, error: Exception, **context):
        """Track pipeline errors"""
        self.track_error(
            error,
            {
                "error_category": "pipeline_error",
                "pipeline_stage": pipeline_stage,
                **context,
            },
        )


def get_log_metrics() -> Dict[str, Any]:
    """Get current logging metrics"""
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if isinstance(handler, MetricsHandler):
            return handler.get_metrics()

    return {}


class LoggerFactory:
    """Factory for creating specialized loggers"""

    @staticmethod
    def get_component_logger(component: str, **context) -> LoggerAdapter:
        """Get logger for specific component"""
        return get_logger(f"llm_router.{component}", component=component, **context)

    @staticmethod
    def get_model_logger(model_name: str, **context) -> LoggerAdapter:
        """Get logger for specific model"""
        return get_logger(
            f"llm_router.models.{model_name}", model=model_name, **context
        )

    @staticmethod
    def get_user_logger(user_id: str, **context) -> LoggerAdapter:
        """Get logger with user context"""
        return get_logger("llm_router.user_activity", user_id=user_id, **context)

    @staticmethod
    def get_api_logger(endpoint: str, **context) -> LoggerAdapter:
        """Get logger for API endpoints"""
        return get_logger(f"llm_router.api.{endpoint}", endpoint=endpoint, **context)


# Pre-configured specialized loggers
security_logger = SecurityLogger()
audit_logger = AuditLogger()
performance_tracker = PerformanceTracker()
error_tracker = ErrorTracker()


# Convenience functions
def log_request_start(logger: LoggerAdapter, method: str, path: str, **context):
    """Log request start"""
    logger.info(
        f"{method} {path} - Request started",
        extra={
            "event": "request_start",
            "http_method": method,
            "path": path,
            **context,
        },
    )


def log_request_end(
    logger: LoggerAdapter,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    **context,
):
    """Log request end"""
    level = logging.WARNING if status_code >= 400 else logging.INFO
    logger.log(
        level,
        f"{method} {path} - Request completed",
        extra={
            "event": "request_end",
            "http_method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "response_time_ms": duration_ms,  # For metrics handler
            **context,
        },
    )


def log_model_inference(
    logger: LoggerAdapter,
    model_name: str,
    tokens: int,
    latency_ms: float,
    cost: float,
    **context,
):
    """Log model inference"""
    logger.info(
        "Model inference completed",
        extra={
            "event": "model_inference",
            "model_name": model_name,
            "token_count": tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "response_time_ms": latency_ms,  # For metrics handler
            **context,
        },
    )


# Export main components
__all__ = [
    "setup_logging",
    "get_logger",
    "set_request_context",
    "clear_request_context",
    "generate_request_id",
    "RequestContextManager",
    "LoggerAdapter",
    "PerformanceLogger",
    "SecurityLogger",
    "AuditLogger",
    "PerformanceTracker",
    "ErrorTracker",
    "LoggerFactory",
    "get_log_metrics",
    "security_logger",
    "audit_logger",
    "performance_tracker",
    "error_tracker",
    "log_request_start",
    "log_request_end",
    "log_model_inference",
]
