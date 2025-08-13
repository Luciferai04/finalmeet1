"""
Production logging configuration with structured logging
"""
import logging
import sys
import os
from typing import Optional
import structlog
from structlog.stdlib import LoggerFactory
from app.core.production_config import get_config

def setup_logging() -> None:
    """Configure structured logging for production"""
    config = get_config()
    
    # Determine log level
    log_level = getattr(logging, config.log_level.value)
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Set up structlog processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Add request ID if available
        structlog.processors.add_log_level,
    ]
    
    if config.log_format == "json":
        # JSON logging for production
        shared_processors.append(
            structlog.processors.JSONRenderer()
        )
    else:
        # Human-readable logging for development
        shared_processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure log file if specified
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(log_level)
        logging.getLogger().addHandler(file_handler)

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance"""
    return structlog.get_logger(name)

# Setup logging when module is imported
setup_logging()
