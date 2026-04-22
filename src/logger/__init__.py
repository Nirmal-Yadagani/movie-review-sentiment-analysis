import os
import sys
import logging
import structlog
from logging.handlers import RotatingFileHandler

# Define paths
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Use a static name for the active log file so RotatingFileHandler works properly.
# The handler will automatically rename older files to pipeline.log.1, pipeline.log.2, etc.
LOG_FILE = os.path.join(LOG_DIR, "sentiment_pipeline.log")

def configure_logger() -> None:
    """
    Configures structured logging for the project. 
    
    Sets up a dual-output logging system:
    1. A colorful console renderer for local development and debugging.
    2. A JSON renderer writing to a rotating file, ideal for production 
       MLOps environments and log parsing.
    """
    
    # 1. Shared Processors: Applied to all logs regardless of output destination
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"), # UTC is often preferred in MLOps: fmt="iso"
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # 2. Standard Logging Handlers
    
    # Console Handler: Formats logs nicely for human readability in terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=processors,
        )
    )

    # File Handler: JSON-friendly format for tracking ML experiments and production monitoring
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=processors,
        )
    )

    # 3. Configure the underlying standard root logger
    logging.basicConfig(
        level=logging.INFO, # Default to INFO level
        handlers=[console_handler, file_handler]
    )

    # 4. Final structlog configuration wrapping the standard library
    structlog.configure(
        processors=processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Initialize configuration when the module is imported
configure_logger()

# Export a configured logger instance to be imported by other modules
logger = structlog.get_logger()