"""Centralized logging configuration"""
import os
import sys
from loguru import logger
from pathlib import Path

def setup_logger(service_name: str, log_level: str = "INFO"):
    """Configure logger with service name and level"""

    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=log_level,
        colorize=True
    )

    # File handler with cross-platform path
    log_dir_env = os.getenv("LOG_DIR")
    log_dir = Path(log_dir_env) if log_dir_env else (Path.cwd() / "logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / f"{service_name}.log",
            rotation="500 MB",
            retention="10 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level
        )
    except Exception:
        # If file logging fails (e.g., read-only filesystem), continue with console only
        pass

    return logger

# Default logger
log = setup_logger("insurance-rag")
