"""Structured logging configuration for IndoorAI."""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "IndoorAI",
    level: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """Configure and return a standardized logger instance.

    Args:
        name: Name of the logger or calling module.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Optional custom format string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        resolved_level_name = level or os.getenv("LOG_LEVEL", "INFO")
        resolved_level = getattr(logging, resolved_level_name.upper(), logging.INFO)
        logger.setLevel(resolved_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(resolved_level)

        default_format = (
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
        )
        formatter = logging.Formatter(log_format or default_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger

