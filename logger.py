"""
logger.py
---------
Production-grade logging configuration with contextual formatting,
timestamps, and log-level controls.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "jane",
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Get or configure a logger with standardized formatting.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = level or logging.INFO
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


# Default application-wide logger
app_logger = setup_logger("jane")
