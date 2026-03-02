"""
Logging utilities.
"""

from __future__ import annotations

import logging
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Log level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    ...
