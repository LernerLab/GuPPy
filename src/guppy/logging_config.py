"""Centralized logging configuration for GuPPy.

This module provides a standardized logging setup that writes to platform-appropriate
log directories following OS conventions:
- Windows: %APPDATA%/LernerLab/guppy/Logs/
- macOS: ~/Library/Logs/LernerLab/guppy/
- Linux: ~/.local/state/LernerLab/guppy/log/

Call setup_logging() once at application startup before importing other modules.
Each module should then create its own logger using: logger = logging.getLogger(__name__)
"""

import logging
import os
from pathlib import Path
from platformdirs import user_log_dir


def get_log_file():
    """Get the platform-appropriate log file path.
    
    Returns
    -------
    Path
        Path to the guppy.log file in the platform-appropriate log directory.
    """
    log_dir = Path(user_log_dir("guppy", "LernerLab"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "guppy.log"


def setup_logging(*, level=None, console_output=True):
    """Configure centralized logging for GuPPy.
    
    This should be called once at application startup, before importing other modules.
    
    Parameters
    ----------
    level : int, optional
        Logging level (e.g., logging.DEBUG, logging.INFO). If None, uses
        environment variable GUPPY_LOG_LEVEL or defaults to INFO.
    console_output : bool, optional
        Whether to also output logs to console. Default is True.
    """
    # Determine log level
    if level is None:
        env_level = os.environ.get('GUPPY_LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, env_level, logging.INFO)
    
    # Get log file path
    log_file = get_log_file()
    
    # Configure root logger for guppy
    logger = logging.getLogger("guppy")
    logger.setLevel(level)
    
    # Prevent duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
