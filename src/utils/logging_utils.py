import logging
import sys
import os

class ColorFormatter(logging.Formatter):
    """
    Logging Formatter to add colors to the log level and message.
    Target: Professional, icon-free, easy to distinguish levels.
    """
    # ANSI Color Codes
    grey = "\x1b[38;21m"
    white = "\x1b[37;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    # Format: [Timestamp] [Logger] [Level] Message
    # We colorize the entire line or just the level? 
    # Standard practice: Level is definitely colored. Message often is too for Errors.
    
    format_str = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        # Use a consistent date format
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def setup_logger(name="AVSR", log_file=None, level=logging.INFO):
    """
    Setup a logger with console and optional file handler.
    
    Args:
        name: Logger name (default: AVSR)
        log_file: Path to log file (optional)
        level: Logging level (default: INFO)
        
    Returns:
        logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.hasHandlers():
        return logger

    # 1. Console Handler (with Colors)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)
    
    # 2. File Handler (Clean, no ANSI codes)
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            # File format: No colors, just structure
            file_formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)
        except Exception as e:
            # Fallback if file cannot be created, just print to console (don't crash)
            print(f"Warning: Could not create log file {log_file}: {e}")
        
    # Prevent propagation to root logger to avoid double logging if root is config'd
    logger.propagate = False
        
    return logger
