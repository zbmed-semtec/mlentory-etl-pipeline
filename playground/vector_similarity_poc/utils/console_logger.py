#!/usr/bin/env python3
"""
Simple Console Logger

This module provides a simple logging utility that captures console output
and saves it to log files while still printing to the console.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Optional


class ConsoleLogger:
    """
    A simple logger that captures console output and saves it to files.
    """
    
    def __init__(self, log_dir: str = "logs", log_prefix: str = "search_cli"):
        """
        Initialize the console logger.
        
        Args:
            log_dir: Directory to save log files (default: "logs")
            log_prefix: Prefix for log file names (default: "search_cli")
        """
        self.log_dir = log_dir
        self.log_prefix = log_prefix
        self.logger = None
        self.original_stdout = None
        self.original_stderr = None
        self.log_file_path = None
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup the logging configuration."""
        # Generate log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.log_prefix}_{timestamp}.log"
        self.log_file_path = os.path.join(self.log_dir, log_filename)
        
        # Create logger
        self.logger = logging.getLogger(f"{self.log_prefix}_{timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Log session start
        self.logger.info("=" * 60)
        self.logger.info(f"Search CLI Session Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)
    
    def start_logging(self):
        """Start capturing console output."""
        if self.logger is None:
            self._setup_logging()
        
        # Store original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create custom stdout that both prints and logs
        sys.stdout = self._LoggingWriter(self.original_stdout, self.logger, 'INFO')
        sys.stderr = self._LoggingWriter(self.original_stderr, self.logger, 'ERROR')
        
        print(f"📝 Logging started. Log file: {self.log_file_path}")
    
    def stop_logging(self):
        """Stop capturing console output."""
        if self.original_stdout is not None:
            sys.stdout = self.original_stdout
        if self.original_stderr is not None:
            sys.stderr = self.original_stderr
        
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info(f"Search CLI Session Ended - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 60)
            
            # Close all handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        
        print(f"📝 Logging stopped. Log saved to: {self.log_file_path}")
    
    def log_message(self, message: str, level: str = 'INFO'):
        """Manually log a message."""
        if self.logger:
            if level.upper() == 'ERROR':
                self.logger.error(message)
            else:
                self.logger.info(message)
    
    def get_log_file_path(self) -> Optional[str]:
        """Get the current log file path."""
        return self.log_file_path
    
    class _LoggingWriter:
        """Custom writer that both prints to console and logs to file."""
        
        def __init__(self, original_stream, logger, level):
            self.original_stream = original_stream
            self.logger = logger
            self.level = level
        
        def write(self, message):
            # Write to original stream (console)
            self.original_stream.write(message)
            self.original_stream.flush()
            
            # Log to file (strip newlines for cleaner logs)
            if message.strip():
                if self.level == 'ERROR':
                    self.logger.error(message.strip())
                else:
                    self.logger.info(message.strip())
        
        def flush(self):
            self.original_stream.flush()
        
        def __getattr__(self, name):
            return getattr(self.original_stream, name)


# Global logger instance
_global_logger = None


def start_console_logging(log_dir: str = "logs", log_prefix: str = "search_cli") -> ConsoleLogger:
    """
    Start console logging with the given parameters.
    
    Args:
        log_dir: Directory to save log files
        log_prefix: Prefix for log file names
        
    Returns:
        ConsoleLogger instance
    """
    global _global_logger
    _global_logger = ConsoleLogger(log_dir, log_prefix)
    _global_logger.start_logging()
    return _global_logger


def stop_console_logging():
    """Stop console logging."""
    global _global_logger
    if _global_logger:
        _global_logger.stop_logging()
        _global_logger = None


def get_logger() -> Optional[ConsoleLogger]:
    """Get the current logger instance."""
    return _global_logger


def log_message(message: str, level: str = 'INFO'):
    """Log a message using the current logger."""
    if _global_logger:
        _global_logger.log_message(message, level)


if __name__ == "__main__":
    # Test the logger
    logger = start_console_logging("test_logs", "test")
    
    print("This is a test message")
    print("Another test message")
    print("Error message", file=sys.stderr)
    
    log_message("This is a manual log message")
    
    stop_console_logging()
    print("Logging stopped, this won't be logged")
