import logging
import os
from datetime import datetime

def get_logger(name: str = "pipeline", log_dir: str = "logs", level=logging.DEBUG):
    os.makedirs(log_dir, exist_ok=True)

    # Create a timestamped log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Define log format
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create formatter
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers when calling get_logger() multiple times
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
