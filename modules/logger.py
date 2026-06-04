import os
import logging
from logging.handlers import RotatingFileHandler

def get_logger(name: str) -> logging.Logger:
    """
    Returns an enterprise-ready logger with both console and rotating file handlers.
    Ensures logs are structured, traceable, and do not crash the server by filling the disk.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s : %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        os.makedirs('logs', exist_ok=True)
        file_handler = RotatingFileHandler(
            'logs/supply_chain_app.log', 
            maxBytes=5 * 1024 * 1024, 
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG) 
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger