import logging
import os


def get_logger(name: str = "memory_sam") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_str = os.environ.get("MEMORY_SAM_LOG", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


