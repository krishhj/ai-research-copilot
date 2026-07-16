"""
Logging configuration for the AI Research Copilot backend.

This module configures the application's logging system.
"""

import logging

from app.core.constants import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# Log Format
LOG_FORMAT = "%(asctime)s | " "%(levelname)-8s | " "%(name)s | " "%(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure Root Logger
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf--8"),
    ],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
