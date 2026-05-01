import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name="portfolio_optimizer"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        ch.setFormatter(ch_formatter)

        # File handler (rotating)
        fh = RotatingFileHandler(
            f"{LOG_DIR}/app.log",
            maxBytes=2_000_000,
            backupCount=5
        )
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        fh.setFormatter(fh_formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
