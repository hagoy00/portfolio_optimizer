import logging
from logging.handlers import RotatingFileHandler
import os
import sys

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# =========================================================
# GLOBAL LOGGER CACHE (prevents duplicate handlers)
# =========================================================
_LOGGER_CACHE = {}


# =========================================================
# CREATE LOGGER
# =========================================================
def get_logger(name="portfolio_optimizer", level=logging.INFO):
    """
    Institutional-grade logger:
    - Rotating file logs
    - Streamlit-safe console logs
    - No duplicate handlers
    - Works across all utils/modules
    """

    # Return cached logger if already created
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent double logging

    # -----------------------------------------------------
    # Prevent duplicate handlers
    # -----------------------------------------------------
    if len(logger.handlers) == 0:

        # ============================
        # Console Handler
        # ============================
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        # ============================
        # Rotating File Handler
        # ============================
        fh = RotatingFileHandler(
            os.path.join(LOG_DIR, "app.log"),
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8"
        )
        fh.setLevel(level)
        fh_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    # Cache logger
    _LOGGER_CACHE[name] = logger
    return logger
