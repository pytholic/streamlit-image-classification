from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
PROJECT_DIR = Path(BASE_DIR, "streamlit_image_classification")
CONFIG_DIR = Path(PROJECT_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
LOGS_DIR = Path(PROJECT_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


import logging
import logging.config
import sys

from rich.logging import RichHandler

# Logging configuration
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
    },
    "root": {
        "handlers": ["console"],
        "level": logging.INFO,
        "propagate": True,
    },
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)  # pretty formatting

if __name__ == "__main__":
    # Sample messages (note that we use configured `logger` now)
    logger.debug("Used for debugging your code.")
    logger.info("Informative messages from your code.")
    logger.warning("Everything works but there is something to be aware of.")
    logger.error("There's been a mistake with the process.")
    logger.critical("There is something terribly wrong and process may terminate.")

    excep = TypeError("Only integers are allowed")
    logger.error(f"got an exception: {excep}")
