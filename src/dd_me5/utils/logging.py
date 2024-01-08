import logging
from pathlib import Path
from sys import stdout

from loguru import logger

from dd_me5.schemas.settings import settings

DEFAULT_LOG_FORMAT = (
    "<level>{level: <8}</level> "
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<level>{message}</level>"
)
JSON_LOGS = True


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def make_logger(
    file_path: Path = None, level: str = "INFO", rotation: str = "1 day", retention: str = "1 week", format: str = DEFAULT_LOG_FORMAT
):
    logger.remove()

    logger.add(stdout, enqueue=True, backtrace=True, level=level.upper(), format=format)

    if file_path is not None:
        logger.add(str(file_path), rotation=rotation, retention=retention, enqueue=True, backtrace=True, level=level.upper(), format=format)

    # disable handlers for specific loggers
    # to redirect their output to the default logger
    loggers = (
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if name.startswith("uvicorn.") or name.startswith("gunicorn.") or name.startswith("fastapi.")
    )
    for uvicorn_logger in loggers:
        uvicorn_logger.handlers = []

    # change handler for default uvicorn logger
    intercept_handler = InterceptHandler()
    for name in ["uvicorn", "gunicorn", "fastapi"]:
        logging.getLogger(name).handlers = [intercept_handler]

    # set logs output, level and format
    logger.configure(handlers=[{"sink": stdout, "level": logging.DEBUG, "format": format}])
    return logger


logger = make_logger(
    file_path=settings.log_file,
    level=settings.log_level,
)
