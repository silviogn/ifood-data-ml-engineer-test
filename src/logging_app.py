import logging
from src.parameters import *

logging.basicConfig(filename=PATH_TO_LOGGING_FILE, level=logging.ERROR)


def log(excep: Exception) -> None:
    logging.error(excep, exc_info=True)

