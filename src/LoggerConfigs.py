import logging
import sys

formatted_string = "%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s"
formatter = logging.Formatter(formatted_string)
basic_configs = {
    "level": logging.DEBUG,
    "format": formatted_string,
    "datefmt": '%Y-%m-%d %H:%M:%S',
}



def initialize_logger():
    logging.basicConfig(**basic_configs)

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

    return logger
