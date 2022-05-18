import logging


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    white = "\x1b[37;20m"
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    cyan = "\x1b[36;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # (%(funcName)s@%(filename)s:%(lineno)d)
    format = "[%(asctime)s]-[%(levelname)s]: %(message)s"

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

    def get_logger(self, level=logging.INFO):
        self.logger = logging.getLogger('fpgaHART')
        self.logger.setLevel(level)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        ch.setFormatter(CustomFormatter())

        self.logger.addHandler(ch)

        self.logger.propagate = False
        return self.logger


_logger = CustomFormatter().get_logger()
