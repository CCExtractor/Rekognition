import sys
import logging
from logging import DEBUG, INFO, ERROR

class RekogntionLogger(logging.getLoggerClass()):
    '''
        
    '''
    def __init__(self, name, format="%(asctime)s | %(levelname)s | %(message)s", level=INFO):

        self.format = format
        self.level = level
        self.name = name

        self.console_formatter = logging.Formatter(self.format)
        self.console_logger = logging.StreamHandler(sys.stdout)
        self.console_logger.setFormatter(self.console_formatter)

        self.file_formatter = logging.Formatter(self.format)
        self.file_logger = logging.FileHandler('RekognitionLogs.log')
        self.file_logger.setFormatter(self.file_formatter)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.addHandler(self.console_logger)
        self.logger.addHandler(self.file_logger)

    def info(self, msg, extra=None):
        self.logger.info(msg, extra=extra)

    def error(self, msg, extra=None):
        self.logger.error(msg, extra=extra)

    def debug(self, msg, extra=None):
        self.logger.debug(msg, extra=extra)

    def warn(self, msg, extra=None):
        self.logger.warn(msg, extra=extra)
