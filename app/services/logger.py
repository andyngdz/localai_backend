import logging
import re

ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
PROGRESS_BAR_PATTERN = re.compile(r'[\d]+%|[#=]+|\[.*\]')


class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self._buffer = ''

    def write(self, message):
        self._buffer += message
        if '\n' in message or '\r' in message:
            self.flush()

    def flush(self):
        if not self._buffer:
            return

        msg = ANSI_ESCAPE.sub('', self._buffer).strip()
        self._buffer = ''

        if not msg:
            return

        log_level = logging.INFO if PROGRESS_BAR_PATTERN.search(msg) else self.level
        self.logger.log(log_level, msg)
