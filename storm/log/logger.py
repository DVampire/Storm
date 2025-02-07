import logging
from accelerate import Accelerator

from storm.utils import is_main_process
from storm.utils import Singleton

__all__ = ['Logger', 'logger']

class Logger(logging.Logger, metaclass=Singleton):
    def __init__(self,
                 name='logger',
                 level=logging.INFO):
        # Initialize the parent class
        super().__init__(name, level)

        # Define a formatter for log messages
        self.formatter = logging.Formatter(
            fmt='\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        self.is_main_process = True  # Default to True; will be updated in `init_logger`

    def init_logger(self,
                    log_path: str,
                    level=logging.INFO,
                    accelerator: Accelerator = None):
        """
        Initialize the logger with a file path and optional main process check.

        Args:
            log_path (str): The log file path.
            level (int, optional): The logging level. Defaults to logging.INFO.
            accelerator (Accelerator, optional): Accelerator instance to determine the main process.
        """

        # Determine if this is the main process
        if accelerator is None:
            self.is_main_process = is_main_process()
        else:
            self.is_main_process = accelerator.is_local_main_process

        if self.is_main_process:
            # Add a console handler for logging to the console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(self.formatter)
            self.addHandler(console_handler)

            # Add a file handler for logging to the file
            file_handler = logging.FileHandler(log_path, mode='a')  # 'a' mode appends to the file
            file_handler.setLevel(level)
            file_handler.setFormatter(self.formatter)
            self.addHandler(file_handler)

        # Prevent duplicate logs from propagating to the root logger
        self.propagate = False

    def info(self, msg, *args, **kwargs):
        """
        Overridden info method with stacklevel adjustment for correct log location.
        """
        if self.is_main_process:
            kwargs.setdefault("stacklevel", 2)  # Adjust stack level to show the actual caller
            super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.is_main_process:
            kwargs.setdefault("stacklevel", 2)
            super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if self.is_main_process:
            kwargs.setdefault("stacklevel", 2)
            super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if self.is_main_process:
            kwargs.setdefault("stacklevel", 2)
            super().critical(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if self.is_main_process:
            kwargs.setdefault("stacklevel", 2)
            super().debug(msg, *args, **kwargs)

logger = Logger()
