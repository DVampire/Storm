import logging
import sys
from colorama import Fore, Style, init as colours_on
from accelerate import Accelerator

colours_on(autoreset=True)

from storm.utils import get_cpu_usage, get_cpu_memory_usage, get_gpu_usage, get_gpu_memory_usage, is_main_process

def warning(message):
    if is_main_process():
        colours_on()
        print(Fore.RED + f' >>> WARNING: {message} ' + Style.RESET_ALL)

class Formatter(logging.Formatter):
    def format(self, record):
        cpu_usage = get_cpu_usage()
        cpu_memory_usage = get_cpu_memory_usage()
        gpu_usage = get_gpu_usage()
        gpu_memory_usage = get_gpu_memory_usage()

        record.cpu_usage = "{:.2f}".format(cpu_usage)
        record.cpu_memory_usage = "{:.2f}".format(cpu_memory_usage)
        record.gpu_usage = "{:.2f}".format(gpu_usage)
        record.gpu_memory_usage = "{:.2f}".format(gpu_memory_usage)

        record.msg = str(record.msg)

        return super().format(record)

class ColorFormatter(Formatter):
    COLORS = {
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "DEBUG": Fore.GREEN,
        "INFO": Fore.WHITE,
        "CRITICAL": Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        message = color + str(record.msg) + Style.RESET_ALL
        record_copy = logging.makeLogRecord(record.__dict__)
        record_copy.msg = message
        return super().format(record_copy)

class Logger():
    def __init__(self, log_path, accelerator: Accelerator = None):

        self.accelerator = accelerator

        if self.accelerator is None:
            self.is_main_process = is_main_process()
        else:
            self.is_main_process = self.accelerator.is_local_main_process

        self.logger = logging.getLogger("Logger")
        self.logger.setLevel(logging.DEBUG)

        formatter = Formatter('%(asctime)s '
                              '- cpu: %(cpu_usage)s%% '
                              '- cpum: %(cpu_memory_usage)s%% '
                              '- gpu: %(gpu_usage)s%% '
                              '- gpum: %(gpu_memory_usage)s%% '
                              '- %(levelname)s '
                              '- %(message)s')
        color_formatter = ColorFormatter('%(asctime)s '
                                         '- cpu: %(cpu_usage)s%% '
                                         '- cpum: %(cpu_memory_usage)s%% '
                                         '- gpu: %(gpu_usage)s%% '
                                         '- gpum: %(gpu_memory_usage)s%% '
                                         '- %(levelname)s '
                                         '- %(message)s')

        if self.is_main_process:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(color_formatter)
            self.logger.addHandler(console_handler)

            if log_path:
                file_handler = logging.FileHandler(log_path)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
        else:
            self.logger.addHandler(logging.NullHandler())

    def debug(self, message):
        if self.is_main_process:
            self.logger.debug(message)

    def info(self, message):
        if self.is_main_process:
            self.logger.info(message)

    def warning(self, message):
        if self.is_main_process:
            self.logger.warning(message)

    def error(self, message):
        if self.is_main_process:
            self.logger.error(message)

    def critical(self, message):
        if self.is_main_process:
            self.logger.critical(message)