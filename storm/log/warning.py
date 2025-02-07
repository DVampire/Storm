from colorama import Fore, Style, init as colours_on
colours_on(autoreset=True)

from storm.utils import is_main_process

__all__ = ['warning']
def warning(message):
    if is_main_process():
        colours_on()
        print(Fore.RED + f' >>> WARNING: {message} ' + Style.RESET_ALL)
