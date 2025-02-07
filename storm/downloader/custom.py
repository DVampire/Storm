from abc import ABC, abstractmethod

class AbstractDownloader(ABC):

    def __init__(self, *args, **kwargs):
        """
        Initialize the downloader class.
        Args:
            *args:
            **kwargs:
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run the downloader.
        Args:
            *args:
            **kwargs:

        Returns:
        """
        pass