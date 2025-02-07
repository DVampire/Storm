from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from storm.utils import is_main_process
from storm.utils import Singleton

__all__= [
    'TensorboardLogger',
    'tensorboard_logger',
]

class TensorboardLogger(SummaryWriter, metaclass=Singleton):
    def __init__(self):
        self.is_main_process = True  # Default to True; will be updated in `init_logger`

    def init_logger(self,
                    log_path: str,
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
            super().__init__(log_path)
        else:
            self.file_writer = None

    def add_scalar(
            self,
            tag,
            scalar_value,
            global_step=None,
            walltime=None,
            new_style=False,
            double_precision=False,
        ):
        """
        Add a scalar value to the logs.
        """
        if self.is_main_process:
            super().add_scalar(tag,
                               scalar_value,
                               global_step,
                               walltime,
                               new_style,
                               double_precision)

    def add_image(self,
                  tag,
                  img_tensor,
                  global_step=None,
                  walltime=None,
                  dataformats='CHW'):
        """
        Add an image to the logs.
        """
        if self.is_main_process:
            super().add_image(tag, img_tensor, global_step, walltime, dataformats)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """
        Add text to the logs.
        """
        if self.is_main_process:
            super().add_text(tag, text_string, global_step, walltime)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        """
        Add a histogram to the logs.
        """
        if self.is_main_process:
            super().add_histogram(tag, values, global_step, bins, walltime, max_bins)

    def add_graph(self, model,
                  input_to_model=None,
                  verbose=False,
                  use_strict_trace=True):
        """
        Add a graph to the logs.
        """
        if self.is_main_process:
            super().add_graph(model, input_to_model, verbose, use_strict_trace)

    def flush(self):
        """
        Flush the logs to disk.
        """
        if self.is_main_process:
            super().flush()

    def close(self):
        """
        Close the logger to release resources.
        """
        if self.is_main_process:
            super().close()

tensorboard_logger = TensorboardLogger()