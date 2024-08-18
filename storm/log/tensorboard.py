from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from storm.utils import is_main_process

class TensorBoardLogger():
    def __init__(self, log_path, accelerator: Accelerator = None):

        self.accelerator = accelerator

        if self.accelerator is None:
            self.is_main_process = is_main_process()
        else:
            self.is_main_process = self.accelerator.is_local_main_process

        if self.is_main_process:
            self.writer = SummaryWriter(log_path)
        else:
            self.writer = None

    def log_scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, image, step):
        if self.writer:
            self.writer.add_image(tag, image, step)

    def log_histogram(self, tag, values, step):
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_text(self, tag, text, step):
        if self.writer:
            self.writer.add_text(tag, text, step)

    def log_graph(self, model, input_tensor):
        if self.writer:
            self.writer.add_graph(model, input_tensor)

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()