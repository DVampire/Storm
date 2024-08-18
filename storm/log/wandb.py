import os
import wandb
from dotenv import load_dotenv
from accelerate import Accelerator

load_dotenv(verbose=True)

from storm.utils import is_main_process

class WandbLogger():
    def __init__(self, project, name, config, dir, accelerator: Accelerator = None):
        self.accelerator = accelerator

        if self.accelerator is None:
            self.is_main_process = is_main_process()
        else:
            self.is_main_process = self.accelerator.is_local_main_process

        self.project = project
        self.name = name
        self.config = config
        self.dir = dir

        if self.is_main_process:
            wandb.login(key=os.environ["WANDB_API_KEY"])
            wandb.init(project=self.project, name=self.name, config=self.config, dir=self.dir)

    def log(self, log_dict):
        if self.is_main_process:
            wandb.log(log_dict)

    def finish(self):
        if self.is_main_process:
            wandb.finish()