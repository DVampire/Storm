import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["MKL_DEBUG_CPU_TYPE"] = '5'
import warnings
warnings.filterwarnings("ignore")
import os
import sys
from pathlib import Path
import multiprocessing
import argparse
from mmengine.config import Config, DictAction

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from storm.registry import PROCESSOR
from storm.utils import assemble_project_path

def parse_args():
    parser = argparse.ArgumentParser(description="Process Prices")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "processor", "processor_day_sp500.py"), help="processor datasets config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--if_remove", action="store_true", default=False)
    args = parser.parse_args()
    return args

class StockProcessorProcess(multiprocessing.Process):
    def __init__(self, assets, processor):
        super().__init__()
        self.assets = assets
        self.processor = processor
    def run(self):
        self.processor.process(self.assets)

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is None:
        args.cfg_options = dict()
    if args.workdir is not None:
        args.cfg_options["workdir"] = args.workdir
    if args.tag is not None:
        args.cfg_options["tag"] = args.tag
    cfg.merge_from_dict(args.cfg_options)

    exp_path = assemble_project_path(os.path.join(cfg.workdir, cfg.tag))
    if args.if_remove is None:
        args.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {exp_path}? ") == 'y')
    if args.if_remove:
        import shutil
        shutil.rmtree(exp_path, ignore_errors=True)
        print(f"| Arguments Remove work_dir: {exp_path}")
    else:
        print(f"| Arguments Keep work_dir: {exp_path}")
    os.makedirs(exp_path, exist_ok=True)

    processor = PROCESSOR.build(cfg.processor)
    assets = processor.assets

    batch_size = cfg.batch_size if cfg.batch_size < len(assets) else 5
    batch_size = min(len(assets), batch_size)

    processes = []
    remaining_assets = assets.copy()

    while remaining_assets:
        batch = remaining_assets[:batch_size]
        remaining_assets = remaining_assets[batch_size:]

        process = StockProcessorProcess(batch, processor)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    main()