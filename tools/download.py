import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import argparse
from mmengine.config import Config, DictAction

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from storm.registry import DOWNLOADER
from storm.utils import assemble_project_path
from storm.log import logger
from storm.log import warning

def parse_args():
    parser = argparse.ArgumentParser(description="Download Prices")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "download", "dj30.py"), help="download datasets config file path")
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
        warning(f"| Arguments Remove work_dir: {exp_path}")
    else:
        warning(f"| Arguments Keep work_dir: {exp_path}")
    os.makedirs(exp_path, exist_ok=True)

    cfg.log_path = os.path.join(exp_path, cfg.log_file)
    logger.init_logger(log_path=cfg.log_path)

    downloader = DOWNLOADER.build(cfg.downloader)

    try:
        downloader.run()
    except KeyboardInterrupt:
        sys.exit()


if __name__ == '__main__':
    main()