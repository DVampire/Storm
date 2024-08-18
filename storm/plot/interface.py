import os
import shutil
import random

import numpy as np
import pandas as pd
from datetime import datetime
from storm.registry import PLOT
from storm.utils import assemble_project_path, init_path
from storm.plot import plot_kline
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot as driver
from PIL import Image

@PLOT.register_module(force=True)
class PlotInterface():
    def __init__(self,
                 timestamp_format = "%Y-%m-%d",
                 sample_num = None,
                 sample_asset = None,
                 suffix = 'jpeg') -> None:
        super(PlotInterface, self).__init__()

        self.timestamp_format = timestamp_format
        self.sample_num = sample_num
        self.sample_asset = sample_asset
        self.suffix = suffix

        self.echarts_js_path = assemble_project_path(os.path.join("res", "echarts-5.4.3" , "dist", "echarts.min.js"))

    def plot_comparison_kline(self,
                              assets: list,
                              start_timestamps: np.ndarray,
                              end_timestamps: np.ndarray,
                              timestamps: np.ndarray,
                              restored_target_prices: np.ndarray,
                              restored_pred_prices: np.ndarray,
                              save_dir: str,
                              save_prefix: str
                              ):

        kline_dir = init_path(save_dir)

        if not os.path.exists(os.path.join(kline_dir, "echarts.min.js")):
            shutil.copy(self.echarts_js_path, kline_dir)

        nums = len(assets)
        assert nums == len(timestamps) == len(restored_target_prices) == len(restored_pred_prices), "The length of assets, timestamps, prices and pred_prices should be the same"

        sample_num = min(nums, self.sample_num) if self.sample_num is not None else nums

        sample_indices = random.sample(list(range(nums)), sample_num)

        for sample_index in sample_indices:

            batch_smaple_asset = assets[sample_index]
            batch_sample_start_timestamp = int(start_timestamps[sample_index])
            batch_sample_start_timestamp = datetime.fromtimestamp(batch_sample_start_timestamp).strftime(self.timestamp_format) # convert timestamp to string
            batch_sample_end_timestamp = int(end_timestamps[sample_index])
            batch_sample_end_timestamp = datetime.fromtimestamp(batch_sample_end_timestamp).strftime(self.timestamp_format) # convert timestamp to string

            smaple_asset_indices = random.sample(list(range(len(batch_smaple_asset))), self.sample_asset)

            for sample_asset_index in smaple_asset_indices:
                sample_asset = batch_smaple_asset[sample_asset_index]
                sample_timestamp = timestamps[sample_index, :, sample_asset_index].astype(int)
                sample_timestamp = [datetime.fromtimestamp(x).strftime(self.timestamp_format) for x in sample_timestamp]

                sample_price = restored_target_prices[sample_index, :, sample_asset_index, :]
                sample_pred_price = restored_pred_prices[sample_index, :, sample_asset_index, :]

                prices = {
                    "Date": list(sample_timestamp),
                    "Open": list(sample_price[:, 0]),
                    "High": list(sample_price[:, 1]),
                    "Low": list(sample_price[:, 2]),
                    "Close": list(sample_price[:, 3]),
                }
                prices = pd.DataFrame(prices, columns=["Date", "Open", "High", "Low", "Close"])
                prices_title = f"Prices of {sample_asset} from {batch_sample_start_timestamp} to {batch_sample_end_timestamp}"
                pred_prices = {
                    "Date": list(sample_timestamp),
                    "Open": list(sample_pred_price[:, 0]),
                    "High": list(sample_pred_price[:, 1]),
                    "Low": list(sample_pred_price[:, 2]),
                    "Close": list(sample_pred_price[:, 3]),
                }
                pred_prices = pd.DataFrame(pred_prices, columns=["Date", "Open", "High", "Low", "Close"])
                pred_prices_title = f"Predicted Prices of {sample_asset} from {batch_sample_start_timestamp} to {batch_sample_end_timestamp}"

                name = f"{save_prefix}_{'{:06d}'.format(sample_index)}_{sample_asset}_{batch_sample_start_timestamp}_{batch_sample_end_timestamp}"

                prices_kline = plot_kline(prices, prices_title, self.timestamp_format)
                pred_prices_kline = plot_kline(pred_prices, pred_prices_title, self.timestamp_format)

                prices_html_save_path = os.path.join(kline_dir, f'{name}_prices.html')
                prices_image_save_path = os.path.join(kline_dir, f'{name}_prices.{self.suffix}')
                make_snapshot(driver, prices_kline.render(path=prices_html_save_path), prices_image_save_path, is_remove_html=False)

                pred_prices_html_save_path = os.path.join(kline_dir, f'{name}_pred_prices.html')
                pred_prices_image_save_path = os.path.join(kline_dir, f'{name}_pred_prices.{self.suffix}')
                make_snapshot(driver, pred_prices_kline.render(path=pred_prices_html_save_path), pred_prices_image_save_path, is_remove_html=False)

                # combine the two klines through Pillow
                prices_image = Image.open(prices_image_save_path)
                pred_prices_image = Image.open(pred_prices_image_save_path)
                width, height = prices_image.size
                new_image = Image.new('RGB', (2 * width, height))
                new_image.paste(prices_image, (0, 0))
                new_image.paste(pred_prices_image, (width, 0))
                new_image.save(os.path.join(kline_dir, f'{name}_comparison.{self.suffix}'))