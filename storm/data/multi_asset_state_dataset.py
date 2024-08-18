import os

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_rows', 100000)
from typing import List, Dict
from einops import rearrange

from storm.utils import assemble_project_path
from storm.utils import load_json
from storm.utils import load_joblib
from storm.registry import DATASET

@DATASET.register_module(force=True)
class MultiAssetStateDataset():
    def __init__(self,
                 data_path: str = None,
                 assets_path: str = None,
                 fields_name: Dict[str, List[str]] = None,
                 states_path: str = None,
                 select_asset: str = None,
                 history_timestamps: int = 64,
                 future_timestamps: int = 32,
                 start_timestamp: str = None,
                 end_timestamp: str = None,
                 timestamp_format: str = "%Y-%m-%d",
                 if_use_cs: bool = True,
                 if_use_ts: bool = True,
                 exp_path: str = None,
                 ):
        super(MultiAssetStateDataset, self).__init__()

        self.data_path = assemble_project_path(data_path)
        self.assets_path = assemble_project_path(assets_path)

        self.fields_name = fields_name

        self.prices_name = self.fields_name["prices"]

        self.states_path = assemble_project_path(states_path)
        assert os.path.exists(self.states_path), f"states_path: {self.states_path} not exists"

        self.history_timestamps = history_timestamps
        self.future_timestamps = future_timestamps

        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

        self.timestamp_format = timestamp_format
        self.if_use_cs = if_use_cs
        self.if_use_ts = if_use_ts

        self.assets = self._init_assets()
        self.select_asset = select_asset
        self.select_asset_index = self.assets.index(self.select_asset)

        self.assets_df = self._load_assets_df()
        self.features, self.prices = self._init_features()
        self.data_info = self._init_data_info()

    def _init_assets(self):
        assets = load_json(self.assets_path)
        assets = [asset["symbol"] for asset in assets]
        return assets

    def _load_assets_df(self):
        start_timestamp = pd.to_datetime(self.start_timestamp, format=self.timestamp_format) if self.start_timestamp else None
        end_timestamp = pd.to_datetime(self.end_timestamp, format=self.timestamp_format) if self.end_timestamp else None

        assets_df = {}
        for asset in self.assets:
            asset_path = os.path.join(self.data_path, "{}.csv".format(asset))
            asset_df = pd.read_csv(asset_path, index_col=0)
            asset_df.index = pd.to_datetime(asset_df.index)

            if start_timestamp and end_timestamp:
                asset_df = asset_df.loc[start_timestamp:end_timestamp]
            elif start_timestamp:
                asset_df = asset_df.loc[start_timestamp:]
            elif end_timestamp:
                asset_df = asset_df.loc[:end_timestamp]
            else:
                pass

            assets_df[asset] = asset_df
        return assets_df

    def _init_features(self):

        states = load_joblib(self.states_path)

        meta = states["meta"]
        items = states["items"]

        features = {}

        for timestamp, item in items.items():

            if self.if_use_cs and self.if_use_ts:

                ts_n_size = meta["ts_n_size"]

                factor_cs = item["factor_cs"]
                factor_ts = item["factor_ts"]
                embed_dim = factor_ts.shape[-1]

                # (n1, n2, n3, embed_dim)
                factor_ts = rearrange(factor_ts, "(n1 n2 n3) c -> n1 n2 n3 c", n1=ts_n_size[0], n2=ts_n_size[1], n3=ts_n_size[2], c = embed_dim)

                select_asset_factor_ts = factor_ts[:, self.select_asset_index, :, :]
                select_asset_factor_ts = rearrange(select_asset_factor_ts, "n1 n3 c -> (n1 n3) c", n1=ts_n_size[0], n3=ts_n_size[2], c = embed_dim)

                factors = np.concatenate([factor_cs, select_asset_factor_ts], axis=0)

                features[timestamp] = factors

            elif self.if_use_cs and not self.if_use_ts:

                factor = item["factor"]
                features[timestamp] = factor

            elif not self.if_use_cs and self.if_use_ts:

                n_size = meta["n_size"]
                factor = item["factor"]

                embed_dim = factor.shape[-1]

                # (n1, n2, n3, embed_dim)
                factor = rearrange(factor, "(n1 n2 n3) c -> n1 n2 n3 c", n1=n_size[0], n2=n_size[1], n3=n_size[2], c = embed_dim)

                select_asset_factor = factor[:, self.select_asset_index, :, :]
                select_asset_factor = rearrange(select_asset_factor, "n1 n3 c -> (n1 n3) c", n1=n_size[0], n3=n_size[2], c = embed_dim)

                features[timestamp] = select_asset_factor

        prices = self.assets_df[self.select_asset][self.prices_name]
        return features, prices

    def _init_data_info(self):
        data_info = {}
        count = 0

        first_asset = self.assets_df[self.assets[0]]
        for i in range(self.history_timestamps, len(first_asset) - self.future_timestamps):
            history_df = first_asset.iloc[i - self.history_timestamps: i]
            future_df = first_asset.iloc[i: i + self.future_timestamps]

            history_info = {
                "start_timestamp": history_df.index[0],
                "end_timestamp": history_df.index[-1],
                "start_index": i - self.history_timestamps,
                "end_index": i - 1,
            }

            future_info = {
                "start_timestamp": future_df.index[0],
                "end_timestamp": future_df.index[-1],
                "start_index": i,
                "end_index": i + self.future_timestamps - 1,
            }

            data_info[count] = {
                "history_info": history_info,
                "future_info": future_info,
            }

            count += 1

        return data_info