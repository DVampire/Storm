import os

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_rows', 100000)
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any

from storm.utils import convert_timestamp_to_int
from storm.utils import load_json
from storm.utils import load_joblib
from storm.utils import save_joblib
from storm.registry import DATASET
from storm.registry import SCALER
from storm.utils import assemble_project_path
from storm.data.collate_fn import MultiAssetPriceTextCollateFn

@DATASET.register_module(force=True)
class MultiAssetDataset(Dataset):

    def __init__(self,
                 *args,
                 data_path: str = None,
                 assets_path: str = None,
                 fields_name: Dict[str, List[str]] = None,
                 if_norm: bool = True,
                 if_norm_temporal: bool = False,
                 if_use_temporal: bool = True,
                 if_use_future: bool = True,
                 scaler_cfg: Dict[str, Any] = None,
                 scaler_file: str = None,
                 scaled_data_file: str = None,
                 history_timestamps: int = 64,
                 future_timestamps: int = 32,
                 start_timestamp: str = None,
                 end_timestamp: str = None,
                 timestamp_format: str = "%Y-%m-%d",
                 exp_path: str = None,
                 **kwargs
                 ):
        super(Dataset, self).__init__()

        self.data_path = assemble_project_path(data_path)
        self.assets_path = assemble_project_path(assets_path)

        self.fields_name = fields_name

        self.features_name = self.fields_name["features"]
        self.prices_name = self.fields_name["prices"]
        self.temporals_name = self.fields_name["temporals"]
        self.labels_name = self.fields_name["labels"]

        self.if_norm = if_norm
        self.if_norm_temporal = if_norm_temporal
        self.if_use_temporal = if_use_temporal
        self.if_use_future = if_use_future

        exp_path = assemble_project_path(exp_path)
        self.scaler_path = os.path.join(exp_path, scaler_file)
        self.scaled_data_path = os.path.join(exp_path, scaled_data_file)

        self.history_timestamps = history_timestamps
        self.future_timestamps = future_timestamps

        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

        self.timestamp_format = timestamp_format

        self.scaler_cfg = scaler_cfg
        if os.path.exists(self.scaler_path):
            self.scalers = load_joblib(self.scaler_path)
        else:
            self.scalers = None

        if os.path.exists(self.scaled_data_path):
            scaled_data = load_joblib(self.scaled_data_path)
            self.assets = scaled_data["assets"]
            self.assets_df = scaled_data["assets_df"]
            self.features = scaled_data["features"]
            self.labels = scaled_data["labels"]
            self.prices = scaled_data["prices"]
            self.original_prices = scaled_data["original_prices"]
            self.prices_mean = scaled_data["prices_mean"]
            self.prices_std = scaled_data["prices_std"]
            self.data_info = scaled_data["data_info"]
        else:
            self.assets = self._init_assets()
            self.assets_df = self._load_assets_df()
            self.features, self.labels, self.prices, self.original_prices, self.scalers, self.prices_mean, self.prices_std = self._init_features()
            save_joblib(self.scalers, self.scaler_path)
            self.data_info = self._init_data_info()

            scaled_data = {
                "assets": self.assets,
                "assets_df": self.assets_df,
                "features": self.features,
                "labels": self.labels,
                "prices": self.prices,
                "original_prices": self.original_prices,
                "prices_mean": self.prices_mean,
                "prices_std": self.prices_std,
                "data_info": self.data_info,
            }

            save_joblib(scaled_data, self.scaled_data_path)

    def _init_assets(self):

        assets = load_json(self.assets_path)
        if isinstance(assets, dict):
            asset_symbols = list(assets.keys())
        elif isinstance(assets, list):
            asset_symbols = [asset["symbol"] for asset in assets]
        else:
            raise ValueError("Unsupported assets format. Expected a dict or a list of dicts.")
        
        return asset_symbols


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

        features = {}
        labels = {}
        scalers = {}
        prices = {} # scaled prices
        original_prices = {} # original prices
        prices_mean = {}
        prices_std = {}

        for asset in self.assets:

            asset_df = self.assets_df[asset]

            price_indices = [self.features_name.index(price_name) for price_name in self.prices_name]
            original_prices[asset] = asset_df[self.prices_name]

            if self.if_norm:
                if self.scalers is None or asset not in self.scalers:
                    scaler = SCALER.build(self.scaler_cfg)

                    if self.if_norm_temporal:
                        asset_df[self.features_name + self.temporals_name] = scaler.fit_transform(asset_df[self.features_name + self.temporals_name])
                    else:
                        asset_df[self.features_name] = scaler.fit_transform(asset_df[self.features_name])
                else:
                    scaler = self.scalers[asset]
                    if self.if_norm_temporal:
                        asset_df[self.features_name + self.temporals_name] = scaler.transform(asset_df[self.features_name + self.temporals_name])
                    else:
                        asset_df[self.features_name] = scaler.transform(asset_df[self.features_name])

                scalers[asset] = scaler

                if len(scaler.mean.shape) == 1 and len(scaler.std.shape) == 1:
                    prices_mean[asset] = scaler.mean[None, 1].repeat(asset_df.shape[0], axis=0)[..., price_indices]
                    prices_std[asset] = scaler.std[None, 1].repeat(asset_df.shape[0], axis=0)[..., price_indices]
                else:
                    prices_mean[asset] = scaler.mean[..., price_indices]
                    prices_std[asset] = scaler.std[..., price_indices]

            else:
                scalers[asset] = None

            if self.if_use_temporal:
                features[asset] = asset_df[self.features_name + self.temporals_name]
            else:
                features[asset] = asset_df[self.features_name]

            prices[asset] = asset_df[self.prices_name]
            labels[asset] = asset_df[self.labels_name]

        return features, labels, prices, original_prices, scalers, prices_mean, prices_std

    def _init_data_info(self):
        data_info = {}
        count = 0

        first_asset = self.assets_df[self.assets[0]]

        future_timestamps = self.future_timestamps if self.if_use_future else 0

        for i in range(self.history_timestamps, len(first_asset) - future_timestamps):
            data_info[count] = {}

            history_df = first_asset.iloc[i - self.history_timestamps: i]
            history_info = {
                "start_timestamp": history_df.index[0],
                "end_timestamp": history_df.index[-1],
                "start_index": i - self.history_timestamps,
                "end_index": i - 1,
            }

            data_info[count].update(
                {"history_info": history_info}
            )

            if self.if_use_future:

                future_df = first_asset.iloc[i: i + self.future_timestamps]
                future_info = {
                    "start_timestamp": future_df.index[0],
                    "end_timestamp": future_df.index[-1],
                    "start_index": i,
                    "end_index": i + self.future_timestamps - 1,
                }

                data_info[count].update(
                    {"future_info": future_info}
                )

            count += 1

        return data_info

    def __str__(self):
        str = f"{'-' * 50} MultiAssetDataset {'-' * 50}\n"
        for asset in self.assets:
            str += f"asset: {asset}\n"
            str += f"features shape: {self.features[asset].shape}\n"
            str += f"labels shape: {self.labels[asset].shape}\n"
            str += f"prices shape: {self.prices[asset].shape}\n"
            str += f"prices mean shape: {self.prices_mean[asset].shape}\n"
            str += f"prices std shape: {self.prices_std[asset].shape}\n"
            if self.scalers[asset]:
                str += f"scaler: mean: {self.scalers[asset].mean.shape}, std: {self.scalers[asset].std.shape}\n"
            else:
                str += "scaler: None\n"
            str += "\n"

        str += f"total data info: {len(self.data_info)}\n"
        str += f"{'-' * 50} MultiAssetDataset {'-' * 50}\n"
        return str


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        item = self.data_info[idx]

        asset = self.assets

        history_info = item["history_info"]
        history_data = {
            "start_timestamp": convert_timestamp_to_int(history_info["start_timestamp"]),
            "end_timestamp": convert_timestamp_to_int(history_info["end_timestamp"]),
            "start_index": history_info["start_index"],
            "end_index": history_info["end_index"],
            "features": np.stack([self.features[asset].loc[history_info["start_timestamp"]:
                                                           history_info["end_timestamp"]].values.astype("float32") for asset in self.assets]),
            "labels": np.stack([self.labels[asset].loc[history_info["start_timestamp"]:
                                                       history_info["end_timestamp"]].values.astype("float32") for asset in self.assets]),
            "original_prices": np.stack([self.original_prices[asset].loc[history_info["start_timestamp"]:
                                                       history_info["end_timestamp"]].values.astype("float32") for asset in self.assets]),
            "prices": np.stack([self.prices[asset].loc[history_info["start_timestamp"]:
                                                       history_info["end_timestamp"]].values.astype("float32") for asset in self.assets]),
            "timestamps": np.stack([self.features[asset].loc[history_info["start_timestamp"]:
                                                             history_info["end_timestamp"]].reset_index(drop=False)["timestamp"].apply(lambda x: convert_timestamp_to_int(x)).values.astype("float32") for asset in self.assets]),
            "prices_mean": np.stack([self.prices_mean[asset][history_info["start_index"]:
                                                             history_info["end_index"] + 1, ...].astype("float32") for asset in self.assets]),
            "prices_std": np.stack([self.prices_std[asset][history_info["start_index"]:
                                                           history_info["end_index"] + 1, ...].astype("float32") for asset in self.assets]),
            "text": ["sample text"] * len(self.assets),
        }

        res = {
            "asset": asset,
            "history": history_data,
        }

        if self.if_use_future:

            future_info = item["future_info"]
            future_data = {
                "start_timestamp": convert_timestamp_to_int(future_info["start_timestamp"]),
                "end_timestamp": convert_timestamp_to_int(future_info["end_timestamp"]),
                "start_index": future_info["start_index"],
                "end_index": future_info["end_index"],
                "features": np.stack([self.features[asset].loc[future_info["start_timestamp"]:
                                                               future_info["end_timestamp"]].values.astype("float32") for asset in self.assets]),
                "labels": np.stack([self.labels[asset].loc[future_info["start_timestamp"]:
                                                             future_info["end_timestamp"]].values.astype("float32") for asset in self.assets]),
                "prices": np.stack([self.prices[asset].loc[future_info["start_timestamp"]:
                                                           future_info["end_timestamp"]].values.astype("float32") for asset in self.assets]),
                "original_prices": np.stack([self.original_prices[asset].loc[future_info["start_timestamp"]:
                                                              future_info["end_timestamp"]].values.astype("float32") for asset in self.assets]),
                "timestamps": np.stack([self.features[asset].loc[future_info["start_timestamp"]:
                                                                 future_info["end_timestamp"]].reset_index(drop=False)["timestamp"].apply(lambda x: convert_timestamp_to_int(x)).values.astype("float32") for asset in self.assets]),
                "prices_mean": np.stack([self.prices_mean[asset][future_info["start_index"]:
                                                                 future_info["end_index"] + 1, ...].astype("float32") for asset in self.assets]),
                "prices_std": np.stack([self.prices_std[asset][future_info["start_index"]:
                                                               future_info["end_index"] + 1, ...].astype("float32") for asset in self.assets]),
                "text": ["sample text"] * len(self.assets),
            }

            res.update({
                "future": future_data
            })

        return res

__all__ = [
    "MultiAssetDataset"
]

if __name__ == '__main__':

    dataset = dict(
        type="MultiAssetDataset",
        data_path="datasets/processd_day_dj30/features",
        assets_path="configs/_asset_list_/dj30.json",
        fields_name={
            "features": [
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "kmid",
                "kmid2",
                "klen",
                "kup",
                "kup2",
                "klow",
                "klow2",
                "ksft",
                "ksft2",
                "roc_5",
                "roc_10",
                "roc_20",
                "roc_30",
                "roc_60",
                "ma_5",
                "ma_10",
                "ma_20",
                "ma_30",
                "ma_60",
                "std_5",
                "std_10",
                "std_20",
                "std_30",
                "std_60",
                "beta_5",
                "beta_10",
                "beta_20",
                "beta_30",
                "beta_60",
                "max_5",
                "max_10",
                "max_20",
                "max_30",
                "max_60",
                "min_5",
                "min_10",
                "min_20",
                "min_30",
                "min_60",
                "qtlu_5",
                "qtlu_10",
                "qtlu_20",
                "qtlu_30",
                "qtlu_60",
                "qtld_5",
                "qtld_10",
                "qtld_20",
                "qtld_30",
                "qtld_60",
                "rank_5",
                "rank_10",
                "rank_20",
                "rank_30",
                "rank_60",
                "imax_5",
                "imax_10",
                "imax_20",
                "imax_30",
                "imax_60",
                "imin_5",
                "imin_10",
                "imin_20",
                "imin_30",
                "imin_60",
                "imxd_5",
                "imxd_10",
                "imxd_20",
                "imxd_30",
                "imxd_60",
                "rsv_5",
                "rsv_10",
                "rsv_20",
                "rsv_30",
                "rsv_60",
                "cntp_5",
                "cntp_10",
                "cntp_20",
                "cntp_30",
                "cntp_60",
                "cntn_5",
                "cntn_10",
                "cntn_20",
                "cntn_30",
                "cntn_60",
                "cntd_5",
                "cntd_10",
                "cntd_20",
                "cntd_30",
                "cntd_60",
                "corr_5",
                "corr_10",
                "corr_20",
                "corr_30",
                "corr_60",
                "cord_5",
                "cord_10",
                "cord_20",
                "cord_30",
                "cord_60",
                "sump_5",
                "sump_10",
                "sump_20",
                "sump_30",
                "sump_60",
                "sumn_5",
                "sumn_10",
                "sumn_20",
                "sumn_30",
                "sumn_60",
                "sumd_5",
                "sumd_10",
                "sumd_20",
                "sumd_30",
                "sumd_60",
                "vma_5",
                "vma_10",
                "vma_20",
                "vma_30",
                "vma_60",
                "vstd_5",
                "vstd_10",
                "vstd_20",
                "vstd_30",
                "vstd_60",
                "wvma_5",
                "wvma_10",
                "wvma_20",
                "wvma_30",
                "wvma_60",
                "vsump_5",
                "vsump_10",
                "vsump_20",
                "vsump_30",
                "vsump_60",
                "vsumn_5",
                "vsumn_10",
                "vsumn_20",
                "vsumn_30",
                "vsumn_60",
                "vsumd_5",
                "vsumd_10",
                "vsumd_20",
                "vsumd_30",
                "vsumd_60",
            ],
            "prices": [
                "open",
                "high",
                "low",
                "close",
                "adj_close",
            ],
            "temporals": [
                "day",
                "weekday",
                "month",
            ],
            "labels": [
                "ret1",
                "mov1"
            ]
        },
        if_norm=True,
        if_norm_temporal=False,
        scaler_cfg = dict(
            type="WindowedScaler"
        ),
        scaler_file = "scalers.joblib",
        scaled_data_file = "scaled_data.joblib",
        history_timestamps = 64,
        future_timestamps = 32,
        start_timestamp="2008-04-01",
        end_timestamp="2024-06-01",
        timestamp_format = "%Y-%m-%d",
        exp_path = assemble_project_path(os.path.join("workdir", "tmp"))
    )

    dataset = DATASET.build(dataset)
    print(len(dataset))

    collate_fn = MultiAssetPriceTextCollateFn()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
    item = next(iter(dataloader))

    print(item)
    print(item["history"]["features"].shape)
    print(item["history"]["labels"].shape)
    print(item["history"]["prices"].shape)
    print(item["history"]["original_prices"].shape)
    print(item["history"]["prices_mean"].shape)
    print(item["history"]["prices_std"].shape)