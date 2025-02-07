import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from typing import List, Any
import random
import gym
import pandas as pd

from storm.registry import ENVIRONMENT
from storm.registry import DATASET
from storm.utils import assemble_project_path
from storm.environment.wrapper import make_env
from copy import deepcopy

@ENVIRONMENT.register_module(force=True)
class EnvironmentTrading(gym.Env):
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 select_asset: str = None,
                 initial_amount: float = 1e3,
                 transaction_cost_pct: float = 1e-3,
                 timestamp_format: str = "%Y-%m-%d",
                 history_timestamps: int = 64,
                 future_timestamps: int = 32,
                 start_timestamp="2008-04-01",
                 end_timestamp="2021-04-01",
                 ):
        super(EnvironmentTrading, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.select_asset = select_asset
        assert self.select_asset in self.dataset.assets and self.select_asset is not None, \
            f"select_asset {self.select_asset} not in assets {self.dataset.assets}"

        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        self.assets = self.dataset.assets
        self.prices_name = self.dataset.prices_name
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.history_timestamps = history_timestamps
        self.future_timestamps = future_timestamps
        self.timestamp_format = timestamp_format

        self.features, self.prices, self.features_df, self.prices_df = self._init_features()

        self.action_labels = {
            "SELL": 0,
            "HOLD": 1,
            "BUY": 2
        }
        self.action_dim = len(self.action_labels)

    def _init_features(self):

        data_info = {}

        self.index2timestamps = {}
        for key, value in self.dataset.data_info.items():
            timestamp = value["history_info"]["end_timestamp"]
            if timestamp >= pd.to_datetime(self.start_timestamp) and timestamp < pd.to_datetime(self.end_timestamp):
                data_info[key] = value
                self.index2timestamps[key] = timestamp

        self.timestamp_min_index = min(data_info.keys())
        self.timestamp_max_index = max(data_info.keys())
        self.num_timestamps = self.timestamp_max_index - self.timestamp_min_index + 1

        assert self.num_timestamps == len(data_info), f"num_timestamps {self.num_timestamps} != len(data_info) {len(data_info)}"

        features = self.dataset.features
        prices = self.dataset.original_prices

        features = features[self.select_asset]
        prices = prices[self.select_asset]

        features_df = features
        prices_df = prices

        part_features = {}
        part_prices = {}
        for key, value in data_info.items():

            start_timestamp = value["history_info"]["start_timestamp"]
            end_timestamp = value["history_info"]["end_timestamp"]

            part_features[end_timestamp] = features.loc[start_timestamp:end_timestamp].values
            part_prices[end_timestamp] = prices.loc[end_timestamp].values

        return part_features, part_prices, features_df, prices_df


    def init_timestamp_index(self):
        if self.mode == "train":
            timestamp_index = random.randint(self.timestamp_min_index, self.timestamp_min_index + 3 * (self.num_timestamps // 4))
        else:
            timestamp_index = self.timestamp_min_index
        return timestamp_index

    def get_current_timestamp(self):
        return self.index2timestamps[self.timestamp_index]

    def current_value(self, price):
        return self.cash + self.position * price

    def get_price(self):
        timestamp = self.get_current_timestamp()
        prices = self.prices[timestamp]

        o, h, l, c, adj = prices[0], prices[1], prices[2], prices[3], prices[4]
        price = adj

        return price

    def reset(self, **kwargs):

        self.timestamp_index = self.init_timestamp_index()
        self.timestamp = self.get_current_timestamp()

        self.price = self.get_price()

        state = self.features[self.timestamp]
        state = {
            "states": state,
        }

        self.ret = 0
        self.cash = self.initial_amount
        self.position = 0
        self.discount = 1.0
        self.value = self.initial_amount
        self.total_return = 0
        self.total_profit = 0
        self.action = 1
        self.action_label = "HOLD"

        info= {
            "timestamp": self.timestamp.strftime(self.timestamp_format),
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": self.action,
            "action_label": self.action_label
        }

        return state, info

    def eval_buy_position(self, price):
        # evaluate buy position
        # price * position + price * position * transaction_cost_pct <= cash
        # position <= cash / price / (1 + transaction_cost_pct)
        return int(np.floor(self.cash / price / (1 + self.transaction_cost_pct)))

    def eval_sell_position(self):
        # evaluate sell position
        return int(self.position)

    def buy(self, price, amount):

        # evaluate buy position
        eval_buy_postion = self.eval_buy_position(price)

        # predict buy position
        buy_position = int(np.floor((1.0 * np.abs(amount)) * eval_buy_postion))

        self.cash -= buy_position * price * (1 + self.transaction_cost_pct)
        self.position += buy_position
        self.value = self.current_value(price)

        if buy_position == 0:
            self.action_label = "HOLD"
            self.action = self.action_labels["HOLD"]
        else:
            self.action_label = "BUY"
            self.action = self.action_labels["BUY"]

    def sell(self, price, amount):

        # evaluate sell position
        eval_sell_postion = self.eval_sell_position()

        # predict sell position
        sell_position = int(np.floor((1.0 * np.abs(amount)) * eval_sell_postion))

        self.cash += sell_position * price * (1 - self.transaction_cost_pct)
        self.position -= sell_position
        self.value = self.current_value(price)

        if sell_position == 0:
            self.action_label = "HOLD"
            self.action = self.action_labels["HOLD"]
        else:
            self.action_label = "SELL"
            self.action = self.action_labels["SELL"]

    def hold(self, price, amount):
        self.value = self.current_value(price)

        self.action_label = "HOLD"
        self.action = self.action_labels["HOLD"]

    def step(self, action: int = 0):

        pre_value = self.value

        action = action - 1 # modify the action to -1, 0, 1

        if action > 0:
            self.buy(self.price, amount=action)
        elif action < 0:
            self.sell(self.price, amount=action)
        else:
            self.hold(self.price, amount=action)

        post_value = self.value

        self.timestamp_index = self.timestamp_index + 1
        self.timestamp = self.get_current_timestamp()
        self.price = self.get_price()

        next_state = self.features[self.timestamp]
        next_state = {
            "states": next_state,
        }
        reward = (post_value - pre_value) / pre_value

        self.ret = reward
        self.discount *= 0.99
        self.total_return += self.discount * reward
        self.total_profit = (self.value - self.initial_amount) / self.initial_amount * 100

        if self.timestamp_index < self.timestamp_max_index:
            self.done = False
            self.truncted = False
        else:
            self.done = True
            self.truncted = True

        info = {
            "timestamp": self.timestamp.strftime(self.timestamp_format),
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": self.action,
            "action_label": self.action_label
        }

        return next_state, reward, self.done, self.truncted, info

if __name__ == '__main__':
    select_asset = "AAPL"
    num_envs = 2
    history_timestamps = 64
    num_features = 152

    transition = ["states", "actions", "logprobs", "rewards", "dones", "values"]
    transition_shape = dict(
        states=dict(shape=(num_envs, history_timestamps, num_features), type="float32"),
        actions=dict(shape=(num_envs,), type="int32"),
        logprobs=dict(shape=(num_envs,), type="float32"),
        rewards=dict(shape=(num_envs,), type="float32"),
        dones=dict(shape=(num_envs,), type="float32"),
        values=dict(shape=(num_envs,), type="float32"),
    )

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
        if_use_future=False,
        scaler_cfg=dict(
            type="WindowedScaler"
        ),
        scaler_file="scalers.joblib",
        scaled_data_file="scaled_data.joblib",
        history_timestamps=64,
        future_timestamps=32,
        start_timestamp="2008-04-01",
        end_timestamp="2024-06-01",
        timestamp_format="%Y-%m-%d",
        exp_path=assemble_project_path(os.path.join("workdir", "tmp"))
    )

    environment = dict(
        type="EnvironmentTrading",
        mode="train",
        dataset=None,
        initial_amount=float(1e5),
        transaction_cost_pct=float(1e-4),
        timestamp_format="%Y-%m-%d",
        start_timestamp="2008-04-01",
        end_timestamp="2024-06-01",
        history_timestamps=64,
        future_timestamps=32,
    )

    dataset = DATASET.build(dataset)

    environment.update({
        "dataset": dataset,
        "select_asset": select_asset
    })

    environment = ENVIRONMENT.build(environment)

    environments = gym.vector.AsyncVectorEnv([
        make_env("Trading-v0",env_params=dict(env = deepcopy(environment),
                                     transition_shape = transition_shape, seed = 2024 + i)) for i in range(num_envs)
    ])

    state, info = environments.reset()

    for key, value in state.items():
        print(f"{key}: {value.shape}")
    print()

    for i in range(100):
        action = [1] * num_envs
        next_state, reward, done, truncted, info = environments.step(action)
        for key, value in next_state.items():
            print(f"{key}: {value.shape}")
        print(info)
        if "final_info" in info:
            break
    environments.close()