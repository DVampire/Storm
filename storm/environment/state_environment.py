import warnings
warnings.filterwarnings("ignore")
import numpy as np
from typing import List, Any
import random
import gym

from storm.registry import ENVIRONMENT

@ENVIRONMENT.register_module(force=True)
class EnvironmentStateTrading(gym.Env):
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 initial_amount: float = 1e3,
                 transaction_cost_pct: float = 1e-3,
                 timestamp_format: str = "%Y-%m-%d",
                 history_timestamps: int = 64,
                 future_timestamps: int = 32,
                 start_timestamp="2008-04-01",
                 end_timestamp="2021-04-01",
                 ):
        super(EnvironmentStateTrading, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        self.assets = self.dataset.assets
        self.prices_name = self.dataset.prices_name
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.history_timestamps = history_timestamps
        self.future_timestamps = future_timestamps
        self.timestamp_format = timestamp_format

        self.features, self.prices = self._init_features()
        self.data_info = self._init_data_info()
        index2timestamps = {key: value["history_info"]["end_timestamp"] for key, value in self.data_info.items()}
        self.index2timestamps = index2timestamps

        self.num_timestamps = len(self.data_info)

        self.hold_on_action = 1 # sell, hold, buy=>-1, 0, 1
        self.action_dim = 2 * self.hold_on_action + 1
        self.actions = ["SELL", "HOLD", "BUY"]

    def _init_features(self):
        features = self.dataset.features
        prices = self.dataset.prices
        prices = prices.loc[self.start_timestamp:self.end_timestamp]

        part_features = {}
        for timestamp in prices.index:
            timestamp = timestamp.strftime(self.timestamp_format)
            if timestamp in features:
                part_features[timestamp] = features[timestamp]

        return part_features, prices

    def _init_data_info(self):
        data_info = {}
        count = 0

        for i in range(self.history_timestamps, len(self.prices) - self.future_timestamps):
            history_df = self.prices.iloc[i - self.history_timestamps: i]
            future_df = self.prices.iloc[i: i + self.future_timestamps]

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

    def init_timestamp_index(self):
        if self.mode == "train":
            timestamp = random.randint(0, 3 * (self.num_timestamps // 4))
        else:
            timestamp = 0
        return timestamp

    def get_current_timestamp_datetime(self):
        return self.index2timestamps[self.timestamp_index]

    def current_value(self, price):
        return self.cash + self.position * price

    def get_price(self):
        timestamp_datetime = self.get_current_timestamp_datetime()
        prices = self.prices.loc[timestamp_datetime.strftime(self.timestamp_format)]

        o, h, l, c, adj = prices[0], prices[1], prices[2], prices[3], prices[4]
        price = adj

        return price

    def reset(self, **kwargs):

        self.timestamp_index = self.init_timestamp_index()
        self.timestamp_datetime = self.get_current_timestamp_datetime()
        self.price = self.get_price()

        state = self.features[self.timestamp_datetime.strftime(self.timestamp_format)]

        self.ret = 0
        self.cash = self.initial_amount
        self.position = 0
        self.discount = 1.0
        self.value = self.initial_amount
        self.total_return = 0
        self.total_profit = 0
        self.action = "HOLD"

        info= {
            "timestamp": self.timestamp_datetime.strftime(self.timestamp_format),
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": self.action
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
        buy_position = int(np.floor((1.0 * np.abs(amount / self.hold_on_action)) * eval_buy_postion))

        self.cash -= buy_position * price * (1 + self.transaction_cost_pct)
        self.position += buy_position
        self.value = self.current_value(price)

        if buy_position == 0:
            self.action = "HOLD"
        else:
            self.action = "BUY"

    def sell(self, price, amount):

        # evaluate sell position
        eval_sell_postion = self.eval_sell_position()

        # predict sell position
        sell_position = int(np.floor((1.0 * np.abs(amount / self.hold_on_action)) * eval_sell_postion))

        self.cash += sell_position * price * (1 - self.transaction_cost_pct)
        self.position -= sell_position
        self.value = self.current_value(price)

        if sell_position == 0:
            self.action = "HOLD"
        else:
            self.action = "SELL"

    def noop(self, price, amount):
        self.value = self.current_value(price)

        self.action = "HOLD"

    def step(self, action: int = 0):

        pre_value = self.value

        action = action - self.hold_on_action

        if action > 0:
            self.buy(self.price, amount=action)
        elif action < 0:
            self.sell(self.price, amount=action)
        else:
            self.noop(self.price, amount=action)

        post_value = self.value

        self.timestamp_index = self.timestamp_index + 1
        self.timestamp_datetime = self.get_current_timestamp_datetime()
        self.price = self.get_price()

        next_state = self.features[self.timestamp_datetime.strftime(self.timestamp_format)]
        reward = (post_value - pre_value) / pre_value

        self.state = next_state

        self.ret = reward
        self.discount *= 0.99
        self.total_return += self.discount * reward
        self.total_profit = (self.value - self.initial_amount) / self.initial_amount * 100

        if self.timestamp_index < self.num_timestamps - 1:
            done = False
            truncted = False
        else:
            done = True
            truncted = True

        info = {
            "timestamp": self.timestamp_datetime.strftime(self.timestamp_format),
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": str(self.action)
        }

        return next_state, reward, done, truncted, info