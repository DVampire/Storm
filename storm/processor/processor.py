from copy import deepcopy
from datetime import datetime
from storm.registry import PROCESSOR
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from storm.utils import assemble_project_path
from storm.utils import load_json

def my_rank(x):
   return pd.Series(x).rank(pct=True).iloc[-1]

def cal_news(df):
    df["title"] = df["title"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    df["text"] = df["text"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    df["source"] = df["source"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return df

def cal_guidance(df):
    df["title"] = df["title"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    df["text"] = df["text"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return df

def cal_sentiment(df, columns):
    for col in columns:
        if "sentiment" not in col:
            df[col] = df.groupby("timestamp")[col].transform("sum")
        else:
            df[col] = df.groupby("timestamp")[col].transform("mean")
            df[col] = df[col].fillna(0.5)
    return df


def cal_factor(df, level="day"):
    # intermediate values
    df['max_oc'] = df[["open", "close"]].max(axis=1)
    df['min_oc'] = df[["open", "close"]].min(axis=1)
    df["kmid"] = (df["close"] - df["open"]) / df["close"]
    df['kmid2'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-12)
    df["klen"] = (df["high"] - df["low"]) / df["open"]
    df['kup'] = (df['high'] - df['max_oc']) / df['open']
    df['kup2'] = (df['high'] - df['max_oc']) / (df['high'] - df['low'] + 1e-12)
    df['klow'] = (df['min_oc'] - df['low']) / df['open']
    df['klow2'] = (df['min_oc'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df["ksft"] = (2 * df["close"] - df["high"] - df["low"]) / df["open"]
    df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df.drop(columns=['max_oc', 'min_oc'], inplace=True)

    window = [5, 10, 20, 30, 60]
    for w in window:
        df['roc_{}'.format(w)] = df['close'].shift(w) / df['close']

    for w in window:
        df['ma_{}'.format(w)] = df['close'].rolling(w).mean() / df['close']

    for w in window:
        df['std_{}'.format(w)] = df['close'].rolling(w).std() / df['close']

    for w in window:
        df['beta_{}'.format(w)] = (df['close'].shift(w) - df['close']) / (w * df['close'])

    for w in window:
        df['max_{}'.format(w)] = df['close'].rolling(w).max() / df['close']

    for w in window:
        df['min_{}'.format(w)] = df['close'].rolling(w).min() / df['close']

    for w in window:
        df['qtlu_{}'.format(w)] = df['close'].rolling(w).quantile(0.8) / df['close']

    for w in window:
        df['qtld_{}'.format(w)] = df['close'].rolling(w).quantile(0.2) / df['close']

    for w in window:
        df['rank_{}'.format(w)] = df['close'].rolling(w).apply(my_rank) / w

    for w in window:
        df['imax_{}'.format(w)] = df['high'].rolling(w).apply(np.argmax) / w

    for w in window:
        df['imin_{}'.format(w)] = df['low'].rolling(w).apply(np.argmin) / w

    for w in window:
        df['imxd_{}'.format(w)] = (df['high'].rolling(w).apply(np.argmax) - df['low'].rolling(w).apply(np.argmin)) / w

    for w in window:
        shift = df['close'].shift(w)
        min = df["low"].where(df["low"] < shift, shift)
        max = df["high"].where(df["high"] > shift, shift)
        df["rsv_{}".format(w)] = (df["close"] - min) / (max - min + 1e-12)

    df['ret1'] = df['close'].pct_change(1)
    for w in window:
        df['cntp_{}'.format(w)] = (df['ret1'].gt(0)).rolling(w).sum() / w

    for w in window:
        df['cntn_{}'.format(w)] = (df['ret1'].lt(0)).rolling(w).sum() / w

    for w in window:
        df['cntd_{}'.format(w)] = df['cntp_{}'.format(w)] - df['cntn_{}'.format(w)]

    for w in window:
        df1 = df["close"].rolling(w)
        df2 = np.log(df["volume"] + 1).rolling(w)
        df["corr_{}".format(w)] = df1.corr(pairwise = df2)

    for w in window:
        df1 = df["close"]
        df_shift1 = df1.shift(1)
        df2 = df["volume"]
        df_shift2 = df2.shift(1)
        df1 = df1 / df_shift1
        df2 = np.log(df2 / df_shift2 + 1)
        df["cord_{}".format(w)] = df1.rolling(w).corr(pairwise = df2.rolling(w))

    df['abs_ret1'] = np.abs(df['ret1'])
    df['pos_ret1'] = df['ret1']
    df['pos_ret1'][df['pos_ret1'].lt(0)] = 0

    for w in window:
        df['sump_{}'.format(w)] = df['pos_ret1'].rolling(w).sum() / (df['abs_ret1'].rolling(w).sum() + 1e-12)

    for w in window:
        df['sumn_{}'.format(w)] = 1 - df['sump_{}'.format(w)]

    for w in window:
        df['sumd_{}'.format(w)] = 2 * df['sump_{}'.format(w)] - 1

    for w in window:
        df["vma_{}".format(w)] = df["volume"].rolling(w).mean() / (df["volume"] + 1e-12)

    for w in window:
        df["vstd_{}".format(w)] = df["volume"].rolling(w).std() / (df["volume"] + 1e-12)

    for w in window:
        shift = np.abs((df["close"] / df["close"].shift(1) - 1)) * df["volume"]
        df1 = shift.rolling(w).std()
        df2 = shift.rolling(w).mean()
        df["wvma_{}".format(w)] = df1 / (df2 + 1e-12)

    df['vchg1'] = df['volume'] - df['volume'].shift(1)
    df['abs_vchg1'] = np.abs(df['vchg1'])
    df['pos_vchg1'] = df['vchg1']
    df['pos_vchg1'][df['pos_vchg1'].lt(0)] = 0

    for w in window:
        df["vsump_{}".format(w)] = df["pos_vchg1"].rolling(w).sum() / (df["abs_vchg1"].rolling(w).sum() + 1e-12)
    for w in window:
        df["vsumn_{}".format(w)] = 1 - df["vsump_{}".format(w)]
    for w in window:
        df["vsumd_{}".format(w)] = 2 * df["vsump_{}".format(w)] - 1

    df["log_volume"] = np.log(df["volume"] + 1)

    df.drop(columns=['ret1', 'abs_ret1', 'pos_ret1', 'vchg1', 'abs_vchg1', 'pos_vchg1', 'volume'], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)

    if level == "minute":
        df["minute"] = pd.to_datetime(df["timestamp"]).dt.minute
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    df["day"] = pd.to_datetime(df["timestamp"]).dt.day
    df["weekday"] = pd.to_datetime(df["timestamp"]).dt.weekday
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    df["year"] = pd.to_datetime(df["timestamp"]).dt.year

    return df

def cal_target(df):
    df['ret1'] = df['close'].pct_change(1).shift(-1)
    df['mov1'] = (df['ret1'] > 0)
    df['mov1'] = df['mov1'].astype(int)
    return df

@PROCESSOR.register_module(force=True)
class Processor():
    def __init__(self,
                 path_params = None,
                 assets_path = None,
                 start_date = None,
                 end_date = None,
                 interval="day",
                 workdir = None,
                 tag = None
                 ):
        self.path_params = path_params
        self.assets_path = assemble_project_path(assets_path)
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.tag = tag
        self.workdir = assemble_project_path(os.path.join(workdir, tag))

        self.assets = self._init_assets()

    def _init_assets(self):
        assets = load_json(self.assets_path)
        assets = [asset["symbol"] for asset in assets]
        return assets

    def _process_price_and_features(self,
                assets = None,
                start_date = None,
                end_date = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        assets = assets if assets else self.assets

        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close"
        ]

        for asset in tqdm(assets):

            try:
                price = self.path_params["prices"][0]
                price_type = price["type"]
                price_path = price["path"]

                price_path = assemble_project_path(os.path.join(price_path, "{}.csv".format(asset)))

                if price_type == "fmp":
                    price_column_map = {
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "volume": "volume",
                        "adjClose": "adj_close",
                    }
                elif price_type == "yahoofinance":
                    price_column_map = {
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                        "Date": "timestamp",
                        "Adj Close": "adj_close",
                    }
                else:
                    price_column_map = {
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "volume": "volume",
                        "adjClose": "adj_close",
                    }

                assert os.path.exists(price_path), "Price path {} does not exist".format(price_path)
                price_df = pd.read_csv(price_path)

                price_df = price_df.rename(columns=price_column_map)[["timestamp"] + price_columns]

                price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
                price_df = price_df[(price_df["timestamp"] >= start_date) & (price_df["timestamp"] < end_date)]

                price_df = price_df.sort_values(by="timestamp")
                price_df = price_df.drop_duplicates(subset=["timestamp"], keep="first")
                price_df = price_df.reset_index(drop=True)

                outpath = os.path.join(self.workdir, "price")
                os.makedirs(outpath, exist_ok=True)
                price_df.to_csv(os.path.join(outpath, "{}.csv".format(asset)), index=False)

                features_df = cal_factor(deepcopy(price_df), level=self.interval)
                features_df = cal_target(features_df)
                outpath = os.path.join(self.workdir, "features")
                os.makedirs(outpath, exist_ok=True)
                features_df.to_csv(os.path.join(outpath, "{}.csv".format(asset)), index=False)
            except Exception as e:
                print("Error processing asset {} due to {}".format(asset, e))
                continue
    def process(self,
                assets = None,
                start_date = None,
                end_date = None):

        print(">" * 30 + "Running price and features..." + ">" * 30)
        self._process_price_and_features(assets=assets, start_date=start_date, end_date=end_date)
        print("<" * 30 + "Finish price and features..." + "<" * 30)