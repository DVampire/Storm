import os
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
import time
import signal
from pandas_market_calendars import get_calendar
from urllib.request import urlopen
import certifi
import json
from dotenv import load_dotenv
load_dotenv(verbose=True)

from storm.registry import DOWNLOADER
from storm.downloader.custom import Downloader
from storm.utils import generate_intervals
from storm.utils import load_json
from storm.utils import assemble_project_path

class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Time out")

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

NYSE = get_calendar('XNYS')

@DOWNLOADER.register_module(force=True)
class FMPDayPriceDownloader(Downloader):
    def __init__(self,
                 token: str = None,
                 delay: int = 1,
                 start_date: str = "2023-04-01",
                 end_date: str = "2023-04-01",
                 interval: str = "minute",
                 assets_path: str = None,
                 workdir: str = "",
                 tag: str = "",
                 **kwargs):

        self.token = token if token is not None else os.environ.get("OA_FMP_KEY")
        self.delay = delay
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.assets_path = assemble_project_path(assets_path)
        self.tag = tag
        self.workdir = assemble_project_path(os.path.join(workdir, tag))

        self.log_path = os.path.join(self.workdir, "{}.txt".format(tag))

        with open(self.log_path, "w") as op:
            op.write("")

        self.assets = self._init_assets()

        self.request_url = "https://financialmodelingprep.com/api/v3/historical-price-full/{}?from={}&to={}&apikey={}"

        super().__init__(**kwargs)

    def _init_assets(self):
        assets = load_json(self.assets_path)
        assets = [asset["symbol"] for asset in assets]
        return assets

    def check_download(self,
                       assets = None,
                       start_date = None,
                       end_date = None):
        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")
        assets = assets if assets else self.assets

        intervals = generate_intervals(start_date, end_date, "year")

        failed_assets = []

        total_count = 0
        total_asset_count = 0

        for asset in assets:
            count = 0
            asset_count = 0

            for item in intervals:

                name = item["name"]

                if os.path.exists(os.path.join(self.workdir, asset, f"{name}.csv")):
                    count += 1
                    total_count += 1
                asset_count += 1
                total_asset_count += 1

            if count != asset_count:
                failed_assets.append(asset)

            print("{}: {}/{}".format(asset, count, asset_count))

        print("Total: {}/{}, failed {}/{}".format(total_count, total_asset_count, total_asset_count - total_count, total_asset_count))

        return failed_assets

    def download(self,
                 assets = None,
                 start_date = None,
                 end_date = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        assets = assets if assets else self.assets

        intervals = generate_intervals(start_date, end_date, "year")

        for asset in assets:

            os.makedirs(os.path.join(self.workdir, asset), exist_ok=True)

            df = pd.DataFrame()

            for item in tqdm(intervals, bar_format="Download {} Prices:".format(asset) + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):

                name = item["name"]
                start = item["start"]
                end = item["end"]

                is_trading_day = NYSE.valid_days(start_date=start, end_date=end).size > 0
                if is_trading_day:
                    if os.path.exists(os.path.join(self.workdir, asset, f"{name}.csv")):
                        chunk_df = pd.read_csv(os.path.join(self.workdir, asset, f"{name}.csv"))
                    else:
                        chunk_df = {
                            "open": [],
                            "high": [],
                            "low": [],
                            "close": [],
                            "volume": [],
                            "timestamp": [],
                            "adjClose": [],
                            "unadjustedVolume": [],
                            "change": [],
                            "changePercent": [],
                            "vwap": [],
                            "label": [],
                            "changeOverTime": []
                        }

                        request_url = self.request_url.format(
                            asset,
                            start.strftime("%Y-%m-%d"),
                            end.strftime("%Y-%m-%d"),
                            self.token)

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)

                        try:
                            time.sleep(self.delay)
                            aggs = get_jsonparsed_data(request_url)
                            aggs = aggs["historical"] if "historical" in aggs else []
                            signal.alarm(0)
                        except TimeoutException:
                            print("Time out")
                            aggs = []

                        if len(aggs) == 0:
                            with open(self.log_path, "a") as op:
                                op.write("{},{}\n".format(asset, start.strftime("%Y-%m-%d")))
                            continue

                        for a in aggs:

                            chunk_df["open"].append(a["open"])
                            chunk_df["high"].append(a["high"])
                            chunk_df["low"].append(a["low"])
                            chunk_df["close"].append(a["close"])
                            chunk_df["volume"].append(a["volume"])
                            chunk_df["timestamp"].append(a["date"])
                            chunk_df["adjClose"].append(a["adjClose"])
                            chunk_df["unadjustedVolume"].append(a["unadjustedVolume"])
                            chunk_df["change"].append(a["change"])
                            chunk_df["changePercent"].append(a["changePercent"])
                            chunk_df["vwap"].append(a["vwap"])
                            chunk_df["label"].append(a["label"])
                            chunk_df["changeOverTime"].append(a["changeOverTime"])

                        chunk_df = pd.DataFrame(chunk_df,index=range(len(chunk_df["timestamp"])))
                        if chunk_df.shape[0] > 0:
                            chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"]).apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
                            chunk_df = chunk_df.sort_values(by="timestamp", ascending=True)
                            chunk_df.to_csv(os.path.join(self.workdir, asset, f"{name}.csv"), index=False)
                        else:
                            print("No data for {}, name: {}".format(asset, name))

                    df = pd.concat([df, chunk_df], axis=0)

            if df.shape[0] > 0:
                df = df.sort_values(by="timestamp", ascending=True)
                df.to_csv(os.path.join(self.workdir, "{}.csv".format(asset)), index=False)
            else:
                print("No data for {}".format(asset))