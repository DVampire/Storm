import os
import json
import signal
import time
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
from typing import List, Optional, Tuple
from multiprocessing import Pool
from pandas_market_calendars import get_calendar

from dotenv import load_dotenv
load_dotenv(verbose=True)

from storm.downloader.custom import AbstractDownloader
from storm.utils import get_jsonparsed_data, generate_intervals, assemble_project_path
from storm.log import logger
from storm.registry import DOWNLOADER

NYSE = get_calendar("XNYS")

class FMPPriceDownloader(AbstractDownloader):
    def __init__(self,
                 api_key: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 symbol: Optional[str] = None,
                 exp_path: Optional[str] = None,
                 **kwargs):
        super().__init__()

        self.api_key = api_key

        if self.api_key is None:
            self.api_key = os.getenv("FMP_API_KEY")

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

        self.request_url = "https://financialmodelingprep.com/api/v3/historical-price-full/{}?from={}&to={}&apikey={}"

        self.exp_path = exp_path
        os.makedirs(self.exp_path, exist_ok=True)

    def _check_download(self,
                              symbol: Optional[str] = None,
                              intervals: Optional[List[Tuple[datetime, datetime]]] = None):

        download_infos = []

        for (start, end) in intervals:
            name = "{}".format(start.strftime("%Y-%m-%d"))
            if os.path.exists(os.path.join(self.exp_path, symbol, name)):
                item = {
                    "name": name,
                    "downloaded": True,
                    "start": start,
                    "end": end
                }
            else:
                item = {
                    "name": name,
                    "downloaded": False,
                    "start": start,
                    "end": end
                }
            download_infos.append(item)

        downloaded_items_num = len([info for info in download_infos if info["downloaded"]])
        total_items_num = len(download_infos)

        logger.info(f"Downloaded / Total: [{downloaded_items_num} / {total_items_num}]")

        return download_infos

    def run(self,
              start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              symbol: Optional[str] = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")
        symbol = symbol if symbol else self.symbol

        intervals = generate_intervals(start_date, end_date, "year")

        download_infos = self._check_download(
            symbol=symbol,
            intervals=intervals,
        )

        slice_dir = os.path.join(self.exp_path, symbol)
        os.makedirs(slice_dir, exist_ok=True)

        df = pd.DataFrame()

        bar_format = f"Download {symbol} Prices:" + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"
        for info in tqdm(download_infos, bar_format=bar_format):

            name = info["name"]
            downloaded = info["downloaded"]
            start = info["start"]
            end = info["end"]

            is_trading_day = NYSE.valid_days(start_date=start, end_date=end).size > 0
            if is_trading_day:
                if downloaded:
                    chunk_df = pd.read_csv(os.path.join(slice_dir, "{}.csv".format(name)))
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
                        symbol,
                        start.strftime("%Y-%m-%d"),
                        end.strftime("%Y-%m-%d"),
                        self.api_key)

                    try:
                        time.sleep(1)
                        aggs = get_jsonparsed_data(request_url)
                        aggs = aggs["historical"] if "historical" in aggs else []
                    except Exception as e:
                        logger.error(e)
                        aggs = []

                    if len(aggs) == 0:
                        logger.error(f"No prices for {name}")
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

                    chunk_df = pd.DataFrame(chunk_df, index=range(len(chunk_df["timestamp"])))
                    chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"]).apply(
                        lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

                    chunk_df = chunk_df[["timestamp"] + [col for col in chunk_df.columns if col != "timestamp"]]
                    chunk_df.to_csv(os.path.join(slice_dir, "{}.csv".format(name)), index=False)

                df = pd.concat([df, chunk_df], axis=0)

        df = df.sort_values(by="timestamp", ascending=True)
        df = df[["timestamp"] + [col for col in df.columns if col != "timestamp"]]
        df.to_csv(os.path.join(self.exp_path, "{}.csv".format(symbol)), index=False)

class FMPNewsDownloader(AbstractDownloader):
    def __init__(self,
                 api_key: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 symbol: Optional[str] = None,
                 exp_path: Optional[str] = None,
                 max_pages: int = 100,
                 **kwargs):
        super().__init__()
        self.api_key = api_key

        if self.api_key is None:
            self.api_key = os.getenv("FMP_API_KEY")

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.max_pages = max_pages

        self.request_url = "https://financialmodelingprep.com/api/v3/stock_news?tickers={}&page={}&limit=100&from={}&to={}&apikey={}"

        self.exp_path = exp_path
        os.makedirs(self.exp_path, exist_ok=True)

    def _check_download(self,
                              symbol: Optional[str] = None,
                              intervals: Optional[List[Tuple[datetime, datetime]]] = None):

        download_infos = []

        for (start, end) in intervals:
            for page in range(1, self.max_pages + 1):
                name = "{}_page_{:04d}".format(start.strftime("%Y-%m-%d"), page)
                if os.path.exists(os.path.join(self.exp_path, symbol, name)):
                    item = {
                        "name": name,
                        "downloaded": True,
                        "start": start,
                        "end": end,
                        "page": page
                    }
                else:
                    item = {
                        "name": name,
                        "downloaded": False,
                        "start": start,
                        "end": end,
                        "page": page
                    }
                download_infos.append(item)

        downloaded_items_num = len([info for info in download_infos if info["downloaded"]])
        total_items_num = len(download_infos)

        logger.info(f"Downloaded / Total: [{downloaded_items_num} / {total_items_num}]")

        return download_infos

    def run(self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            symbol: Optional[str] = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")
        symbol = symbol if symbol else self.symbol

        intervals = generate_intervals(start_date, end_date, "year")

        download_infos = self._check_download(
            symbol=symbol,
            intervals=intervals,
        )

        slice_dir = os.path.join(self.exp_path, symbol)
        os.makedirs(slice_dir, exist_ok=True)

        df = pd.DataFrame()

        bar_format = f"Download {symbol} News:" + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"

        for info in tqdm(download_infos, bar_format=bar_format):

            name = info["name"]
            downloaded = info["downloaded"]
            start = info["start"]
            end = info["end"]
            page = info["page"]

            is_trading_day = NYSE.valid_days(start_date=start, end_date=end).size > 0
            if is_trading_day:

                if downloaded:
                    chunk_df = pd.read_csv(os.path.join(slice_dir, "{}.csv".format(name)))
                else:
                    chunk_df = {
                        "symbol": [],
                        "publishedDate": [],
                        "title": [],
                        "image": [],
                        "site": [],
                        "text": [],
                        "url": []
                    }

                    request_url = self.request_url.format(
                        symbol,
                        page,
                        start.strftime("%Y-%m-%d"),
                        end.strftime("%Y-%m-%d"),
                        self.api_key)
                    try:
                        time.sleep(1)
                        aggs = get_jsonparsed_data(request_url)
                    except Exception as e:
                        logger.error(e)
                        aggs = []

                    if len(aggs) == 0:
                        logger.error(f"No news for {name}")
                        continue

                    for a in aggs:
                        chunk_df["symbol"].append(a["symbol"])
                        chunk_df["publishedDate"].append(a["publishedDate"])
                        chunk_df["title"].append(a["title"])
                        chunk_df["image"].append(a["image"])
                        chunk_df["site"].append(a["site"])
                        chunk_df["text"].append(a["text"])
                        chunk_df["url"].append(a["url"])

                    chunk_df = pd.DataFrame(chunk_df, index=range(len(chunk_df["publishedDate"])))
                    chunk_df["timestamp"] = pd.to_datetime(chunk_df["publishedDate"]).apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

                    chunk_df = chunk_df[["timestamp"] + [col for col in chunk_df.columns if col != "timestamp"]]
                    chunk_df.to_csv(os.path.join(slice_dir, "{}.csv".format(name)), index=False)

                df = pd.concat([df, chunk_df], axis=0)

        df = df.sort_values(by="timestamp", ascending=True)
        df = df.drop_duplicates(subset=["publishedDate", "title"], keep="first")
        df = df[["timestamp"] + [col for col in df.columns if col != "timestamp"]]
        df.to_csv(os.path.join(self.exp_path, "{}.csv".format(symbol)), index=False)

@DOWNLOADER.register_module(force=True)
class Downloader(AbstractDownloader):
    def __init__(self,
                 assets_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 exp_path: Optional[str] = None,
                 batch_size: int = 10,
                 ):
        super().__init__()

        self.assets_path = assemble_project_path(assets_path)
        self.start_date = start_date
        self.end_date = end_date

        self.assets_info, self.symbols = self._load_assets()

        assert len(self.symbols) > 0, "No symbols to download"
        if len(self.symbols) <= batch_size:
            batch_size = len(self.symbols)
        self.batch_size = batch_size

        self.exp_path = assemble_project_path(exp_path)
        os.makedirs(self.exp_path, exist_ok=True)

    def _load_assets(self):
        """
        Load assets from the assets file.
        :return:
        """
        with open(self.assets_path) as f:
            assets_info = json.load(f)
        symbols = [asset for asset in assets_info]
        logger.info(f"Loaded {len(symbols)} assets from {self.assets_path}")
        return assets_info, symbols

    @staticmethod
    def _run_task(downloader):
        """Static method to run a single downloader task."""
        downloader.run()

    def _download_fmp_price(self, save_dir: str):
        """
        Download price data from FMP API.
        :return:
        """

        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for symbol in self.symbols:
            downloader = FMPPriceDownloader(
                api_key=os.getenv("FMP_API_KEY"),
                start_date=self.start_date,
                end_date=self.end_date,
                symbol=symbol,
                exp_path=save_dir
            )
            tasks.append(downloader)

        with Pool(self.batch_size) as p:
            p.map(self._run_task, tasks)

    def _download_fmp_news(self, save_dir: str):
        """
        Download news data from FMP API.
        :return:
        """
        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for symbol in self.symbols:
            downloader = FMPNewsDownloader(
                api_key=os.getenv("FMP_API_KEY"),
                start_date=self.start_date,
                end_date=self.end_date,
                symbol=symbol,
                exp_path=save_dir
            )
            tasks.append(downloader)

        with Pool(self.batch_size) as p:
            p.map(self._run_task, tasks)

    def run(self):
        save_dir = os.path.join(self.exp_path, "price")
        self._download_fmp_price(save_dir = save_dir)

        news_dir = os.path.join(self.exp_path, "news")
        os.makedirs(news_dir, exist_ok=True)

        save_dir = os.path.join(news_dir, "fmp")
        self._download_fmp_news(save_dir = save_dir)

