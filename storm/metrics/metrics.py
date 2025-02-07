import numpy as np
from typing import List

def ARR(ret):
    res = (np.cumprod(ret + 1.0)[-1] - 1.0) / ret.shape[0] * 252
    return res

def VOL(ret):
    res = np.std(ret)
    return res

def DD(ret):
    res = np.std(ret[np.where(ret<0, True, False)])
    return res

def MDD(ret):
    iter_ret = np.cumprod(ret + 1.0)
    peak = iter_ret[0]
    mdd = 0
    for value in iter_ret:
        if value > peak:
            peak =value
        dd = (peak - value)/peak
        if dd > mdd:
            mdd =dd
    return mdd

def SR(ret):
    res = 1.0 * np.mean(ret) * np.sqrt(ret.shape[0]) / np.std(ret)
    return res

def CR(ret, mdd):
    res = np.mean(ret) * 252 / mdd
    return res

def SOR(ret, dd):
    res = 1.0 * np.mean(ret) * 252 / dd
    return res

def NumTrades(actions: np.ndarray) -> int:
    # Count all actions that are not 'hold' (1), i.e., 'sell' (0) or 'buy' (2)
    actions = actions.astype(int)
    res = np.count_nonzero(actions != 1)
    return res

def NumBuys(actions: np.ndarray) -> int:
    # Count all 'buy' (2) actions
    actions = actions.astype(int)
    res = np.count_nonzero(actions == 2)
    return res

def NumSells(actions: np.ndarray) -> int:
    # Count all 'sell' (0) actions
    actions = actions.astype(int)
    res = np.count_nonzero(actions == 0)
    return res

def AvgHoldPeriod(actions: np.ndarray) -> float:
    # Treat hold as 1, buy as 2, and sell as 0
    positions = np.where(actions == 1, 1, np.where(actions == 2, -1, 0))
    cumulative_position = np.cumsum(positions)
    # Identify where cumulative positions change, indicating the start or end of a holding period
    holding_periods = np.diff(np.flatnonzero(np.diff(cumulative_position) != 0))
    # Calculate the average holding period
    average_holding_period = holding_periods.mean() if holding_periods.size > 0 else 0
    return average_holding_period

def TurnoverRate(positions: np.ndarray) -> float:
    # Calculate turnover rate based on position changes
    position_changes = np.abs(np.diff(positions))
    total_position_change = position_changes.sum()
    average_position = positions.mean()

    turnover_rate = total_position_change / average_position if average_position != 0 else 0
    return turnover_rate

def ActivityRate(actions: np.ndarray) -> float:
    # Calculate activity rate as proportion of non-hold (1) actions
    actions = actions.astype(int)
    res = np.count_nonzero(actions != 1) / actions.size
    return res

def AvgTradeInterval(actions: np.ndarray) -> float:
    # Calculate the average interval between trades (non-hold actions)
    actions = actions.astype(int)
    trade_indices = np.flatnonzero(actions != 1)
    trade_intervals = np.diff(trade_indices)
    average_trade_interval = trade_intervals.mean() if trade_intervals.size > 0 else 0
    return average_trade_interval

def BuyToSellRatio(actions: np.ndarray) -> float:
    # Calculate buy-to-sell ratio, where buy is 2 and sell is 0
    actions = actions.astype(int)
    buy_count = np.count_nonzero(actions == 2)
    sell_count = np.count_nonzero(actions == 0)
    buy_to_sell_ratio = buy_count / sell_count if sell_count != 0 else float('inf')
    return buy_to_sell_ratio

