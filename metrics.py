from datetime import datetime, timedelta
from pytz import timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MetricCalculator:
    def __init__(
        self, data, time_freq="M", sharpe_quantiles=[], start_time=None, end_time=None
    ):
        assert time_freq.upper() in ["D", "M"]
        self.data = (
            data.squeeze()
            .truncate(before=start_time, after=end_time)
            .dropna(how="all")
            .asfreq("B")
            .fillna(0)
        )

        self.time_freq = time_freq.upper().replace("M", "ME")
        self.sharpe_quantiles = sharpe_quantiles
        self.clean_return = (
            self.data.copy()
            if self.time_freq == "D"
            else self.data.resample(self.time_freq).agg(lambda x: (x + 1).prod() - 1)
        )

        self.T = 252 if self.time_freq == "D" else 12
        self.sqrt_T = np.sqrt(self.T)
        self.AUM = (1 + self.clean_return).cumprod()
        self.dd = self.AUM / self.AUM.cummax() - 1

        self.rolling_sharpe_cache = {}

    def rolling_sharpe(self, window):
        if window not in self.rolling_sharpe_cache:
            ret = (self.AUM / self.AUM.shift(window)) ** (self.T / window) - 1
            std = self.clean_return.rolling(window).std() * self.sqrt_T
            self.rolling_sharpe_cache[window] = ret / std
        return self.rolling_sharpe_cache[window]

    def compute_stat(
        self, window: int = 252, plot=False, start_time=None, end_time=None
    ):
        if window == 0:
            window = self.T * 100
        if start_time is None:
            start_time = self.data.index.min()
        if end_time is None:
            end_time = self.data.index.max()
        if self.time_freq == "D":
            ret = self.clean_return.loc[start_time:end_time]
        else:
            ret = self.clean_return.loc[
                start_time.strftime("%Y-%m") : end_time.strftime("%Y-%m")
            ]
        annualized_return = (ret + 1).prod() ** (self.T / len(ret)) - 1
        annualized_volatility = ret.std() * self.sqrt_T
        annualized_downside_std = np.sqrt(
            (ret.clip(upper=0) ** 2).sum() * self.T / len(ret)
        )
        sharpe_ratio = annualized_return / annualized_volatility
        sortino_ratio = annualized_return / annualized_downside_std
        rolling_sharpe = self.rolling_sharpe(window).loc[start_time:end_time]
        min_rolling_sharpe = rolling_sharpe.min()
        max_drawdown = self.dd.loc[start_time:end_time].min()
        smdd = max_drawdown / annualized_volatility
        res = {
            "start_time": start_time,
            "end_time": end_time,
            "annualized_return": round(annualized_return * 100, 2),
            "annualized_volatility": round(annualized_volatility * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "min_rolling_sharpe": round(min_rolling_sharpe, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "standardized_max_drawdown": round(smdd, 2),
        }
        for q in self.sharpe_quantiles:
            res[f"qsharpe_{q}"] = round(rolling_sharpe.quantile(q), 2)

        if plot:
            (
                self.AUM.loc[start_time:end_time]
                / self.AUM.loc[start_time:end_time].iloc[0]
            ).plot()
            plt.grid(True)
            plt.show()
        if isinstance(self.data, pd.Series):
            res = [res]
        return pd.DataFrame(res)


class BacktestMetricCalculator:
    def __init__(
        self,
        data,
        time_freq="M",
        sub_period: int = 3,
        sharpe_quantiles: list = [],
        additional_periods: list[tuple] = [],
    ):
        data = data.squeeze()
        self.data = data
        self.time_freq = time_freq
        self.sub_period = sub_period
        self.sharpe_quantiles = sharpe_quantiles
        self.additional_periods = additional_periods

    def summarize(self, plot=False):
        metric = MetricCalculator(self.data, self.time_freq, self.sharpe_quantiles)
        data = metric.data
        roll_window = self.sub_period * metric.T
        res = [metric.compute_stat(roll_window, plot)]

        if self.sub_period:
            last_year = data.index.year.max()
            first_year = data.index.year.min()
            while True:
                start_year = last_year - self.sub_period + 1
                sub = data.loc[str(start_year) : str(last_year)].index
                res.append(metric.compute_stat(roll_window, plot, sub[0], sub[-1]))
                if start_year <= first_year:
                    break
                else:
                    last_year = start_year - 1
        for start_time, end_time in self.additional_periods:
            sub = data.loc[start_time:end_time].index
            res.append(metric.compute_stat(roll_window, plot, sub[0], sub[-1]))

        return pd.concat(res, axis=0)
