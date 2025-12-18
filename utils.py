from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


###########################################################################################
###################### Utilities for Extreme Value Theory Analysis ########################
###########################################################################################
class EmpiricalDistribution:
    def __init__(self, data, smoothing=1e-10):
        self.data = np.sort(np.asarray(data))
        self.n = len(self.data)

        self.ecdf_y = np.linspace(0, 1, self.n, endpoint=False) + 1 / self.n
        self.ecdf_x = self.data

        self._cdf = interp1d(self.ecdf_x, self.ecdf_y, kind="linear",
                             bounds_error=False, fill_value=(0.0, 1.0), assume_sorted=True)
        self._ppf = interp1d(self.ecdf_y, self.ecdf_x, kind="linear",
                             bounds_error=False,
                             fill_value=(self.ecdf_x[0], self.ecdf_x[-1]),
                             assume_sorted=True)

        dx = np.diff(self.ecdf_x)
        dy = np.diff(self.ecdf_y)
        pdf_vals = np.zeros_like(self.ecdf_x)
        pdf_vals[1:] = dy / (dx + smoothing)
        pdf_vals[0] = pdf_vals[1]

        self._pdf = interp1d(self.ecdf_x, pdf_vals, kind="linear",
                             bounds_error=False, fill_value=0.0, assume_sorted=True)

    def cdf(self, x): return self._cdf(x)
    def pdf(self, x): return self._pdf(x)
    def ppf(self, q): return self._ppf(q)


def fit_evt(dist: EmpiricalDistribution, n: int | np.ndarray | pd.Series):
    # Fitting Extreme Value theory using Gumbel domain
    b_n = dist.ppf(1 - 1 / n)
    a_n = 1 / (n * dist.pdf(b_n))
    expected = b_n + a_n * 0.5772156649  # Euler-Mascheroni constant
    cdf_func = lambda x: np.exp(-np.exp(-(x - b_n) / a_n))
    return a_n, b_n, expected, cdf_func


###########################################################################################
############################## Utilities for Data Processing ##############################
###########################################################################################
def add_time(df, **kwargs):
    df = df.merge(
        df.reset_index().groupby("date")["time"].min().rename("open_time"),
        left_on="date",
        right_index=True,
    )
    df = df.merge(
        df.reset_index().groupby("date")["time"].max().rename("close_time"),
        left_on="date",
        right_index=True,
    )
    for k, v in kwargs.items():
        df[k] = (
            df["open_time"] + timedelta(minutes=int(v[1:]))
            if v[0] == "+"
            else df["close_time"] - timedelta(minutes=int(v[1:]))
        )
    return df


def resample(df, freq="5min"):
    _ = df.resample(freq, origin="end_day")
    D = _["date"].first()
    O = _["open"].first()
    H = _["high"].max()
    L = _["low"].min()
    C = _["close"].last()
    V = _["volume"].sum()
    df = pd.concat([D, O, H, L, C, V], axis=1)
    return df
