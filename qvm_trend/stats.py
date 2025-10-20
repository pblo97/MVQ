import numpy as np
import pandas as pd
from scipy import stats


def future_return(close: pd.Series, horizon=20):
    return close.shift(-horizon) / close - 1


def information_coefficient(df: pd.DataFrame, score_col="BreakoutScore", ret_col="ret_20"):
    s = df[[score_col, ret_col]].dropna()
    if len(s) < 10:
        return np.nan
    return stats.spearmanr(s[score_col], s[ret_col]).correlation