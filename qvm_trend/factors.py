import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class BreakoutFeatures:
    rvol20: float
    closepos: float
    p52: float
    tsmom20: float
    tsmom63: float
    ma20_slope: float
    obv_slope20: float
    adl_slope20: float
    updown_vol_ratio20: float
    rs_ma20_slope: Optional[float]
    atr_pct_rank: float
    gap_hold: bool
    float_velocity: Optional[float]


def _slope_linear(y: pd.Series) -> float:
    y = y.dropna()
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float(np.polyfit(x, y, 1)[0])


def rolling_slope(y: pd.Series, win=20) -> pd.Series:
    return y.rolling(win).apply(lambda s: _slope_linear(pd.Series(s)), raw=False)


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['close'].diff().fillna(0.0))
    return (direction * df['volume']).fillna(0.0).cumsum()


def accumulation_distribution_line(df: pd.DataFrame) -> pd.Series:
    clv = ( (df['close'] - df['low']) - (df['high'] - df['close']) ) / ((df['high'] - df['low']).replace(0, np.nan))
    clv = clv.fillna(0.0)
    return (clv * df['volume']).cumsum()


def atr(df: pd.DataFrame, n=14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def percent_rank(s: pd.Series, lookback: int) -> pd.Series:
    return s.rolling(lookback).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])


def compute_breakout_features(
    df: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    shares_float: Optional[float] = None
) -> Tuple[BreakoutFeatures, Dict[str, pd.Series]]:
    d = df.copy()
    rvol20_series = d['volume'] / d['volume'].shift(1).rolling(20).median()
    rvol20 = float(rvol20_series.iloc[-1])

    rng = (d['high'] - d['low']).replace(0, np.nan)
    closepos_series = (d['close'] - d['low']) / rng
    closepos = float(closepos_series.iloc[-1])

    p52_series = d['close'] / d['high'].rolling(252).max()
    p52 = float(p52_series.iloc[-1])

    tsmom20_series = d['close'] / d['close'].shift(20) - 1
    tsmom63_series = d['close'] / d['close'].shift(63) - 1
    tsmom20 = float(tsmom20_series.iloc[-1])
    tsmom63 = float(tsmom63_series.iloc[-1])

    ma20 = d['close'].rolling(20).mean()
    ma20_slope_series = rolling_slope(ma20, win=20)
    ma20_slope = float(ma20_slope_series.iloc[-1])

    obv = on_balance_volume(d)
    adl = accumulation_distribution_line(d)
    obv_slope20_series = rolling_slope(obv, win=20)
    adl_slope20_series = rolling_slope(adl, win=20)
    obv_slope20 = float(obv_slope20_series.iloc[-1])
    adl_slope20 = float(adl_slope20_series.iloc[-1])

    up = d['volume'].where(d['close'] > d['close'].shift(), 0.0)
    dn = d['volume'].where(d['close'] < d['close'].shift(), 0.0)
    updown_vol_ratio20_series = up.rolling(20).sum() / (dn.rolling(20).sum().replace(0, np.nan))
    updown_vol_ratio20 = float(updown_vol_ratio20_series.iloc[-1])

    rs_ma20_slope_val = None
    rs_ma20_slope_series = None
    if benchmark is not None:
        rs = (d['close'] / benchmark).dropna()
        rs_ma20 = rs.rolling(20).mean()
        rs_ma20_slope_series = rolling_slope(rs_ma20, win=20)
        rs_ma20_slope_val = float(rs_ma20_slope_series.iloc[-1])

    atr14 = atr(d, 14)
    atr_pct_rank_series = percent_rank(atr14, lookback=252)
    atr_pct_rank_val = float(atr_pct_rank_series.iloc[-1])

    prev_high = d['high'].shift()
    gap = d['open'] > prev_high
    gap_hold_series = gap & (closepos_series >= 0.6) & (rvol20_series >= 1.5)
    gap_hold = bool(gap_hold_series.iloc[-1])

    float_velocity_val = None
    med_dollar_vol_60 = None
    if shares_float is not None and shares_float > 0:
        med_dollar_vol_60 = (d['close']*d['volume']).rolling(60).median()
        float_velocity_series = med_dollar_vol_60 / (shares_float * d['close'])
        float_velocity_val = float(float_velocity_series.iloc[-1])

    features = BreakoutFeatures(
        rvol20=rvol20,
        closepos=closepos,
        p52=p52,
        tsmom20=tsmom20,
        tsmom63=tsmom63,
        ma20_slope=ma20_slope,
        obv_slope20=obv_slope20,
        adl_slope20=adl_slope20,
        updown_vol_ratio20=updown_vol_ratio20,
        rs_ma20_slope=rs_ma20_slope_val,
        atr_pct_rank=atr_pct_rank_val,
        gap_hold=gap_hold,
        float_velocity=float_velocity_val
    )
    series_map = {
        "rvol20": rvol20_series,
        "closepos": closepos_series,
        "p52": p52_series,
        "tsmom20": tsmom20_series,
        "tsmom63": tsmom63_series,
        "ma20": ma20,
        "ma20_slope": ma20_slope_series,
        "obv": obv,
        "adl": adl,
        "obv_slope20": obv_slope20_series,
        "adl_slope20": adl_slope20_series,
        "updown_vol_ratio20": updown_vol_ratio20_series,
        "atr14": atr14,
        "atr_pct_rank": atr_pct_rank_series,
        "gap_hold": gap_hold_series
    }
    if rs_ma20_slope_series is not None:
        series_map["rs_ma20_slope"] = rs_ma20_slope_series
    if shares_float is not None and shares_float > 0 and med_dollar_vol_60 is not None:
        series_map["float_velocity"] = med_dollar_vol_60 / (shares_float * d['close'])
    return features, series_map
