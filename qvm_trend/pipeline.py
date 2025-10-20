# qvm_trend/pipeline.py
import numpy as np
import pandas as pd
from typing import Dict, List
from .data_io import load_prices_panel, load_benchmark, get_prices_fmp, DEFAULT_START, DEFAULT_END
from .factors import compute_breakout_features
from .scoring import breakout_score, entry_signal, DEFAULT_TH

def _ma(s: pd.Series, n=200): return s.rolling(n, min_periods=n).mean()
def _mom_12_1(s: pd.Series):  return s.shift(21) / s.shift(252) - 1

def apply_trend_filter(panel: Dict[str, pd.DataFrame], use_and=False) -> List[str]:
    import numpy as np
    elig = []
    for sym, df in panel.items():
        if df is None or df.empty: 
            continue
        close = df['close']
        ma200 = _ma(close, 200)
        mom   = _mom_12_1(close)

        last_close = float(close.dropna().iloc[-1]) if not close.dropna().empty else np.nan
        last_ma200 = float(ma200.dropna().iloc[-1]) if not ma200.dropna().empty else np.nan
        last_mom   = float(mom.dropna().iloc[-1])   if not mom.dropna().empty   else np.nan

        cond_ma = (not np.isnan(last_ma200)) and (last_close > last_ma200)
        cond_mo = (not np.isnan(last_mom))   and (last_mom > 0)

        pass_sig = (cond_ma and cond_mo) if use_and else (cond_ma or cond_mo)
        if pass_sig: 
            elig.append(sym)
    return elig

def enrich_with_breakout(df_sel: pd.DataFrame, price_map: Dict[str, pd.DataFrame],
                         benchmark_series=None, float_map=None, th=DEFAULT_TH, min_score=0.6) -> pd.DataFrame:
    rows = []
    for sym in df_sel["symbol"].tolist():
        dfp = price_map.get(sym)
        if dfp is None or dfp.empty: continue
        bench = benchmark_series.reindex(dfp.index).ffill() if benchmark_series is not None else None
        shares_float = float_map.get(sym) if float_map is not None else None
        feat, _ = compute_breakout_features(dfp, benchmark=bench, shares_float=shares_float)
        score, tests = breakout_score(feat, th)
        signal = entry_signal(score, tests, min_score=min_score)
        rows.append({
            "symbol": sym, "BreakoutScore": score, "EntrySignal": signal,
            "RVOL20": feat.rvol20, "ClosePos": feat.closepos, "P52": feat.p52,
            "TSMOM20": feat.tsmom20, "TSMOM63": feat.tsmom63, "MA20_slope": feat.ma20_slope,
            "OBV_slope20": feat.obv_slope20, "ADL_slope20": feat.adl_slope20,
            "UDVolRatio20": feat.updown_vol_ratio20, "RS_MA20_slope": feat.rs_ma20_slope,
            "ATR_pct_rank": feat.atr_pct_rank, "GapHold": feat.gap_hold, "FloatVelocity": feat.float_velocity
        })
    out = pd.DataFrame(rows)
    return df_sel.merge(out, on="symbol", how="left")




