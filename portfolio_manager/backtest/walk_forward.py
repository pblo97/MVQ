# portfolio_manager/backtest/walk_forward.py
"""
Walk-Forward Backtest Framework

Implements rigorous out-of-sample validation:
1. Rolling window: Train on historical, test on future
2. Expanding window: Cumulative training, test on next period
3. Cross-validation: Multiple train/test splits

Academic Foundation:
- Bailey et al. (2014): "The Probability of Backtest Overfitting"
- Harvey et al. (2016): "... and the Cross-Section of Expected Returns"
- López de Prado (2018): "Advances in Financial Machine Learning"

Critical for avoiding:
- Overfitting (strategy looks good in-sample but fails out-of-sample)
- Data snooping (testing many strategies, publishing only winners)
- Look-ahead bias (using future information)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    weights_history: pd.DataFrame
    metrics: Dict[str, float]
    train_periods: List[Tuple[str, str]]
    test_periods: List[Tuple[str, str]]


def walk_forward_backtest(
    returns_df: pd.DataFrame,
    strategy_func: Callable,
    train_window: int = 756,  # 3 years daily
    test_window: int = 63,    # 3 months daily
    step_size: int = 21,      # 1 month step
    min_train_obs: int = 252,
    benchmark_weights: Optional[np.ndarray] = None,
    **strategy_kwargs
) -> BacktestResult:
    """
    Walk-forward backtest with rolling windows.

    Process:
    1. Train on [t-train_window, t]
    2. Test on [t+1, t+test_window]
    3. Step forward by step_size
    4. Repeat until end of data

    Args:
        returns_df: DataFrame of asset returns (index = dates, columns = assets)
        strategy_func: Function that takes (returns_train, **kwargs) → weights
        train_window: Training window size (days)
        test_window: Test window size (days)
        step_size: Step size between windows (days)
        min_train_obs: Minimum training observations required
        benchmark_weights: Benchmark weights (if None, uses equal weight)
        **strategy_kwargs: Additional arguments for strategy_func

    Returns:
        BacktestResult object with metrics and history
    """
    returns_df = returns_df.dropna(how='all')

    if len(returns_df) < train_window + test_window:
        raise ValueError(f"Insufficient data: need {train_window + test_window}, have {len(returns_df)}")

    # Benchmark (equal weight if not specified)
    if benchmark_weights is None:
        benchmark_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

    # Storage
    strategy_returns_list = []
    benchmark_returns_list = []
    weights_history_list = []
    train_periods = []
    test_periods = []

    # Walk forward
    start_idx = train_window
    end_idx = len(returns_df)

    while start_idx + test_window <= end_idx:
        # Train period
        train_start_idx = max(0, start_idx - train_window)
        train_end_idx = start_idx

        train_data = returns_df.iloc[train_start_idx:train_end_idx]

        if len(train_data) < min_train_obs:
            # Skip if insufficient training data
            start_idx += step_size
            continue

        # Test period
        test_start_idx = start_idx
        test_end_idx = min(start_idx + test_window, end_idx)

        test_data = returns_df.iloc[test_start_idx:test_end_idx]

        if test_data.empty:
            break

        # Train strategy
        try:
            weights = strategy_func(train_data, **strategy_kwargs)

            # Ensure weights are np.array and match columns
            if isinstance(weights, pd.Series):
                weights = weights.reindex(returns_df.columns, fill_value=0.0).values
            elif isinstance(weights, dict):
                weights = np.array([weights.get(col, 0.0) for col in returns_df.columns])

            # Normalize weights
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

        except Exception as e:
            print(f"Warning: Strategy failed in period {test_data.index[0]} - {test_data.index[-1]}: {e}")
            # Fallback to equal weight
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

        # Evaluate on test period
        strategy_ret = (test_data.values * weights).sum(axis=1)
        benchmark_ret = (test_data.values * benchmark_weights).sum(axis=1)

        strategy_returns_list.extend(strategy_ret)
        benchmark_returns_list.extend(benchmark_ret)

        # Store weights
        for date in test_data.index:
            weights_history_list.append({
                'date': date,
                **{col: w for col, w in zip(returns_df.columns, weights)}
            })

        # Store periods
        train_periods.append((
            train_data.index[0].strftime('%Y-%m-%d'),
            train_data.index[-1].strftime('%Y-%m-%d')
        ))
        test_periods.append((
            test_data.index[0].strftime('%Y-%m-%d'),
            test_data.index[-1].strftime('%Y-%m-%d')
        ))

        # Step forward
        start_idx += step_size

    # Combine results
    test_dates = [d for period in test_periods for d in returns_df.loc[period[0]:period[1]].index]

    strategy_returns = pd.Series(strategy_returns_list, index=test_dates)
    benchmark_returns = pd.Series(benchmark_returns_list, index=test_dates)

    weights_history = pd.DataFrame(weights_history_list).set_index('date')

    # Calculate metrics
    metrics = calculate_backtest_metrics(strategy_returns, benchmark_returns)

    return BacktestResult(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        weights_history=weights_history,
        metrics=metrics,
        train_periods=train_periods,
        test_periods=test_periods
    )


def expanding_window_backtest(
    returns_df: pd.DataFrame,
    strategy_func: Callable,
    initial_train_window: int = 756,
    test_window: int = 63,
    step_size: int = 21,
    benchmark_weights: Optional[np.ndarray] = None,
    **strategy_kwargs
) -> BacktestResult:
    """
    Expanding window backtest (cumulative training).

    Difference from walk-forward:
    - Training window expands over time (uses all historical data)
    - Test window stays constant

    More realistic for strategies that benefit from longer history.

    Args:
        returns_df: DataFrame of asset returns
        strategy_func: Strategy function
        initial_train_window: Initial training window size
        test_window: Test window size
        step_size: Step size
        benchmark_weights: Benchmark weights
        **strategy_kwargs: Strategy arguments

    Returns:
        BacktestResult object
    """
    returns_df = returns_df.dropna(how='all')

    if benchmark_weights is None:
        benchmark_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

    strategy_returns_list = []
    benchmark_returns_list = []
    weights_history_list = []
    train_periods = []
    test_periods = []

    start_idx = initial_train_window
    end_idx = len(returns_df)

    while start_idx + test_window <= end_idx:
        # Train period (from beginning to current)
        train_data = returns_df.iloc[:start_idx]

        # Test period
        test_start_idx = start_idx
        test_end_idx = min(start_idx + test_window, end_idx)
        test_data = returns_df.iloc[test_start_idx:test_end_idx]

        if test_data.empty:
            break

        # Train strategy
        try:
            weights = strategy_func(train_data, **strategy_kwargs)

            if isinstance(weights, pd.Series):
                weights = weights.reindex(returns_df.columns, fill_value=0.0).values
            elif isinstance(weights, dict):
                weights = np.array([weights.get(col, 0.0) for col in returns_df.columns])

            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

        except Exception as e:
            print(f"Warning: Strategy failed: {e}")
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

        # Evaluate
        strategy_ret = (test_data.values * weights).sum(axis=1)
        benchmark_ret = (test_data.values * benchmark_weights).sum(axis=1)

        strategy_returns_list.extend(strategy_ret)
        benchmark_returns_list.extend(benchmark_ret)

        for date in test_data.index:
            weights_history_list.append({
                'date': date,
                **{col: w for col, w in zip(returns_df.columns, weights)}
            })

        train_periods.append((
            train_data.index[0].strftime('%Y-%m-%d'),
            train_data.index[-1].strftime('%Y-%m-%d')
        ))
        test_periods.append((
            test_data.index[0].strftime('%Y-%m-%d'),
            test_data.index[-1].strftime('%Y-%m-%d')
        ))

        start_idx += step_size

    test_dates = [d for period in test_periods for d in returns_df.loc[period[0]:period[1]].index]

    strategy_returns = pd.Series(strategy_returns_list, index=test_dates)
    benchmark_returns = pd.Series(benchmark_returns_list, index=test_dates)
    weights_history = pd.DataFrame(weights_history_list).set_index('date')

    metrics = calculate_backtest_metrics(strategy_returns, benchmark_returns)

    return BacktestResult(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        weights_history=weights_history,
        metrics=metrics,
        train_periods=train_periods,
        test_periods=test_periods
    )


def calculate_backtest_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive backtest performance metrics.

    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Periods per year for annualization

    Returns:
        Dict of metrics
    """
    # Cumulative returns
    strat_cum_ret = (1 + strategy_returns).prod() - 1
    bench_cum_ret = (1 + benchmark_returns).prod() - 1

    # Annualized returns
    n_years = len(strategy_returns) / periods_per_year
    strat_ann_ret = (1 + strat_cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    bench_ann_ret = (1 + bench_cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility
    strat_vol = strategy_returns.std() * np.sqrt(periods_per_year)
    bench_vol = benchmark_returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio
    strat_sharpe = strat_ann_ret / strat_vol if strat_vol > 0 else 0
    bench_sharpe = bench_ann_ret / bench_vol if bench_vol > 0 else 0

    # Max drawdown
    strat_cum = (1 + strategy_returns).cumprod()
    strat_dd = (strat_cum / strat_cum.cummax() - 1).min()

    bench_cum = (1 + benchmark_returns).cumprod()
    bench_dd = (bench_cum / bench_cum.cummax() - 1).min()

    # Downside deviation (Sortino)
    downside_strat = strategy_returns[strategy_returns < 0].std() * np.sqrt(periods_per_year)
    sortino = strat_ann_ret / downside_strat if downside_strat > 0 else 0

    # Information ratio (excess return / tracking error)
    excess_ret = strategy_returns - benchmark_returns
    tracking_error = excess_ret.std() * np.sqrt(periods_per_year)
    info_ratio = excess_ret.mean() * periods_per_year / tracking_error if tracking_error > 0 else 0

    # Win rate
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0

    # Calmar ratio (return / max drawdown)
    calmar = strat_ann_ret / abs(strat_dd) if strat_dd != 0 else 0

    return {
        'Total Return (%)': strat_cum_ret * 100,
        'Annual Return (%)': strat_ann_ret * 100,
        'Volatility (%)': strat_vol * 100,
        'Sharpe Ratio': strat_sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': strat_dd * 100,
        'Calmar Ratio': calmar,
        'Information Ratio': info_ratio,
        'Win Rate (%)': win_rate * 100,
        'Benchmark Return (%)': bench_ann_ret * 100,
        'Benchmark Sharpe': bench_sharpe,
        'Benchmark Max DD (%)': bench_dd * 100,
        'Excess Return (%)': (strat_ann_ret - bench_ann_ret) * 100
    }


def plot_backtest_results(result: BacktestResult):
    """
    Create summary plots of backtest results.

    Args:
        result: BacktestResult object

    Returns:
        None (displays plots)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cumulative returns
    strat_cum = (1 + result.strategy_returns).cumprod()
    bench_cum = (1 + result.benchmark_returns).cumprod()

    axes[0, 0].plot(strat_cum.index, strat_cum.values, label='Strategy')
    axes[0, 0].plot(bench_cum.index, bench_cum.values, label='Benchmark')
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Drawdowns
    strat_dd = strat_cum / strat_cum.cummax() - 1
    bench_dd = bench_cum / bench_cum.cummax() - 1

    axes[0, 1].fill_between(strat_dd.index, strat_dd.values, 0, alpha=0.3, label='Strategy')
    axes[0, 1].fill_between(bench_dd.index, bench_dd.values, 0, alpha=0.3, label='Benchmark')
    axes[0, 1].set_title('Drawdowns')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Rolling Sharpe (252-day)
    rolling_window = min(252, len(result.strategy_returns) // 4)
    if rolling_window > 20:
        strat_roll_sharpe = (result.strategy_returns.rolling(rolling_window).mean() * 252 /
                            (result.strategy_returns.rolling(rolling_window).std() * np.sqrt(252)))
        axes[1, 0].plot(strat_roll_sharpe.index, strat_roll_sharpe.values)
        axes[1, 0].set_title(f'Rolling Sharpe Ratio ({rolling_window}-day)')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True)

    # Weights over time (top 5 assets)
    if not result.weights_history.empty:
        top_5_assets = result.weights_history.iloc[-1].nlargest(5).index
        result.weights_history[top_5_assets].plot(ax=axes[1, 1])
        axes[1, 1].set_title('Portfolio Weights (Top 5 Assets)')
        axes[1, 1].legend(fontsize='small')
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()
