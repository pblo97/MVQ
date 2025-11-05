# portfolio_manager/monitor/persistence.py
"""
Portfolio State Persistence: Save/Load completo del estado en CSVs
Permite reconstruir portfolio en cualquier fecha y trackear cambios históricos.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path


class PortfolioStatePersistence:
    """
    Gestiona persistencia completa del portfolio en CSVs.

    Archivos generados:
    1. portfolio_state_YYYY-MM-DD.csv: posiciones actuales + metadata
    2. macro_scores_YYYY-MM-DD.csv: z-scores y régimen macro
    3. quality_scores_YYYY-MM-DD.csv: quality 3D por activo
    4. exit_signals_YYYY-MM-DD.csv: señales de salida (MA200, Mom, VFQ)
    5. risk_metrics_YYYY-MM-DD.csv: métricas de riesgo diarias
    6. allocation_history.csv: histórico de pesos (append)
    7. rebalance_orders_YYYY-MM-DD.csv: órdenes ejecutadas
    """

    def __init__(self, snapshots_dir: str = "snapshots/"):
        self.dir = Path(snapshots_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _format_date(self, date) -> str:
        """Convierte fecha a formato YYYY-MM-DD"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return date.strftime("%Y-%m-%d")

    def save_portfolio_state(
        self,
        date,
        portfolio_df: pd.DataFrame,
        metadata: Optional[Dict] = None
    ):
        """
        Guarda estado del portfolio en CSV.

        Columnas esperadas en portfolio_df:
        - symbol, weight, shares, price, value, quality_score, kelly_fraction,
          sector, beta, position_cap, etc.
        """
        d = self._format_date(date)
        filepath = self.dir / f"portfolio_state_{d}.csv"

        # Agrega fecha y metadata como columnas
        df = portfolio_df.copy()
        df['date'] = d

        if metadata:
            for key, val in metadata.items():
                df[f'meta_{key}'] = val

        df.to_csv(filepath, index=False)
        return filepath

    def save_macro_scores(
        self,
        date,
        macro_z: float,
        regime: str,
        M_macro: float,
        beta_cap_sug: float,
        pos_cap_sug: float,
        overlay_signal: Optional[int] = None,
        composite_z: Optional[float] = None
    ):
        """Guarda scores macro en CSV"""
        d = self._format_date(date)
        filepath = self.dir / f"macro_scores_{d}.csv"

        data = {
            'date': [d],
            'macro_z': [macro_z],
            'regime': [regime],
            'M_macro': [M_macro],
            'beta_cap_sug': [beta_cap_sug],
            'pos_cap_sug': [pos_cap_sug],
            'overlay_signal': [overlay_signal if overlay_signal is not None else np.nan],
            'composite_z': [composite_z if composite_z is not None else np.nan]
        }

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return filepath

    def save_quality_scores(
        self,
        date,
        quality_df: pd.DataFrame
    ):
        """
        Guarda quality scores en CSV.

        Columnas esperadas:
        - symbol, quality_score, liq_score, fund_score, tech_score,
          position_cap, ADV, spread_bps, days_to_liquidate
        """
        d = self._format_date(date)
        filepath = self.dir / f"quality_scores_{d}.csv"

        df = quality_df.copy()
        df['date'] = d

        df.to_csv(filepath, index=False)
        return filepath

    def save_exit_signals(
        self,
        date,
        exit_df: pd.DataFrame
    ):
        """
        Guarda señales de salida en CSV.

        Columnas esperadas:
        - symbol, price, MA200, ma_flag, mom_12_1, mom_flag,
          vfq_last, vfq_delta, quality_flag, action, reason, next_review
        """
        d = self._format_date(date)
        filepath = self.dir / f"exit_signals_{d}.csv"

        df = exit_df.copy()
        df['date'] = d

        df.to_csv(filepath, index=False)
        return filepath

    def save_risk_metrics(
        self,
        date,
        sharpe: float,
        sortino: float,
        max_dd: float,
        cvar_95: float,
        volatility: float,
        turnover: Optional[float] = None,
        tracking_error: Optional[float] = None
    ):
        """Guarda métricas de riesgo en CSV"""
        d = self._format_date(date)
        filepath = self.dir / f"risk_metrics_{d}.csv"

        data = {
            'date': [d],
            'sharpe': [sharpe],
            'sortino': [sortino],
            'max_dd': [max_dd],
            'cvar_95': [cvar_95],
            'volatility': [volatility],
            'turnover': [turnover if turnover is not None else np.nan],
            'tracking_error': [tracking_error if tracking_error is not None else np.nan]
        }

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return filepath

    def append_to_allocation_history(
        self,
        date,
        portfolio_df: pd.DataFrame
    ):
        """
        Append pesos actuales al histórico.

        allocation_history.csv acumula todos los snapshots históricos.
        """
        filepath = self.dir / "allocation_history.csv"

        df = portfolio_df[['symbol', 'weight']].copy() if 'symbol' in portfolio_df.columns else portfolio_df.copy()
        df['date'] = self._format_date(date)

        # Append (o create si no existe)
        if filepath.exists():
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, index=False)

        return filepath

    def save_rebalance_orders(
        self,
        date,
        orders_df: pd.DataFrame
    ):
        """
        Guarda órdenes de rebalanceo ejecutadas.

        Columnas esperadas:
        - symbol, action (BUY/SELL), shares, price_exec, slippage_bps, cost_usd
        """
        d = self._format_date(date)
        filepath = self.dir / f"rebalance_orders_{d}.csv"

        df = orders_df.copy()
        df['date'] = d

        df.to_csv(filepath, index=False)
        return filepath

    def save_complete_state(
        self,
        date,
        portfolio_df: pd.DataFrame,
        macro_data: Dict,
        quality_df: Optional[pd.DataFrame] = None,
        exit_df: Optional[pd.DataFrame] = None,
        risk_metrics: Optional[Dict] = None,
        orders_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Path]:
        """
        Guarda estado completo en un solo llamado (conveniente).

        Args:
            date: fecha del snapshot
            portfolio_df: DataFrame con posiciones
            macro_data: Dict con {macro_z, regime, M_macro, beta_cap_sug, pos_cap_sug, ...}
            quality_df: DataFrame con quality scores (opcional)
            exit_df: DataFrame con exit signals (opcional)
            risk_metrics: Dict con {sharpe, sortino, max_dd, cvar_95, volatility, ...}
            orders_df: DataFrame con órdenes ejecutadas (opcional)

        Returns:
            Dict con paths de archivos generados
        """
        paths = {}

        # 1) Portfolio state
        paths['portfolio'] = self.save_portfolio_state(date, portfolio_df)

        # 2) Macro scores
        paths['macro'] = self.save_macro_scores(
            date=date,
            macro_z=macro_data.get('macro_z', 0.0),
            regime=macro_data.get('regime', 'NEUTRAL'),
            M_macro=macro_data.get('M_macro', 1.0),
            beta_cap_sug=macro_data.get('beta_cap_sug', 1.0),
            pos_cap_sug=macro_data.get('pos_cap_sug', 0.05),
            overlay_signal=macro_data.get('overlay_signal'),
            composite_z=macro_data.get('composite_z')
        )

        # 3) Quality scores (opcional)
        if quality_df is not None and not quality_df.empty:
            paths['quality'] = self.save_quality_scores(date, quality_df)

        # 4) Exit signals (opcional)
        if exit_df is not None and not exit_df.empty:
            paths['exit'] = self.save_exit_signals(date, exit_df)

        # 5) Risk metrics (opcional)
        if risk_metrics:
            paths['risk'] = self.save_risk_metrics(
                date=date,
                sharpe=risk_metrics.get('sharpe', np.nan),
                sortino=risk_metrics.get('sortino', np.nan),
                max_dd=risk_metrics.get('max_dd', np.nan),
                cvar_95=risk_metrics.get('cvar_95', np.nan),
                volatility=risk_metrics.get('volatility', np.nan),
                turnover=risk_metrics.get('turnover'),
                tracking_error=risk_metrics.get('tracking_error')
            )

        # 6) Allocation history (append)
        paths['history'] = self.append_to_allocation_history(date, portfolio_df)

        # 7) Rebalance orders (opcional)
        if orders_df is not None and not orders_df.empty:
            paths['orders'] = self.save_rebalance_orders(date, orders_df)

        return paths

    # ===== LOAD FUNCTIONS =====

    def load_portfolio_state(self, date) -> Optional[pd.DataFrame]:
        """Carga portfolio state desde CSV"""
        d = self._format_date(date)
        filepath = self.dir / f"portfolio_state_{d}.csv"

        if not filepath.exists():
            return None

        return pd.read_csv(filepath)

    def load_macro_scores(self, date) -> Optional[pd.DataFrame]:
        """Carga macro scores desde CSV"""
        d = self._format_date(date)
        filepath = self.dir / f"macro_scores_{d}.csv"

        if not filepath.exists():
            return None

        return pd.read_csv(filepath)

    def load_quality_scores(self, date) -> Optional[pd.DataFrame]:
        """Carga quality scores desde CSV"""
        d = self._format_date(date)
        filepath = self.dir / f"quality_scores_{d}.csv"

        if not filepath.exists():
            return None

        return pd.read_csv(filepath)

    def load_exit_signals(self, date) -> Optional[pd.DataFrame]:
        """Carga exit signals desde CSV"""
        d = self._format_date(date)
        filepath = self.dir / f"exit_signals_{d}.csv"

        if not filepath.exists():
            return None

        return pd.read_csv(filepath)

    def load_risk_metrics(self, date) -> Optional[pd.DataFrame]:
        """Carga risk metrics desde CSV"""
        d = self._format_date(date)
        filepath = self.dir / f"risk_metrics_{d}.csv"

        if not filepath.exists():
            return None

        return pd.read_csv(filepath)

    def load_allocation_history(self) -> Optional[pd.DataFrame]:
        """Carga histórico completo de allocations"""
        filepath = self.dir / "allocation_history.csv"

        if not filepath.exists():
            return None

        return pd.read_csv(filepath, parse_dates=['date'])

    def load_complete_state(self, date) -> Dict[str, pd.DataFrame]:
        """
        Carga estado completo desde CSVs.

        Returns:
            Dict con keys: portfolio, macro, quality, exit, risk
        """
        state = {
            'portfolio': self.load_portfolio_state(date),
            'macro': self.load_macro_scores(date),
            'quality': self.load_quality_scores(date),
            'exit': self.load_exit_signals(date),
            'risk': self.load_risk_metrics(date)
        }

        return state

    def list_available_dates(self) -> List[str]:
        """Lista todas las fechas con snapshots disponibles"""
        files = list(self.dir.glob("portfolio_state_*.csv"))
        dates = [f.stem.replace("portfolio_state_", "") for f in files]
        return sorted(dates)

    def detect_changes(self, date_prev, date_curr) -> Dict:
        """
        Detecta cambios entre dos estados (útil para alertas).

        Returns:
            Dict con cambios detectados en exit signals, weights, etc.
        """
        changes = {
            'exit_changes': [],
            'weight_changes': [],
            'new_positions': [],
            'closed_positions': []
        }

        # Load estados
        prev_state = self.load_complete_state(date_prev)
        curr_state = self.load_complete_state(date_curr)

        # 1) Exit signal changes
        if prev_state['exit'] is not None and curr_state['exit'] is not None:
            prev_exit = prev_state['exit'].set_index('symbol')['action']
            curr_exit = curr_state['exit'].set_index('symbol')['action']

            for sym in set(prev_exit.index) & set(curr_exit.index):
                if prev_exit[sym] != curr_exit[sym]:
                    changes['exit_changes'].append({
                        'symbol': sym,
                        'from': prev_exit[sym],
                        'to': curr_exit[sym]
                    })

        # 2) Weight changes (>5% cambio relativo)
        if prev_state['portfolio'] is not None and curr_state['portfolio'] is not None:
            prev_port = prev_state['portfolio'].set_index('symbol')['weight']
            curr_port = curr_state['portfolio'].set_index('symbol')['weight']

            all_symbols = set(prev_port.index) | set(curr_port.index)

            for sym in all_symbols:
                w_prev = prev_port.get(sym, 0.0)
                w_curr = curr_port.get(sym, 0.0)

                if w_prev == 0 and w_curr > 0:
                    changes['new_positions'].append({'symbol': sym, 'weight': w_curr})
                elif w_prev > 0 and w_curr == 0:
                    changes['closed_positions'].append({'symbol': sym, 'weight_prev': w_prev})
                elif abs(w_curr - w_prev) / max(abs(w_prev), 0.01) > 0.05:  # 5% threshold
                    changes['weight_changes'].append({
                        'symbol': sym,
                        'weight_prev': w_prev,
                        'weight_curr': w_curr,
                        'delta': w_curr - w_prev,
                        'delta_pct': (w_curr - w_prev) / abs(w_prev) if w_prev != 0 else np.inf
                    })

        return changes
