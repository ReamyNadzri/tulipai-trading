from __future__ import annotations

from ..config import StrategyConfig
from .base import SIGNAL_COLUMNS, Strategy
from .ensemble import Ensemble
from .mean_reversion import MeanReversion
from .session_breakout import SessionBreakout
from .trend_pullback import TrendPullback

STRATEGIES: dict[str, type[Strategy]] = {
    SessionBreakout.name: SessionBreakout,
    TrendPullback.name: TrendPullback,
    MeanReversion.name: MeanReversion,
}


def build_strategy(cfg: StrategyConfig) -> Strategy:
    params = cfg.params or {}
    unknown = set(params) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"strategy.params has unknown strategies {sorted(unknown)}")
    if cfg.name == "ensemble":
        return Ensemble([STRATEGIES[m](**params.get(m, {})) for m in cfg.members])
    if cfg.name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {cfg.name!r}; choose ensemble or one of {sorted(STRATEGIES)}")
    return STRATEGIES[cfg.name](**params.get(cfg.name, {}))


__all__ = ["SIGNAL_COLUMNS", "STRATEGIES", "Strategy", "Ensemble", "build_strategy"]
