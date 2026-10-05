"""Meta-labelling: a model that learns WHEN the strategy's own signals tend to work.

The strategies decide direction; this model only answers "is this particular signal, in
this market context, likely to hit its target before its stop?". Signals below the learnt
probability threshold are skipped. Training is strictly chronological (older data trains,
newer data tests), and the report shows test-period results with and without the filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..execution import simulate_trade, spread_array
from ..indicators import Features
from ..risk import RiskManager

DIRECTIONAL = ["dist_ema20", "dist_ema50", "dist_ema200", "ema50_slope", "di_diff", "bb_pos", "ret_4", "ret_16",
               "ret_96", "body_atr", "htf_trend", "rsi_c", "range_pos_c"]
NEUTRAL = ["atr_pct", "atr_ratio", "adx", "hour_sin", "hour_cos", "dow"]


def feature_frame(f: Features) -> pd.DataFrame:
    c, o, h, l = f.close, f.open, f.high, f.low
    a = f.atr(14)
    adx, pdi, mdi = f.adx(14)
    mid, up, _ = f.bollinger(20, 2.0)
    out = pd.DataFrame(index=f.index)
    out["atr_pct"] = a / c * 100
    out["atr_ratio"] = a / f.atr(100)
    out["adx"] = adx
    out["di_diff"] = pdi - mdi
    for n in (20, 50, 200):
        out[f"dist_ema{n}"] = (c - f.ema(n)) / a
    out["ema50_slope"] = (f.ema(50) - f.ema(50).shift(10)) / a
    out["bb_pos"] = (c - mid) / (up - mid).replace(0, np.nan)
    for k in (4, 16, 96):
        out[f"ret_{k}"] = (c - c.shift(k)) / a
    out["body_atr"] = (c - o) / a
    out["htf_trend"] = f.htf_ema_slope("H4", 50)
    out["rsi_c"] = f.rsi(14) - 50
    hi96, lo96 = h.rolling(96).max(), l.rolling(96).min()
    out["range_pos_c"] = (c - lo96) / (hi96 - lo96).replace(0, np.nan) - 0.5
    hour = f.index.hour + f.index.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow"] = f.index.weekday
    return out


def signal_vector(feat_row: pd.Series, side: int, sl_dist: float, tp_dist: float, atr: float, strategy: str,
                  strategies: list[str]) -> np.ndarray:
    d = [float(feat_row[k]) * side for k in DIRECTIONAL]
    n = [float(feat_row[k]) for k in NEUTRAL]
    extra = [float(side), sl_dist / atr if atr > 0 else 0.0, tp_dist / sl_dist if sl_dist > 0 else 0.0]
    onehot = [1.0 if strategy == s else 0.0 for s in strategies]
    return np.array(d + n + extra + onehot, dtype=float)


def build_dataset(df: pd.DataFrame, signals: pd.DataFrame, cfg: Config, feats: Optional[Features] = None) -> pd.DataFrame:
    """One row per in-session signal, labelled by a stand-alone SL/TP simulation."""
    feats = feats or Features(df, cfg.symbol.timeframe)
    ff = feature_frame(feats)
    atr = feats.atr(14).to_numpy()
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    spread = spread_array(df, cfg.backtest.spread)
    risk = RiskManager(cfg.risk, cfg.tf_minutes)
    bar = pd.Timedelta(minutes=cfg.tf_minutes)
    max_bars = cfg.management.max_bars_in_trade or 96
    rows = []
    for i in np.flatnonzero(signals["signal"].to_numpy() != 0):
        if i + 1 >= len(df) or not np.isfinite(atr[i]) or ff.iloc[i].isna().any():
            continue
        if not risk.in_session(df.index[i] + bar):
            continue
        s = signals.iloc[i]
        ok, _, sl_d, tp_d = risk.validate_levels(float(s.sl_dist), float(s.tp_dist), float(atr[i]))
        if not ok:
            continue
        res = simulate_trade(o, h, l, c, spread, i + 1, int(s.signal), sl_d, tp_d, cfg.backtest.slippage, max_bars)
        if res is None:
            continue
        rows.append({"i": i, "time": df.index[i], "side": int(s.signal), "sl_dist": sl_d, "tp_dist": tp_d,
                     "atr": float(atr[i]), "strategy": str(s.strategy), "r": float(res[4]), "exit_reason": res[3]})
    return pd.DataFrame(rows)


def _matrix(ds: pd.DataFrame, ff: pd.DataFrame, strategies: list[str]) -> np.ndarray:
    return np.vstack([signal_vector(ff.iloc[r.i], r.side, r.sl_dist, r.tp_dist, r.atr, r.strategy, strategies)
                      for r in ds.itertuples()])


def _best_threshold(prob: np.ndarray, r: np.ndarray, min_keep: float = 0.25) -> float:
    best_t, best_v = 0.0, r.sum()
    for t in np.unique(np.round(prob, 3)):
        keep = prob >= t
        if keep.mean() < min_keep:
            continue
        v = r[keep].sum()
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t


@dataclass
class TrainReport:
    n_train: int
    n_test: int
    test_auc: float
    threshold: float
    test_avg_r_all: float
    test_avg_r_kept: float
    test_kept_frac: float
    test_total_r_all: float
    test_total_r_kept: float

    def verdict(self) -> tuple[bool, str]:
        """Does the filter genuinely help? It must rank trades better than chance, actually skip
        some signals, and lift the average result by a meaningful amount on the unseen test period."""
        if not (self.test_auc == self.test_auc) or self.test_auc < 0.55:
            return False, f"no predictive skill (test AUC {self.test_auc:.2f}; needs >= 0.55)"
        if self.test_kept_frac > 0.9:
            return False, f"it barely filters anything (keeps {self.test_kept_frac:.0%} of signals)"
        gain = self.test_avg_r_kept - self.test_avg_r_all
        if gain < 0.03:
            return False, f"improvement too small ({gain:+.3f} R per trade; needs >= +0.03)"
        return True, f"test AUC {self.test_auc:.2f}, keeps {self.test_kept_frac:.0%} of signals, {gain:+.3f} R per trade"

    def text(self) -> str:
        return (
            f"signals: train {self.n_train}, test {self.n_test}\n"
            f"test AUC: {self.test_auc:.3f} (0.5 = no skill)\n"
            f"threshold: {self.threshold:.3f}\n"
            f"test avg R  - all signals: {self.test_avg_r_all:+.3f}   filtered: {self.test_avg_r_kept:+.3f} "
            f"(kept {self.test_kept_frac:.0%})\n"
            f"test total R - all signals: {self.test_total_r_all:+.1f}   filtered: {self.test_total_r_kept:+.1f}"
        )


def train(df: pd.DataFrame, signals: pd.DataFrame, cfg: Config, train_frac: float = 0.7,
          out_path: str | Path | None = None, seed: int = 0) -> tuple["MetaLabelModel", TrainReport]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    feats = Features(df, cfg.symbol.timeframe)
    ds = build_dataset(df, signals, cfg, feats)
    if len(ds) < 150:
        raise ValueError(f"Only {len(ds)} labelled signals - need at least 150. Use more history.")
    ff = feature_frame(feats)
    strategies = sorted(ds["strategy"].unique())
    X, y, r = _matrix(ds, ff, strategies), (ds["r"].to_numpy() > 0).astype(int), ds["r"].to_numpy()
    cut = int(len(ds) * train_frac)
    make = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                  l2_regularization=1.0, min_samples_leaf=30, random_state=seed)
    # Out-of-fold probabilities on the train part choose the threshold (never the test part).
    oof = np.full(cut, np.nan)
    for tr, va in TimeSeriesSplit(n_splits=4).split(X[:cut]):
        if len(np.unique(y[tr])) < 2:
            continue
        oof[va] = make().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    mask = np.isfinite(oof)
    threshold = _best_threshold(oof[mask], r[:cut][mask]) if mask.sum() > 50 else 0.5
    model = make().fit(X[:cut], y[:cut])
    p_test = model.predict_proba(X[cut:])[:, 1]
    keep = p_test >= threshold
    auc = roc_auc_score(y[cut:], p_test) if len(np.unique(y[cut:])) > 1 else float("nan")
    rep = TrainReport(cut, len(ds) - cut, float(auc), threshold, float(r[cut:].mean()),
                      float(r[cut:][keep].mean()) if keep.any() else 0.0, float(keep.mean()),
                      float(r[cut:].sum()), float(r[cut:][keep].sum()))
    # Final model for live use: refit on everything.
    final = make().fit(X, y)
    bundle = {"model": final, "strategies": strategies, "threshold": threshold, "report": rep.__dict__,
              "timeframe": cfg.symbol.timeframe, "trained_until": str(df.index[-1])}
    mlm = MetaLabelModel(bundle)
    if out_path:
        mlm.save(out_path)
    return mlm, rep


class MetaLabelModel:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = bundle["model"]
        self.strategies = bundle["strategies"]
        self.threshold = float(bundle["threshold"])

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.bundle, path)

    @classmethod
    def load(cls, path: str | Path) -> "MetaLabelModel":
        import joblib

        return cls(joblib.load(path))

    def predict(self, feat_row: pd.Series, side: int, sl_dist: float, tp_dist: float, atr: float, strategy: str) -> float:
        if feat_row.isna().any():
            return float("nan")
        x = signal_vector(feat_row, side, sl_dist, tp_dist, atr, strategy, self.strategies)
        return float(self.model.predict_proba(x.reshape(1, -1))[0, 1])

    def gate(self, feats: Features) -> Callable[[int, int, float, float, str], float]:
        ff = feature_frame(feats)
        atr = feats.atr(14).to_numpy()

        def _gate(i: int, side: int, sl_d: float, tp_d: float, strategy: str) -> float:
            p = self.predict(ff.iloc[i], side, sl_d, tp_d, atr[i], strategy)
            return 1.0 if not np.isfinite(p) else p

        return _gate
