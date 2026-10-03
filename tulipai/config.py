"""Typed configuration loaded from YAML.

Every option has a safe default here; ``config/default.yaml`` documents them and a
user file only needs the keys it wants to change. Unknown keys raise an error so a
typo in a risk setting can never be silently ignored.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def tf_minutes(tf: str) -> int:
    try:
        return TF_MINUTES[tf.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown timeframe {tf!r}; use one of {sorted(TF_MINUTES)}") from exc


@dataclass
class AccountConfig:
    broker: str = "mt5"  # mt5 | paper (paper = MT5 prices, simulated fills)
    # Cent accounts are REAL-money accounts in MT5. The bot refuses to send orders
    # to a non-demo account unless this is explicitly switched on.
    allow_real_account: bool = False
    magic: int = 26100301
    deviation_points: int = 50
    comment: str = "TulipAI"


@dataclass
class SymbolConfig:
    name: str = "auto"  # auto -> XAUUSDc / XAUUSDm / XAUUSD / GOLD ...
    timeframe: str = "M15"
    server_utc_offset_hours: Any = "auto"  # "auto", a fixed number (0, 2, 3...) or "ny+7"
    history_bars: int = 6000  # enough M15 bars for the H4 EMA50 trend filter to fully converge
    # Backtest/paper contract model. Live trading reads the real spec from MT5.
    contract_size: float = 100.0  # ounces per 1.0 lot
    value_multiplier: float = 1.0  # account-currency units per (price unit x oz); 1.0 for USD and USC
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 100.0


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 20.0  # kill switch measured from peak equity
    max_open_positions: int = 1
    max_trades_per_day: int = 4
    max_lot: float = 5.0
    max_spread: float = 1.0  # price units (USD/oz)
    max_spread_atr: float = 0.25  # spread must also be below this fraction of ATR
    min_rr: float = 1.0
    min_sl_atr: float = 0.8
    max_sl_atr: float = 4.0
    min_lot_risk_overshoot: float = 1.5  # allow the minimum lot if it risks <= 1.5x budget
    loss_streak_pause: int = 3
    cooldown_bars: int = 8
    trade_hours_utc: list = field(default_factory=lambda: [[7, 20]])
    trade_days: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    friday_cutoff_hour_utc: int = 18
    news_blackout_before_min: int = 30
    news_blackout_after_min: int = 30


@dataclass
class ManagementConfig:
    breakeven_at_r: float = 1.0  # 0 disables
    breakeven_lock_r: float = 0.1
    trail_start_r: float = 1.5  # 0 disables
    trail_atr_mult: float = 2.0
    max_bars_in_trade: int = 96  # 0 disables
    close_before_weekend: bool = True
    weekend_close_hour_utc: int = 20


@dataclass
class StrategyConfig:
    name: str = "ensemble"
    members: list = field(default_factory=lambda: ["session_breakout", "trend_pullback", "mean_reversion"])
    params: dict = field(default_factory=dict)  # {strategy_name: {param: value}}


@dataclass
class AIConfig:
    mode: str = "filter"  # off | filter | autonomous
    model: str = "claude-opus-5-5"
    effort: str = "medium"  # low | medium | high | xhigh | max
    max_tokens: int = 16000
    min_confidence: float = 0.6
    max_calls_per_day: int = 60
    autonomous_every_bars: int = 4
    on_error: str = "skip"  # skip (no trade) | quant (trade the quant signal anyway)
    use_fallbacks: bool = True
    timeout_s: float = 180.0
    context_symbols: list = field(default_factory=lambda: ["DXY", "USDX", "US10Y", "XAGUSD"])
    price_input_per_mtok: float = 4.0  # only used for the cost estimate in reports
    price_output_per_mtok: float = 20.0


@dataclass
class MLConfig:
    enabled: bool = False
    model_path: str = "models/meta_label.joblib"
    threshold: float = 0.0  # 0 -> use the threshold stored with the model


@dataclass
class NewsConfig:
    enabled: bool = True
    calendar_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    calendar_csv: str = ""  # optional offline calendar for backtests
    currencies: list = field(default_factory=lambda: ["USD"])
    impacts: list = field(default_factory=lambda: ["High"])
    feeds: list = field(
        default_factory=lambda: [
            "https://www.fxstreet.com/rss/news",
            "https://www.forexlive.com/feed/news",
            "https://www.investing.com/rss/news_11.rss",
        ]
    )
    keywords: list = field(
        default_factory=lambda: [
            "gold", "xau", "bullion", "fed", "fomc", "powell", "inflation", "cpi", "pce",
            "payroll", "nfp", "yield", "treasur", "dollar", "dxy", "rate cut", "rate hike",
            "tariff", "war", "geopolit", "central bank", "safe haven", "recession",
        ]
    )
    max_headlines: int = 15
    refresh_minutes: int = 30
    cache_dir: str = "data/cache"


@dataclass
class BacktestConfig:
    initial_balance: float = 10000.0  # account currency; 10000 USC = 100 USD on a cent account
    spread: float = 0.35  # price units; used when data has no spread, and as a floor under MT5's bar spread
    slippage: float = 0.05
    commission_per_lot: float = 0.0  # round-turn, account currency
    warmup_bars: int = 300


@dataclass
class LiveConfig:
    poll_seconds: float = 5.0
    journal_path: str = "runs/journal.db"
    stop_file: str = "STOP"
    log_dir: str = "logs"
    panel_port: int = 8765


@dataclass
class Config:
    account: AccountConfig = field(default_factory=AccountConfig)
    symbol: SymbolConfig = field(default_factory=SymbolConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    management: ManagementConfig = field(default_factory=ManagementConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    @property
    def tf_minutes(self) -> int:
        return tf_minutes(self.symbol.timeframe)

    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> "Config":
        return copy.deepcopy(self)


def _apply(obj: Any, data: dict, path: str) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key '{path}{key}'. Valid keys: {sorted(known)}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config key '{path}{key}' must be a mapping")
            _apply(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load defaults, then a YAML file (if given), then a dict of overrides."""
    cfg = Config()
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        _apply(cfg, data, "")
    if overrides:
        _apply(cfg, overrides, "")
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    tf_minutes(cfg.symbol.timeframe)
    if cfg.ai.mode not in ("off", "filter", "autonomous"):
        raise ValueError("ai.mode must be off, filter or autonomous")
    if cfg.ai.on_error not in ("skip", "quant"):
        raise ValueError("ai.on_error must be skip or quant")
    if cfg.account.broker not in ("mt5", "paper"):
        raise ValueError("account.broker must be mt5 or paper")
    r = cfg.risk
    if not 0 < r.risk_per_trade_pct <= 5:
        raise ValueError("risk.risk_per_trade_pct must be in (0, 5]")
    if r.min_sl_atr <= 0 or r.max_sl_atr <= r.min_sl_atr:
        raise ValueError("risk.min_sl_atr must be > 0 and < risk.max_sl_atr")
    if r.max_open_positions < 1:
        raise ValueError("risk.max_open_positions must be >= 1")
    for window in r.trade_hours_utc:
        if len(window) != 2 or not (0 <= window[0] < window[1] <= 24):
            raise ValueError(f"Bad risk.trade_hours_utc window {window}; use [start, end) hours 0-24")


def dump_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
