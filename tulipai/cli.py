"""Command line interface: `python -m tulipai <command>` (or `tulip <command>`)."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path

import pandas as pd

from .config import Config, load_config

DEFAULT_CONFIG = "config/config.yaml"

NOTES = [
    "Backtests are estimates. Real fills, spreads around news, swaps and broker outages will differ; demo-trade "
    "for several weeks and compare the live journal with this report before risking money.",
    "Only the walk-forward (out-of-sample) numbers are an honest estimate of future performance. In-sample "
    "results after optimisation are always too optimistic.",
    "The random-entry benchmark answers 'is this skill or luck?'. A profitable strategy that does not beat most "
    "random runs has no demonstrated edge.",
    "Claude's decisions cannot be backtested honestly: the model has read about historical gold prices, so a "
    "historical test would leak hindsight. The AI layer is evaluated forward-only through the live journal and "
    "the veto scorecard.",
]


def _setup_logging(cfg: Config | None = None, verbose: bool = False) -> None:
    log = logging.getLogger("tulipai")
    if log.handlers:
        return
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if cfg is not None:
        Path(cfg.live.log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(Path(cfg.live.log_dir) / "tulipai.log", maxBytes=5_000_000,
                                                  backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _cfg(args) -> Config:
    path = args.config
    if path is None and Path(DEFAULT_CONFIG).exists():
        path = DEFAULT_CONFIG
    cfg = load_config(path)
    if getattr(args, "params", None):
        import yaml

        extra = yaml.safe_load(Path(args.params).read_text(encoding="utf-8")) or {}
        cfg.strategy.params.update(extra.get("strategy", {}).get("params", extra))
    return cfg


def _load_data(args, cfg: Config) -> pd.DataFrame:
    from .data.io import load_csv, resample, slice_time

    df = load_csv(args.data, utc_offset_hours=getattr(args, "utc_offset", 0.0) or 0.0)
    df = slice_time(df, getattr(args, "start", None), getattr(args, "end", None))
    step = (df.index[1:] - df.index[:-1]).min() if len(df) > 2 else pd.Timedelta(minutes=cfg.tf_minutes)
    if step < pd.Timedelta(minutes=cfg.tf_minutes):
        df = resample(df, cfg.symbol.timeframe)
    if len(df) < 2000:
        print(f"WARNING: only {len(df)} bars - results will be statistically meaningless. Use 1+ years of data.")
    return df


def _mt5_broker(cfg: Config):
    from .broker.mt5 import MT5Broker

    return MT5Broker(cfg, login=os.environ.get("MT5_LOGIN") or None, password=os.environ.get("MT5_PASSWORD") or None,
                     server=os.environ.get("MT5_SERVER") or None, path=os.environ.get("MT5_PATH") or None)


# ---------------------------------------------------------------------------- commands
def cmd_doctor(args) -> int:
    cfg = _cfg(args)
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    ok = True
    for mod, why in (("numpy", "core"), ("pandas", "core"), ("yaml", "core"), ("requests", "news"),
                     ("anthropic", "Claude AI layer"), ("sklearn", "ML filter"), ("MetaTrader5", "MT5 trading (Windows)")):
        try:
            __import__(mod)
            print(f"  [ok]   {mod:<12} ({why})")
        except ImportError:
            print(f"  [miss] {mod:<12} ({why})")
            ok = ok and mod not in ("numpy", "pandas", "yaml")
    from .ai.brain import ClaudeBrain

    ai_ok, why = ClaudeBrain.available()
    print(f"  Claude API: {'ready' if ai_ok else why}; ai.mode = {cfg.ai.mode}, model = {cfg.ai.model}")
    print(f"  Risk: {cfg.risk.risk_per_trade_pct}% per trade, daily loss cap {cfg.risk.max_daily_loss_pct}%, "
          f"drawdown kill switch {cfg.risk.max_drawdown_pct}%, real accounts allowed: {cfg.account.allow_real_account}")
    if args.mt5:
        try:
            b = _mt5_broker(cfg)
            acct = b.connect()
            spec = b.spec()
            tick = b.tick()
            print(f"  MT5: account {acct.login} {acct.currency} {'DEMO' if acct.is_demo else 'REAL'} on {acct.server}; "
                  f"balance {acct.balance:.2f}, leverage 1:{acct.leverage}")
            print(f"  Symbol {spec.name}: digits {spec.digits}, contract {spec.contract_size}, volume "
                  f"{spec.volume_min}-{spec.volume_max} step {spec.volume_step}, stops level {spec.stops_level}")
            print(f"  Tick: bid {tick.bid} ask {tick.ask} spread {tick.spread:.3f}; server time {b.offset_note}")
            lpl = b.loss_per_lot(1, tick.ask, tick.ask - 5.0)
            print(f"  A $5 stop on 1.0 lot loses {lpl:.2f} {acct.currency}; 0.01 lot loses {lpl / 100:.2f} {acct.currency}")
            for p in b.preflight():
                print(f"  [!] {p}")
            b.shutdown()
        except Exception as exc:
            print(f"  MT5 check failed: {exc}")
            ok = False
    return 0 if ok else 1


def cmd_fetch(args) -> int:
    from .data.io import save_csv

    cfg = _cfg(args)
    tf = args.tf or cfg.symbol.timeframe
    if args.source == "mt5":
        b = _mt5_broker(cfg)
        b.connect()
        df = b.history(pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC") if args.end else
                       pd.Timestamp.now(tz="UTC"), tf)
        b.shutdown()
    elif args.source == "dukascopy":
        from .data.dukascopy import download

        df = download(args.start, args.end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"), tf)
    else:
        from .data.synthetic import synthetic_gold

        days = (pd.Timestamp(args.end or "2025-01-01") - pd.Timestamp(args.start)).days
        df = synthetic_gold(start=args.start, days=days, timeframe=tf, seed=args.seed)
    path = save_csv(df, args.out)
    print(f"Saved {len(df)} {tf} bars {df.index[0]} -> {df.index[-1]} to {path}")
    return 0


def _run_backtest(args, cfg, df):
    from .backtest import Backtester
    from .indicators import Features
    from .strategies import build_strategy

    feats = Features(df, cfg.symbol.timeframe)
    strat = build_strategy(cfg.strategy)
    sig = strat.generate(feats)
    gate, thr = None, 0.5
    if getattr(args, "ml", False):
        from .ml import MetaLabelModel

        model = MetaLabelModel.load(cfg.ml.model_path)
        gate, thr = model.gate(feats), (cfg.ml.threshold or model.threshold)
    events = None
    cal_path = getattr(args, "calendar", None) or cfg.news.calendar_csv
    if cal_path:
        from .news import EconomicCalendar

        events = EconomicCalendar(cfg.news).load_csv(cal_path).event_times()
    start = min(cfg.backtest.warmup_bars, len(df) - 1)
    res = Backtester(cfg).run(df, sig, start=start, features=feats, ml_gate=gate, ml_threshold=thr, news_events=events,
                              label=strat.describe())
    return res, feats, start, strat


def cmd_backtest(args) -> int:
    from .backtest.benchmark import buy_and_hold, random_entry_benchmark
    from .backtest.metrics import format_metrics
    from .report import build_report, write_report

    cfg = _cfg(args)
    df = _load_data(args, cfg)
    res, feats, start, strat = _run_backtest(args, cfg, df)
    m = res.metrics()
    print(f"\n=== Backtest {df.index[start].date()} -> {df.index[-1].date()} ({len(df) - start} bars) ===")
    print(strat.describe())
    print(format_metrics(m))
    print(f"Signals blocked: {res.blocked}")
    bh_eq, bh = buy_and_hold(df, res.initial_balance, start=start, bar_minutes=cfg.tf_minutes)
    print(f"\nBuy & hold gold: {bh['return_pct']:+.2f}% (max DD {bh['max_dd_pct']:.1f}%)")
    rb = None
    if args.mc:
        print(f"Running {args.mc} random-entry simulations...")
        rb = random_entry_benchmark(df, cfg, res, n_sims=args.mc, start=start, features=feats, progress=True)
        if rb.get("n_sims"):
            print(f"Random entries: mean {rb['random_net_mean']:+.2f}, 5-95% [{rb['random_net_p5']:+.2f}, "
                  f"{rb['random_net_p95']:+.2f}]; strategy beats {rb['strategy_percentile']:.0f}% (p={rb['p_value']:.3f})")
            print(f"VERDICT: {rb['verdict']}")
    if args.report:
        page = build_report(res, f"TulipAI backtest - {Path(args.data).name}",
                            f"{df.index[start]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} · {cfg.symbol.timeframe} · "
                            f"{strat.describe()} · risk {cfg.risk.risk_per_trade_pct}%/trade · spread "
                            f"{cfg.backtest.spread} · AI not included (see notes)",
                            buy_hold=bh_eq, random_bench=rb, notes=NOTES)
        print(f"Report: {write_report(args.report, page)}")
    return 0


def cmd_walkforward(args) -> int:
    from .backtest.benchmark import buy_and_hold
    from .backtest.metrics import format_metrics
    from .backtest.walkforward import walk_forward
    from .report import build_report, write_report

    cfg = _cfg(args)
    df = _load_data(args, cfg)
    print(f"Walk-forward: train {args.train_months}m / test {args.test_months}m, up to {args.max_combos} combos per strategy")
    wf = walk_forward(df, cfg, args.train_months, args.test_months, args.max_combos, args.min_trades)
    m = wf.metrics()
    print("\n=== OUT-OF-SAMPLE (stitched test windows) ===")
    print(format_metrics(m))
    if args.save_params:
        import yaml

        Path(args.save_params).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_params).write_text(yaml.safe_dump({"strategy": {"params": wf.best_params}}, sort_keys=False),
                                          encoding="utf-8")
        print(f"Latest window's parameters saved to {args.save_params} (use with --params)")
    if args.report:
        first = wf.oos.equity.index[0]
        sub = df[df.index >= first - pd.Timedelta(minutes=cfg.tf_minutes)]
        bh_eq, _ = buy_and_hold(sub, wf.oos.initial_balance, bar_minutes=cfg.tf_minutes)
        page = build_report(wf.oos, f"TulipAI walk-forward - {Path(args.data).name}",
                            f"Out-of-sample only · train {args.train_months} months / test {args.test_months} month(s) · "
                            f"{len(wf.windows)} windows", buy_hold=bh_eq, windows=wf.windows, notes=NOTES)
        print(f"Report: {write_report(args.report, page)}")
    return 0


def cmd_train_ml(args) -> int:
    from .indicators import Features
    from .ml import train
    from .strategies import build_strategy

    cfg = _cfg(args)
    df = _load_data(args, cfg)
    sig = build_strategy(cfg.strategy).generate(Features(df, cfg.symbol.timeframe))
    out = args.out or cfg.ml.model_path
    _, rep = train(df, sig, cfg, out_path=out)
    print(rep.text())
    print(f"Model saved to {out}. Enable it with ml.enabled: true (and test it: tulip backtest --ml).")
    if rep.test_avg_r_kept <= rep.test_avg_r_all:
        print("NOTE: the filter did not improve the test period - keep ml.enabled: false.")
    return 0


def cmd_replay(args) -> int:
    from .backtest.metrics import format_metrics
    from .live import run_replay
    from .report import journal_result

    cfg = _cfg(args)
    df = _load_data(args, cfg)
    if args.ai:
        from .ai.brain import ClaudeBrain

        brain = ClaudeBrain(cfg.ai, cfg.risk, cfg.management, cfg.symbol.timeframe)
    else:
        cfg.ai.mode, brain = "off", None
    path = Path(args.journal)
    if path.exists():
        path.unlink()
    eng = run_replay(df, cfg, path, brain=brain, progress=True)
    res = journal_result(eng.journal, "replay")
    print(format_metrics(res.metrics()))
    return 0


def cmd_live(args) -> int:
    from .ai.brain import ClaudeBrain
    from .broker.paper import PaperBroker
    from .journal import Journal
    from .live import LiveEngine
    from .news import EconomicCalendar, HeadlineFeed
    from .panel.server import _load_ml

    cfg = _cfg(args)
    _setup_logging(cfg, args.verbose)
    mt5b = _mt5_broker(cfg)
    broker = PaperBroker(mt5b, cfg) if args.paper or cfg.account.broker == "paper" else mt5b
    brain = ClaudeBrain(cfg.ai, cfg.risk, cfg.management, cfg.symbol.timeframe) if cfg.ai.mode != "off" else None
    eng = LiveEngine(cfg, broker, Journal(cfg.live.journal_path), brain=brain,
                     calendar=EconomicCalendar(cfg.news) if cfg.news.enabled else None,
                     headlines=HeadlineFeed(cfg.news) if cfg.news.enabled else None, ml=_load_ml(cfg),
                     mode_label="paper" if broker is not mt5b else "mt5")
    eng.start()
    print(f"Running. Create a file named '{cfg.live.stop_file}' to pause new entries; Ctrl+C to stop.")
    stop = threading.Event()
    try:
        eng.run(stop)
    except KeyboardInterrupt:
        stop.set()
        print("Stopped. Open positions keep their stop-loss and take-profit at the broker.")
    return 0


def cmd_panel(args) -> int:
    from .panel import serve

    cfg = _cfg(args)
    _setup_logging(cfg, args.verbose)
    serve(cfg, args.port, open_browser=not args.no_browser)
    return 0


def cmd_report(args) -> int:
    from .backtest.metrics import format_metrics
    from .journal import Journal
    from .report import build_report, journal_result, write_report
    from .stats import journal_stats

    cfg = _cfg(args)
    j = Journal(args.journal)
    res = journal_result(j, args.mode)
    print(format_metrics(res.metrics()))
    st = journal_stats(j, args.mode, cfg.ai)
    print(f"AI: {st['ai']['calls_total']} calls, est. ${st['ai']['est_cost_total_usd']}; vetoes: {st['vetoes']}")
    page = build_report(res, "TulipAI live journal", f"{args.journal} · mode {args.mode or 'all'}", notes=NOTES)
    print(f"Report: {write_report(args.out, page)}")
    return 0


def cmd_review(args) -> int:
    from .ai.brain import ClaudeBrain
    from .backtest.metrics import format_metrics
    from .journal import Journal
    from .report import journal_result
    from .stats import journal_stats

    cfg = _cfg(args)
    ok, why = ClaudeBrain.available()
    if not ok:
        print(why)
        return 1
    j = Journal(args.journal)
    res = journal_result(j, args.mode)
    st = journal_stats(j, args.mode, cfg.ai)
    st.pop("equity_curve", None)
    trades = j.frame("trades").tail(60).drop(columns=["extra"], errors="ignore")
    text = (f"## Metrics\n{format_metrics(res.metrics())}\n\n## Journal stats\n{st}\n\n"
            f"## Config\nrisk={cfg.risk}\nmanagement={cfg.management}\nstrategy={cfg.strategy}\nai.mode={cfg.ai.mode}\n\n"
            f"## Last trades\n{trades.to_string()}")
    review = ClaudeBrain(cfg.ai, cfg.risk, cfg.management, cfg.symbol.timeframe).review(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(review, encoding="utf-8")
    print(review)
    print(f"\nSaved to {out}")
    return 0


def cmd_research(args) -> int:
    """Fetch broker history, backtest + benchmarks, walk-forward and ML in one go, and write
    everything worth sharing to reports/research_summary.txt."""
    import time as _time

    import yaml

    from .backtest.benchmark import buy_and_hold, random_entry_benchmark
    from .backtest.metrics import format_metrics
    from .backtest.walkforward import summarize_windows, walk_forward
    from .data.io import load_csv, save_csv
    from .report import build_report, write_report

    cfg = _cfg(args)
    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    out(f"TulipAI research summary - generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC")
    out("(No passwords or account numbers are included in this file.)")
    t0 = _time.time()

    # 1. data -------------------------------------------------------------------------
    out("\n== 1. Data ==")
    if args.data:
        df = load_csv(args.data)
        out(f"Loaded {args.data}")
    else:
        b = _mt5_broker(cfg)
        acct = b.connect()
        spec = b.spec()
        out(f"Broker: {acct.company} | server {acct.server} | currency {acct.currency} | "
            f"{'DEMO' if acct.is_demo else 'REAL'} | leverage 1:{acct.leverage}")
        out(f"Symbol: {spec.name} | digits {spec.digits} | contract {spec.contract_size} | volume "
            f"{spec.volume_min}-{spec.volume_max} step {spec.volume_step} | min stop distance {spec.stops_level}")
        out(f"Server time: {b.offset_note}")
        try:
            t = b.tick()
            out(f"Last tick: {t.time:%Y-%m-%d %H:%M} UTC bid {t.bid} ask {t.ask} spread {t.spread:.3f}")
            lpl = b.loss_per_lot(1, t.ask, t.ask - 10.0)
            out(f"Sizing check: a $10 stop on 0.01 lot loses {lpl / 100:.2f} {acct.currency}")
        except Exception as exc:  # pragma: no cover - depends on the broker
            out(f"Tick unavailable: {exc}")
        end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
        df = b.history(pd.Timestamp(args.start, tz="UTC"), end, cfg.symbol.timeframe)
        b.shutdown()
        save_csv(df, args.out)
        out(f"Saved to {args.out}")
    gaps = (df.index[1:] - df.index[:-1]) > pd.Timedelta(days=4)
    out(f"{len(df)} {cfg.symbol.timeframe} bars from {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} "
        f"| price {df['close'].min():.2f}-{df['close'].max():.2f} | gaps > 4 days: {int(gaps.sum())}")
    if "spread" in df.columns:
        sp = df["spread"]
        out(f"Spread (from MT5 bars): median {sp.median():.3f}, 90th pct {sp.quantile(0.9):.3f}; backtest floor "
            f"{cfg.backtest.spread}")
    if args.start and df.index[0] > pd.Timestamp(args.start, tz="UTC") + pd.Timedelta(days=20):
        out(f"NOTE: MT5 only had history from {df.index[0]:%Y-%m-%d}. For more, set MT5 Tools > Options > Charts > "
            "Max bars in chart to Unlimited, scroll the gold M15 chart back, and run again.")
    hours = df.index.hour.value_counts().reindex(range(24), fill_value=0)
    quiet = hours.idxmin()
    out(f"Quietest UTC hour in the data: {quiet:02d}:00 (gold's daily break is 21:00-22:00 UTC in summer, "
        "22:00-23:00 in winter; a big mismatch means the server time zone is wrong)")

    # 2. backtest + benchmarks --------------------------------------------------------
    out("\n== 2. Backtest with default settings (in-sample) ==")
    res, feats, start, strat = _run_backtest(argparse.Namespace(ml=False, calendar=None), cfg, df)
    out(format_metrics(res.metrics()))
    out(f"Signals blocked: {res.blocked}")
    bh_eq, bh = buy_and_hold(df, res.initial_balance, start=start, bar_minutes=cfg.tf_minutes)
    out(f"Buy & hold gold: {bh['return_pct']:+.2f}% (max DD {bh['max_dd_pct']:.1f}%, Sharpe {bh['sharpe']:.2f})")
    rb = None
    if args.mc:
        out(f"Random-entry benchmark ({args.mc} runs)...")
        rb = random_entry_benchmark(df, cfg, res, n_sims=args.mc, start=start, features=feats, progress=True)
        if rb.get("n_sims"):
            out(f"Random entries: mean {rb['random_net_mean']:+.2f}, 5-95% [{rb['random_net_p5']:+.2f}, "
                f"{rb['random_net_p95']:+.2f}]; strategy beats {rb['strategy_percentile']:.0f}% (p={rb['p_value']:.3f})")
            out(f"VERDICT: {rb['verdict']}")
        else:
            out(rb.get("note", ""))
    page = build_report(res, "TulipAI backtest (broker data)", f"{df.index[start]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}",
                        buy_hold=bh_eq, random_bench=rb, notes=NOTES)
    out(f"Report: {write_report('reports/backtest.html', page)}")

    # 3. walk-forward -----------------------------------------------------------------
    out(f"\n== 3. Walk-forward, out-of-sample (train {args.train_months}m / test {args.test_months}m) ==")
    try:
        wf = walk_forward(df, cfg, args.train_months, args.test_months, args.max_combos, 15)
        out(summarize_windows(wf).to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
        out(format_metrics(wf.metrics()))
        Path("config").mkdir(exist_ok=True)
        Path("config/optimized_params.yaml").write_text(
            yaml.safe_dump({"strategy": {"params": wf.best_params}}, sort_keys=False), encoding="utf-8")
        out("Latest parameters: " + str(wf.best_params))
        sub = df[df.index >= wf.oos.equity.index[0] - pd.Timedelta(minutes=cfg.tf_minutes)]
        wf_bh, _ = buy_and_hold(sub, wf.oos.initial_balance, bar_minutes=cfg.tf_minutes)
        page = build_report(wf.oos, "TulipAI walk-forward (broker data)", "Out-of-sample only", buy_hold=wf_bh,
                            windows=wf.windows, notes=NOTES)
        out(f"Report: {write_report('reports/walkforward.html', page)}")
    except ValueError as exc:
        out(f"Walk-forward skipped: {exc}")

    # 4. ML filter --------------------------------------------------------------------
    out("\n== 4. ML meta-label filter ==")
    try:
        from .ml import train

        _, rep = train(df, strat.generate(feats), cfg, out_path=cfg.ml.model_path)
        out(rep.text())
        out("Filter helps on the test period - consider ml.enabled: true" if rep.test_avg_r_kept > rep.test_avg_r_all
            else "Filter does NOT help on the test period - keep ml.enabled: false")
    except Exception as exc:
        out(f"ML skipped: {exc}")

    out(f"\nFinished in {(_time.time() - t0) / 60:.1f} min.")
    path = Path("reports/research_summary.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {path}. Send that file (or paste its text) to Claude.")
    return 0


def cmd_close_all(args) -> int:
    cfg = _cfg(args)
    b = _mt5_broker(cfg)
    b.connect()
    for r in b.close_all():
        print(f"close {r.ticket}: {'ok' if r.ok else r.message}")
    b.shutdown()
    return 0


def cmd_demo(args) -> int:
    """End-to-end demo on SYNTHETIC data - shows the workflow, says nothing about real profitability."""
    from .data.io import save_csv
    from .data.synthetic import synthetic_gold

    path = save_csv(synthetic_gold(start="2024-01-01", days=args.days, seed=args.seed), "data/synthetic_m15.csv")
    print(f"Synthetic data written to {path} (NOT real prices)")
    ns = argparse.Namespace(config=args.config, data=str(path), start=None, end=None, mc=args.mc, ml=False,
                            calendar=None, report="reports/demo_backtest.html", params=None, utc_offset=0.0)
    return cmd_backtest(ns)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tulip", description="TulipAI - autonomous AI gold trading on MT5")
    p.add_argument("--config", default=None, help=f"YAML config (default: {DEFAULT_CONFIG} if present)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="check installation, API key and (with --mt5) the MT5 connection")
    s.add_argument("--mt5", action="store_true")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("fetch", help="download candles to CSV")
    s.add_argument("--source", choices=["mt5", "dukascopy", "synthetic"], default="mt5")
    s.add_argument("--start", required=True)
    s.add_argument("--end")
    s.add_argument("--tf")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="data/xauusd_m15.csv")
    s.set_defaults(fn=cmd_fetch)

    def data_args(s):
        s.add_argument("--data", required=True, help="candle CSV (see `fetch`)")
        s.add_argument("--start")
        s.add_argument("--end")
        s.add_argument("--utc-offset", type=float, default=0.0, help="hours to subtract (MT5 exports are server time)")
        s.add_argument("--params", help="YAML with strategy params (from walkforward --save-params)")

    s = sub.add_parser("backtest", help="backtest + benchmarks + HTML report")
    data_args(s)
    s.add_argument("--mc", type=int, default=100, help="random-entry simulations (0 = skip)")
    s.add_argument("--ml", action="store_true", help="apply the trained ML filter")
    s.add_argument("--calendar", help="CSV of news events (time,currency,impact,title) for the blackout")
    s.add_argument("--report", default="reports/backtest.html")
    s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("walkforward", help="out-of-sample parameter optimisation")
    data_args(s)
    s.add_argument("--train-months", type=int, default=6)
    s.add_argument("--test-months", type=int, default=1)
    s.add_argument("--max-combos", type=int, default=27)
    s.add_argument("--min-trades", type=int, default=15)
    s.add_argument("--save-params", default="config/optimized_params.yaml")
    s.add_argument("--report", default="reports/walkforward.html")
    s.set_defaults(fn=cmd_walkforward)

    s = sub.add_parser("train-ml", help="train the meta-label filter")
    data_args(s)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_train_ml)

    s = sub.add_parser("replay", help="run the LIVE engine over historical data (simulated fills)")
    data_args(s)
    s.add_argument("--journal", default="runs/replay.db")
    s.add_argument("--ai", action="store_true", help="call Claude on every signal (costs API credits)")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("live", help="run the bot headless (credentials from .env)")
    s.add_argument("--paper", action="store_true", help="simulate fills, send no orders")
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("panel", help="open the browser control panel (login + dashboard)")
    s.add_argument("--port", type=int)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_panel)

    s = sub.add_parser("report", help="HTML report from the live journal")
    s.add_argument("--journal", default="runs/journal.db")
    s.add_argument("--mode", default=None, help="mt5 | paper | replay (default: all)")
    s.add_argument("--out", default="reports/live.html")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("review", help="ask Claude to review performance and suggest improvements")
    s.add_argument("--journal", default="runs/journal.db")
    s.add_argument("--mode", default=None)
    s.add_argument("--out", default="reports/review.md")
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("research", help="fetch MT5 history + backtest + benchmarks + walk-forward + ML, one summary")
    s.add_argument("--start", default="2024-09-01")
    s.add_argument("--end")
    s.add_argument("--data", help="use this CSV instead of downloading from MT5")
    s.add_argument("--out", default="data/xauusd_m15.csv")
    s.add_argument("--mc", type=int, default=200)
    s.add_argument("--train-months", type=int, default=6)
    s.add_argument("--test-months", type=int, default=1)
    s.add_argument("--max-combos", type=int, default=27)
    s.set_defaults(fn=cmd_research)

    s = sub.add_parser("close-all", help="close every position opened by the bot")
    s.set_defaults(fn=cmd_close_all)

    s = sub.add_parser("demo", help="synthetic-data demo of backtest + benchmark + report")
    s.add_argument("--days", type=int, default=540)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--mc", type=int, default=60)
    s.set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    _load_env()
    args = build_parser().parse_args(argv)
    if args.cmd not in ("live", "panel"):
        _setup_logging(None, args.verbose)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
