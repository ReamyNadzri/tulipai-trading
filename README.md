# TulipAI: autonomous gold trading for MetaTrader 5

TulipAI trades **gold (XAUUSD)** on a MetaTrader 5 **demo or cent account** on its own. It reads the
candles, runs three rule-based strategies, checks a set of safety rules (sessions, spread, news
blackout, loss limits), sizes every position from a fixed % risk, places the stop-loss and
take-profit, manages the trade, and logs every decision so you can measure whether it actually
makes money.

**By default it runs on rules only: no AI and no API costs.** Every decision is fixed logic that
the backtester reproduces exactly. Claude can optionally be switched on as an extra approve/veto
filter (section 5).

```
 MT5 candles ─► strategies ─► risk pre-checks ─► [optional] ML filter ─► [optional] Claude
 (closed bars)  breakout       session, spread,
                pullback       news blackout,
                mean-revert    daily loss, ...
                                     │
                                     ▼
              position size (1% risk) ─► MT5 order with SL + TP ─► break-even / trailing /
                                                                   time stop / weekend flat
                                     │
                                     ▼
              journal (SQLite) ─► control panel, live report, benchmarks
```

> **Read this first.** No trading system can promise profit, and this one does not. Gold moves
> $50-100 a day at $4,000+. Cent accounts are **real money**. The point of TulipAI is to trade
> with strict limits *and* tell you honestly whether it has an edge, using benchmarks that are hard
> to fool. Run it on **demo first**.

---

## 1. Quick start (Windows, about 10 minutes)

1. **MetaTrader 5.** Install MT5 from your broker. Log in to your demo or cent account once and
   click **Algo Trading** in the toolbar (it must be green).
2. **Python.** Install Python 3.11 or newer from [python.org](https://www.python.org/downloads/windows/).
   Tick *"Add python.exe to PATH"* during setup.
3. **Get TulipAI.** Download this repository (*Code → Download ZIP*) and unzip it, or `git clone` it.
4. **Double-click `TulipAI.bat`.** The first run installs everything, then your browser opens the
   **control panel** at `http://127.0.0.1:8765`.
5. **Log in.** Enter your MT5 **login number, password and server** (exactly as in your broker's
   email, e.g. `Exness-MT5Trial8`). Click **Connect & open MT5**. TulipAI opens the MT5 terminal and logs in by itself.
6. **Decisions: leave "Rules only (no AI)"** (the default). To try Claude instead, pick a Claude
   option and paste an API key from [console.anthropic.com](https://console.anthropic.com).
7. Click **Start bot**. The dashboard shows equity, open positions, every decision (and why) and
   every trade.

You don't need TeamViewer or remote access. The bot runs on your own PC next to MT5. Keep the PC
on and awake while it trades, or use a cheap Windows VPS.

**Your password stays on your PC.** It goes only to your MT5 terminal and is never saved to disk.
"Remember" stores only login, server, path and the decision mode. **Never paste passwords or API keys into chats or GitHub.**

---

## 2. Prove it before you trust it

With MT5 open and logged in, double-click **`scripts\research.bat`**. It does everything in one go
on **your broker's own gold history** (about 10-30 minutes):

1. downloads about 2 years of M15 bars from MT5 and checks the broker's server time zone;
2. backtests the strategies and compares them with buy & hold and 200 random-entry runs;
3. runs walk-forward optimisation (tuned on old data, tested on newer data it has never seen);
4. trains the optional ML filter and reports whether it helps.

It writes **`reports\research_summary.txt`** (send this to Claude; it contains no passwords or
account numbers) plus `reports\backtest.html` and `reports\walkforward.html`.

The same steps one by one:

```bat
.venv\Scripts\python -m tulipai fetch --source mt5 --start 2024-09-01        :: ~2 years of M15 bars
.venv\Scripts\python -m tulipai backtest --data data\xauusd_m15.csv --mc 200 :: + benchmarks + report
.venv\Scripts\python -m tulipai walkforward --data data\xauusd_m15.csv       :: out-of-sample "training"
.venv\Scripts\python -m tulipai train-ml --data data\xauusd_m15.csv          :: optional ML filter
```

Each command writes an HTML report to `reports\`. Read it in this order:

| Benchmark | Question it answers | What "good" looks like |
|---|---|---|
| **Random-entry Monte Carlo** | Is the profit skill or luck? 200 runs with the *same* sessions, risk, sizing and stop/target geometry but random entries. | Verdict **EDGE** (p ≤ 0.05). **NO EDGE** means don't trust the profit, even if it's positive. |
| **Walk-forward (out-of-sample)** | Does it still work on data it was *not* tuned on? | Positive total R, and most test windows not negative. This is the honest number. |
| **Buy & hold gold** | Would simply holding gold have done better? | Better risk-adjusted return (Sharpe, drawdown) than holding. |
| **Max drawdown / loss streak** | Can you stomach it? | Within what you would accept with real money. |

The backtester is strict: bid/ask spread, slippage, stop-first when a candle hits both stop and
target, gaps filled at the open, and no look-ahead. The tests check this: on random-walk prices
the strategies must *not* make money, and the live engine replayed over history must produce
**exactly** the backtester's trades.

`walkforward` saves the latest parameters to `config\optimized_params.yaml`. Use them with
`--params config\optimized_params.yaml`, or copy them into `config\config.yaml`.

### No internet data source? Use Dukascopy

`python -m tulipai fetch --source dukascopy --start 2023-01-01 --end 2025-01-01` downloads free spot
XAUUSD history. MT5 data is still preferred because it matches your broker.

---

## 3. Measuring live: is it making money?

- **Control panel:** equity curve, today's P&L, win rate, profit factor, average R, max drawdown,
  per-strategy stats and the decision log. If you switch on Claude or the ML filter, a **veto
  scorecard** appears: every signal they blocked is later scored *as if it had been taken*.
  Negative "Avg R if taken" means the vetoes saved money; positive means they are filtering out
  winners.
- **`python -m tulipai report --mode mt5`** builds the same HTML report from the live journal, so
  you can compare live results against the backtest.
- **`python -m tulipai review`** (optional, needs an API key) asks Claude to review the journal
  offline and propose up to 5 testable changes. It never touches live trading. Test each
  suggestion with `backtest`/`walkforward` before applying it.

Everything lives in `runs\journal.db` (SQLite: decisions, trades, equity, AI calls with token
usage, shadow trades).

### Demo → cent account checklist

Switch only when **all** are true:

- ≥ 4 weeks and ≥ 50 closed trades on demo;
- live average R ≥ 0, and live results look like the walk-forward report;
- the backtest verdict is EDGE or WEAK EDGE (not NO EDGE), and the cost stress test is still positive;
- if you use Claude: the veto scorecard shows its vetoes are not costing money.

On the cent account, start at `risk_per_trade_pct: 0.5` (or lower) until live results confirm the
backtest. With no proven edge, smaller is safer.

Then set `account.allow_real_account: true` in `config\config.yaml`. Until then the bot **refuses
to send orders to any non-demo account**, including cent accounts.

---

## 4. Risk limits (`config\config.yaml`)

Neither the strategies nor Claude can override these:

| Setting | Default | Meaning |
|---|---|---|
| `risk_per_trade_pct` | 1.0 | Equity lost if the stop is hit. On a 10,000 USC (= $100) cent account that's 100 USC. |
| `max_daily_loss_pct` | 3.0 | No new trades for the rest of the UTC day. |
| `max_drawdown_pct` | 20.0 | Kill switch from peak equity; reset from the panel. |
| `max_open_positions` / `max_trades_per_day` | 1 / 4 | Exposure limits. |
| `trade_hours_utc` | 07:00-20:00 | London open → New York afternoon (gold's liquid hours). |
| `news_blackout_*_min` | 30 / 30 | No entries around high-impact USD events (NFP, CPI, FOMC, ...). |
| `max_spread`, `max_spread_atr` | 1.0, 0.25 | Skip entries when the spread blows out. |
| `min_sl_atr` / `max_sl_atr`, `min_rr` | 0.8 / 4.0, 1.0 | Stop clamped to ATR multiples; minimum reward:risk. |
| `loss_streak_pause`, `cooldown_bars` | 3, 8 | Pause after 3 losses in a row. |
| `max_lot` | 5.0 | Hard volume cap. |

Open trades: stop to break-even (+0.1R) at +1R, ATR trailing stop from +1.5R, 24h time stop,
flat before the weekend (Friday 20:00 UTC). Every order carries a stop-loss at the broker, so
trades stay protected even if your PC crashes.

**Sizing on a cent account.** Lots come from MT5's own `order_calc_profit` in your account
currency (USC), so the maths is exact for `XAUUSDc` or whichever gold symbol your broker uses. The
bot finds the symbol automatically (`XAUUSDc`, `XAUUSDm`, `XAUUSD`, `GOLD`, ...).

**Pausing.** Use **Pause new trades** in the panel, or create a file named `STOP` in the folder.
**Close all positions** flattens everything immediately.

---

## 5. Rules only vs. AI (optional)

**Rules only (`ai.mode: off`, the default)** is the recommended way to run TulipAI:

- every decision is deterministic: the same candles always give the same trade;
- the backtester reproduces the live logic exactly (only the news blackout is missing from
  backtests unless you pass `--calendar`), so the research reports describe what the bot will do;
- no API costs (on a $100 cent account a Claude call can cost more than the spread), no outage risk.

Research on systematic gold trading found no published evidence that an LLM layer improves a
rule-based gold system. The things an AI would add, like reading news, are covered by rules: the
news blackout around high-impact US releases (with a built-in official schedule as a fallback), the
spread filter and the loss limits.

### The optional Claude layer

**Claude** (model `claude-opus-5-5`, set in `ai.model`) gets a compact briefing at each decision point:

- M15 / H1 / H4 trend, ATR, ADX, RSI, EMAs, last 8 daily candles and the last 16 bars;
- key levels: Asian range, today's and yesterday's high/low, 20-day range;
- related markets your broker offers (DXY, US10Y, XAGUSD), because real yields and the dollar drive gold;
- the economic calendar for the next 24h (ForexFactory feed) and recent gold-relevant headlines (RSS);
- the quant signal, the ML probability, account state and the last 8 trade results (feedback).

It answers with schema-validated JSON: action, confidence, stop/target in ATR, a risk multiplier
(it can *lower* risk, never raise it), market bias, reasoning and key risks.

| `ai.mode` | Behaviour |
|---|---|
| `off` (default) | Rules only. No API key needed. |
| `filter` | Claude approves or vetoes each strategy signal and may tighten stop/target and risk. It can't reverse direction. |
| `autonomous` | Experimental. Claude can also propose its own trades (asked every 4 bars when flat) and overrule signals. |

Pick the mode in the panel's **Decisions** list, in `config\config.yaml`, or with `--ai` on `live`/`panel`.

Safety around the AI: minimum confidence 0.6; a daily call cap (`max_calls_per_day`); if the API
fails, the bot does *not* trade (`on_error: skip`); a declined request falls back server-side to
another model. **Cost:** filter mode calls Claude only when a signal survives the risk checks,
typically a handful of times a day. The panel shows token usage and an estimated cost.

**Why the AI is not in the backtest:** Claude has read about historical gold prices, so asking it
about 2024 candles would leak hindsight. The AI is judged **forward-only**, through the live
journal and the veto scorecard.

**ML filter (optional):** `train-ml` learns *when* the strategy signals work (trend alignment,
volatility, session, distance to EMAs, ...). It trains chronologically on older data and reports
the newer test period with and without the filter. Enable it (`ml.enabled: true`) only if the filter
improved the test period.

---

## 6. The strategies

From research on what has worked for intraday gold (sources at the bottom):

1. **Asian-range breakout at the London open** (`session_breakout`): mark the overnight range
   (00:00-08:00 London time) and take the first close beyond it during the London morning
   (08:00-14:00 London time), only with (or not against) the H4 trend. The stop sits at the middle of
   the range. London time follows UK daylight saving, so in UTC the window is 07:00-13:00 in summer
   and 08:00-14:00 in winter.
2. **Trend pullback** (`trend_pullback`): EMA50 > EMA200, EMA200 rising, ADX ≥ 20 and H4 trend up.
   Buy when price dips into the EMA20 and closes back up (mirror for shorts). Stop 1.5 ATR, target 2R.
3. **Range mean reversion** (`mean_reversion`): with ADX < 20, fade closes back inside a 2.2σ
   Bollinger band, targeting the middle band.

The **ensemble** runs all three. Breakout has priority, and opposite signals on the same bar cancel.

---

## 7. TradingView

`pine/tulipai_gold.pine` is a TradingView strategy with the same breakout and pullback rules.
Paste it into the Pine Editor on an XAUUSD M15 chart to *see* signals, the Asian range and EMAs,
cross-check results in the Strategy Tester, or set alerts. It doesn't include the Python bot's
trade management, news blackout, Claude or ML, so its numbers will differ from the bot's.

---

## 8. Commands

```
python -m tulipai panel                 # browser control panel (what TulipAI.bat runs)
python -m tulipai live [--paper]        # headless bot using .env credentials (scripts\run_headless.bat)
python -m tulipai research              # one-shot: MT5 history + backtest + benchmarks + walk-forward + ML
python -m tulipai doctor --mt5          # check packages, API key, MT5 connection, symbol spec, sizing
python -m tulipai fetch --source mt5|dukascopy|synthetic --start YYYY-MM-DD [--end ...]
python -m tulipai backtest --data CSV [--mc 200] [--ml] [--calendar events.csv] [--params file]
python -m tulipai walkforward --data CSV [--train-months 6 --test-months 1]
python -m tulipai train-ml --data CSV
python -m tulipai replay --data CSV [--ai]  # the live engine over history (simulated fills)
python -m tulipai report [--mode mt5]   # HTML report from the live journal
python -m tulipai review                # Claude reviews performance, suggests changes
python -m tulipai close-all             # flatten every bot position
python -m tulipai demo                  # synthetic-data demo of the research workflow
```

`--paper` (or *Orders: Paper* in the panel) uses live MT5 prices but simulates fills, so no orders
are sent.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| "AutoTrading is OFF" / retcode 10027 | Click **Algo Trading** in the MT5 toolbar (it must be green). |
| "Could not start/connect MetaTrader 5" | Check login, password and server. Set the terminal path if MT5 isn't in the default folder. |
| "REAL account detected" | It's a cent/live account. Demo-test first, then set `allow_real_account: true`. |
| No gold symbol found | Show all symbols in MT5 (*Market Watch → right click → Show All*), or set `symbol.name`. |
| "could not detect the server time zone" | Set `symbol.server_utc_offset_hours`: a fixed number (Exness uses `0`) or `ny+7` for brokers on UTC+2 in winter / UTC+3 in summer. Session times depend on this. |
| "invalid stops" (10016) | Your broker's minimum stop distance is large. Raise `risk.min_sl_atr`. |
| Not enough history | In MT5 *Tools → Options → Charts*, set *Max bars in chart* to Unlimited and scroll the gold chart back. |

Logs are in `logs\tulipai.log`.

---

## 10. Project layout

```
tulipai/
  strategies/         session_breakout, trend_pullback, mean_reversion, ensemble
  indicators.py       EMA, RSI, ATR, ADX, Bollinger, MACD, HTF trend (all causal)
  risk.py             guardrails + position sizing     management.py   BE / trailing / time / weekend
  execution.py        bid/ask fill & exit model shared by backtest and replay
  backtest/           engine, metrics, benchmarks (buy & hold, random-entry MC), walk-forward
  broker/             mt5.py (MetaTrader 5), paper.py (simulated fills), replay.py (history)
  ai/                 brain.py (Claude decisions, structured output), context.py (market briefing)
  news/               calendar.py (high-impact events, blackout), headlines.py (RSS)
  ml/                 meta_label.py (when do signals work?)
  live.py             the live engine           journal.py   SQLite record of everything
  panel/              local login + dashboard   report.py    HTML reports
pine/                 TradingView companion script
tests/                pytest suite (incl. a fake MetaTrader5 module)
```

Run the tests with `python -m pytest`.

---

## Research notes and sources

- Gold is most active in the London session and the London/New York overlap; the Asian session
  mostly ranges. US releases (NFP, CPI, FOMC) cause spread blow-outs and whipsaws:
  [MQL5: Gold trading sessions](https://www.mql5.com/en/blogs/post/773690),
  [dev.to: When does gold actually move](https://dev.to/xauusdrobot/when-does-gold-actually-move-xauusd-trading-sessions-explained-i4a).
- Real yields and the US dollar are the dominant drivers of the gold price:
  [PIMCO: Understanding gold prices](https://www.pimco.com/gbl/en/resources/education/understanding-gold-prices),
  [Axiory: Trading gold with real yields & dollar strength](https://www.axiory.com/education/metals-trading-series/advanced-guide-to-trading-gold-xau-usd-how-to-use/).
- Gold and Treasury momentum together signal the macro regime:
  [Quantpedia: Cross-asset price-based regimes for gold](https://quantpedia.com/cross-asset-price-based-regimes-for-gold/).
- Backtests of session breakouts and trend following on XAUUSD (treat headline win rates with
  scepticism; that's why the random-entry benchmark exists):
  [quant-signals: 3 backtested XAUUSD approaches](https://quant-signals.com/xauusd-trading-strategies/).
- Market context when this was written (October 2026): gold around $4,140 after a September
  sell-off from the January 2026 record, with US 10-year yields at 5.34% before NFP:
  [Vantage: XAUUSD and 10-year yield](https://vantagemarkets.com/market-analysis/xauusd-gold-price-news-10-year-yield-nfp-september-28-october-2-2026).

*Not financial advice. You are responsible for every trade the bot makes on your account.*
