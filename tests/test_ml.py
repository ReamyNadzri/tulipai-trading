import numpy as np
import pytest

pytest.importorskip("sklearn")

from tulipai.backtest import Backtester
from tulipai.data.synthetic import synthetic_gold
from tulipai.indicators import Features
from tulipai.ml import MetaLabelModel, build_dataset, train
from tulipai.strategies import build_strategy


@pytest.fixture(scope="module")
def long_df():
    return synthetic_gold(days=330, seed=9)


def test_dataset_only_uses_in_session_signals(long_df, cfg):
    sig = build_strategy(cfg.strategy).generate(Features(long_df))
    ds = build_dataset(long_df, sig, cfg)
    assert len(ds) > 150
    close_hours = (ds["time"] + np.timedelta64(15, "m")).dt.hour
    assert close_hours.between(7, 19).all()
    assert set(ds["exit_reason"]) <= {"sl", "tp", "time", "trail"}


def test_train_save_load_and_gate(long_df, cfg, tmp_path):
    f = Features(long_df)
    sig = build_strategy(cfg.strategy).generate(f)
    model, rep = train(long_df, sig, cfg, out_path=tmp_path / "m.joblib")
    assert rep.n_train > rep.n_test > 0 and 0 <= rep.test_kept_frac <= 1
    loaded = MetaLabelModel.load(tmp_path / "m.joblib")
    assert loaded.threshold == model.threshold
    gate = loaded.gate(f)
    res = Backtester(cfg).run(long_df, sig, start=300, features=f, ml_gate=gate, ml_threshold=0.99)
    base = Backtester(cfg).run(long_df, sig, start=300, features=f)
    assert len(res.trades) < len(base.trades)  # a strict threshold filters signals out
    assert res.blocked.get("ml filter", 0) > 0
