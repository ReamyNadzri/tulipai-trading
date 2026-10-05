import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tulipai.config import Config  # noqa: E402
from tulipai.data.synthetic import synthetic_gold  # noqa: E402


@pytest.fixture(scope="session")
def gold_df():
    return synthetic_gold(days=120, seed=5)


@pytest.fixture()
def cfg():
    c = Config()
    c.ai.mode = "off"
    c.news.enabled = False
    c.strategy.auto_retune = False  # tests that want it switch it on with a temp params file
    return c
