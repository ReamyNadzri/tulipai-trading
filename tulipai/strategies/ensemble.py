"""Combine several strategies. Conflicting directions on the same bar cancel out;
otherwise the first member (priority order) that fires wins."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import Features
from .base import Strategy


def combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge member signal frames given in priority order (first wins, conflicts cancel)."""
    sigs = np.vstack([fr["signal"].to_numpy() for fr in frames])
    conflict = (sigs > 0).any(axis=0) & (sigs < 0).any(axis=0)
    out = frames[-1].copy()
    for fr in reversed(frames[:-1]):  # earlier members overwrite later ones
        hit = fr["signal"].to_numpy() != 0
        out.loc[hit, :] = fr.loc[hit, :]
    out.loc[conflict, "signal"] = 0
    out.loc[out["signal"] == 0, ["sl_dist", "tp_dist", "strength"]] = 0.0
    out.loc[out["signal"] == 0, "strategy"] = ""
    out["signal"] = out["signal"].astype(int)
    return out


class Ensemble(Strategy):
    name = "ensemble"
    default_params: dict = {}

    def __init__(self, members: list[Strategy]):
        super().__init__()
        if not members:
            raise ValueError("Ensemble needs at least one member strategy")
        self.members = members

    @property
    def warmup(self) -> int:
        return max(m.warmup for m in self.members)

    def generate(self, f: Features) -> pd.DataFrame:
        return combine_frames([m.generate(f) for m in self.members])

    def describe(self, verbose: bool = False) -> str:
        return "ensemble[" + ", ".join(m.describe(verbose) for m in self.members) + "]"
