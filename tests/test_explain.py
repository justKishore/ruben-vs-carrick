"""Every Explain button has text, and the text follows the UI rules."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import stats as S
from src.explain import EXPLAIN
from src.load import load_match_data

PAGES = Path(__file__).resolve().parents[1] / "pages"


def section_keys() -> set[str]:
    pattern = re.compile(r'ui\.section\([^\n]*?,\s*"([a-z_]+)"')
    return {k for p in PAGES.glob("*.py") for k in pattern.findall(p.read_text(encoding="utf-8"))}


def test_every_section_has_an_explanation() -> None:
    keys = section_keys()
    assert len(keys) >= 30
    assert keys <= set(EXPLAIN), keys - set(EXPLAIN)


def test_explanations_have_no_dashes() -> None:
    for key, text in EXPLAIN.items():
        assert not re.search("[\u2013\u2014]", text), key


@pytest.mark.parametrize(("d", "label"), [(0.1, "negligible"), (-0.3, "small"),
                                          (0.6, "medium"), (-1.2, "large"), (np.nan, "n/a")])
def test_effect_size_label(d: float, label: str) -> None:
    assert S.effect_size_label(d) == label


@pytest.mark.parametrize(
    ("p_raw", "p_holm", "lo", "hi", "expected"),
    [(0.001, 0.01, 0.2, 0.9, S.CLEAR),
     (0.01, 0.6, 0.2, 0.9, S.POSSIBLE),
     (0.01, 0.01, -0.1, 0.9, S.NONE),  # CI spans zero
     (0.3, 1.0, -0.5, 0.9, S.NONE)],
)
def test_verdict(p_raw: float, p_holm: float, lo: float, hi: float, expected: str) -> None:
    assert S.verdict(p_raw, p_holm, lo, hi) == expected


def test_simple_summary() -> None:
    df = S.per_game_frame(load_match_data(cache_dir=None))
    table = S.comparison_table(df, ["xG", "Opp goals", "Throw-ins", "Possession %"],
                               n_resamples=1000)
    out = S.simple_summary(table)
    assert len(out) == 4
    assert set(out["Verdict"]) <= {S.CLEAR, S.POSSIBLE, S.NONE}
    order = out["Verdict"].map(S.VERDICT_ORDER)
    assert order.is_monotonic_increasing
    assert out.loc[out["metric"] == "Opp goals", "Better under"].item() == "Carrick"
    assert out.loc[out["metric"] == "Possession %", "Better under"].item() == "Neutral stat"
    assert out.loc[out["metric"] == "Possession %", "Amorim"].item().endswith("%")
