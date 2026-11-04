"""Tests for src/stats.py."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from src import stats as S
from src.load import AMORIM, CARRICK, FLETCHER, STAT_COLS, load_match_data

FAST = 2_000  # resamples for tests that do not check exact reproducibility


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return S.per_game_frame(load_match_data(cache_dir=None))


def toy(a: list[float], b: list[float], metric: str = "x") -> pd.DataFrame:
    """Minimal frame with Amorim values a, Carrick values b, one Fletcher game."""
    managers = [AMORIM] * len(a) + [CARRICK] * len(b) + [FLETCHER]
    return pd.DataFrame({"Manager": managers, metric: list(a) + list(b) + [999.0]})


# --- module rules -------------------------------------------------------------
def test_stats_does_not_import_streamlit() -> None:
    tree = ast.parse(Path(S.__file__).read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(str(n).startswith("streamlit") for n in names)


def test_sections_cover_every_stat_column() -> None:
    covered = [c for cols in S.SECTIONS.values() for c in cols]
    assert sorted(covered) == sorted(STAT_COLS)


@pytest.mark.parametrize(
    ("key", "direction"),
    [
        ("Opp goals", -1), ("Opp xG", -1), ("Opp Total shots", -1),
        ("Opp Shots on target", -1), ("Fouls", -1), ("Yellow cards", -1),
        ("Red cards", -1), ("Errors leading to goal", -1),
        ("xG", 1), ("Points", 1), ("MU goals", 1),
        ("Clearances", 0), ("Opp Big chances missed", 0),
    ],
)
def test_direction(key: str, direction: int) -> None:
    assert S.metric_direction(key) == direction


def test_redundant_metrics_not_offered() -> None:
    assert "Tackles" not in S.METRICS
    assert "Opp Possession %" not in S.METRICS
    assert "Opp Aerial duels total" not in S.METRICS


# --- frame and filters ----------------------------------------------------------
@pytest.mark.parametrize(
    ("form", "expected"),
    [("WWWWW", 3.0), ("LDWLW", 7 / 5), ("D", 1.0), ("LW", 1.5), (np.nan, np.nan)],
)
def test_form_ppg(form: object, expected: float) -> None:
    assert S.form_ppg(form) == pytest.approx(expected, nan_ok=True)


def test_per_game_frame_derived(df: pd.DataFrame) -> None:
    row = df.set_index("GW").loc[26]
    assert row["BC conversion"] == pytest.approx(0.5)  # (2 - 1) / 2
    assert row["BC diff"] == row["Big chances"] - row["Opp Big chances"]
    assert np.isnan(df.set_index("GW").at[33, "Sprints"])
    zero_bc = df["Big chances"] == 0
    assert df.loc[zero_bc, "BC conversion"].isna().all()


def test_filter_keeps_gw1_only_for_full_position_range(df: pd.DataFrame) -> None:
    assert 1 in S.filter_games(df, opp_position=(1, 20))["GW"].values
    assert 1 not in S.filter_games(df, opp_position=(1, 10))["GW"].values


def test_filter_venue_and_manager(df: pd.DataFrame) -> None:
    out = S.filter_games(df, managers=[CARRICK], venue="Home")
    assert set(out["Manager"]) == {CARRICK}
    assert set(out["venue"]) == {"Home"}


def test_group_values_excludes_fletcher_and_nan(df: pd.DataFrame) -> None:
    a, b = S.group_values(df, "Sprints")
    assert (len(a), len(b)) == (20, 16)  # GW33 (Carrick) has no tracking
    a, b = S.group_values(df, "xG")
    assert (len(a), len(b)) == (20, 17)


# --- effect sizes, bootstrap, permutation ---------------------------------------
def test_cohens_d_known_value() -> None:
    a, b = np.array([1.0, 2, 3, 4, 5]), np.array([3.0, 4, 5, 6, 7])
    assert S.cohens_d(a, b) == pytest.approx(2 / np.sqrt(2.5))


def test_cohens_d_zero_sd_is_nan() -> None:
    assert np.isnan(S.cohens_d(np.zeros(5), np.zeros(5)))


def test_cliffs_delta_bounds() -> None:
    assert S.cliffs_delta(np.array([1.0, 2]), np.array([3.0, 4])) == 1.0
    assert S.cliffs_delta(np.array([3.0, 4]), np.array([1.0, 2])) == -1.0
    assert S.cliffs_delta(np.array([1.0, 2]), np.array([1.0, 2])) == 0.0


def test_bootstrap_ci_contains_observed_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 20), rng.normal(1, 1, 17)
    lo, hi = S.bootstrap_ci(a, b)
    assert lo < S.mean_diff(a, b) < hi
    assert S.bootstrap_ci(a, b) == (lo, hi)


def test_permutation_test_seeded_and_sane() -> None:
    a, b = np.arange(20.0), np.arange(20.0) + 100
    r1, r2 = S.permutation_test(a, b), S.permutation_test(a, b)
    assert r1.p_value == r2.p_value
    assert r1.p_value < 0.001
    assert len(r1.null_distribution) == S.N_RESAMPLES
    same = S.permutation_test(np.arange(10.0), np.arange(10.0), n_resamples=FAST)
    assert same.p_value == pytest.approx(1.0)


def test_holm_matches_manual_and_keeps_nan() -> None:
    p = pd.Series([0.01, 0.04, np.nan, 0.03])
    adj = S.holm(p)
    assert adj[0] == pytest.approx(0.03)  # 0.01 * 3
    assert adj[3] == pytest.approx(0.06)  # max(0.03, 0.03 * 2)
    assert adj[1] == pytest.approx(0.06)  # max(0.06, 0.04 * 1)
    assert np.isnan(adj[2])


def test_compare_on_toy_data_excludes_fletcher() -> None:
    c = S.compare(toy([1, 2, 3, 4], [5, 6, 7, 8]), "x", n_resamples=FAST)
    assert (c.n_amorim, c.n_carrick) == (4, 4)
    assert c.diff == pytest.approx(4.0)
    assert c.cliffs_delta == 1.0
    assert c.diff_ci_low <= 4.0 <= c.diff_ci_high
    assert c.better is None  # unknown metric is neutral


def test_compare_better_is_direction_aware(df: pd.DataFrame) -> None:
    c = S.compare(df, "Opp goals", n_resamples=FAST)
    assert c.mean_carrick < c.mean_amorim
    assert c.better == CARRICK


def test_comparison_table_has_raw_and_holm(df: pd.DataFrame) -> None:
    t = S.comparison_table(df, ["xG", "Opp xG", "Points"], n_resamples=FAST)
    for col in ("p_welch", "p_mann_whitney", "p_permutation"):
        assert (t[f"{col}_holm"] >= t[col]).all()
    assert {"n_amorim", "n_carrick", "cohens_d", "cliffs_delta"} <= set(t.columns)


# --- results ---------------------------------------------------------------------
def test_fisher_2xk_matches_scipy_on_2x2() -> None:
    table = np.array([[8, 2], [1, 5]])
    assert S.fisher_exact_2xk(table) == pytest.approx(sps.fisher_exact(table).pvalue)


def test_wdl_table_matches_points(df: pd.DataFrame) -> None:
    t = S.wdl_table(df)
    assert list(t.index) == [AMORIM, CARRICK]
    assert t.sum(axis=1).tolist() == [20, 17]
    for mgr in t.index:
        pts = df.loc[df["Manager"] == mgr, "Points"].sum()
        assert 3 * t.at[mgr, "W"] + t.at[mgr, "D"] == pts


def test_wdl_test_outputs(df: pd.DataFrame) -> None:
    r = S.wdl_test(df, n_boot=FAST)
    assert 0 <= r.p_fisher <= 1
    assert 0 <= r.cramers_v <= 1
    assert r.win_rate_ci_low <= r.win_rate_diff <= r.win_rate_ci_high


def test_cumulative_points_recounts_games(df: pd.DataFrame) -> None:
    c = S.cumulative_points(df)
    last = c.groupby("Manager").last()
    assert last.at[CARRICK, "games_managed"] == 17
    assert last.at[AMORIM, "cum_points"] == df.loc[df["Manager"] == AMORIM, "Points"].sum()


# --- models ---------------------------------------------------------------------
def test_adjusted_model_drops_gw1(df: pd.DataFrame) -> None:
    r = S.adjusted_model(df, "xG", n_boot=FAST)
    assert (r.n_amorim, r.n_carrick) == (19, 17)
    assert r.ci_low <= r.estimate <= r.ci_high
    assert set(r.covariates) == {"home", "opp_position", "opp_form_ppg"}


def test_adjusted_model_drops_constant_venue(df: pd.DataFrame) -> None:
    r = S.adjusted_model(S.filter_games(df, venue="Home"), "xG", n_boot=FAST)
    assert "home" not in r.covariates


def test_count_model_rate_ratio_matches_means(df: pd.DataFrame) -> None:
    r = S.count_model(df, "Opp goals", "poisson", n_boot=FAST)
    a, b = S.group_values(df, "Opp goals")
    assert r.estimate == pytest.approx(b.mean() / a.mean(), rel=1e-6)
    assert r.ci_low < r.estimate < r.ci_high
    nb = S.count_model(df, "Big chances missed", "negbin", n_boot=FAST)
    assert np.isfinite(nb.estimate)


# --- time series --------------------------------------------------------------------
def test_autocorrelation_excludes_fletcher_and_nan(df: pd.DataFrame) -> None:
    r = S.autocorrelation(df, "Sprints")
    assert r.n == 36  # 37 games minus GW33
    assert r.acf[0] == pytest.approx(1.0)
    assert {"lag", "lb_stat", "lb_pvalue"} <= set(r.ljung_box.columns)


def test_change_points_detects_obvious_shift() -> None:
    frame = pd.DataFrame({
        "GW": range(1, 21),
        "Manager": [AMORIM] * 20,
        "x": [0.0] * 10 + [10.0] * 10,
    })
    assert S.change_points(frame, "x") == [11]


def test_rolling_mean_within_manager(df: pd.DataFrame) -> None:
    r = S.rolling_mean(df, "Points", window=3)
    first_carrick = df.index[df["GW"] == 22][0]
    assert r[first_carrick] == df.at[first_carrick, "Points"]


# --- luck vs skill -------------------------------------------------------------------
def test_match_outcome_probs_sum_and_symmetry() -> None:
    w, d, l = S.match_outcome_probs(np.array([1.4]), np.array([1.4]))
    assert w[0] + d[0] + l[0] == pytest.approx(1.0, abs=1e-6)
    assert w[0] == pytest.approx(l[0])


def test_expected_points(df: pd.DataFrame) -> None:
    xp = S.expected_points(df)
    assert np.isnan(xp.set_index("GW").at[1, "xPts"])
    known = xp["xPts"].dropna()
    assert ((known >= 0) & (known <= 3)).all()


def test_pythagorean_points(df: pd.DataFrame) -> None:
    p = S.pythagorean_points(df, n_boot=FAST).set_index("Manager")
    assert p.at[AMORIM, "n"] == 20 and p.at[CARRICK, "n"] == 17
    assert (p["ci_low"] <= p["Pythagorean PPG"]).all()
    assert (p["Pythagorean PPG"] <= p["ci_high"]).all()


def test_bc_conversion_counts(df: pd.DataFrame) -> None:
    r = S.bc_conversion(df, "mu", n_boot=FAST)
    a = df[df["Manager"] == AMORIM]
    assert r.created_amorim == a["Big chances"].sum()
    assert r.scored_amorim == r.created_amorim - r.missed_amorim
    assert r.conversion_carrick == pytest.approx(r.scored_carrick / r.created_carrick)
    assert 0 <= r.p_fisher <= 1


@pytest.mark.parametrize(
    ("bc", "xg", "tag"),
    [(-1, -0.5, "Deserved"), (0, -0.5, "Unlucky"), (-1, 0.0, "Unlucky"), (2, 1, "Unlucky")],
)
def test_tag_loss(bc: float, xg: float, tag: str) -> None:
    assert S.tag_loss(bc, xg) == tag


def test_loss_table_and_counts(df: pd.DataFrame) -> None:
    losses = S.loss_table(df)
    assert len(losses) == (df["Result"] == "L").sum()
    counts = S.loss_tag_counts(df)
    assert (counts["Deserved"] + counts["Unlucky"] == counts["Losses"]).all()


def test_ppg_by_bc_bucket_totals(df: pd.DataFrame) -> None:
    t = S.ppg_by_bc_bucket(df)
    totals = t.groupby("Manager")["games"].sum()
    assert totals[AMORIM] == 20 and totals[CARRICK] == 17


# --- exploratory ------------------------------------------------------------------------
def test_driver_metrics_exclude_outcomes() -> None:
    assert not set(S.driver_metrics()) & S.OUTCOME_METRICS


def test_pca_kmeans_runs(df: pd.DataFrame) -> None:
    r = S.pca_kmeans(df, ["xG", "Opp xG", "Possession %", "Total shots"], n_resamples=200)
    assert r.n_amorim + r.n_carrick == len(r.scores)
    assert -1 <= r.adjusted_rand <= 1


def test_correlation_with_points(df: pd.DataFrame) -> None:
    c = S.correlation_with_points(df, ["xG", "BC diff"], n_boot=FAST)
    assert ((c["rho"] >= -1) & (c["rho"] <= 1)).all()
    assert (c["ci_low"] <= c["ci_high"]).all()


# --- wording ----------------------------------------------------------------------------
def test_describe_has_n_and_no_em_dash(df: pd.DataFrame) -> None:
    text = S.describe_comparison(S.compare(df, "Points", n_resamples=FAST))
    assert "n=17 vs 20" in text
    assert "—" not in text and "–" not in text
    model_text = S.describe_model(S.adjusted_model(df, "xG", n_boot=FAST), "xG")
    assert "confounded" in model_text and "—" not in model_text


# --- findings, areas and style ---------------------------------------------------------
def test_area_groups_are_valid_and_unique() -> None:
    for area, ms in S.STYLE_GROUPS.items():
        assert len(ms) == 8, area
        assert set(ms) <= set(S.METRICS), area
    assert len(S.area_metrics()) == len(set(S.area_metrics()))
    assert S.area_of("Throw-ins") == ""
    assert S.area_of("Possession %") == "Passing"
    assert S.area_of("xG") == "Attack"  # style areas win over Results
    assert S.area_of("Points") == "Results"


def test_plain_finding_wording(df: pd.DataFrame) -> None:
    text = S.plain_finding(S.compare(df, "Points", n_resamples=FAST))
    assert text.startswith("Points went up under Carrick")
    assert "better for Man Utd" in text
    assert "\u2014" not in text


def test_group_finding_counts_each_metric_once(df: pd.DataFrame) -> None:
    table = S.comparison_table(df, S.area_metrics(), n_resamples=FAST)
    text = S.group_finding(table, S.STYLE_GROUPS["Attack"])
    assert "of 8" in text or "the 8" in text


def test_improvement_table_and_summary(df: pd.DataFrame) -> None:
    t = S.improvement_table(df, n_resamples=FAST)
    assert list(t["metric"]) == list(S.KEY_METRICS)
    assert (t["p_permutation_holm"] >= t["p_permutation"]).all()
    headline, detail = S.improvement_summary(t, S.wdl_test(df, n_boot=FAST))
    assert headline.split(".")[0] in {"Not proven", "Yes, in part", "No"}
    assert "Points per game went from 1.55 to 2.29" in detail


def test_style_classifier_runs(df: pd.DataFrame) -> None:
    r = S.style_classifier(df, S.style_metrics(), n_permutations=20)
    assert 0 <= r.accuracy <= 1
    assert r.baseline == pytest.approx(20 / 37)
    assert (r.n_amorim, r.n_carrick) == (20, 17)
    assert len(r.null_scores) == 20
