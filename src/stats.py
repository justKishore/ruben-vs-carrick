"""Statistics for the Amorim vs Carrick comparison.

Conventions:
- Comparisons are always Carrick minus Amorim. Fletcher is never part of a test.
- NaN means missing and is dropped per metric, never filled.
- Every test result carries n per manager, an effect size and a bootstrap CI.
- Resampling uses N_RESAMPLES draws and a fixed SEED so results are reproducible.

This module must not import streamlit.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from scipy.special import gammaln
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tsa.stattools import acf

from src.load import AMORIM, CARRICK, COUNT_COLS, PCT_COLS, STAT_COLS, MatchData

SEED = 2026
N_RESAMPLES = 10_000
CI_LEVEL = 0.95
ALPHA = 0.05
TESTED = (AMORIM, CARRICK)

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------
SECTIONS: dict[str, tuple[str, ...]] = {
    "Match overview": (
        "Possession %", "Distance covered (km)", "Sprints", "xG", "Big chances",
        "Total shots", "GK saves", "Corners", "Fouls", "Passes", "Tackles",
        "Free kicks", "Yellow cards", "Red cards",
    ),
    "Shots": (
        "Shots on target", "xG on target (xGOT)", "Hit woodwork",
        "Shots off target", "Blocked shots", "Shots inside box",
        "Shots outside box",
    ),
    "Attack": (
        "Big chances scored", "Big chances missed", "Through balls",
        "Touches in opp box", "Fouled in final third", "Offsides",
    ),
    "Duels": (
        "Duels won %", "Dispossessed", "Ground duels won", "Ground duels total",
        "Ground duels %", "Aerial duels won", "Aerial duels total",
        "Aerial duels %", "Dribbles successful", "Dribbles attempted",
        "Dribbles %",
    ),
    "Passes": (
        "Accurate passes", "Throw-ins", "Final third entries",
        "Passes in final third accurate", "Passes in final third total",
        "Passes in final third %", "Long balls accurate", "Long balls total",
        "Long balls %", "Crosses accurate", "Crosses total", "Crosses %",
        "Pass accuracy %",
    ),
    "Defending": (
        "Tackles won %", "Total tackles", "Interceptions", "Recoveries",
        "Clearances", "Errors leading to shot", "Errors leading to goal",
    ),
    "Goalkeeping": (
        "Goals prevented", "Big saves", "High claims", "Punches", "Goal kicks",
        "Penalty saves",
    ),
}

# "Tackles" is identical to "Total tackles" in every row, so it is not offered
# as a separate metric (it would double count in Holm corrections).
DUPLICATE_COLS = frozenset({"Tackles"})
# Opponent versions that equal (totals) or mirror (100% minus) the MU stat in
# every game, so they carry no extra information.
REDUNDANT_AGAINST = frozenset({
    "Possession %", "Duels won %", "Ground duels total", "Ground duels %",
    "Aerial duels total", "Aerial duels %",
})

# Direction for Man Utd's own stats: +1 higher is better (the default),
# -1 lower is better, 0 neutral (volume or style stats with no clear "better").
LOWER_IS_BETTER = frozenset({
    "Fouls", "Yellow cards", "Red cards", "Errors leading to shot",
    "Errors leading to goal", "Dispossessed", "Big chances missed",
})
NEUTRAL = frozenset({
    "Possession %", "Distance covered (km)", "Sprints", "GK saves",
    "Free kicks", "Offsides", "Throw-ins", "Hit woodwork", "Shots off target",
    "Blocked shots", "Ground duels total", "Aerial duels total",
    "Dribbles attempted", "Long balls accurate", "Long balls total",
    "Long balls %", "Crosses total", "Total tackles", "Interceptions",
    "Recoveries", "Clearances", "Big saves", "High claims", "Punches",
    "Goal kicks", "Penalty saves",
})

MetricKind = Literal["count", "pct", "continuous", "points"]


@dataclass(frozen=True)
class Metric:
    key: str  # column name in the per-game frame
    label: str  # UI label
    section: str
    direction: int  # +1 higher better, -1 lower better, 0 neutral
    kind: MetricKind


def _mu_direction(col: str) -> int:
    if col in LOWER_IS_BETTER:
        return -1
    if col in NEUTRAL:
        return 0
    return 1


def _kind(col: str) -> MetricKind:
    if col in PCT_COLS:
        return "pct"
    if col in COUNT_COLS:
        return "count"
    return "continuous"


def _build_metrics() -> dict[str, Metric]:
    derived = [
        Metric("Points", "Points", "Results", 1, "points"),
        Metric("MU goals", "Goals scored", "Results", 1, "count"),
        Metric("Opp goals", "Goals conceded", "Results", -1, "count"),
        Metric("Goal difference", "Goal difference", "Results", 1, "continuous"),
        Metric("xG difference", "xG difference", "Chance quality", 1, "continuous"),
        Metric("Goals minus xG", "Goals minus xG", "Chance quality", 1, "continuous"),
        Metric("Goals minus xGOT", "Goals minus xGOT", "Chance quality", 1, "continuous"),
        Metric("Conceded minus xG against", "Goals conceded minus xG against",
               "Chance quality", -1, "continuous"),
        Metric("BC diff", "Big chance difference", "Chance quality", 1, "continuous"),
        Metric("BC conversion", "Big chance conversion", "Chance quality", 1, "pct"),
        # Opponent finishing is luck from Man Utd's side, so it gets no colour.
        Metric("Opp BC conversion", "Big chance conversion (against)",
               "Chance quality", 0, "pct"),
    ]
    out = {m.key: m for m in derived}
    for section, cols in SECTIONS.items():
        for col in cols:
            if col in DUPLICATE_COLS:
                continue
            d = _mu_direction(col)
            out[col] = Metric(col, col, section, d, _kind(col))
            if col in REDUNDANT_AGAINST:
                continue
            # Against: what Man Utd allow. Only "higher is better for MU" stats
            # flip to "lower is better"; everything else is neutral.
            out[f"Opp {col}"] = Metric(
                f"Opp {col}", f"{col} (against)", f"{section} (against)",
                -1 if d == 1 else 0, _kind(col),
            )
    return out


METRICS: dict[str, Metric] = _build_metrics()

CONTEXT_LABELS = {
    "opp_position": "Opponent pre-match position",
    "opp_form_ppg": "Opponent form (points per game, last 5)",
}


def metric_label(key: str) -> str:
    if key in METRICS:
        return METRICS[key].label
    return CONTEXT_LABELS.get(key, key)


def metric_direction(key: str) -> int:
    return METRICS[key].direction if key in METRICS else 0


def metric_kind(key: str) -> MetricKind:
    return METRICS[key].kind if key in METRICS else "continuous"


def metrics_by_section() -> dict[str, list[str]]:
    """Metric keys grouped by section, in display order, for the picker."""
    out: dict[str, list[str]] = {}
    for m in METRICS.values():
        out.setdefault(m.section, []).append(m.key)
    return out


# ---------------------------------------------------------------------------
# Per-game frame and filters
# ---------------------------------------------------------------------------
FORM_POINTS = {"W": 3, "D": 1, "L": 0}


def form_ppg(form: object) -> float:
    """Points per game from a form string like 'WDL'. NaN if missing."""
    if not isinstance(form, str) or not form:
        return np.nan
    return float(np.mean([FORM_POINTS[c] for c in form]))


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den).where(den > 0)


def per_game_frame(data: MatchData) -> pd.DataFrame:
    """One row per GW: context, MU stats, opponent stats ("Opp " prefix), derived."""
    mu, opp = data.mu, data.opp
    if not mu["GW"].equals(opp["GW"]):
        raise ValueError("MU and opponent frames are not aligned on GW")
    context = mu[[
        "GW", "Date", "Manager", "Opponent", "H/A", "Stadium", "Result",
        "Points", "MU goals", "Opp goals",
    ]].copy()
    context["venue"] = mu["H/A"].map({"H": "Home", "A": "Away"})
    context["opp_position"] = mu["Opp position"]
    context["opp_form_ppg"] = mu["Opp form (last 5)"].map(form_ppg)
    context["games_managed"] = context.groupby("Manager").cumcount() + 1

    own = mu[list(STAT_COLS)]
    against = opp[list(STAT_COLS)].add_prefix("Opp ")
    df = pd.concat([context, own, against], axis=1)

    bc, bc_missed = df["Big chances"], df["Big chances missed"]
    obc, obc_missed = df["Opp Big chances"], df["Opp Big chances missed"]
    derived = pd.DataFrame({
        "Goal difference": df["MU goals"] - df["Opp goals"],
        "xG difference": df["xG"] - df["Opp xG"],
        "Goals minus xG": df["MU goals"] - df["xG"],
        "Goals minus xGOT": df["MU goals"] - df["xG on target (xGOT)"],
        "Conceded minus xG against": df["Opp goals"] - df["Opp xG"],
        "BC diff": bc - obc,
        "BC conversion": _safe_ratio(bc - bc_missed, bc),
        "Opp BC conversion": _safe_ratio(obc - obc_missed, obc),
    })
    df = pd.concat([df, derived], axis=1)
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns: {df.columns[df.columns.duplicated()].tolist()}")
    return df


def filter_games(
    df: pd.DataFrame,
    managers: Sequence[str] | None = None,
    venue: Literal["All", "Home", "Away"] = "All",
    opp_position: tuple[int, int] = (1, 20),
    gw_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Apply sidebar filters. GW1 (no pre-match position) is kept only when the
    position range is the full 1 to 20."""
    mask = pd.Series(True, index=df.index)
    if managers is not None:
        mask &= df["Manager"].isin(managers)
    if venue != "All":
        mask &= df["venue"] == venue
    lo, hi = opp_position
    pos_ok = df["opp_position"].between(lo, hi)
    if lo <= 1 and hi >= 20:
        pos_ok |= df["opp_position"].isna()
    mask &= pos_ok
    if gw_range is not None:
        mask &= df["GW"].between(*gw_range)
    return df.loc[mask]


def group_values(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """(Amorim values, Carrick values) for a metric, NaN dropped, Fletcher excluded."""
    a = df.loc[df["Manager"] == AMORIM, metric].dropna().to_numpy(float)
    b = df.loc[df["Manager"] == CARRICK, metric].dropna().to_numpy(float)
    return a, b


# ---------------------------------------------------------------------------
# Effect sizes, bootstrap, permutation
# ---------------------------------------------------------------------------
def _scalar(x: np.ndarray) -> float | np.ndarray:
    x = np.asarray(x, float)
    return float(x) if x.ndim == 0 else x


def mean_diff(a: np.ndarray, b: np.ndarray) -> float | np.ndarray:
    """mean(b) - mean(a) along the last axis (vectorised for bootstrap)."""
    return _scalar(np.mean(b, axis=-1) - np.mean(a, axis=-1))


def ratio_of_means(a: np.ndarray, b: np.ndarray) -> float | np.ndarray:
    ma, mb = np.mean(a, axis=-1), np.mean(b, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _scalar(np.where(ma > 0, mb / ma, np.nan))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float | np.ndarray:
    """Standardised mean difference (b - a) / pooled SD. NaN if SD is zero."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.shape[-1], b.shape[-1]
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(axis=-1, ddof=1), b.var(axis=-1, ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (b.mean(axis=-1) - a.mean(axis=-1)) / sp
    return _scalar(np.where(sp > 0, d, np.nan))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(b > a) - P(b < a), from -1 to 1."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.sign(np.subtract.outer(b, a)).mean())


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    stat: Callable[[np.ndarray, np.ndarray], float | np.ndarray] = mean_diff,
    n_boot: int = N_RESAMPLES,
    seed: int = SEED,
    level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Percentile CI, resampling games within each manager."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    A = a[rng.integers(0, len(a), (n_boot, len(a)))]
    B = b[rng.integers(0, len(b), (n_boot, len(b)))]
    vals = np.asarray(stat(A, B), float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 0.5 * n_boot:
        return (np.nan, np.nan)
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(vals, [tail, 100 - tail])
    return (float(lo), float(hi))


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    null_distribution: np.ndarray
    n_amorim: int
    n_carrick: int


def permutation_test(
    a: np.ndarray, b: np.ndarray, n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> PermutationResult:
    """Two-sided permutation test on the difference in means (b - a)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return PermutationResult(np.nan, np.nan, np.array([]), len(a), len(b))
    res = sps.permutation_test(
        (a, b),
        lambda x, y, axis: np.mean(y, axis=axis) - np.mean(x, axis=axis),
        permutation_type="independent",
        vectorized=True,
        n_resamples=n_resamples,
        alternative="two-sided",
        rng=np.random.default_rng(seed),
    )
    return PermutationResult(
        float(res.statistic), float(res.pvalue), res.null_distribution, len(a), len(b)
    )


def permutation_for_metric(
    df: pd.DataFrame, metric: str, n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> PermutationResult:
    """Permutation test of Carrick minus Amorim for one metric of a frame."""
    return permutation_test(*group_values(df, metric), n_resamples=n_resamples, seed=seed)


def holm(p: pd.Series) -> pd.Series:
    """Holm-adjusted p-values, leaving NaN where the raw p is NaN."""
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    if ok.any():
        out[ok] = multipletests(p[ok].to_numpy(float), method="holm")[1]
    return out


# ---------------------------------------------------------------------------
# Two-group comparison (default test set)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Comparison:
    metric: str
    label: str
    direction: int
    n_amorim: int
    n_carrick: int
    mean_amorim: float
    mean_carrick: float
    median_amorim: float
    median_carrick: float
    diff: float  # Carrick minus Amorim
    diff_ci_low: float
    diff_ci_high: float
    cohens_d: float
    d_ci_low: float
    d_ci_high: float
    cliffs_delta: float
    p_welch: float
    p_mann_whitney: float
    p_permutation: float
    better: str | None  # descriptive direction only, not a significance claim


def _better(diff: float, direction: int) -> str | None:
    if direction == 0 or not np.isfinite(diff) or diff == 0:
        return None
    return CARRICK if diff * direction > 0 else AMORIM


def compare(
    df: pd.DataFrame, metric: str, n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> Comparison:
    """Welch t, Mann-Whitney and permutation test, with effect sizes and CIs."""
    a, b = group_values(df, metric)
    enough = len(a) >= 2 and len(b) >= 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_welch = float(sps.ttest_ind(b, a, equal_var=False).pvalue) if enough else np.nan
        p_mw = (
            float(sps.mannwhitneyu(b, a, alternative="two-sided").pvalue)
            if enough else np.nan
        )
    perm = permutation_test(a, b, n_resamples, seed) if enough else None
    diff = mean_diff(a, b) if enough else np.nan
    direction = metric_direction(metric)
    return Comparison(
        metric=metric,
        label=metric_label(metric),
        direction=direction,
        n_amorim=len(a),
        n_carrick=len(b),
        mean_amorim=float(np.mean(a)) if len(a) else np.nan,
        mean_carrick=float(np.mean(b)) if len(b) else np.nan,
        median_amorim=float(np.median(a)) if len(a) else np.nan,
        median_carrick=float(np.median(b)) if len(b) else np.nan,
        diff=diff,
        diff_ci_low=(ci := bootstrap_ci(a, b, mean_diff, n_resamples, seed))[0],
        diff_ci_high=ci[1],
        cohens_d=cohens_d(a, b),
        d_ci_low=(dci := bootstrap_ci(a, b, cohens_d, n_resamples, seed))[0],
        d_ci_high=dci[1],
        cliffs_delta=cliffs_delta(a, b),
        p_welch=p_welch,
        p_mann_whitney=p_mw,
        p_permutation=perm.p_value if perm else np.nan,
        better=_better(diff, direction),
    )


def comparison_table(
    df: pd.DataFrame,
    metrics: Sequence[str],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> pd.DataFrame:
    """One row per metric with raw and Holm-adjusted p for all three tests.
    Holm is applied across the metrics passed in."""
    rows = [asdict(compare(df, m, n_resamples, seed)) for m in metrics]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for col in ("p_welch", "p_mann_whitney", "p_permutation"):
        table[f"{col}_holm"] = holm(table[col])
    return table


CLEAR = "Clear difference"
POSSIBLE = "Possible difference"
NONE = "No reliable difference"
VERDICT_ORDER = {CLEAR: 0, POSSIBLE: 1, NONE: 2}


def effect_size_label(d: float) -> str:
    """Cohen's conventional words for |d|."""
    if not np.isfinite(d):
        return "n/a"
    size = abs(d)
    if size < 0.2:
        return "negligible"
    if size < 0.5:
        return "small"
    return "medium" if size < 0.8 else "large"


def verdict(p_raw: float, p_holm: float, ci_low: float, ci_high: float) -> str:
    """Three-level plain verdict. 'Clear' needs Holm p < alpha and a CI that
    excludes zero; 'Possible' only meets that without the Holm correction."""
    if is_reliable(p_holm, ci_low, ci_high):
        return CLEAR
    if is_reliable(p_raw, ci_low, ci_high):
        return POSSIBLE
    return NONE


def simple_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Reader-friendly view of comparison_table: one row per stat with averages,
    the change and its range, an effect size word and a verdict."""
    rows = []
    for r in table.itertuples():
        kind = metric_kind(r.metric)
        rows.append({
            "metric": r.metric,
            "Stat": r.label,
            "Section": METRICS[r.metric].section if r.metric in METRICS else "",
            "Amorim": fmt_value(r.mean_amorim, kind),
            "Carrick": fmt_value(r.mean_carrick, kind),
            "Change": fmt_value(r.diff, kind, signed=True),
            "Plausible range (95%)": (f"{fmt_value(r.diff_ci_low, kind, True)} to "
                                      f"{fmt_value(r.diff_ci_high, kind, True)}"),
            "Effect size": effect_size_label(r.cohens_d),
            "Better under": (r.better.split()[-1] if isinstance(r.better, str)
                             else "Neutral stat" if r.direction == 0 else "No change"),
            "Verdict": verdict(r.p_permutation, r.p_permutation_holm,
                               r.diff_ci_low, r.diff_ci_high),
            "Games (A / C)": f"{r.n_amorim} / {r.n_carrick}",
            "_abs_d": abs(r.cohens_d) if np.isfinite(r.cohens_d) else -1.0,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["Verdict", "_abs_d"], ascending=[True, False],
                          key=lambda s: s.map(VERDICT_ORDER) if s.name == "Verdict" else s)
    return out.drop(columns="_abs_d").reset_index(drop=True)


def forest_data(table: pd.DataFrame) -> pd.DataFrame:
    """Comparison table sorted for a forest plot. d_better is d signed so that
    positive favours Carrick (raw d for neutral stats)."""
    out = table.copy()
    sign = out["direction"].replace(0, 1)
    out["d_better"] = out["cohens_d"] * sign
    lo, hi = out["d_ci_low"] * sign, out["d_ci_high"] * sign
    out["d_better_low"], out["d_better_high"] = np.minimum(lo, hi), np.maximum(lo, hi)
    out["verdict"] = [verdict(r.p_permutation, r.p_permutation_holm, r.diff_ci_low, r.diff_ci_high)
                      for r in out.itertuples()]
    return out.sort_values("d_better", key=np.abs, ascending=False, na_position="last")


# ---------------------------------------------------------------------------
# Results: W/D/L, KPIs, cumulative points
# ---------------------------------------------------------------------------
def wdl_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of W/D/L for Amorim and Carrick."""
    games = df[df["Manager"].isin(TESTED)]
    return (
        pd.crosstab(games["Manager"], games["Result"])
        .reindex(index=list(TESTED), columns=["W", "D", "L"], fill_value=0)
        .fillna(0)
        .astype(int)
    )


def fisher_exact_2xk(table: np.ndarray) -> float:
    """Exact Fisher-Freeman-Halton test for a 2 x k table (two-sided)."""
    t = np.asarray(table, dtype=int)
    if t.shape[0] != 2:
        raise ValueError("table must have 2 rows")
    t = t[:, t.sum(axis=0) > 0]
    cols, r1 = t.sum(axis=0), int(t[0].sum())
    n = int(cols.sum())

    def log_comb(nn: np.ndarray, kk: np.ndarray) -> np.ndarray:
        return gammaln(nn + 1) - gammaln(kk + 1) - gammaln(nn - kk + 1)

    def log_p(row: np.ndarray) -> float:
        return float(log_comb(cols, row).sum() - log_comb(np.array(n), np.array(r1)))

    observed = log_p(t[0])
    total = 0.0
    for row in itertools.product(*(range(c + 1) for c in cols)):
        if sum(row) != r1:
            continue
        lp = log_p(np.array(row))
        if lp <= observed + 1e-7:
            total += np.exp(lp)
    return float(min(1.0, total))


@dataclass(frozen=True)
class WDLTest:
    table: pd.DataFrame
    p_fisher: float
    cramers_v: float
    win_rate_amorim: float
    win_rate_carrick: float
    win_rate_diff: float
    win_rate_ci_low: float
    win_rate_ci_high: float
    n_amorim: int
    n_carrick: int


def wdl_test(df: pd.DataFrame, n_boot: int = N_RESAMPLES, seed: int = SEED) -> WDLTest:
    """Fisher exact on the W/D/L table, Cramer's V, and win rate difference CI."""
    table = wdl_table(df)
    t = table.to_numpy()
    nonzero = t[:, t.sum(axis=0) > 0]
    n = int(t.sum())
    if nonzero.shape[1] >= 2 and (t.sum(axis=1) > 0).all():
        chi2 = sps.chi2_contingency(nonzero, correction=False)[0]
        v = float(np.sqrt(chi2 / (n * (min(nonzero.shape) - 1))))
        p = fisher_exact_2xk(t)
    else:
        v, p = np.nan, np.nan
    a, b = group_values(df.assign(_win=(df["Result"] == "W").astype(float)), "_win")
    lo, hi = bootstrap_ci(a, b, mean_diff, n_boot, seed)
    return WDLTest(
        table=table,
        p_fisher=p,
        cramers_v=v,
        win_rate_amorim=float(a.mean()) if len(a) else np.nan,
        win_rate_carrick=float(b.mean()) if len(b) else np.nan,
        win_rate_diff=mean_diff(a, b) if len(a) and len(b) else np.nan,
        win_rate_ci_low=lo,
        win_rate_ci_high=hi,
        n_amorim=len(a),
        n_carrick=len(b),
    )


KPI_SPEC: dict[str, tuple[str, int]] = {
    "PPG": ("Points", 1),
    "GF": ("MU goals", 1),
    "GA": ("Opp goals", -1),
    "xG": ("xG", 1),
    "xGA": ("Opp xG", -1),
    "Win %": ("_win", 1),
}


def kpi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-manager per-game KPIs with n. Includes every manager present."""
    games = df.assign(_win=(df["Result"] == "W").astype(float))
    out = games.groupby("Manager", sort=False).agg(
        n=("GW", "size"), **{k: (col, "mean") for k, (col, _) in KPI_SPEC.items()}
    )
    return out


def cumulative_points(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative points by games managed (recounted within the filtered set)."""
    out = df.sort_values("GW")[["GW", "Manager", "Opponent", "Result", "Points"]].copy()
    out["games_managed"] = out.groupby("Manager").cumcount() + 1
    out["cum_points"] = out.groupby("Manager")["Points"].cumsum()
    out["cum_ppg"] = out["cum_points"] / out["games_managed"]
    return out


# ---------------------------------------------------------------------------
# Count models (goals, big chances missed): Poisson and NegBin
# ---------------------------------------------------------------------------
COVARIATES = ("home", "opp_position", "opp_form_ppg")


def _design(df: pd.DataFrame, metric: str, adjusted: bool) -> tuple[pd.Series, pd.DataFrame]:
    games = df[df["Manager"].isin(TESTED)]
    X = pd.DataFrame({"carrick": (games["Manager"] == CARRICK).astype(float)})
    if adjusted:
        X["home"] = (games["venue"] == "Home").astype(float)
        X["opp_position"] = games["opp_position"]
        X["opp_form_ppg"] = games["opp_form_ppg"]
    data = pd.concat([games[metric].rename("y"), X], axis=1).dropna()
    X = data.drop(columns="y")
    # Drop covariates that are constant after filtering (e.g. home-only view).
    X = X.loc[:, (X.nunique() > 1) | (X.columns == "carrick")]
    return data["y"].astype(float), sm.add_constant(X, has_constant="add")


@dataclass(frozen=True)
class ModelResult:
    metric: str
    family: str  # "ols", "poisson" or "negbin"
    adjusted: bool
    estimate: float  # OLS: difference per game. Counts: rate ratio Carrick/Amorim
    ci_low: float
    ci_high: float
    p_value: float
    n_amorim: int
    n_carrick: int
    covariates: tuple[str, ...]
    params: pd.DataFrame  # every term: coef, ci_low, ci_high, p
    dispersion: float = np.nan  # Poisson Pearson chi2 / df; > 1 means overdispersed
    alpha: float = np.nan  # NegBin dispersion parameter
    boot_ci_low: float = np.nan  # bootstrap CI (unadjusted ratio of means)
    boot_ci_high: float = np.nan
    note: str = ""


def _params_frame(names: Sequence[str], coef, ci, p) -> pd.DataFrame:
    ci = np.asarray(ci)
    return pd.DataFrame(
        {"coef": np.asarray(coef), "ci_low": ci[:, 0], "ci_high": ci[:, 1],
         "p": np.asarray(p)},
        index=list(names),
    )


def _empty_model(metric, family, adjusted, n_a, n_b, note) -> ModelResult:
    return ModelResult(metric, family, adjusted, np.nan, np.nan, np.nan, np.nan,
                       n_a, n_b, (), pd.DataFrame(), note=note)


def count_model(
    df: pd.DataFrame,
    metric: str,
    family: Literal["poisson", "negbin"] = "poisson",
    adjusted: bool = False,
    n_boot: int = N_RESAMPLES,
    seed: int = SEED,
) -> ModelResult:
    """Rate comparison for a per-game count (exposure 1 game, so no offset).
    Estimate is the rate ratio Carrick / Amorim."""
    y, X = _design(df, metric, adjusted)
    n_b = int(X["carrick"].sum())
    n_a = len(X) - n_b
    covs = tuple(c for c in X.columns if c not in ("const", "carrick"))
    if n_a < 2 or n_b < 2 or y.sum() == 0:
        return _empty_model(metric, family, adjusted, n_a, n_b, "not enough data")
    a, b = y[X["carrick"] == 0].to_numpy(), y[X["carrick"] == 1].to_numpy()
    boot = bootstrap_ci(a, b, ratio_of_means, n_boot, seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if family == "poisson":
                fit = sm.GLM(y, X, family=sm.families.Poisson()).fit()
                params = _params_frame(X.columns, fit.params, fit.conf_int(), fit.pvalues)
                dispersion, alpha = float(fit.pearson_chi2 / fit.df_resid), np.nan
            else:
                fit = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=500)
                names = list(X.columns) + ["alpha"]
                params = _params_frame(names, fit.params, fit.conf_int(), fit.pvalues)
                alpha = float(params.at["alpha", "coef"])
                params = params.drop(index="alpha")
                dispersion = np.nan
    except (np.linalg.LinAlgError, ValueError) as exc:
        return _empty_model(metric, family, adjusted, n_a, n_b, f"model failed: {exc}")
    row = params.loc["carrick"]
    return ModelResult(
        metric=metric, family=family, adjusted=adjusted,
        estimate=float(np.exp(row["coef"])),
        ci_low=float(np.exp(row["ci_low"])), ci_high=float(np.exp(row["ci_high"])),
        p_value=float(row["p"]), n_amorim=n_a, n_carrick=n_b, covariates=covs,
        params=params, dispersion=dispersion, alpha=alpha,
        boot_ci_low=boot[0], boot_ci_high=boot[1],
    )


def adjusted_model(
    df: pd.DataFrame,
    metric: str,
    family: Literal["ols", "poisson", "negbin"] = "ols",
    n_boot: int = N_RESAMPLES,
    seed: int = SEED,
) -> ModelResult:
    """metric ~ manager + venue + opp_position + opp_form.

    GW1 has no pre-match position or form, so it drops out (Amorim n=19 at
    most). OLS uses HC3 robust standard errors. For counts, use poisson/negbin.
    """
    if family != "ols":
        return count_model(df, metric, family, adjusted=True, n_boot=n_boot, seed=seed)
    y, X = _design(df, metric, adjusted=True)
    n_b = int(X["carrick"].sum())
    n_a = len(X) - n_b
    if n_a < 3 or n_b < 3 or len(y) <= X.shape[1] + 1:
        return _empty_model(metric, "ols", True, n_a, n_b, "not enough data")
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    params = _params_frame(X.columns, fit.params, fit.conf_int(), fit.pvalues)
    row = params.loc["carrick"]
    a, b = y[X["carrick"] == 0].to_numpy(), y[X["carrick"] == 1].to_numpy()
    boot = bootstrap_ci(a, b, mean_diff, n_boot, seed)
    return ModelResult(
        metric=metric, family="ols", adjusted=True,
        estimate=float(row["coef"]), ci_low=float(row["ci_low"]),
        ci_high=float(row["ci_high"]), p_value=float(row["p"]),
        n_amorim=n_a, n_carrick=n_b,
        covariates=tuple(c for c in X.columns if c not in ("const", "carrick")),
        params=params, boot_ci_low=boot[0], boot_ci_high=boot[1],
        note="bootstrap CI is for the unadjusted difference",
    )


# ---------------------------------------------------------------------------
# Time series: rolling means, era means, autocorrelation, change points
# ---------------------------------------------------------------------------
def series(df: pd.DataFrame, metric: str, managers: Sequence[str] = TESTED) -> pd.DataFrame:
    """GW-ordered (GW, Manager, value) with NaN dropped."""
    s = df[df["Manager"].isin(managers)].sort_values("GW")[["GW", "Manager", metric]]
    return s.dropna().rename(columns={metric: "value"}).reset_index(drop=True)


def rolling_mean(
    df: pd.DataFrame, metric: str, window: int, within_manager: bool = True
) -> pd.Series:
    """Trailing rolling mean indexed like df (sorted by GW). Missing games are
    skipped, not filled. Within manager by default so eras do not blend."""
    s = df.sort_values("GW")
    roll = lambda x: x.rolling(window, min_periods=1).mean()  # noqa: E731
    if within_manager:
        return s.groupby("Manager")[metric].transform(roll)
    return roll(s[metric])


def era_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-manager mean, n (non-missing games) and GW span."""
    g = df.dropna(subset=[metric]).groupby("Manager", sort=False)
    return g.agg(mean=(metric, "mean"), n=(metric, "size"),
                 first_gw=("GW", "min"), last_gw=("GW", "max")).reset_index()


@dataclass(frozen=True)
class AutocorrelationResult:
    metric: str
    n: int
    acf: np.ndarray  # lags 0..nlags
    conf_band: float  # +/- approximate 95% band for white noise
    ljung_box: pd.DataFrame  # lag, lb_stat, lb_pvalue
    demeaned: bool


def autocorrelation(
    df: pd.DataFrame, metric: str, nlags: int = 8, demean_by_manager: bool = True
) -> AutocorrelationResult:
    """ACF and Ljung-Box on the Amorim + Carrick series in GW order.

    Demeaning by manager removes the era shift itself, so what remains is
    game-to-game dependence within each era. Missing games are skipped.
    """
    s = series(df, metric)
    x = s["value"].to_numpy(float)
    if demean_by_manager:
        x = x - s.groupby("Manager")["value"].transform("mean").to_numpy(float)
    n = len(x)
    nlags = int(max(1, min(nlags, n // 2 - 1)))
    if n < 6 or np.allclose(x, x[0]):
        return AutocorrelationResult(metric, n, np.array([]), np.nan,
                                     pd.DataFrame(), demean_by_manager)
    values = acf(x, nlags=nlags, fft=False)
    lb = acorr_ljungbox(x, lags=list(range(1, nlags + 1)))
    lb = lb.rename_axis("lag").reset_index()
    return AutocorrelationResult(metric, n, values, 1.96 / np.sqrt(n), lb, demean_by_manager)


def cusum(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Cumulative sum of deviations from the season mean (exploratory)."""
    s = series(df, metric)
    s["cusum"] = (s["value"] - s["value"].mean()).cumsum()
    return s


def change_points(
    df: pd.DataFrame, metric: str, penalty: float | None = None, min_size: int = 3
) -> list[int]:
    """GWs where a new segment starts, by PELT on the mean (exploratory only).

    Default penalty is BIC-like: 2 * log(n) * variance.
    """
    import ruptures as rpt

    s = series(df, metric)
    x = s["value"].to_numpy(float)
    if len(x) < 2 * min_size or np.allclose(x, x[0]):
        return []
    pen = penalty if penalty is not None else 2 * np.log(len(x)) * np.var(x)
    bkps = rpt.Pelt(model="l2", min_size=min_size).fit(x.reshape(-1, 1)).predict(pen=pen)
    return [int(s["GW"].iloc[i]) for i in bkps[:-1]]


# ---------------------------------------------------------------------------
# Head to head: radar, venue split, schedule difficulty
# ---------------------------------------------------------------------------
def radar_z(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Manager means as z-scores against the Amorim + Carrick game distribution.
    Signed by direction so further out is better; neutral metrics are unsigned."""
    games = df[df["Manager"].isin(TESTED)]
    rows = []
    for m in metrics:
        col = games[m]
        mu, sd = col.mean(), col.std(ddof=1)
        sign = metric_direction(m) or 1
        for mgr in TESTED:
            vals = col[games["Manager"] == mgr].dropna()
            z = (vals.mean() - mu) / sd * sign if sd > 0 else np.nan
            rows.append({"metric": m, "label": metric_label(m), "Manager": mgr,
                         "mean": vals.mean(), "n": len(vals), "z": z,
                         "direction": metric_direction(m)})
    return pd.DataFrame(rows)


def mean_ci(values: np.ndarray, n_boot: int = N_RESAMPLES, seed: int = SEED,
            level: float = CI_LEVEL) -> tuple[float, float]:
    """Percentile bootstrap CI for one group's mean."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    tail = (1 - level) / 2 * 100
    return (float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail)))


def venue_split(df: pd.DataFrame, metrics: Sequence[str], n_boot: int = N_RESAMPLES,
                seed: int = SEED) -> pd.DataFrame:
    """Mean, n and a 95% bootstrap CI per manager and venue for each metric (long format)."""
    games = df[df["Manager"].isin(TESTED)]
    rows = []
    for m in metrics:
        for (mgr, venue), g in games.dropna(subset=[m]).groupby(["Manager", "venue"]):
            lo, hi = mean_ci(g[m].to_numpy(), n_boot, seed)
            rows.append({"metric": m, "Manager": mgr, "venue": venue, "mean": float(g[m].mean()),
                         "n": len(g), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def xy_means(df: pd.DataFrame, x: str, y: str, n_boot: int = N_RESAMPLES,
             seed: int = SEED) -> pd.DataFrame:
    """Per-manager mean of x and y with 95% bootstrap CIs (games resampled)."""
    rows = []
    for mgr, g in df[df["Manager"].isin(TESTED)].dropna(subset=[x, y]).groupby("Manager"):
        xl, xh = mean_ci(g[x].to_numpy(), n_boot, seed)
        yl, yh = mean_ci(g[y].to_numpy(), n_boot, seed)
        rows.append({"Manager": mgr, "n": len(g), "x_mean": float(g[x].mean()), "x_low": xl,
                     "x_high": xh, "y_mean": float(g[y].mean()), "y_low": yl, "y_high": yh})
    return pd.DataFrame(rows)


def schedule_difficulty(df: pd.DataFrame, n_resamples: int = N_RESAMPLES,
                        seed: int = SEED) -> pd.DataFrame:
    """Average opponent pre-match position and form faced, with tests.
    Lower position number means a stronger opponent."""
    rows = [asdict(compare(df, m, n_resamples, seed))
            for m in ("opp_position", "opp_form_ppg")]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Luck vs skill: expected points and Pythagorean points
# ---------------------------------------------------------------------------
MAX_GOALS = 10


def match_outcome_probs(
    lam_for: np.ndarray, lam_against: np.ndarray, max_goals: int = MAX_GOALS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(win), P(draw), P(loss) from independent Poisson goal rates (vectorised)."""
    lf = np.atleast_1d(np.asarray(lam_for, float))
    la = np.atleast_1d(np.asarray(lam_against, float))
    k = np.arange(max_goals + 1)
    pf = sps.poisson.pmf(k[None, :], lf[:, None])
    pa = sps.poisson.pmf(k[None, :], la[:, None])
    grid = pf[:, :, None] * pa[:, None, :]  # [game, goals for, goals against]
    win = np.tril(np.ones((len(k), len(k))), -1).astype(bool)
    draw = np.eye(len(k), dtype=bool)
    loss = ~(win | draw)
    return grid[:, win].sum(1), grid[:, draw].sum(1), grid[:, loss].sum(1)


def expected_points(df: pd.DataFrame) -> pd.DataFrame:
    """Expected points per game from opponent strength and venue only.

    Two Poisson models (goals for, goals against) ~ home + opp_position, fitted
    on Amorim and Carrick games pooled with no manager term, so xPts reflects
    the schedule and not who was in charge. Pass the unfiltered season frame.
    GW1 has no pre-match position, so its xPts is NaN.
    """
    fit_rows = df[df["Manager"].isin(TESTED) & df["opp_position"].notna()]

    def design(d: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame({"home": (d["venue"] == "Home").astype(float),
                          "opp_position": d["opp_position"]})
        return sm.add_constant(X, has_constant="add")

    poisson = sm.families.Poisson()
    gf = sm.GLM(fit_rows["MU goals"].astype(float), design(fit_rows), family=poisson).fit()
    ga = sm.GLM(fit_rows["Opp goals"].astype(float), design(fit_rows), family=poisson).fit()

    out = df[["GW", "Manager", "Opponent", "venue", "opp_position", "Points"]].copy()
    known = out["opp_position"].notna()
    out["xGF_model"] = np.nan
    out["xGA_model"] = np.nan
    out["xPts"] = np.nan
    if known.any():
        X = design(df.loc[known])
        lf, la = gf.predict(X).to_numpy(), ga.predict(X).to_numpy()
        w, d, _ = match_outcome_probs(lf, la)
        out.loc[known, "xGF_model"] = lf
        out.loc[known, "xGA_model"] = la
        out.loc[known, "xPts"] = 3 * w + d
    out["Points minus xPts"] = out["Points"] - out["xPts"]
    return out


def pythagorean_points(
    df: pd.DataFrame, n_boot: int = N_RESAMPLES, seed: int = SEED
) -> pd.DataFrame:
    """Points per game implied by each manager's GF and GA per game.

    Uses the Poisson form of the Pythagorean idea: treat GF and GA per game as
    independent Poisson rates and compute expected points. This avoids fitting
    an exponent on 37 games. CI resamples games within manager.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for mgr in TESTED:
        g = df[df["Manager"] == mgr]
        gf, ga = g["MU goals"].to_numpy(float), g["Opp goals"].to_numpy(float)
        n = len(g)
        if n == 0:
            continue
        w, d, _ = match_outcome_probs(gf.mean(), ga.mean())
        pyth = float(3 * w[0] + d[0])
        lo = hi = np.nan
        if n >= 2:
            idx = rng.integers(0, n, (n_boot, n))
            bw, bd, _ = match_outcome_probs(gf[idx].mean(1), ga[idx].mean(1))
            lo, hi = np.percentile(3 * bw + bd, [2.5, 97.5])
        rows.append({"Manager": mgr, "n": n, "GF per game": gf.mean(),
                     "GA per game": ga.mean(), "Actual PPG": g["Points"].mean(),
                     "Pythagorean PPG": pyth, "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Actual minus Pythagorean"] = out["Actual PPG"] - out["Pythagorean PPG"]
    return out


# ---------------------------------------------------------------------------
# Luck vs skill: big chances
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConversionResult:
    side: Literal["mu", "opp"]
    created_amorim: int
    missed_amorim: int
    scored_amorim: int
    created_carrick: int
    missed_carrick: int
    scored_carrick: int
    conversion_amorim: float
    conversion_carrick: float
    diff: float  # Carrick minus Amorim
    boot_ci_low: float  # resamples games, not chances
    boot_ci_high: float
    cohens_h: float
    p_fisher: float
    p_ztest: float
    n_games_amorim: int
    n_games_carrick: int


def bc_conversion(
    df: pd.DataFrame,
    side: Literal["mu", "opp"] = "mu",
    n_boot: int = N_RESAMPLES,
    seed: int = SEED,
) -> ConversionResult:
    """Pooled big chance conversion, (created - missed) / created, by manager."""
    p = "" if side == "mu" else "Opp "
    created_col, missed_col = f"{p}Big chances", f"{p}Big chances missed"
    rng = np.random.default_rng(seed)
    per: dict[str, dict] = {}
    boots = {}
    for mgr in TESTED:
        g = df[df["Manager"] == mgr]
        c = g[created_col].to_numpy(float)
        m = g[missed_col].to_numpy(float)
        created, missed = int(c.sum()), int(m.sum())
        per[mgr] = {"created": created, "missed": missed, "scored": created - missed,
                    "conv": (created - missed) / created if created else np.nan,
                    "n": len(g)}
        if len(g) >= 2:
            idx = rng.integers(0, len(g), (n_boot, len(g)))
            bc, bm = c[idx].sum(1), m[idx].sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                boots[mgr] = np.where(bc > 0, (bc - bm) / bc, np.nan)
    a, b = per[AMORIM], per[CARRICK]
    lo = hi = np.nan
    if AMORIM in boots and CARRICK in boots:
        d = boots[CARRICK] - boots[AMORIM]
        d = d[np.isfinite(d)]
        if len(d) >= 0.5 * n_boot:
            lo, hi = np.percentile(d, [2.5, 97.5])
    p_fisher = p_z = h = np.nan
    if a["created"] and b["created"]:
        table = [[a["scored"], a["missed"]], [b["scored"], b["missed"]]]
        p_fisher = float(sps.fisher_exact(table).pvalue)
        h = float(2 * np.arcsin(np.sqrt(b["conv"])) - 2 * np.arcsin(np.sqrt(a["conv"])))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                p_z = float(proportions_ztest(
                    [b["scored"], a["scored"]], [b["created"], a["created"]])[1])
            except ZeroDivisionError:
                p_z = np.nan
    return ConversionResult(
        side=side,
        created_amorim=a["created"], missed_amorim=a["missed"], scored_amorim=a["scored"],
        created_carrick=b["created"], missed_carrick=b["missed"], scored_carrick=b["scored"],
        conversion_amorim=a["conv"], conversion_carrick=b["conv"],
        diff=b["conv"] - a["conv"], boot_ci_low=float(lo), boot_ci_high=float(hi),
        cohens_h=h, p_fisher=p_fisher, p_ztest=p_z,
        n_games_amorim=a["n"], n_games_carrick=b["n"],
    )


def tag_loss(bc_diff: float, xg_diff: float) -> str:
    """'Deserved' if out-chanced on both big chances and xG, else 'Unlucky'."""
    return "Deserved" if bc_diff < 0 and xg_diff < 0 else "Unlucky"


def loss_table(df: pd.DataFrame) -> pd.DataFrame:
    """Every defeat with big chance and xG context and a Deserved/Unlucky tag."""
    losses = df[df["Result"] == "L"].sort_values("GW")
    out = pd.DataFrame({
        "GW": losses["GW"],
        "Manager": losses["Manager"],
        "Opponent": losses["Opponent"],
        "venue": losses["venue"],
        "Score": losses["MU goals"].astype(str) + "-" + losses["Opp goals"].astype(str),
        "BC created": losses["Big chances"],
        "BC missed": losses["Big chances missed"],
        "BC conceded": losses["Opp Big chances"],
        "xG": losses["xG"],
        "xG against": losses["Opp xG"],
        "BC diff": losses["BC diff"],
        "xG diff": losses["xG difference"],
    })
    out["Tag"] = [tag_loss(b, x) for b, x in zip(out["BC diff"], out["xG diff"])]
    return out.reset_index(drop=True)


def loss_tag_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Deserved / Unlucky loss counts per manager, with total losses and games."""
    losses = loss_table(df)
    counts = (pd.crosstab(losses["Manager"], losses["Tag"])
              .reindex(columns=["Deserved", "Unlucky"], fill_value=0))
    counts["Losses"] = counts.sum(axis=1)
    counts["Games"] = df.groupby("Manager").size().reindex(counts.index)
    return counts


def bc_bucket(bc_diff: float) -> str:
    if bc_diff < 0:
        return "Negative"
    return "Level" if bc_diff == 0 else "Positive"


def ppg_by_bc_bucket(
    df: pd.DataFrame, n_boot: int = N_RESAMPLES, seed: int = SEED
) -> pd.DataFrame:
    """Games, points and PPG per manager by big chance difference bucket, with a
    95% percentile bootstrap CI for PPG (games resampled within each cell)."""
    buckets = pd.CategoricalDtype(["Negative", "Level", "Positive"], ordered=True)
    games = df[df["Manager"].isin(TESTED)].assign(
        bucket=lambda d: d["BC diff"].map(bc_bucket).astype(buckets))
    out = games.groupby(["Manager", "bucket"], observed=False).agg(
        games=("GW", "size"), points=("Points", "sum")).reset_index()
    out["PPG"] = (out["points"] / out["games"]).where(out["games"] > 0)
    rng = np.random.default_rng(seed)
    lows, highs = [], []
    for r in out.itertuples():
        pts = games.loc[(games["Manager"] == r.Manager) & (games["bucket"] == r.bucket),
                        "Points"].to_numpy(dtype=float)
        if len(pts) == 0:
            lows.append(np.nan)
            highs.append(np.nan)
            continue
        means = rng.choice(pts, size=(n_boot, len(pts)), replace=True).mean(axis=1)
        lows.append(float(np.percentile(means, 2.5)))
        highs.append(float(np.percentile(means, 97.5)))
    out["ci_low"], out["ci_high"] = lows, highs
    return out


# ---------------------------------------------------------------------------
# Style and drivers (exploratory)
# ---------------------------------------------------------------------------
# Outcomes or direct functions of goals: excluded from "what predicts points".
OUTCOME_METRICS = frozenset({
    "Points", "MU goals", "Opp goals", "Goal difference", "Goals minus xG",
    "Goals minus xGOT", "Conceded minus xG against", "Big chances scored",
    "Opp Big chances scored", "BC conversion", "Opp BC conversion",
    "Goals prevented", "Opp Goals prevented",
})


def driver_metrics() -> list[str]:
    """Default candidate predictors: all metrics except outcomes."""
    return [k for k in METRICS if k not in OUTCOME_METRICS]


def standardised_matrix(
    df: pd.DataFrame, metrics: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Z-scored Amorim + Carrick games. Drops constant metrics, then any game
    with a missing value (e.g. GW33 if tracking metrics are included).
    Returns (Z, games kept, dropped metrics)."""
    games = df[df["Manager"].isin(TESTED)]
    sub = games[list(metrics)].astype(float)
    constant = [c for c in sub.columns if sub[c].nunique(dropna=True) <= 1]
    sub = sub.drop(columns=constant).dropna()
    z = (sub - sub.mean()) / sub.std(ddof=1)
    return z, games.loc[z.index], constant


@dataclass(frozen=True)
class ClusterResult:
    scores: pd.DataFrame  # GW, Manager, PC1, PC2, cluster
    explained_variance: np.ndarray
    loadings: pd.DataFrame  # metric x component
    crosstab: pd.DataFrame  # manager x cluster
    adjusted_rand: float
    p_adjusted_rand: float  # permutation p for ARI vs manager labels
    n_amorim: int
    n_carrick: int
    dropped_metrics: list[str]


def pca_kmeans(
    df: pd.DataFrame,
    metrics: Sequence[str],
    k: int = 2,
    n_components: int = 2,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> ClusterResult:
    """PCA for display and KMeans on the full standardised matrix (exploratory).
    ARI near 0 means clusters do not line up with managers."""
    z, games, dropped = standardised_matrix(df, metrics)
    pca = PCA(n_components=n_components, random_state=seed).fit(z)
    pcs = pca.transform(z)
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(z)
    managers = games["Manager"].to_numpy()
    ari = float(adjusted_rand_score(managers, labels))
    rng = np.random.default_rng(seed)
    null = np.array([adjusted_rand_score(rng.permutation(managers), labels)
                     for _ in range(n_resamples)])
    p = float((np.sum(null >= ari) + 1) / (n_resamples + 1))
    scores = pd.DataFrame({"GW": games["GW"].to_numpy(), "Manager": managers,
                           "cluster": labels})
    for i in range(n_components):
        scores[f"PC{i + 1}"] = pcs[:, i]
    loadings = pd.DataFrame(pca.components_.T, index=z.columns,
                            columns=[f"PC{i + 1}" for i in range(n_components)])
    return ClusterResult(
        scores=scores,
        explained_variance=pca.explained_variance_ratio_,
        loadings=loadings,
        crosstab=pd.crosstab(scores["Manager"], scores["cluster"]),
        adjusted_rand=ari,
        p_adjusted_rand=p,
        n_amorim=int((managers == AMORIM).sum()),
        n_carrick=int((managers == CARRICK).sum()),
        dropped_metrics=dropped,
    )


def correlation_with_points(
    df: pd.DataFrame,
    metrics: Sequence[str],
    n_boot: int = N_RESAMPLES,
    seed: int = SEED,
) -> pd.DataFrame:
    """Spearman rho of each metric with points (Amorim + Carrick games), with
    bootstrap CI, raw and Holm p. Sorted by |rho|."""
    games = df[df["Manager"].isin(TESTED)]
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics:
        d = games[[m, "Points"]].dropna()
        x, y = d[m].to_numpy(float), d["Points"].to_numpy(float)
        n = len(d)
        if n < 4 or np.ptp(x) == 0:
            rows.append({"metric": m, "label": metric_label(m), "n": n, "rho": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "p": np.nan})
            continue
        rho, p = sps.spearmanr(x, y)
        idx = rng.integers(0, n, (n_boot, n))
        rx = sps.rankdata(x[idx], axis=1)
        ry = sps.rankdata(y[idx], axis=1)
        rx -= rx.mean(1, keepdims=True)
        ry -= ry.mean(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (rx * ry).sum(1) / np.sqrt((rx**2).sum(1) * (ry**2).sum(1))
        r = r[np.isfinite(r)]
        lo, hi = np.percentile(r, [2.5, 97.5]) if len(r) else (np.nan, np.nan)
        rows.append({"metric": m, "label": metric_label(m), "n": n, "rho": float(rho),
                     "ci_low": float(lo), "ci_high": float(hi), "p": float(p)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out.sort_values("rho", key=np.abs, ascending=False, na_position="last")


@dataclass(frozen=True)
class LassoResult:
    coefficients: pd.DataFrame  # metric, label, coef, selection_freq
    alpha: float
    cv_r2: float  # mean out-of-fold R squared; near or below 0 means no skill
    n: int
    dropped_metrics: list[str]


def regularised_points_model(
    df: pd.DataFrame,
    metrics: Sequence[str],
    n_boot: int = 500,
    seed: int = SEED,
) -> LassoResult:
    """Lasso of points on standardised stats (exploratory).

    alpha chosen by 5-fold CV; selection frequency is how often each stat is
    kept across bootstrap refits at that alpha.
    """
    z, games, dropped = standardised_matrix(df, metrics)
    y = games["Points"].to_numpy(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LassoCV(cv=cv, random_state=seed, max_iter=50_000).fit(z, y)
        alpha = float(model.alpha_)
        r2 = float(np.mean(cross_val_score(Lasso(alpha=alpha, max_iter=50_000),
                                           z, y, cv=cv, scoring="r2")))
        rng = np.random.default_rng(seed)
        kept = np.zeros(z.shape[1])
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            fit = Lasso(alpha=alpha, max_iter=50_000).fit(z.iloc[idx], y[idx])
            kept += fit.coef_ != 0
    coefs = pd.DataFrame({
        "metric": z.columns,
        "label": [metric_label(c) for c in z.columns],
        "coef": model.coef_,
        "selection_freq": kept / n_boot,
    }).sort_values("coef", key=np.abs, ascending=False)
    return LassoResult(coefs.reset_index(drop=True), alpha, r2, len(y), dropped)


# ---------------------------------------------------------------------------
# Curated groups: areas of play and the key "did it improve" family
# ---------------------------------------------------------------------------
STYLE_GROUPS: dict[str, tuple[str, ...]] = {
    "Attack": ("xG", "Total shots", "Shots inside box", "Shots outside box",
               "Touches in opp box", "Big chances", "Through balls", "Corners"),
    "Defence": ("Opp xG", "Opp Total shots", "Opp Big chances", "Opp Touches in opp box",
                "Total tackles", "Interceptions", "Recoveries", "Clearances"),
    "Passing": ("Possession %", "Passes", "Pass accuracy %", "Final third entries",
                "Passes in final third %", "Long balls total", "Long balls %", "Crosses total"),
    "Duels": ("Duels won %", "Ground duels %", "Aerial duels total", "Aerial duels %",
              "Dribbles attempted", "Dribbles %", "Fouls", "Dispossessed"),
}
KEY_METRICS: tuple[str, ...] = (
    "Points", "MU goals", "Opp goals", "xG", "Opp xG", "xG difference", "BC diff",
)
AREA_GROUPS: dict[str, tuple[str, ...]] = {"Results": KEY_METRICS, **STYLE_GROUPS}


def style_metrics() -> list[str]:
    return [m for ms in STYLE_GROUPS.values() for m in ms]


def area_metrics() -> list[str]:
    """Every metric in AREA_GROUPS once, in group order."""
    return list(dict.fromkeys(m for ms in AREA_GROUPS.values() for m in ms))


def area_of(metric: str) -> str:
    """Area of play for a metric; style areas win over the Results family."""
    for area, ms in [*STYLE_GROUPS.items(), ("Results", KEY_METRICS)]:
        if metric in ms:
            return area
    return ""


def manager_short(name: str | None) -> str:
    return name.split()[-1] if isinstance(name, str) else ""


def _lower_first(label: str) -> str:
    """'Total shots' -> 'total shots' but 'xG' stays 'xG'."""
    if len(label) > 1 and label[0].isupper() and label[1].islower():
        return label[0].lower() + label[1:]
    return label


def plain_change(metric: str, label: str, diff: float, mean_a: float, mean_c: float) -> str:
    """e.g. 'fewer throw-ins (14.1 vs 20.0)' describing Carrick relative to Amorim."""
    kind = metric_kind(metric)
    if kind in ("count", "points"):
        word = "more" if diff > 0 else "fewer"
    else:
        word = "higher" if diff > 0 else "lower"
    return (f"{word} {_lower_first(label)} ({fmt_value(mean_c, kind)} vs "
            f"{fmt_value(mean_a, kind)})")


def plain_finding(c: Comparison, p_holm: float | None = None) -> str:
    """One plain-English sentence about a comparison, then how sure we can be."""
    if c.n_amorim < 2 or c.n_carrick < 2 or not np.isfinite(c.diff):
        return "Not enough games to compare with these filters."
    kind = metric_kind(c.metric)
    a, b = fmt_value(c.mean_amorim, kind), fmt_value(c.mean_carrick, kind)
    if np.isfinite(c.cohens_d) and abs(c.cohens_d) < 0.2:
        head = f"{c.label} stayed about the same: {a} per game under Amorim, {b} under Carrick."
    else:
        word = "went up" if c.diff > 0 else "went down"
        head = f"{c.label} {word} under Carrick, from {a} to {b} per game"
        head += (", which is better for Man Utd." if c.better == CARRICK else
                 ", which is worse for Man Utd." if c.better == AMORIM else ".")
    p = p_holm if p_holm is not None else c.p_permutation
    if is_reliable(p, c.diff_ci_low, c.diff_ci_high):
        sure = "The gap is big enough that chance alone is unlikely to explain it."
    elif is_reliable(c.p_permutation, c.diff_ci_low, c.diff_ci_high):
        sure = ("On its own the gap looks unusual, but after allowing for testing many stats "
                "it could be chance.")
    else:
        sure = f"With {c.n_amorim} and {c.n_carrick} games, a gap like this could easily be chance."
    return f"{head} {sure}"


def group_finding(table: pd.DataFrame, metrics: Sequence[str]) -> str:
    """Plain summary of one area: the biggest shifts and how many are reliable."""
    sub = table[table["metric"].isin(metrics)].drop_duplicates("metric").copy()
    if sub.empty:
        return "No stats to compare."
    sub["verdict"] = [verdict(r.p_permutation, r.p_permutation_holm, r.diff_ci_low,
                              r.diff_ci_high) for r in sub.itertuples()]
    sub["abs_d"] = sub["cohens_d"].abs()
    moved = sub[sub["abs_d"] >= 0.5].sort_values("abs_d", ascending=False)
    if moved.empty:
        text = ("The two managers look very similar in this area: no stat moved by a medium "
                "or large amount.")
    else:
        changes = [plain_change(r.metric, r.label, r.diff, r.mean_amorim, r.mean_carrick)
                   for r in moved.head(3).itertuples()]
        text = "Biggest shifts under Carrick: " + "; ".join(changes) + "."
    n_clear = int((sub["verdict"] == CLEAR).sum())
    n_possible = int((sub["verdict"] == POSSIBLE).sum())
    if n_clear:
        names = ", ".join(_lower_first(lab) for lab in sub.loc[sub["verdict"] == CLEAR, "label"])
        text += f" {n_clear} of {len(sub)} is a clear difference ({names})."
    elif n_possible:
        text += (f" None of the {len(sub)} is a clear difference; {n_possible} look unusual on "
                 "their own but may be chance after correcting for many tests.")
    else:
        text += f" None of the {len(sub)} gaps is statistically reliable."
    return text


# ---------------------------------------------------------------------------
# Did performance improve? A small, pre-chosen family of key measures
# ---------------------------------------------------------------------------
def improvement_table(
    df: pd.DataFrame, n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> pd.DataFrame:
    """Key results and chance-quality measures, Holm-corrected within this family
    only (7 tests), with a verdict and whether Carrick's direction is better."""
    table = comparison_table(df, KEY_METRICS, n_resamples, seed)
    table["verdict"] = [verdict(r.p_permutation, r.p_permutation_holm, r.diff_ci_low,
                                r.diff_ci_high) for r in table.itertuples()]
    table["improved"] = table["better"] == CARRICK
    return table


RESULTS_PROCESS: dict[str, tuple[str, ...]] = {
    "Results": ("Points", "MU goals", "Opp goals"),
    "Process": ("xG", "Opp xG", "xG difference", "BC diff"),
}


def results_vs_process(
    df: pd.DataFrame, n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> pd.DataFrame:
    """The 7 key measures as forest rows (d signed so + is better under Carrick),
    tagged Results or Process. Holm is applied across the same 7 as the Stats lab."""
    fd = forest_data(comparison_table(df, KEY_METRICS, n_resamples, seed))
    group_of = {m: g for g, ms in RESULTS_PROCESS.items() for m in ms}
    fd["group"] = fd["metric"].map(group_of)
    order = {m: i for i, m in enumerate(m for ms in RESULTS_PROCESS.values() for m in ms)}
    return fd.sort_values("metric", key=lambda s: s.map(order)).reset_index(drop=True)


def improvement_summary(table: pd.DataFrame, wdl: WDLTest) -> tuple[str, str]:
    """(headline answer, explanation) to 'did performance improve?'."""
    t = table.set_index("metric")
    pts, xgd = t.loc["Points"], t.loc["xG difference"]
    clear_better = t[(t["verdict"] == CLEAR) & t["improved"]]
    clear_worse = t[(t["verdict"] == CLEAR) & ~t["improved"] & t["better"].notna()]
    n = len(t)
    if len(clear_better) and not len(clear_worse):
        names = ", ".join(_lower_first(x) for x in clear_better["label"])
        headline = (f"Yes, in part. {len(clear_better)} of {n} key measures improved by more "
                    f"than chance would explain ({names}).")
    elif len(clear_worse) and not len(clear_better):
        headline = "No. Some key measures got reliably worse."
    else:
        headline = ("Not proven. Results looked better under Carrick, but none of the "
                    f"{n} key measures improved by more than chance can explain.")
    better_dirs = int(t["improved"].sum())
    detail = (
        f"Points per game went from {pts.mean_amorim:.2f} to {pts.mean_carrick:.2f} "
        f"({pts['diff']:+.2f}, plausible range {pts.diff_ci_low:+.2f} to {pts.diff_ci_high:+.2f}, "
        f"Holm p={fmt_p(pts.p_permutation_holm)}). Win rate went from "
        f"{wdl.win_rate_amorim:.0%} to {wdl.win_rate_carrick:.0%} (Fisher p on W/D/L "
        f"{fmt_p(wdl.p_fisher)}). "
        f"The underlying chance balance (xG difference) went from {xgd.mean_amorim:+.2f} to "
        f"{xgd.mean_carrick:+.2f} per game, "
        + ("so results improved more than xG did. That pattern fits finishing, goalkeeping "
           "or luck better than a clear change in xG, though this data cannot say which. "
           if pts["diff"] > 0 and abs(xgd.cohens_d) < 0.2 else
           "in line with the change in results. " if np.sign(xgd["diff"]) == np.sign(pts["diff"])
           else "moving the opposite way to results. ")
        + (f"The big chance difference went from {t.at['BC diff', 'mean_amorim']:+.2f} to "
           f"{t.at['BC diff', 'mean_carrick']:+.2f} per game. " if "BC diff" in t.index else "")
        + f"{better_dirs} of {n} measures moved in Carrick's favour. With 20 and 17 games, only "
          "very large changes can be confirmed, and the change of manager also coincides with "
          "a different part of the season."
    )
    return headline, detail


# ---------------------------------------------------------------------------
# Did the playing style change? Supervised check
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassifierResult:
    accuracy: float  # cross-validated share of games assigned to the right manager
    baseline: float  # accuracy from always guessing the bigger group
    p_value: float  # permutation test: shuffled labels score this well how often
    null_scores: np.ndarray
    n_amorim: int
    n_carrick: int
    weights: pd.DataFrame  # metric, label, weight (+ points to Carrick)
    dropped_metrics: list[str]


def style_classifier(
    df: pd.DataFrame,
    metrics: Sequence[str],
    n_permutations: int = 1000,
    seed: int = SEED,
) -> ClassifierResult:
    """Can a simple model tell Amorim games from Carrick games using style stats?

    Ridge-penalised logistic regression, 5-fold stratified CV, scaling inside
    each fold. Above-chance accuracy means the styles differ (or something
    else that changed with the season does).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, permutation_test_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    z, games, dropped = standardised_matrix(df, metrics)
    X = games[z.columns].astype(float)
    y = (games["Manager"] == CARRICK).astype(int).to_numpy()
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.5, solver="liblinear", max_iter=5000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    score, null, p = permutation_test_score(
        model, X, y, cv=cv, n_permutations=n_permutations, random_state=seed,
        scoring="accuracy", n_jobs=-1)
    coef = model.fit(X, y)[-1].coef_[0]
    weights = pd.DataFrame({"metric": z.columns, "label": [metric_label(c) for c in z.columns],
                            "weight": coef}).sort_values("weight", key=np.abs, ascending=False)
    return ClassifierResult(
        accuracy=float(score), baseline=float(max(y.mean(), 1 - y.mean())), p_value=float(p),
        null_scores=np.asarray(null), n_amorim=int((y == 0).sum()), n_carrick=int((y == 1).sum()),
        weights=weights.reset_index(drop=True), dropped_metrics=dropped,
    )


# ---------------------------------------------------------------------------
# Plain-English wording (no em dashes, never causal)
# ---------------------------------------------------------------------------
UNCERTAIN = "Small sample, effect not reliably different from zero."
ASSOCIATION = (
    "The gap is larger than chance alone would usually produce, but it is an "
    "association across two parts of the season, not proof of a manager effect."
)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_value(v: float, kind: MetricKind, signed: bool = False) -> str:
    if not np.isfinite(v):
        return "n/a"
    sign = "+" if signed else ""
    if kind == "pct":
        return f"{v * 100:{sign}.1f}{' pp' if signed else '%'}"
    return f"{v:{sign}.2f}"


def is_reliable(p: float, ci_low: float, ci_high: float, alpha: float = ALPHA) -> bool:
    """True only if p < alpha and the CI excludes zero."""
    return bool(np.isfinite(p) and p < alpha and (ci_low > 0 or ci_high < 0))


def describe_comparison(c: Comparison, p_holm: float | None = None) -> str:
    """One-line insight for a comparison. Uses Holm p for the verdict if given."""
    kind = metric_kind(c.metric)
    p = p_holm if p_holm is not None else c.p_permutation
    p_txt = f"permutation p={fmt_p(c.p_permutation)}"
    if p_holm is not None:
        p_txt += f", Holm p={fmt_p(p_holm)}"
    verdict = ASSOCIATION if is_reliable(p, c.diff_ci_low, c.diff_ci_high) else UNCERTAIN
    return (
        f"{c.label}: Carrick {fmt_value(c.mean_carrick, kind)} vs Amorim "
        f"{fmt_value(c.mean_amorim, kind)} per game (n={c.n_carrick} vs {c.n_amorim}). "
        f"Difference {fmt_value(c.diff, kind, signed=True)} (95% CI "
        f"{fmt_value(c.diff_ci_low, kind, True)} to {fmt_value(c.diff_ci_high, kind, True)}), "
        f"{p_txt}. {verdict}"
    )


def describe_model(r: ModelResult, label: str) -> str:
    """Plain-English reading of the manager coefficient."""
    n = f"n={r.n_carrick} Carrick vs {r.n_amorim} Amorim games"
    if not np.isfinite(r.estimate):
        return f"{label}: model could not be estimated ({r.note or 'not enough data'}, {n})."
    adj = ("After allowing for venue, opponent pre-match position and opponent form, "
           if r.adjusted else "")
    if r.family == "ols":
        kind = metric_kind(r.metric)
        body = (f"{adj}Carrick's games are associated with "
                f"{fmt_value(r.estimate, kind, signed=True)} {label} per game compared "
                f"with Amorim's (95% CI {fmt_value(r.ci_low, kind, True)} to "
                f"{fmt_value(r.ci_high, kind, True)}, p={fmt_p(r.p_value)}, {n}).")
        reliable = is_reliable(r.p_value, r.ci_low, r.ci_high)
    else:
        body = (f"{adj}the {label} rate under Carrick is {r.estimate:.2f} times "
                f"Amorim's (95% CI {r.ci_low:.2f} to {r.ci_high:.2f}, "
                f"p={fmt_p(r.p_value)}, {r.family}, {n}).")
        body = body[0].upper() + body[1:]
        reliable = bool(np.isfinite(r.p_value) and r.p_value < ALPHA
                        and (r.ci_low > 1 or r.ci_high < 1))
    tail = ASSOCIATION if reliable else UNCERTAIN
    return f"{body} {tail} Manager is confounded with season phase."
