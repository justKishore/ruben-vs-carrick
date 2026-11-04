"""Smoke tests for src/charts.py: builders run on real data and follow the rules."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from src import charts as C
from src import stats as S
from src.load import AMORIM, CARRICK, FLETCHER, load_match_data

FAST = 500


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    frame = S.per_game_frame(load_match_data(cache_dir=None))
    return frame.merge(S.expected_points(frame)[["GW", "xPts"]], on="GW")


def test_charts_does_not_import_streamlit() -> None:
    tree = ast.parse(Path(C.__file__).read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(str(n).startswith("streamlit") for n in names)


def test_manager_colours() -> None:
    assert C.COLORS[CARRICK] == "#DA291C"
    assert C.COLORS[FLETCHER] == C.NEUTRAL
    assert C.COLORS[AMORIM].startswith("#")


def test_era_divider_at_gw22_boundary(df: pd.DataFrame) -> None:
    fig = C.form_strip(df)
    xs = [s.x0 for s in fig.layout.shapes]
    assert C.ERA_BOUNDARY in xs and 21 < C.ERA_BOUNDARY < 22


def test_builders_return_figures(df: pd.DataFrame) -> None:
    t = df[df["Manager"].isin(S.TESTED)]
    valid = df.dropna(subset=["xG"]).sort_values("GW")
    pts = valid[["GW", "Manager", "Opponent", "venue", "opp_position"]].assign(
        value=valid["xG"], rolling=S.rolling_mean(valid, "xG", 4))
    table = S.comparison_table(t, ["xG", "Opp xG", "Possession %"], n_resamples=FAST)
    perm = S.permutation_for_metric(t, "xG", n_resamples=FAST)
    ac = S.autocorrelation(t, "xG")
    xp = t.dropna(subset=["xPts"]).assign(
        games_managed=lambda d: d.groupby("Manager").cumcount() + 1,
        cum_points=lambda d: d.groupby("Manager")["Points"].cumsum(),
        cum_xpts=lambda d: d.groupby("Manager")["xPts"].cumsum())
    cl = S.pca_kmeans(t, ["xG", "Opp xG", "Possession %", "Total shots"], n_resamples=50)
    cor = S.correlation_with_points(t, ["xG", "BC diff"], n_boot=FAST)
    lasso = S.regularised_points_model(t, ["xG", "Opp xG", "BC diff", "Possession %"], n_boot=20)
    card = pd.DataFrame({"label": ["xG", "Possession %"], "kind": ["continuous", "pct"],
                         "left": [1.2, 0.6], "right": [0.8, 0.4]})
    figs = [
        C.form_strip(df),
        C.cumulative_points_chart(S.cumulative_points(df)),
        C.trend_chart(pts, S.era_means(df, "xG"), "xG", "continuous", [22]),
        C.cusum_chart(S.cusum(df, "xG"), "xG"),
        C.radar_chart(S.radar_z(t, ["xG", "Opp xG", "Possession %"])),
        C.forest_plot(S.forest_data(table)),
        C.violin_chart(df, "xG", "xG", "continuous"),
        C.venue_split_chart(S.venue_split(df, ["xG"]), "xG", "continuous"),
        C.schedule_chart(df),
        C.permutation_histogram(perm.null_distribution, perm.observed, "xG", "continuous"),
        C.acf_chart(ac.acf, ac.conf_band),
        C.xy_scatter(df, "xG", "MU goals", "xG", "Goals"),
        C.xpts_chart(xp),
        C.ppg_interval_chart(S.pythagorean_points(t, n_boot=FAST)),
        C.bar_with_dots(df, "Big chances missed", "Big chances missed"),
        C.loss_dot_plot(S.loss_table(df)),
        C.bucket_chart(S.ppg_by_bc_bucket(t)),
        C.pca_chart(cl.scores, cl.explained_variance),
        C.interval_chart(cor, "rho", "ci_low", "ci_high", "rho"),
        C.coefficient_chart(lasso.coefficients),
        C.match_card(card, "Home", "Away", C.OPPONENT, C.COLORS[CARRICK]),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)
    for f in figs:
        f.to_dict()  # validates the whole figure


def test_traces_use_manager_colours(df: pd.DataFrame) -> None:
    fig = C.cumulative_points_chart(S.cumulative_points(df))
    by_name = {t.name: t.line.color for t in fig.data}
    assert by_name["Carrick"] == C.COLORS[CARRICK]
    assert by_name["Amorim"] == C.COLORS[AMORIM]


def test_forest_plot_drops_neutral_and_marks_verdict(df: pd.DataFrame) -> None:
    t = df[df["Manager"].isin(S.TESTED)]
    table = S.comparison_table(t, ["xG", "Opp goals", "Possession %"], n_resamples=FAST)
    fd = S.forest_data(table)
    assert "verdict" in fd
    fig = C.forest_plot(fd)
    labels = list(fig.data[0].y)
    assert S.metric_label("Possession %") not in labels
    assert len(labels) == 2


def test_style_shift_chart_has_no_manager_colours(df: pd.DataFrame) -> None:
    t = df[df["Manager"].isin(S.TESTED)]
    table = S.comparison_table(t, ["xG", "Possession %"], n_resamples=FAST)
    table["verdict"] = S.NONE
    table["area"] = "Attack"
    line = C.style_shift_chart(table).data[0].marker.line.color
    assert line not in (C.COLORS[AMORIM], C.COLORS[CARRICK])


def test_new_comparison_charts(df: pd.DataFrame) -> None:
    t = df[df["Manager"].isin(S.TESTED)]
    figs = [
        C.results_process_chart(S.results_vs_process(t, n_resamples=200)),
        C.estimate_chart(pd.DataFrame({"label": ["Raw", "Adjusted"], "estimate": [0.9, 1.1],
                                       "ci_low": [0.6, 0.7], "ci_high": [1.3, 1.6]}),
                         1.0, "Rate ratio", log=True),
        C.xy_scatter(df, "xG", "MU goals", "xG", "Goals",
                     means=S.xy_means(df, "xG", "MU goals", n_boot=200)),
        C.venue_split_chart(S.venue_split(df, ["xG"], n_boot=200), "xG", "continuous"),
    ]
    for f in figs:
        f.to_dict()


def test_scatter_jitter_keeps_true_values_in_hover(df: pd.DataFrame) -> None:
    fig = C.xy_scatter(df, "xG", "MU goals", "xG", "Goals")
    tr = [t for t in fig.data if t.name and t.name.startswith("Carrick")][0]
    true_y = [float(v) for v in tr.customdata[:, 3]]
    assert all(float(v).is_integer() for v in true_y)
    assert max(abs(a - b) for a, b in zip(tr.y, true_y)) <= C.JITTER


def test_radar_labels_wrap_long_names() -> None:
    assert C._wrap("Touches in opp box") == "Touches in<br>opp box"
    assert C._wrap("xG") == "xG"
    assert "<br>" in C._wrap("Passes in final third %")
