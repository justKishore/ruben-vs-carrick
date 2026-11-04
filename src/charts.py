"""Plotly figure builders. Pure functions: data in, go.Figure out.

Colours are fixed per manager everywhere. The Amorim green was chosen with a
colour-vision-deficiency check against the Carrick red on the dark surface.
This module must not import streamlit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.load import AMORIM, CARRICK, FLETCHER

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
SURFACE = "#0E1117"
PANEL = "#1B1F27"
GRID = "#232733"
AXIS = "#3A3F4B"
TEXT = "#E6E8EB"
MUTED = "#9AA0A6"
NEUTRAL = "#8A8F98"
OPPONENT = "#B8BDC6"

COLORS = {AMORIM: "#3FAE5A", CARRICK: "#DA291C", FLETCHER: NEUTRAL}
SHORT = {AMORIM: "Amorim", CARRICK: "Carrick", FLETCHER: "Fletcher"}
ORDER = (AMORIM, FLETCHER, CARRICK)
ERA_BOUNDARY = 21.5  # Carrick's first game is GW22
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def short(manager: str) -> str:
    return SHORT.get(manager, manager)


def _managers_in(df: pd.DataFrame) -> list[str]:
    present = set(df["Manager"])
    return [m for m in ORDER if m in present]


def _tickformat(kind: str) -> str | None:
    return ".0%" if kind == "pct" else None


def _hoverfmt(kind: str) -> str:
    return ".1%" if kind == "pct" else ".2f"


def base_layout(
    fig: go.Figure,
    height: int = 380,
    x_title: str | None = None,
    y_title: str | None = None,
    legend: bool = True,
) -> go.Figure:
    """Dark theme, hairline grid, legend on top, recessive axes."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=TEXT),
        height=height,
        margin=dict(l=60, r=24, t=36, b=52),
        showlegend=legend,
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=AXIS, font=dict(color=TEXT, family=FONT)),
    )
    axis = dict(gridcolor=GRID, gridwidth=1, linecolor=AXIS, zerolinecolor=AXIS,
                tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_xaxes(**axis, title_text=x_title)
    fig.update_yaxes(**axis, title_text=y_title)
    return fig


def add_era_divider(fig: go.Figure, label: bool = True) -> go.Figure:
    """Green vertical line at the GW22 boundary (Carrick's first game)."""
    fig.add_vline(x=ERA_BOUNDARY, line=dict(color=COLORS[AMORIM], width=1.5))
    if label:
        fig.add_annotation(x=ERA_BOUNDARY, y=1, yref="paper", text="Carrick from GW22",
                           showarrow=False, xanchor="left", yanchor="top", xshift=4,
                           font=dict(color=MUTED, size=11))
    return fig


def _ring() -> dict:
    """2px surface ring so overlapping markers stay separable."""
    return dict(color=SURFACE, width=1.5)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
# Result colours (status job, not identity). Separated by lightness as well as hue so
# W and L stay distinct for red-green colour-blind readers; letters are a second cue.
RESULT_COLORS = {"W": "#6FD394", "D": "#E8B931", "L": "#B92B2B"}
RESULT_TEXT = {"W": SURFACE, "D": SURFACE, "L": "#FFFFFF"}
RESULT_NAMES = {"W": "Win", "D": "Draw", "L": "Loss"}


def form_strip(df: pd.DataFrame) -> go.Figure:
    """One square per game coloured by result (W/D/L), on a background band
    tinted and labelled by the manager in charge."""
    fig = go.Figure()
    for mgr in _managers_in(df):
        g = df[df["Manager"] == mgr]
        x0, x1 = g["GW"].min() - 0.5, g["GW"].max() + 0.5
        fig.add_vrect(x0=x0, x1=x1, fillcolor=_rgba(COLORS[mgr], 0.16), line_width=0,
                      layer="below")
        span = (f"GW{g['GW'].min()}" if x1 - x0 == 1 else
                f"GW{g['GW'].min()} to {g['GW'].max()}")
        fig.add_annotation(x=(x0 + x1) / 2, y=0.62, text=f"<b>{short(mgr)}</b> {span}"
                           if x1 - x0 > 3 else f"<b>{short(mgr)[0]}</b>",
                           showarrow=False, font=dict(color=COLORS[mgr], size=12))
    for res in ("W", "D", "L"):
        g = df[df["Result"] == res].sort_values("GW")
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g["GW"], y=[0] * len(g), mode="markers+text", name=RESULT_NAMES[res],
            marker=dict(symbol="square", size=24, color=RESULT_COLORS[res], line=_ring()),
            text=g["Result"], textfont=dict(color=RESULT_TEXT[res], size=12, family=FONT),
            customdata=np.stack([g["Opponent"], g["venue"],
                                 g["MU goals"].astype(str) + "-" + g["Opp goals"].astype(str),
                                 g["opp_position"].fillna(-1).astype(int).astype(str)
                                 .replace("-1", "n/a"),
                                 g["Date"].dt.strftime("%d %b %Y"),
                                 g["Manager"].map(short)], axis=-1),
            hovertemplate=("GW%{x} %{customdata[4]}<br>%{customdata[0]} (%{customdata[1]})"
                           "<br>Score %{customdata[2]}, %{text}"
                           "<br>Opponent pre-match position %{customdata[3]}"
                           "<extra>%{customdata[5]}</extra>"),
        ))
    base_layout(fig, height=190, x_title="Gameweek")
    fig.update_yaxes(visible=False, range=[-0.8, 0.85])
    fig.update_xaxes(dtick=1, showgrid=False, range=[0.3, 38.7], zeroline=False)
    fig.update_layout(margin=dict(l=24, r=24, t=36, b=44))
    return add_era_divider(fig, label=False)


def cumulative_points_chart(cum: pd.DataFrame, pace_ppg: float = 2.0) -> go.Figure:
    """Cumulative points by games managed with a reference pace line."""
    fig = go.Figure()
    max_games = int(cum["games_managed"].max()) if len(cum) else 1
    fig.add_trace(go.Scatter(
        x=[0, max_games], y=[0, pace_ppg * max_games], mode="lines",
        name=f"{pace_ppg:g} PPG pace", line=dict(color=NEUTRAL, width=1),
        hoverinfo="skip",
    ))
    for mgr in _managers_in(cum):
        g = cum[cum["Manager"] == mgr]
        fig.add_trace(go.Scatter(
            x=g["games_managed"], y=g["cum_points"],
            mode="lines+markers", name=short(mgr),
            line=dict(color=COLORS[mgr], width=2),
            marker=dict(size=8, color=COLORS[mgr], line=_ring()),
            customdata=np.stack([g["GW"], g["Opponent"], g["Result"]], axis=-1),
            hovertemplate=("Game %{x}: %{y} pts<br>GW%{customdata[0]} %{customdata[1]} "
                           f"%{{customdata[2]}}<extra>{short(mgr)}</extra>"),
        ))
        last = g.iloc[-1]
        fig.add_annotation(x=last["games_managed"], y=last["cum_points"],
                           text=f"{short(mgr)} {int(last['cum_points'])}", showarrow=False,
                           xanchor="left", xshift=8, font=dict(color=TEXT, size=12))
    return base_layout(fig, height=380, x_title="Games managed", y_title="Cumulative points")


# ---------------------------------------------------------------------------
# Trend explorer
# ---------------------------------------------------------------------------
def trend_chart(
    points: pd.DataFrame,
    eras: pd.DataFrame,
    label: str,
    kind: str,
    change_points: Sequence[int] = (),
    size_by_opponent: bool = True,
) -> go.Figure:
    """Per-game values, rolling mean per era and era mean segments.

    points needs GW, Manager, Opponent, venue, opp_position, value, rolling.
    """
    fig = go.Figure()
    fmt = _hoverfmt(kind)
    for mgr in _managers_in(points):
        g = points[points["Manager"] == mgr].sort_values("GW")
        if size_by_opponent:
            size = (8 + (21 - g["opp_position"].fillna(10.5)) * 0.7).to_numpy()
        else:
            size = np.full(len(g), 9.0)
        fig.add_trace(go.Scatter(
            x=g["GW"], y=g["value"], mode="markers", name=short(mgr),
            legendgroup=mgr, marker=dict(size=size, color=COLORS[mgr], opacity=0.75,
                                         line=_ring()),
            customdata=np.stack([g["Opponent"], g["venue"],
                                 g["opp_position"].fillna(-1).astype(int).astype(str)
                                 .replace("-1", "n/a")], axis=-1),
            hovertemplate=(f"GW%{{x}} %{{customdata[0]}} (%{{customdata[1]}})<br>{label} "
                           f"%{{y:{fmt}}}<br>Opponent pre-match position %{{customdata[2]}}"
                           f"<extra>{short(mgr)}</extra>"),
        ))
        if len(g) > 1:
            fig.add_trace(go.Scatter(
                x=g["GW"], y=g["rolling"], mode="lines", name=f"{short(mgr)} rolling mean",
                legendgroup=mgr, showlegend=False,
                line=dict(color=COLORS[mgr], width=2),
                hovertemplate=f"Rolling mean %{{y:{fmt}}}<extra>{short(mgr)}</extra>",
            ))
    for _, era in eras.iterrows():
        mgr = era["Manager"]
        if mgr not in COLORS or era["n"] < 2:
            continue
        fig.add_shape(type="line", x0=era["first_gw"] - 0.4, x1=era["last_gw"] + 0.4,
                      y0=era["mean"], y1=era["mean"],
                      line=dict(color=COLORS[mgr], width=1.5), opacity=0.55)
        value = f"{era['mean']:.1%}" if kind == "pct" else f"{era['mean']:.2f}"
        fig.add_annotation(x=era["last_gw"] + 0.4, y=era["mean"], showarrow=False,
                           text=f"{short(mgr)} mean {value}", xanchor="right",
                           yanchor="bottom", yshift=10, bgcolor=_rgba(SURFACE, 0.8),
                           font=dict(color=MUTED, size=11))
    for gw in change_points:
        fig.add_vline(x=gw - 0.5, line=dict(color=NEUTRAL, width=1))
        fig.add_annotation(x=gw - 0.5, y=0, yref="paper", text="change point (exploratory)",
                           showarrow=False, textangle=-90, xanchor="right", yanchor="bottom",
                           font=dict(color=MUTED, size=10))
    base_layout(fig, height=430, x_title="Gameweek", y_title=label)
    fig.update_yaxes(tickformat=_tickformat(kind))
    fig.update_xaxes(range=[0.3, 38.7], dtick=2)
    fig.update_layout(dragmode="select", selectdirection="h")
    return add_era_divider(fig)


def cusum_chart(cus: pd.DataFrame, label: str) -> go.Figure:
    """CUSUM of deviations from the season mean (exploratory)."""
    fig = go.Figure(go.Scatter(
        x=cus["GW"], y=cus["cusum"], mode="lines", line=dict(color=NEUTRAL, width=2),
        showlegend=False, hoverinfo="skip"))
    for mgr in _managers_in(cus):
        g = cus[cus["Manager"] == mgr]
        fig.add_trace(go.Scatter(
            x=g["GW"], y=g["cusum"], mode="markers", name=short(mgr),
            marker=dict(size=8, color=COLORS[mgr], line=_ring()),
            hovertemplate=f"GW%{{x}} CUSUM %{{y:.2f}}<extra>{short(mgr)}</extra>"))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    base_layout(fig, height=280, x_title="Gameweek",
                y_title=f"CUSUM of {label}")
    fig.update_xaxes(range=[0.3, 38.7], dtick=2)
    return add_era_divider(fig, label=False)


# ---------------------------------------------------------------------------
# Head to head
# ---------------------------------------------------------------------------
def _wrap(label: str, width: int = 14) -> str:
    """Break a long axis label onto two lines at the space nearest the middle."""
    if len(label) <= width or " " not in label:
        return label
    spaces = [i for i, ch in enumerate(label) if ch == " "]
    cut = min(spaces, key=lambda i: abs(i - len(label) / 2))
    return label[:cut] + "<br>" + label[cut + 1:]


def radar_chart(rz: pd.DataFrame, height: int = 460) -> go.Figure:
    """Z-scored manager means; further out is better for directional stats.
    Long spoke labels wrap onto two lines so they are not clipped."""
    fig = go.Figure()
    for mgr in _managers_in(rz):
        g = rz[rz["Manager"] == mgr]
        theta = [_wrap(x) for x in g["label"]] + [_wrap(g["label"].iloc[0])]
        r = list(g["z"]) + [g["z"].iloc[0]]
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta, name=short(mgr), fill="toself",
            line=dict(color=COLORS[mgr], width=2), fillcolor=_rgba(COLORS[mgr], 0.12),
            customdata=list(g["n"]) + [g["n"].iloc[0]],
            hovertemplate=f"%{{theta}}: z %{{r:.2f}} (n=%{{customdata}})<extra>{short(mgr)}</extra>",
        ))
    zmax = float(np.nanmax(np.abs(rz["z"]))) if len(rz) else 1
    fig.update_layout(polar=dict(
        bgcolor=SURFACE,
        radialaxis=dict(range=[-max(1.0, zmax) * 1.1, max(1.0, zmax) * 1.1],
                        gridcolor=GRID, linecolor=AXIS, showticklabels=False),
        angularaxis=dict(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=TEXT, size=11)),
    ))
    base_layout(fig, height=height)
    fig.update_layout(margin=dict(l=100, r=100, t=40, b=50),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.06, yanchor="top"))
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def forest_plot(fd: pd.DataFrame) -> go.Figure:
    """Standardised differences (d) with bootstrap CIs, signed so that positive
    is better for Man Utd under Carrick. Only directional stats are drawn (neutral
    style stats have no 'better' and live on the Style page). Dots are coloured by
    the manager they favour; filled = clear or possible difference, hollow = could
    be chance. customdata[0] is the metric key."""
    fd = fd[fd["direction"] != 0].dropna(subset=["d_better"]).iloc[::-1]  # biggest on top
    colour = np.where(fd["d_better"] > 0, COLORS[CARRICK], COLORS[AMORIM])
    solid = (fd["verdict"] != "No reliable difference") if "verdict" in fd else np.ones(len(fd), bool)
    fig = go.Figure(go.Scatter(
        x=fd["d_better"], y=fd["label"], mode="markers",
        marker=dict(size=11, color=np.where(solid, colour, SURFACE), line=dict(color=colour, width=2)),
        error_x=dict(type="data", symmetric=False,
                     array=fd["d_better_high"] - fd["d_better"],
                     arrayminus=fd["d_better"] - fd["d_better_low"],
                     color=MUTED, thickness=1.2, width=0),
        customdata=np.stack([fd["metric"], fd["n_amorim"], fd["n_carrick"],
                             fd["p_permutation"], fd["p_permutation_holm"],
                             fd["d_better_low"], fd["d_better_high"]], axis=-1),
        hovertemplate=("%{y}<br>d %{x:.2f} (95% CI %{customdata[5]:.2f} to %{customdata[6]:.2f})"
                       "<br>n Amorim %{customdata[1]}, Carrick %{customdata[2]}"
                       "<br>permutation p %{customdata[3]:.3f}, Holm %{customdata[4]:.3f}"
                       "<br>Click to open trend<extra></extra>"),
        showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    for mgr, x, anchor in ((AMORIM, -0.02, "right"), (CARRICK, 0.02, "left")):
        fig.add_annotation(x=x, y=1.0, xref="x", yref="paper", text=f"better under {short(mgr)}",
                           showarrow=False, xanchor=anchor, yanchor="bottom",
                           font=dict(color=COLORS[mgr], size=12))
    base_layout(fig, height=max(320, 26 * len(fd) + 110), legend=False,
                x_title="Size of change (gap compared with normal game-to-game swings). Right = better under Carrick")
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_layout(margin=dict(l=24, r=24, t=40, b=56))
    return fig


def violin_chart(df: pd.DataFrame, metric: str, label: str, kind: str) -> go.Figure:
    """Distribution per manager with every game as a jittered point."""
    fig = go.Figure()
    for mgr in _managers_in(df):
        vals = df.loc[df["Manager"] == mgr, [metric, "GW", "Opponent"]].dropna()
        fig.add_trace(go.Violin(
            x=[short(mgr)] * len(vals), y=vals[metric], name=f"{short(mgr)} (n={len(vals)})",
            line=dict(color=COLORS[mgr], width=1.5), fillcolor=_rgba(COLORS[mgr], 0.15),
            points="all", jitter=0.35, pointpos=0,
            marker=dict(color=COLORS[mgr], size=8, line=_ring()),
            box=dict(visible=False), meanline=dict(visible=True, color=TEXT, width=1),
            customdata=np.stack([vals["GW"], vals["Opponent"]], axis=-1),
            hovertemplate=f"GW%{{customdata[0]}} %{{customdata[1]}}: %{{y:{_hoverfmt(kind)}}}<extra></extra>",
            spanmode="hard",
        ))
    base_layout(fig, height=400, y_title=label)
    fig.update_yaxes(tickformat=_tickformat(kind))
    return fig


def venue_split_chart(vs: pd.DataFrame, label: str, kind: str) -> go.Figure:
    """Mean per manager by venue with 95% bootstrap CI whiskers and n labels.
    Managers are nudged sideways so their whiskers do not overlap."""
    fig = go.Figure()
    present = [m for m in (AMORIM, CARRICK) if m in set(vs["Manager"])]
    for i, mgr in enumerate(present):
        g = vs[vs["Manager"] == mgr].set_index("venue").reindex(["Home", "Away"]).reset_index()
        offset = (i - (len(present) - 1) / 2) * 0.12
        has_ci = {"ci_low", "ci_high"} <= set(g.columns)
        fig.add_trace(go.Scatter(
            x=np.array([0.0, 1.0]) + offset, y=g["mean"], mode="lines+markers+text",
            name=short(mgr), line=dict(color=COLORS[mgr], width=1.5, dash="dot"),
            marker=dict(size=12, color=COLORS[mgr], line=_ring()),
            text=[f"n={int(n)}" if pd.notna(n) else "" for n in g["n"]],
            textposition="middle right" if i else "middle left",
            textfont=dict(color=MUTED, size=11),
            error_y=(dict(type="data", symmetric=False, array=g["ci_high"] - g["mean"],
                          arrayminus=g["mean"] - g["ci_low"], color=COLORS[mgr],
                          thickness=1.5, width=5) if has_ci else None),
            customdata=np.stack([g["n"], g.get("ci_low", g["mean"]), g.get("ci_high", g["mean"])],
                                axis=-1),
            hovertemplate=(f"%{{y:{_hoverfmt(kind)}}} (95% CI %{{customdata[1]:{_hoverfmt(kind)}}} "
                           f"to %{{customdata[2]:{_hoverfmt(kind)}}}, n=%{{customdata[0]}})"
                           f"<extra>{short(mgr)}</extra>"),
        ))
    base_layout(fig, height=360, y_title=f"{label} (whisker: 95% bootstrap CI)")
    fig.update_xaxes(tickvals=[0, 1], ticktext=["Home", "Away"], range=[-0.5, 1.5], showgrid=False,
                     zeroline=False)
    fig.update_yaxes(tickformat=_tickformat(kind), rangemode="tozero")
    return fig


def schedule_chart(df: pd.DataFrame) -> go.Figure:
    """Opponent pre-match position faced, each game a dot, mean as a bar mark."""
    fig = go.Figure()
    rng = np.random.default_rng(0)
    present = _managers_in(df)
    for row, mgr in enumerate(present):
        g = df[(df["Manager"] == mgr) & df["opp_position"].notna()]
        # small vertical jitter so tied positions remain visible
        fig.add_trace(go.Scatter(
            x=g["opp_position"], y=row + rng.uniform(-0.15, 0.15, len(g)), mode="markers",
            name=f"{short(mgr)} (n={len(g)})",
            marker=dict(size=10, color=COLORS[mgr], opacity=0.8, line=_ring()),
            customdata=np.stack([g["GW"], g["Opponent"]], axis=-1),
            hovertemplate="GW%{customdata[0]} %{customdata[1]}: position %{x}<extra></extra>",
        ))
        if len(g):
            mean = g["opp_position"].mean()
            fig.add_shape(type="line", x0=mean, x1=mean, y0=row - 0.32, y1=row + 0.32,
                          line=dict(color=TEXT, width=2))
    base_layout(fig, height=280, x_title="Opponent pre-match position (1 = top of table, white bar = mean)")
    fig.update_yaxes(tickvals=list(range(len(present))), ticktext=[short(m) for m in present],
                     showgrid=False, zeroline=False, range=[-0.6, len(present) - 0.4])
    fig.update_xaxes(range=[0.5, 20.5], dtick=1)
    return fig


# ---------------------------------------------------------------------------
# Stats lab
# ---------------------------------------------------------------------------
def permutation_histogram(null: np.ndarray, observed: float, label: str, kind: str) -> go.Figure:
    """Null distribution of Carrick minus Amorim with the observed gap marked."""
    scale = 100 if kind == "pct" else 1
    fig = go.Figure(go.Histogram(
        x=null * scale, nbinsx=60, marker=dict(color=NEUTRAL, line=dict(color=SURFACE, width=1)),
        name="Shuffled manager labels", hovertemplate="%{x}: %{y} shuffles<extra></extra>"))
    for x, txt in ((observed, "observed"), (-observed, "mirror")):
        fig.add_vline(x=x * scale, line=dict(color=TEXT if txt == "observed" else MUTED,
                                             width=2 if txt == "observed" else 1))
    fig.add_annotation(x=observed * scale, y=1, yref="paper", showarrow=False, xanchor="left",
                       yanchor="top", xshift=4, text=f"observed {observed * scale:+.2f}",
                       font=dict(color=TEXT, size=12))
    unit = " (percentage points)" if kind == "pct" else ""
    return base_layout(fig, height=300, x_title=f"Carrick minus Amorim, {label}{unit}",
                       y_title="Shuffles", legend=False)


def null_histogram(null: np.ndarray, observed: float, x_title: str,
                   observed_text: str) -> go.Figure:
    """Shuffled-label scores with the real score marked (one-sided, higher = more extreme)."""
    fig = go.Figure(go.Histogram(
        x=null, nbinsx=30, marker=dict(color=NEUTRAL, line=dict(color=SURFACE, width=1)),
        hovertemplate="%{x}: %{y} shuffles<extra></extra>"))
    fig.add_vline(x=observed, line=dict(color=TEXT, width=2))
    fig.add_annotation(x=observed, y=1, yref="paper", showarrow=False, xanchor="right",
                       yanchor="top", xshift=-4, text=observed_text,
                       font=dict(color=TEXT, size=12))
    return base_layout(fig, height=280, x_title=x_title, y_title="Shuffles", legend=False)


def results_process_chart(fd: pd.DataFrame) -> go.Figure:
    """Results vs process: standardised gaps (d signed so right = better under
    Carrick) for the key measures, grouped Results / Process. Colour = manager
    favoured, filled = clear or possible difference, hollow = could be chance.

    fd needs group, label, d_better, d_better_low, d_better_high, verdict."""
    t = fd.dropna(subset=["d_better"]).iloc[::-1]
    colour = np.where(t["d_better"] > 0, COLORS[CARRICK], COLORS[AMORIM])
    solid = t["verdict"] != "No reliable difference"
    fig = go.Figure(go.Scatter(
        x=t["d_better"], y=[t["group"], t["label"]], mode="markers", showlegend=False,
        marker=dict(size=13, color=np.where(solid, colour, SURFACE), line=dict(color=colour, width=2)),
        error_x=dict(type="data", symmetric=False, array=t["d_better_high"] - t["d_better"],
                     arrayminus=t["d_better"] - t["d_better_low"], color=MUTED, thickness=1.5,
                     width=0),
        customdata=np.stack([t["d_better_low"], t["d_better_high"], t["verdict"],
                             t["n_amorim"], t["n_carrick"]], axis=-1),
        hovertemplate=("%{y}<br>d %{x:.2f} (95% CI %{customdata[0]:.2f} to %{customdata[1]:.2f})"
                       "<br>%{customdata[2]}<br>n Amorim %{customdata[3]}, Carrick %{customdata[4]}"
                       "<extra></extra>"),
    ))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    for mgr, x, anchor in ((AMORIM, -0.02, "right"), (CARRICK, 0.02, "left")):
        fig.add_annotation(x=x, y=1.0, xref="x", yref="paper", text=f"better under {short(mgr)}",
                           showarrow=False, xanchor=anchor, yanchor="bottom",
                           font=dict(color=COLORS[mgr], size=12))
    base_layout(fig, height=360, legend=False,
                x_title="Size of change (gap compared with normal game-to-game swings). Right = better under Carrick")
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(color=TEXT))
    fig.update_layout(margin=dict(l=24, r=24, t=40, b=56))
    return fig


def estimate_chart(rows: pd.DataFrame, null: float, x_title: str, log: bool = False) -> go.Figure:
    """Neutral dot-and-whisker chart for model estimates (one row each), with a
    reference line at the 'no difference' value. rows: label, estimate, ci_low,
    ci_high, and optional n_amorim, n_carrick."""
    t = rows.dropna(subset=["estimate"]).iloc[::-1]
    solid = ~((t["ci_low"] <= null) & (null <= t["ci_high"]))
    n_text = (t["n_amorim"].astype(str) + " / " + t["n_carrick"].astype(str)
              if {"n_amorim", "n_carrick"} <= set(t.columns) else pd.Series("", index=t.index))
    fig = go.Figure(go.Scatter(
        x=t["estimate"], y=t["label"], mode="markers", showlegend=False,
        marker=dict(size=12, color=np.where(solid, TEXT, SURFACE), line=dict(color=TEXT, width=2)),
        error_x=dict(type="data", symmetric=False, array=t["ci_high"] - t["estimate"],
                     arrayminus=t["estimate"] - t["ci_low"], color=MUTED, thickness=1.5, width=0),
        customdata=np.stack([t["ci_low"], t["ci_high"], n_text], axis=-1),
        hovertemplate=("%{y}: %{x:.2f} (95% CI %{customdata[0]:.2f} to %{customdata[1]:.2f})"
                       "<br>games Amorim / Carrick: %{customdata[2]}<extra></extra>"),
    ))
    fig.add_vline(x=null, line=dict(color=AXIS, width=1.5, dash="dot"))
    base_layout(fig, height=max(220, 52 * len(t) + 110), x_title=x_title, legend=False)
    fig.update_xaxes(zeroline=False, type="log" if log else None)
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(color=TEXT))
    fig.update_layout(margin=dict(l=24, r=24, t=24, b=56))
    return fig


def style_shift_chart(table: pd.DataFrame) -> go.Figure:
    """Standardised change per style stat, grouped by area. Dot on the right =
    more of it in Carrick's games. Filled = reliable or possible, hollow = could be chance.

    table needs metric, label, area, cohens_d, d_ci_low, d_ci_high, verdict.
    """
    t = table.dropna(subset=["cohens_d"]).iloc[::-1]
    solid = t["verdict"] != "No reliable difference"
    # One neutral colour: "more" is not better here, so manager colours would mislead.
    fig = go.Figure(go.Scatter(
        x=t["cohens_d"], y=[t["area"], t["label"]], mode="markers", showlegend=False,
        marker=dict(size=11, color=np.where(solid, TEXT, SURFACE),
                    line=dict(color=TEXT, width=2)),
        error_x=dict(type="data", symmetric=False, array=t["d_ci_high"] - t["cohens_d"],
                     arrayminus=t["cohens_d"] - t["d_ci_low"], color=MUTED, thickness=1.2, width=0),
        customdata=np.stack([t["d_ci_low"], t["d_ci_high"], t["verdict"]], axis=-1),
        hovertemplate=("%{y}<br>d %{x:.2f} (95% CI %{customdata[0]:.2f} to %{customdata[1]:.2f})"
                       "<br>%{customdata[2]}<extra></extra>"),
    ))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    for mgr, x, anchor in ((AMORIM, -0.02, "right"), (CARRICK, 0.02, "left")):
        fig.add_annotation(x=x, y=1.0, xref="x", yref="paper", text=f"more in {short(mgr)} games",
                           showarrow=False, xanchor=anchor, yanchor="bottom",
                           font=dict(color=MUTED, size=12))
    base_layout(fig, height=max(360, 24 * len(t) + 120), legend=False,
                x_title="Size of change (gap compared with normal game-to-game swings). Right = more in Carrick's games")
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(color=TEXT))
    fig.update_layout(margin=dict(l=24, r=24, t=40, b=56))
    return fig


def acf_chart(acf_values: np.ndarray, band: float) -> go.Figure:
    """Autocorrelation by lag with the approximate 95% white-noise band."""
    lags = np.arange(1, len(acf_values))
    vals = acf_values[1:]
    fig = go.Figure()
    fig.add_hrect(y0=-band, y1=band, fillcolor=_rgba(NEUTRAL, 0.15), line_width=0)
    fig.add_trace(go.Bar(x=lags, y=vals, width=0.25, marker=dict(color=TEXT),
                         hovertemplate="Lag %{x}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    base_layout(fig, height=260, x_title="Lag (games)", y_title="Autocorrelation", legend=False)
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[-1, 1])
    return fig


# ---------------------------------------------------------------------------
# Luck vs skill
# ---------------------------------------------------------------------------
JITTER = 0.15  # spread for whole-number axes so tied games stay visible


def _is_whole(s: pd.Series) -> bool:
    v = s.dropna().to_numpy(dtype=float)
    return len(v) > 0 and bool(np.all(np.isclose(v, np.round(v))))


def xy_scatter(
    df: pd.DataFrame, x: str, y: str, x_label: str, y_label: str, identity: bool = True,
    means: pd.DataFrame | None = None,
) -> go.Figure:
    """Per-game scatter by manager with an optional y = x line. Whole-number axes
    (goals, big chances) get a small fixed jitter so tied games do not hide each
    other; hover shows the true values. `means` (from stats.xy_means) adds each
    manager's average as a large marker with 95% CI whiskers."""
    fig = go.Figure()
    if identity:
        hi = float(np.nanmax([df[x].max(), df[y].max(), 1])) * 1.05
        fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="y = x",
                                 line=dict(color=NEUTRAL, width=1), hoverinfo="skip"))
    rng = np.random.default_rng(0)
    jx, jy = _is_whole(df[x]), _is_whole(df[y])
    for mgr in _managers_in(df):
        g = df[df["Manager"] == mgr].dropna(subset=[x, y])
        xs = g[x] + (rng.uniform(-JITTER, JITTER, len(g)) if jx else 0)
        ys = g[y] + (rng.uniform(-JITTER, JITTER, len(g)) if jy else 0)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=f"{short(mgr)} games (n={len(g)})",
            legendgroup=mgr,
            marker=dict(size=9, color=COLORS[mgr], opacity=0.7, line=_ring()),
            customdata=np.stack([g["GW"], g["Opponent"], g[x], g[y]], axis=-1),
            hovertemplate=(f"GW%{{customdata[0]}} %{{customdata[1]}}<br>{x_label} "
                           f"%{{customdata[2]:.2f}}<br>{y_label} %{{customdata[3]}}"
                           f"<extra>{short(mgr)}</extra>"),
        ))
    if means is not None and not means.empty:
        for r in means.itertuples():
            fig.add_trace(go.Scatter(
                x=[r.x_mean], y=[r.y_mean], mode="markers", name=f"{short(r.Manager)} average",
                legendgroup=r.Manager,
                marker=dict(size=16, symbol="diamond", color=COLORS[r.Manager],
                            line=dict(color=TEXT, width=2)),
                error_x=dict(type="data", symmetric=False, array=[r.x_high - r.x_mean],
                             arrayminus=[r.x_mean - r.x_low], color=COLORS[r.Manager],
                             thickness=2, width=6),
                error_y=dict(type="data", symmetric=False, array=[r.y_high - r.y_mean],
                             arrayminus=[r.y_mean - r.y_low], color=COLORS[r.Manager],
                             thickness=2, width=6),
                hovertemplate=(f"{short(r.Manager)} average (n={r.n})<br>{x_label} "
                               f"{r.x_mean:.2f} (95% CI {r.x_low:.2f} to {r.x_high:.2f})"
                               f"<br>{y_label} {r.y_mean:.2f} (95% CI {r.y_low:.2f} to "
                               f"{r.y_high:.2f})<extra></extra>"),
            ))
    return base_layout(fig, height=420, x_title=x_label, y_title=y_label)


def xpts_chart(cum: pd.DataFrame) -> go.Figure:
    """Cumulative actual points (solid) vs expected points (dotted) by games managed."""
    fig = go.Figure()
    for mgr in _managers_in(cum):
        g = cum[cum["Manager"] == mgr]
        fig.add_trace(go.Scatter(
            x=g["games_managed"], y=g["cum_points"], mode="lines+markers",
            name=f"{short(mgr)} actual", line=dict(color=COLORS[mgr], width=2),
            marker=dict(size=7, color=COLORS[mgr], line=_ring()),
            hovertemplate=f"Game %{{x}}: %{{y}} pts<extra>{short(mgr)} actual</extra>"))
        fig.add_trace(go.Scatter(
            x=g["games_managed"], y=g["cum_xpts"], mode="lines",
            name=f"{short(mgr)} expected", line=dict(color=COLORS[mgr], width=2, dash="dot"),
            hovertemplate=f"Game %{{x}}: %{{y:.1f}} xPts<extra>{short(mgr)} expected</extra>"))
    return base_layout(fig, height=380, x_title="Games managed (games with a pre-match position)",
                       y_title="Cumulative points")


def ppg_interval_chart(p: pd.DataFrame) -> go.Figure:
    """Pythagorean PPG with bootstrap CI (dot and whisker) and actual PPG (diamond)."""
    fig = go.Figure()
    for _, row in p.iterrows():
        mgr = row["Manager"]
        name = f"{short(mgr)} (n={int(row['n'])})"
        fig.add_trace(go.Scatter(
            x=[row["Pythagorean PPG"]], y=[name], mode="markers", name=f"{short(mgr)} Pythagorean",
            marker=dict(size=12, color=COLORS[mgr], line=_ring()),
            error_x=dict(type="data", symmetric=False, color=COLORS[mgr], thickness=1.5, width=0,
                         array=[row["ci_high"] - row["Pythagorean PPG"]],
                         arrayminus=[row["Pythagorean PPG"] - row["ci_low"]]),
            hovertemplate=(f"Pythagorean %{{x:.2f}} PPG (95% CI {row['ci_low']:.2f} to "
                           f"{row['ci_high']:.2f})<extra>{short(mgr)}</extra>")))
        fig.add_trace(go.Scatter(
            x=[row["Actual PPG"]], y=[name], mode="markers", name=f"{short(mgr)} actual",
            marker=dict(size=13, symbol="diamond-open", color=TEXT, line=dict(width=2)),
            hovertemplate=f"Actual %{{x:.2f}} PPG<extra>{short(mgr)}</extra>"))
    base_layout(fig, height=240, x_title="Points per game (dot: Pythagorean with 95% CI, diamond: actual)",
                legend=False)
    fig.update_yaxes(showgrid=False)
    return fig


def bar_with_dots(df: pd.DataFrame, metric: str, label: str) -> go.Figure:
    """Per-game mean as a thin bar with every game as a jittered dot."""
    fig = go.Figure()
    rng = np.random.default_rng(1)
    for i, mgr in enumerate(_managers_in(df)):
        g = df[df["Manager"] == mgr].dropna(subset=[metric])
        fig.add_trace(go.Bar(
            x=[i], y=[g[metric].mean()], width=0.55, name=f"{short(mgr)} (n={len(g)})",
            marker=dict(color=_rgba(COLORS[mgr], 0.35), line=dict(width=0)),
            hovertemplate=f"Mean %{{y:.2f}} per game (n={len(g)})<extra>{short(mgr)}</extra>"))
        fig.add_trace(go.Scatter(
            x=i + rng.uniform(-0.18, 0.18, len(g)), y=g[metric], mode="markers",
            showlegend=False, marker=dict(size=8, color=COLORS[mgr], line=_ring()),
            customdata=np.stack([g["GW"], g["Opponent"]], axis=-1),
            hovertemplate=f"GW%{{customdata[0]}} %{{customdata[1]}}: %{{y}}<extra>{short(mgr)}</extra>"))
    present = _managers_in(df)
    base_layout(fig, height=360, y_title=label)
    fig.update_xaxes(tickvals=list(range(len(present))), ticktext=[short(m) for m in present],
                     showgrid=False)
    return fig


def loss_dot_plot(losses: pd.DataFrame) -> go.Figure:
    """Each defeat by BC difference and xG difference; lower-left is 'Deserved'."""
    fig = go.Figure()
    lim_x = max(1.0, float(losses["BC diff"].abs().max() or 1)) + 0.8
    lim_y = max(0.5, float(losses["xG diff"].abs().max() or 0.5)) + 0.3
    fig.add_shape(type="rect", x0=-lim_x, x1=0, y0=-lim_y, y1=0, line_width=0,
                  fillcolor=_rgba(NEUTRAL, 0.12), layer="below")
    fig.add_annotation(x=-lim_x, y=-lim_y, text="Deserved: out-chanced on both", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(color=MUTED, size=11))
    for mgr in _managers_in(losses):
        g = losses[losses["Manager"] == mgr]
        fig.add_trace(go.Scatter(
            x=g["BC diff"], y=g["xG diff"], mode="markers+text", name=short(mgr),
            text=g["Opponent"], textposition="top center", textfont=dict(color=MUTED, size=11),
            marker=dict(size=12, color=COLORS[mgr], line=_ring()),
            customdata=np.stack([g["GW"], g["Score"], g["Tag"]], axis=-1),
            hovertemplate=("GW%{customdata[0]} %{text} %{customdata[1]}<br>BC diff %{x}, "
                           "xG diff %{y:.2f}<br>%{customdata[2]}<extra></extra>")))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    base_layout(fig, height=400, x_title="Big chance difference (MU minus opponent)",
                y_title="xG difference (MU minus opponent)")
    fig.update_xaxes(range=[-lim_x, lim_x], dtick=1)
    fig.update_yaxes(range=[-lim_y, lim_y])
    return fig


def bucket_chart(t: pd.DataFrame) -> go.Figure:
    """PPG per big chance difference bucket per manager, n above each bar."""
    fig = go.Figure()
    for mgr in _managers_in(t):
        g = t[t["Manager"] == mgr]
        fig.add_trace(go.Bar(
            x=g["bucket"].astype(str), y=g["PPG"], name=short(mgr),
            marker=dict(color=COLORS[mgr], line=dict(color=SURFACE, width=2)),
            text=[f"n={n}" for n in g["games"]], textposition="inside", insidetextanchor="start",
            textfont=dict(color=TEXT, size=11), cliponaxis=False,
            error_y=(dict(type="data", symmetric=False, array=g["ci_high"] - g["PPG"],
                          arrayminus=g["PPG"] - g["ci_low"], color=TEXT, thickness=1.2, width=4)
                     if "ci_low" in g else None),
            customdata=np.stack([g["games"], g["points"],
                                 g.get("ci_low", g["PPG"]), g.get("ci_high", g["PPG"])], axis=-1),
            hovertemplate=("%{x}: %{y:.2f} PPG (95% CI %{customdata[2]:.2f} to %{customdata[3]:.2f})"
                           "<br>%{customdata[1]} pts from %{customdata[0]} games"
                           f"<extra>{short(mgr)}</extra>")))
    base_layout(fig, height=360, x_title="Big chance difference in the game",
                y_title="Points per game (whisker: 95% bootstrap CI)")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 3.1])
    return fig


# ---------------------------------------------------------------------------
# Style and drivers
# ---------------------------------------------------------------------------
CLUSTER_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "cross", "x")


def pca_chart(scores: pd.DataFrame, explained: np.ndarray) -> go.Figure:
    """PC1 vs PC2, colour = manager, symbol = KMeans cluster."""
    fig = go.Figure()
    for mgr in _managers_in(scores):
        g = scores[scores["Manager"] == mgr]
        fig.add_trace(go.Scatter(
            x=g["PC1"], y=g["PC2"], mode="markers", name=short(mgr), showlegend=False,
            marker=dict(size=11, color=COLORS[mgr], line=_ring(),
                        symbol=[CLUSTER_SYMBOLS[c % len(CLUSTER_SYMBOLS)] for c in g["cluster"]]),
            customdata=np.stack([g["GW"], g["cluster"] + 1], axis=-1),
            hovertemplate=f"GW%{{customdata[0]}} ({short(mgr)}), group %{{customdata[1]}}<extra></extra>"))
    # Legend-only entries: colour = manager, shape = KMeans group.
    for mgr in _managers_in(scores):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=short(mgr),
                                 marker=dict(size=11, color=COLORS[mgr], symbol="circle")))
    for c in sorted(scores["cluster"].unique()):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Group {c + 1}",
                                 marker=dict(size=11, color=NEUTRAL,
                                             symbol=CLUSTER_SYMBOLS[c % len(CLUSTER_SYMBOLS)])))
    return base_layout(fig, height=420,
                       x_title=f"Style score 1 (captures {explained[0]:.0%} of the variation)",
                       y_title=f"Style score 2 ({explained[1]:.0%})")


def interval_chart(t: pd.DataFrame, value: str, low: str, high: str, x_title: str) -> go.Figure:
    """Generic neutral dot-and-whisker chart (e.g. correlations with points)."""
    t = t.dropna(subset=[value]).iloc[::-1]
    fig = go.Figure(go.Scatter(
        x=t[value], y=t["label"], mode="markers", showlegend=False,
        marker=dict(size=9, color=TEXT, line=_ring()),
        error_x=dict(type="data", symmetric=False, array=t[high] - t[value],
                     arrayminus=t[value] - t[low], color=MUTED, thickness=1.2, width=0),
        customdata=np.stack([t[low], t[high], t["n"]], axis=-1),
        hovertemplate=("%{y}: %{x:.2f} (95% CI %{customdata[0]:.2f} to %{customdata[1]:.2f}, "
                       "n=%{customdata[2]})<extra></extra>")))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    base_layout(fig, height=max(300, 24 * len(t) + 100), x_title=x_title, legend=False)
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_layout(margin=dict(l=24, r=24, t=24, b=52))
    return fig


def coefficient_chart(coefs: pd.DataFrame) -> go.Figure:
    """Lasso coefficients (neutral bars); hover shows bootstrap selection frequency."""
    c = coefs[coefs["coef"] != 0].iloc[::-1]
    fig = go.Figure(go.Bar(
        x=c["coef"], y=c["label"], orientation="h", width=0.5,
        marker=dict(color=TEXT, line=dict(width=0)),
        customdata=c["selection_freq"],
        hovertemplate="%{y}: %{x:.3f} points per SD<br>kept in %{customdata:.0%} of refits<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    base_layout(fig, height=max(240, 30 * len(c) + 90),
                x_title="Change in points per game for +1 SD of the stat (Lasso)", legend=False)
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_layout(margin=dict(l=24, r=24, t=24, b=52))
    return fig


# ---------------------------------------------------------------------------
# Match explorer
# ---------------------------------------------------------------------------
def match_card(rows: pd.DataFrame, left_name: str, right_name: str,
               left_colour: str, right_colour: str) -> go.Figure:
    """Sofascore-style mirrored bars. rows: label, kind, left, right."""
    r = rows.iloc[::-1].reset_index(drop=True)
    total = (r["left"].abs() + r["right"].abs()).replace(0, np.nan)
    left_share = (r["left"].abs() / total).fillna(0)
    right_share = (r["right"].abs() / total).fillna(0)

    def fmt(v: float, kind: str) -> str:
        if pd.isna(v):
            return "n/a"
        if kind == "pct":
            return f"{v:.0%}"
        return f"{v:.2f}".rstrip("0").rstrip(".")

    left_txt = [fmt(v, k) for v, k in zip(r["left"], r["kind"])]
    right_txt = [fmt(v, k) for v, k in zip(r["right"], r["kind"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=-left_share, y=r["label"], orientation="h", name=left_name, width=0.55,
        marker=dict(color=left_colour, line=dict(width=0)), text=left_txt,
        textposition="outside", textfont=dict(color=TEXT, size=11), cliponaxis=False,
        hovertemplate=f"%{{y}}: %{{text}}<extra>{left_name}</extra>"))
    fig.add_trace(go.Bar(
        x=right_share, y=r["label"], orientation="h", name=right_name, width=0.55,
        marker=dict(color=right_colour, line=dict(width=0)), text=right_txt,
        textposition="outside", textfont=dict(color=TEXT, size=11), cliponaxis=False,
        hovertemplate=f"%{{y}}: %{{text}}<extra>{right_name}</extra>"))
    base_layout(fig, height=24 * len(r) + 110)
    fig.update_layout(barmode="relative", bargap=0.35, margin=dict(l=24, r=24, t=40, b=24))
    fig.update_xaxes(range=[-1.25, 1.25], visible=False)
    fig.update_yaxes(showgrid=False, automargin=True)
    return fig
