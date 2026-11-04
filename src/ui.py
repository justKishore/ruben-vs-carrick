"""Streamlit glue shared by app.py and the pages: cached data, sidebar filters
(bound to the URL), a cached stats runner, and chart/table helpers.

Stats and chart logic live in src/stats.py and src/charts.py, which never
import streamlit. This module is the only place they meet the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import charts as C
from src import stats as S
from src.explain import EXPLAIN
from src.load import AMORIM, CARRICK, FLETCHER, load_match_data

MANAGERS = (AMORIM, CARRICK, FLETCHER)
SECTION_OF = {k: m.section for k, m in S.METRICS.items()}
SECTIONS = list(S.metrics_by_section())
DEFAULT_METRIC = "xG"
PENDING_METRIC = "_pending_metric"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def season() -> pd.DataFrame:
    """Per-game frame for the full season, with schedule-based expected points."""
    df = S.per_game_frame(load_match_data())
    xp = S.expected_points(df)[["GW", "xPts", "Points minus xPts"]]
    return df.merge(xp, on="GW", how="left")


@dataclass(frozen=True)
class Filters:
    metric: str
    managers: tuple[str, ...]
    venue: str
    opp_pos: tuple[int, int]
    window: int


def games(f: Filters) -> pd.DataFrame:
    """Season filtered by every sidebar filter (for charts)."""
    return S.filter_games(season(), f.managers, f.venue, f.opp_pos)


def test_games(f: Filters) -> pd.DataFrame:
    """Season filtered by venue and opponent strength. Tests always compare
    Amorim with Carrick, whatever the manager filter says."""
    return S.filter_games(season(), (AMORIM, CARRICK), f.venue, f.opp_pos)


@st.cache_data(show_spinner="Running resampling tests...")
def run(fn: str, venue: str, opp_pos: tuple[int, int], *args: Any, **kwargs: Any) -> Any:
    """Cached call of src.stats.<fn>(filtered season, ...). Keyed on the
    function, the filters and the arguments."""
    df = S.filter_games(season(), (AMORIM, CARRICK), venue, opp_pos)
    return getattr(S, fn)(df, *args, **kwargs)


def run_f(f: Filters, fn: str, *args: Any, **kwargs: Any) -> Any:
    return run(fn, f.venue, f.opp_pos, *args, **kwargs)


# ---------------------------------------------------------------------------
# Sidebar (rendered once in app.py so it persists across pages)
# ---------------------------------------------------------------------------
def _sync_section_to_metric() -> None:
    """Make sure the section matches the metric before the widgets render."""
    pending = st.session_state.pop(PENDING_METRIC, None)
    if pending in S.METRICS:
        st.session_state["section"] = SECTION_OF[pending]
        st.session_state["metric"] = pending
        return
    if "section" not in st.session_state:
        url_metric = st.query_params.get("metric")
        metric = url_metric if url_metric in S.METRICS else DEFAULT_METRIC
        st.session_state["section"] = SECTION_OF[metric]
        st.session_state["metric"] = metric


def _on_section_change() -> None:
    st.session_state["metric"] = S.metrics_by_section()[st.session_state["section"]][0]


def sidebar() -> Filters:
    _sync_section_to_metric()
    sb = st.sidebar
    sb.header("Filters")
    sb.selectbox("Metric section", SECTIONS, key="section", on_change=_on_section_change,
                 bind="query-params")
    options = S.metrics_by_section()[st.session_state["section"]]
    if st.session_state.get("metric") not in options:
        st.session_state["metric"] = options[0]
    sb.selectbox("Metric", options, key="metric", format_func=S.metric_label,
                 bind="query-params")
    sb.multiselect("Managers", MANAGERS, default=[AMORIM, CARRICK], key="managers",
                   format_func=C.short, bind="query-params",
                   help="Fletcher (1 game) is a neutral marker only and never part of a test.")
    sb.segmented_control("Venue", ["All", "Home", "Away"], default="All", key="venue",
                         required=True, bind="query-params")
    sb.slider("Opponent pre-match position", 1, 20, (1, 20), key="opp_pos",
              bind="query-params",
              help="1 = top of the table. GW1 has no pre-match table, so it is only "
                   "included when the full 1 to 20 range is selected.")
    sb.slider("Smoothing window (games)", 1, 8, 4, key="window", bind="query-params")
    sb.caption("Filters are saved in the URL, so a view can be shared by copying the link.")
    return current_filters()


def current_filters() -> Filters:
    ss = st.session_state
    managers = tuple(m for m in MANAGERS if m in (ss.get("managers") or []))
    return Filters(
        metric=ss.get("metric", DEFAULT_METRIC),
        managers=managers,
        venue=ss.get("venue") or "All",
        opp_pos=tuple(ss.get("opp_pos", (1, 20))),
        window=int(ss.get("window", 4)),
    )


def open_metric_trend(metric: str) -> None:
    """Switch to the trend page with this metric selected."""
    st.session_state[PENDING_METRIC] = metric
    st.switch_page("views/trend.py")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
_DASHES = re.compile("[–—]")


def clean(text: str) -> str:
    """UI text never contains en or em dashes."""
    return _DASHES.sub(",", text)


def insight(text: str) -> None:
    """One-line plain-English takeaway under a chart."""
    st.caption(clean(text))


def section(title: str, explain_key: str, level: str = "subheader") -> None:
    """Section heading with an 'Explain' button that opens a plain-English note."""
    left, right = st.columns([5, 1], vertical_alignment="bottom")
    getattr(left, level)(title)
    with right.popover("Explain", icon=":material/help:", width="stretch"):
        st.markdown(clean(EXPLAIN[explain_key]))


def chip(manager: str, text: str) -> str:
    colour = C.COLORS.get(manager, C.NEUTRAL)
    return (f"<span style='display:inline-block;padding:2px 10px;margin-right:6px;"
            f"border-radius:999px;border:1px solid {colour};color:#E6E8EB;font-size:0.9rem'>"
            f"<span style='color:{colour}'>&#9679;</span> {text}</span>")


def n_banner(n_amorim: int, n_carrick: int, note: str = "") -> None:
    """n per manager, shown next to any test or comparison."""
    html = chip(AMORIM, f"Amorim n={n_amorim}") + chip(CARRICK, f"Carrick n={n_carrick}")
    if note:
        html += f"<span style='color:#9AA0A6;font-size:0.85rem'>{clean(note)}</span>"
    st.markdown(html, unsafe_allow_html=True)


def filter_note(f: Filters) -> str:
    parts = []
    if f.venue != "All":
        parts.append(f"{f.venue.lower()} games only")
    if f.opp_pos != (1, 20):
        parts.append(f"opponents placed {f.opp_pos[0]} to {f.opp_pos[1]}")
    return ("Filtered: " + ", ".join(parts) + ".") if parts else ""


def finding(text: str, detail: str = "") -> None:
    """Findings card: the plain-English result, then optional statistical detail."""
    with st.container(border=True):
        st.markdown(f":material/lightbulb: **What we found.** {clean(text)}")
        if detail:
            st.caption(clean(detail))


def chart(
    fig: go.Figure,
    key: str,
    takeaway: str,
    detail: str = "",
    on_select: str = "ignore",
    selection_mode: tuple[str, ...] | str = ("points", "box", "lasso"),
) -> Any:
    """Plotly chart, its findings card, and a lazily rendered PNG download."""
    event = st.plotly_chart(
        fig, theme=None, key=key, on_select=on_select, selection_mode=selection_mode,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]},
        width="stretch",
    )
    finding(takeaway, detail)
    st.download_button(
        "PNG", data=lambda: fig.to_image(format="png", scale=2), file_name=f"{key}.png",
        mime="image/png", key=f"{key}_png", icon=":material/download:", type="tertiary",
    )
    return event


def table(df: pd.DataFrame, key: str, display: Any = None, **kwargs: Any) -> Any:
    """Dataframe with a CSV download button. `display` (e.g. a Styler) is shown
    instead of df when given; the CSV is always the plain df."""
    event = st.dataframe(df if display is None else display, key=key,
                         hide_index=kwargs.pop("hide_index", True), **kwargs)
    st.download_button(
        "CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=f"{key}.csv",
        mime="text/csv", key=f"{key}_csv", icon=":material/download:", type="tertiary",
    )
    return event


def fmt_ci(lo: float, hi: float, kind: str = "continuous") -> str:
    return f"{S.fmt_value(lo, kind, True)} to {S.fmt_value(hi, kind, True)}"


def need_both(df: pd.DataFrame, minimum: int = 2) -> bool:
    """Warn and return False if either manager has too few games for a test."""
    counts = df["Manager"].value_counts()
    a, c = int(counts.get(AMORIM, 0)), int(counts.get(CARRICK, 0))
    if a < minimum or c < minimum:
        st.info(f"Not enough games for a test with these filters (Amorim n={a}, "
                f"Carrick n={c}). Widen the filters.")
        return False
    return True


SMALL_SAMPLE = ("Exploratory. With 37 games and many stats, patterns here can easily be "
                "noise. Treat as hypotheses, not findings.")
