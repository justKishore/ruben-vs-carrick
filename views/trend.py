"""Trend explorer: per-game values, rolling means, era means, change points."""

import numpy as np
import streamlit as st

from src import charts as C
from src import stats as S
from src import ui
from src.load import AMORIM, CARRICK

f = ui.current_filters()
df = ui.games(f)
label, kind = S.metric_label(f.metric), S.metric_kind(f.metric)

st.title(f"Trend: {label}")
st.markdown(
    "Each dot is a game; bigger dots are stronger opponents. Lines are rolling means "
    f"over {f.window} games, computed within each manager so eras do not blend. "
    "Drag across the chart to select a range of gameweeks; the panels below use that range."
)

if df.empty or df[f.metric].notna().sum() == 0:
    st.warning("No data for this metric with these filters.")
    st.stop()

ui.section("Game by game", "trend")
c1, c2, c3 = st.columns(3)
size_by_opp = c1.toggle("Size dots by opponent strength", value=True)
show_cp = c2.toggle("Change points (exploratory)", value=False)
show_cusum = c3.toggle("CUSUM panel (exploratory)", value=False)

valid = df.dropna(subset=[f.metric]).sort_values("GW")
points = valid[["GW", "Manager", "Opponent", "venue", "opp_position"]].assign(
    value=valid[f.metric], rolling=S.rolling_mean(valid, f.metric, f.window))
eras = S.era_means(df, f.metric)
cps = S.change_points(df, f.metric) if show_cp else []

eras_text = "; ".join(
    f"{C.short(r.Manager)} mean {S.fmt_value(r.mean, kind)} (n={r.n})" for r in eras.itertuples())
season_comp = ui.run_f(f, "compare", f.metric)
spread_vs_gap = (
    "Single games swing much more than the gap between the two averages, so the dots of the "
    "two spells overlap heavily."
    if np.isfinite(season_comp.cohens_d) and abs(season_comp.cohens_d) < 0.5 else
    "The gap between the two averages is large compared with game-to-game swings."
    if np.isfinite(season_comp.cohens_d) else "")
event = ui.chart(
    C.trend_chart(points, eras, label, kind, cps, size_by_opp), "trend_chart",
    f"The flat lines show each spell's average: "
    + "; ".join(f"{C.short(r.Manager)} {S.fmt_value(r.mean, kind)} per game"
                for r in eras.itertuples())
    + f". {spread_vs_gap} The test is in the panel below.",
    f"{eras_text}. Rolling means smooth noise, but with this few games a run of 3 or 4 unusual "
    "matches moves them a lot.",
    on_select="rerun", selection_mode=("box",),
)
if show_cp:
    ui.insight(
        "Change points: " + (", ".join(f"new segment from GW{g}" for g in cps) if cps else
                             "none found at the default penalty")
        + ". PELT on the mean with a BIC-style penalty. Exploratory only: it does not "
          "test whether the manager caused a shift.")

def _lower(text: str) -> str:
    return text[0].lower() + text[1:] if len(text) > 1 and text[1].islower() else text


# --- Range selection -----------------------------------------------------------
xs = [p["x"] for p in (event.selection.points if event and event.selection else [])]
gw_range = (int(min(xs)), int(max(xs))) if xs else None
ui.section("Selected range" if gw_range else "Whole season", "trend_range")
if gw_range:
    st.caption(f"GW{gw_range[0]} to GW{gw_range[1]}. Clear the selection on the chart "
               "(double-click) to return to the whole season.")
sel = S.filter_games(df, gw_range=gw_range) if gw_range else df
sel_tests = sel[sel["Manager"].isin((AMORIM, CARRICK))]

left, right = st.columns([1, 2])
with left:
    ui.table(S.era_means(sel, f.metric).assign(
        Manager=lambda d: d["Manager"].map(C.short)), "trend_era_means",
        column_config={"mean": st.column_config.NumberColumn(label, format="%.3f"),
                       "n": st.column_config.NumberColumn("Games"),
                       "first_gw": st.column_config.NumberColumn("First GW"),
                       "last_gw": st.column_config.NumberColumn("Last GW")})
with right:
    if ui.need_both(sel_tests):
        comp = S.compare(sel_tests, f.metric)
        ui.n_banner(comp.n_amorim, comp.n_carrick, ui.filter_note(f))
        ui.finding(S.plain_finding(comp),
                   S.describe_comparison(comp) + f" Cohen's d {comp.cohens_d:.2f}, Cliff's delta "
                   f"{comp.cliffs_delta:.2f}, Welch p={S.fmt_p(comp.p_welch)}, Mann-Whitney "
                   f"p={S.fmt_p(comp.p_mann_whitney)}.")

if show_cusum:
    ui.section("CUSUM (exploratory)", "cusum")
    cus = S.cusum(sel, f.metric)
    turn = cus.loc[cus["cusum"].abs().idxmax()] if len(cus) else None
    if turn is not None:
        before = "above" if turn["cusum"] > 0 else "below"
        after = "below" if before == "above" else "above"
        cus_text = (f"The running total turns at GW{int(turn['GW'])}: games up to there were mostly "
                    f"{before} the average for {_lower(label)}, and games after it mostly {after}. "
                    + ("That turning point sits close to the change of manager (GW22)."
                       if abs(turn["GW"] - 22) <= 2 else
                       "That turning point is not at the change of manager (GW22)."))
    else:
        cus_text = "Not enough games."
    ui.chart(C.cusum_chart(cus, label), "trend_cusum", cus_text,
             "Exploratory description, not evidence of a manager effect.")
