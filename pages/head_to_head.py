"""Head to head: area radars, forest plot, violin, venue split, schedule difficulty."""

import numpy as np
import streamlit as st

from src import charts as C
from src import stats as S
from src import ui
from src.load import AMORIM, CARRICK

f = ui.current_filters()
df = ui.games(f)
tdf = ui.test_games(f)

st.title("Head to head")
counts = tdf["Manager"].value_counts()
ui.n_banner(int(counts.get(AMORIM, 0)), int(counts.get(CARRICK, 0)), ui.filter_note(f))
if not ui.need_both(tdf):
    st.stop()

# One table for every curated stat, so Holm is applied once across all areas.
area_table = ui.run_f(f, "comparison_table", tuple(S.area_metrics()))

# --- Area radars --------------------------------------------------------------------
ui.section("Profile by area", "radar")
st.caption("Four fixed areas of play, 8 stats each. Further out is better for Man Utd; "
           "neutral style stats (like possession) are not flipped.")
areas = list(S.STYLE_GROUPS.items())
for row in range(0, len(areas), 2):
    cols = st.columns(2)
    for col, (area, metrics) in zip(cols, areas[row:row + 2]):
        with col:
            st.markdown(f"**{area}**")
            rz = S.radar_z(tdf, metrics)
            ui.chart(C.radar_chart(rz, height=380), f"radar_{area.lower()}",
                     S.group_finding(area_table, metrics))

# --- Forest plot ---------------------------------------------------------------------
ui.section("Standardised differences (forest plot)", "forest")
area = st.segmented_control("Area", ["All areas", *S.AREA_GROUPS], default="All areas",
                            key="forest_area", required=True)
metrics = S.area_metrics() if area == "All areas" else list(S.AREA_GROUPS[area])
fd = S.forest_data(area_table[area_table["metric"].isin(metrics)])
event = ui.chart(
    C.forest_plot(fd), f"forest_{area.replace(' ', '_').lower()}",
    S.group_finding(area_table, metrics),
    f"{len(fd)} stats shown; Holm correction applied across all {len(area_table)} area stats. "
    "Whiskers are 95% bootstrap CIs; a whisker crossing zero means no difference is still "
    "plausible. Click a dot to open its trend.",
    on_select="rerun", selection_mode=("points",),
)
picked = event.selection.points if event and event.selection else []
clicked = picked[0]["customdata"][0] if picked else None
# The chart keeps its selection, so only act on a new click.
if clicked != st.session_state.get("_forest_last"):
    st.session_state["_forest_last"] = clicked
    if clicked:
        ui.open_metric_trend(clicked)

# --- Violin --------------------------------------------------------------------------
ui.section(f"Distribution: {S.metric_label(f.metric)}", "violin")
kind = S.metric_kind(f.metric)
comp = ui.run_f(f, "compare", f.metric)
spread_a = tdf.loc[tdf["Manager"] == AMORIM, f.metric].std()
spread_c = tdf.loc[tdf["Manager"] == CARRICK, f.metric].std()
consistency = ("Carrick's games were more consistent" if spread_c < 0.8 * spread_a else
               "Amorim's games were more consistent" if spread_a < 0.8 * spread_c else
               "Both spells were about equally consistent")
ui.chart(C.violin_chart(df, f.metric, S.metric_label(f.metric), kind), "violin",
         f"{S.plain_finding(comp)} {consistency} from game to game.",
         S.describe_comparison(comp))

# --- Venue split -------------------------------------------------------------------------
ui.section("Home vs away", "venue_split")
venue_df = S.filter_games(ui.season(), f.managers, "All", f.opp_pos)
vs = S.venue_split(venue_df, [f.metric])
if not vs.empty:
    ha = vs.pivot_table(index="Manager", columns="venue", values="mean")
    parts = []
    for mgr in (AMORIM, CARRICK):
        if mgr in ha.index and {"Home", "Away"} <= set(ha.columns):
            h, a = ha.at[mgr, "Home"], ha.at[mgr, "Away"]
            parts.append(f"{C.short(mgr)} {S.fmt_value(h, kind)} at home vs {S.fmt_value(a, kind)} "
                         "away")
    gap_a = ha.at[AMORIM, "Home"] - ha.at[AMORIM, "Away"] if AMORIM in ha.index else np.nan
    gap_c = ha.at[CARRICK, "Home"] - ha.at[CARRICK, "Away"] if CARRICK in ha.index else np.nan
    bigger = ("The home and away gap was bigger under Carrick" if abs(gap_c) > abs(gap_a) * 1.25
              else "The home and away gap was bigger under Amorim" if abs(gap_a) > abs(gap_c) * 1.25
              else "Both managers had a similar home and away gap")
    ui.chart(C.venue_split_chart(vs, S.metric_label(f.metric), kind), "venue_split",
             f"{S.metric_label(f.metric)}: " + "; ".join(parts) + f". {bigger}.",
             "Each cell has only 8 to 10 games, so these gaps are very noisy. The venue filter "
             "is ignored here.")
    vs_all = S.venue_split(venue_df, S.style_metrics())
    vs_all["Stat"] = vs_all["metric"].map(S.metric_label)
    vs_all["Area"] = vs_all["metric"].map(S.area_of)
    vs_all["Manager"] = vs_all["Manager"].map(C.short)
    with st.expander("Home vs away for every area stat"):
        ui.table(vs_all[["Area", "Stat", "Manager", "venue", "mean", "n"]], "venue_split_table",
                 column_config={"mean": st.column_config.NumberColumn(format="%.3f")})

# --- Schedule difficulty -------------------------------------------------------------------
ui.section("Schedule difficulty", "schedule")
sched = ui.run_f(f, "schedule_difficulty")
pos = sched.iloc[0]
harder = "Carrick" if pos.mean_carrick < pos.mean_amorim else "Amorim"
ui.chart(C.schedule_chart(df), "schedule",
         f"{harder} faced slightly stronger opponents on average (league position "
         f"{min(pos.mean_carrick, pos.mean_amorim):.1f} vs {max(pos.mean_carrick, pos.mean_amorim):.1f}). "
         + ("The difference is small enough to be chance, so the schedules were broadly comparable."
            if not S.is_reliable(pos.p_permutation, pos.diff_ci_low, pos.diff_ci_high) else
            "The schedules differed in strength more than chance would suggest, which matters "
            "when comparing results."),
         f"Average opponent pre-match position: Amorim {pos.mean_amorim:.1f} (n={pos.n_amorim}), "
         f"Carrick {pos.mean_carrick:.1f} (n={pos.n_carrick}). Difference {pos['diff']:+.1f} places "
         f"(95% CI {pos.diff_ci_low:+.1f} to {pos.diff_ci_high:+.1f}, permutation "
         f"p={S.fmt_p(pos.p_permutation)}). GW1 has no pre-match table.")
