"""Overview: KPI cards, form strip, cumulative points."""

import streamlit as st

from src import charts as C
from src import stats as S
from src import ui
from src.load import AMORIM, CARRICK, FLETCHER

f = ui.current_filters()
df = ui.games(f)
tdf = ui.test_games(f)

st.title("Amorim vs Carrick")
st.markdown(
    "Same Man Utd squad, Premier League 2025-26. Ruben Amorim managed GW1 to 20, "
    "Darren Fletcher was interim for GW21, and Michael Carrick managed GW22 to 38. "
    "Every comparison shows n per manager. Differences are associations, not causes."
)
counts = tdf["Manager"].value_counts()
ui.n_banner(int(counts.get(AMORIM, 0)), int(counts.get(CARRICK, 0)), ui.filter_note(f))

if df.empty:
    st.warning("No games match these filters.")
    st.stop()
if not ui.need_both(tdf):
    st.stop()

# --- KPI cards -----------------------------------------------------------------
ui.section("Per-game KPIs", "kpis")
kpis = S.kpi_summary(tdf)
cols = st.columns(len(S.KPI_SPEC))
for col, (name, (_, direction)) in zip(cols, S.KPI_SPEC.items()):
    with col.container(border=True):
        a, c = kpis.at[AMORIM, name], kpis.at[CARRICK, name]
        kind = "pct" if name == "Win %" else "continuous"
        st.metric(
            f"{name}, Carrick",
            S.fmt_value(c, kind),
            delta=f"{S.fmt_value(c - a, kind, signed=True)} vs Amorim",
            delta_color="normal" if direction > 0 else "inverse",
        )
        st.caption(f"Amorim {S.fmt_value(a, kind)}. n={int(kpis.at[CARRICK, 'n'])} vs "
                   f"{int(kpis.at[AMORIM, 'n'])}")

kpi_comps = {k: ui.run_f(f, "compare", col) for k, (col, _) in S.KPI_SPEC.items() if col != "_win"}
better = [k for k, c in kpi_comps.items() if c.better == CARRICK]
worse = [k for k, c in kpi_comps.items() if c.better == AMORIM]
n_reliable = sum(S.is_reliable(c.p_permutation, c.diff_ci_low, c.diff_ci_high)
                 for c in kpi_comps.values())
ui.finding(
    (f"Under Carrick, {', '.join(better) or 'none of the KPIs'} improved"
     + (f", while {', '.join(worse)} got worse" if worse else "")
     + (". Results improved more than chance creation did. "
        if "PPG" in better and "xG" in worse else ". ")
     + ("None of these gaps is large enough to rule out chance with 20 and 17 games."
        if n_reliable == 0 else
        f"{n_reliable} of {len(kpi_comps)} gaps are larger than chance would usually produce.")),
    "Green deltas are better for Man Utd (for GA and xGA, lower is better). Colour shows "
    "direction only; the Stats lab tests each gap.",
)
if FLETCHER in df["Manager"].values:
    fl = df[df["Manager"] == FLETCHER].iloc[0]
    st.caption(f"Fletcher (neutral marker, not tested): GW21 at {fl['Opponent']}, "
               f"{fl['MU goals']}-{fl['Opp goals']}.")

# --- Form strip -----------------------------------------------------------------
ui.section("Form", "form")
wdl = ui.run_f(f, "wdl_test")
t = wdl.table
na, nc = int(t.loc[AMORIM].sum()), int(t.loc[CARRICK].sum())
ui.chart(
    C.form_strip(df), "form_strip",
    f"Carrick won {t.at[CARRICK, 'W']} of {nc} games ({wdl.win_rate_carrick:.0%}) and lost "
    f"{t.at[CARRICK, 'L']}; Amorim won {t.at[AMORIM, 'W']} of {na} ({wdl.win_rate_amorim:.0%}) and "
    f"lost {t.at[AMORIM, 'L']}. "
    + ("That is a big swing, but over this few games the mix of wins, draws and losses could "
       "still be chance." if wdl.p_fisher >= S.ALPHA else
       "The change in the mix of results is larger than chance would usually produce."),
    f"Fisher exact test on W/D/L p={S.fmt_p(wdl.p_fisher)}, Cramer's V {wdl.cramers_v:.2f}. "
    f"Win rate difference {wdl.win_rate_diff * 100:+.0f} pp (95% CI "
    f"{wdl.win_rate_ci_low * 100:+.0f} to {wdl.win_rate_ci_high * 100:+.0f} pp).",
)

# --- Cumulative points ------------------------------------------------------------
ui.section("Cumulative points by games managed", "cumulative")
cum = S.cumulative_points(df)
pts = ui.run_f(f, "compare", "Points")
pace = "above" if pts.mean_carrick > 2 else "below"
ui.chart(
    C.cumulative_points_chart(cum), "cumulative_points",
    f"Carrick's side collected points at {pts.mean_carrick:.2f} per game, {pace} a 2 points "
    f"per game pace, compared with {pts.mean_amorim:.2f} under Amorim. "
    + ("The gap looks large but is not statistically significant: the plausible range still "
       "includes zero." if not S.is_reliable(pts.p_permutation, pts.diff_ci_low, pts.diff_ci_high)
       else "The gap is larger than chance would usually produce."),
    f"Difference {pts.diff:+.2f} PPG (95% CI {ui.fmt_ci(pts.diff_ci_low, pts.diff_ci_high)}, "
    f"permutation p={S.fmt_p(pts.p_permutation)}, n={pts.n_carrick} vs {pts.n_amorim}).",
)
