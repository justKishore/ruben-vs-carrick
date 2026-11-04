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
kpi_comps = {k: ui.run_f(f, "compare", col) for k, (col, _) in S.KPI_SPEC.items() if col != "_win"}
wdl = ui.run_f(f, "wdl_test")


def kpi_card(name: str, direction: int) -> str:
    """Both managers at equal weight, the change with its 95% range, and which
    manager the direction favours (in that manager's colour)."""
    kind = "pct" if name == "Win %" else "continuous"
    a, c = kpis.at[AMORIM, name], kpis.at[CARRICK, name]
    if name == "Win %":
        lo, hi = wdl.win_rate_ci_low, wdl.win_rate_ci_high
    else:
        lo, hi = kpi_comps[name].diff_ci_low, kpi_comps[name].diff_ci_high
    diff = c - a
    fav = None if diff == 0 else (CARRICK if diff * direction > 0 else AMORIM)
    row = ("<div style='display:flex;justify-content:space-between;align-items:baseline'>"
           "<span style='color:{col}'>&#9679;</span><span style='flex:1;margin-left:6px;"
           "color:#E6E8EB;font-size:0.85rem'>{who}</span>"
           "<span style='font-size:1.35rem;font-weight:600;color:#E6E8EB'>{val}</span></div>")
    rows = "".join(row.format(col=C.COLORS[m], who=C.short(m), val=S.fmt_value(v, kind))
                   for m, v in ((AMORIM, a), (CARRICK, c)))
    fav_html = (f"<span style='color:{C.COLORS[fav]}'>better under {C.short(fav)}</span>"
                if fav else "no change")
    return (f"<div style='font-size:0.9rem;color:#9AA0A6;margin-bottom:4px'>{name}</div>{rows}"
            f"<div style='margin-top:6px;font-size:0.8rem;color:#E6E8EB'>"
            f"Change {S.fmt_value(diff, kind, signed=True)}, {fav_html}</div>"
            f"<div style='font-size:0.75rem;color:#9AA0A6'>95% range "
            f"{S.fmt_value(lo, kind, True)} to {S.fmt_value(hi, kind, True)}</div>")


for row_items in (list(S.KPI_SPEC.items())[:3], list(S.KPI_SPEC.items())[3:]):
    for col, (name, (_, direction)) in zip(st.columns(3), row_items):
        with col.container(border=True):
            st.markdown(kpi_card(name, direction), unsafe_allow_html=True)

better = [k for k, c in kpi_comps.items() if c.better == CARRICK]
worse = [k for k, c in kpi_comps.items() if c.better == AMORIM]
n_reliable = sum(S.is_reliable(c.p_permutation, c.diff_ci_low, c.diff_ci_high)
                 for c in kpi_comps.values())
ui.finding(
    (f"Under Carrick, {', '.join(better) or 'none of the KPIs'} moved in Man Utd's favour"
     + (f", while {', '.join(worse)} moved the other way" if worse else "")
     + ". "
     + ("None of these gaps is large enough to rule out chance with 20 and 17 games."
        if n_reliable == 0 else
        f"{n_reliable} of {len(kpi_comps)} gaps are larger than chance would usually produce.")),
    f"n={int(kpis.at[AMORIM, 'n'])} Amorim, {int(kpis.at[CARRICK, 'n'])} Carrick games. "
    "For GA and xGA lower is better. 'Better under' shows direction only; a 95% range that "
    "includes zero means no difference is still plausible.",
)

# --- Results vs process ----------------------------------------------------------
ui.section("Results vs process", "results_process")
rvp = ui.run_f(f, "results_vs_process")
res, proc = rvp[rvp["group"] == "Results"], rvp[rvp["group"] == "Process"]
n_res_c = int((res["d_better"] > 0).sum())
n_proc_c = int((proc["d_better"] > 0).sum())
gap = res["d_better"].mean() - proc["d_better"].mean()
n_rel = int((rvp["verdict"] != S.NONE).sum())
ui.chart(
    C.results_process_chart(rvp), "results_vs_process",
    f"Under Carrick, {n_res_c} of {len(res)} results measures and {n_proc_c} of {len(proc)} "
    "chance measures moved in Man Utd's favour. "
    + ("On average, results moved further towards Carrick than the chances did. "
       if gap > 0.2 else
       "On average, the chances moved further towards Carrick than results did. "
       if gap < -0.2 else
       "Results and chances moved by similar amounts. ")
    + ("None of the 7 gaps is reliable, so this could all be chance with 20 and 17 games."
       if n_rel == 0 else f"{n_rel} of the 7 gaps look larger than chance on their own."),
    f"Same 7 key measures as the Stats lab, Holm-corrected across those 7. Whiskers are 95% "
    f"bootstrap CIs on Cohen's d. n={int(rvp['n_amorim'].iloc[0])} Amorim, "
    f"{int(rvp['n_carrick'].iloc[0])} Carrick games.",
)
if FLETCHER in df["Manager"].values:
    fl = df[df["Manager"] == FLETCHER].iloc[0]
    st.caption(f"Fletcher (neutral marker, not tested): GW21 at {fl['Opponent']}, "
               f"{fl['MU goals']}-{fl['Opp goals']}.")

# --- Form strip -----------------------------------------------------------------
ui.section("Form", "form")
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
    + ("The gap looks large, but the plausible range for it still includes zero, so it "
       "could be chance." if not S.is_reliable(pts.p_permutation, pts.diff_ci_low, pts.diff_ci_high)
       else "The gap is larger than chance would usually produce."),
    f"Difference {pts.diff:+.2f} PPG (95% CI {ui.fmt_ci(pts.diff_ci_low, pts.diff_ci_high)}, "
    f"permutation p={S.fmt_p(pts.p_permutation)}, n={pts.n_carrick} vs {pts.n_amorim}).",
)
