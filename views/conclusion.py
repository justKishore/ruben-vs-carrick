"""Conclusion: what we found, in a few lines. Whole season, ignores sidebar filters."""

import streamlit as st

from src import stats as S
from src import ui
from src.load import AMORIM, CARRICK

ALL = ("All", (1, 20))  # unfiltered: venue All, every opponent position


def run(fn: str, *args):
    return ui.run(fn, *ALL, *args)


def ok(c: S.Comparison) -> bool:
    return S.is_reliable(c.p_permutation, c.diff_ci_low, c.diff_ci_high)


def chance_note(*comps: S.Comparison) -> str:
    return ("Not a reliable difference." if not any(ok(c) for c in comps)
            else "Larger than chance would usually produce.")


pts, gf, ga = run("compare", "Points"), run("compare", "MU goals"), run("compare", "Opp goals")
xg, xga, xgd = run("compare", "xG"), run("compare", "Opp xG"), run("compare", "xG difference")
fin, keep = run("compare", "Goals minus xG"), run("compare", "Conceded minus xG against")
wdl = run("wdl_test")
sched = run("schedule_difficulty").iloc[0]
style = run("comparison_table", tuple(S.style_metrics()))
style_clear = [r.label for r in style.itertuples()
               if S.verdict(r.p_permutation, r.p_permutation_holm, r.diff_ci_low,
                            r.diff_ci_high) == S.CLEAR]

st.title("Conclusion")
ui.n_banner(pts.n_amorim, pts.n_carrick, "Whole season. Sidebar filters do not apply here.")

results_up = pts.diff > 0
chances_up = xgd.diff > 0 and ok(xgd)
with st.container(border=True):
    st.markdown("### " + (
        "Better results under Carrick, matched by better chances."
        if results_up and chances_up else
        ("Better results under Carrick" + (", but not proven" if not ok(pts) else "")
         + ", and not matched by better chances.")
        if results_up else
        "No better results under Carrick."))


def finding(title: str, line: str, note: str) -> None:
    st.markdown(f"**{title}.** {ui.clean(line)}  \n"
                f"<span style='color:#9AA0A6;font-size:0.9rem'>{ui.clean(note)}</span>",
                unsafe_allow_html=True)


finding("Results",
        f"{pts.mean_amorim:.2f} to {pts.mean_carrick:.2f} points per game, and a win rate of "
        f"{wdl.win_rate_amorim:.0%} to {wdl.win_rate_carrick:.0%}.",
        chance_note(pts) + f" The plausible range for the points gap is "
        f"{pts.diff_ci_low:+.2f} to {pts.diff_ci_high:+.2f} per game.")

finding("Chances",
        f"xG {xg.mean_amorim:.2f} to {xg.mean_carrick:.2f} per game, xG against "
        f"{xga.mean_amorim:.2f} to {xga.mean_carrick:.2f}. The balance of chances (xG "
        f"difference) went from {xgd.mean_amorim:+.2f} to {xgd.mean_carrick:+.2f}.",
        chance_note(xg, xga, xgd)
        + (" The chances created and allowed changed only a little."
           if max(abs(c.cohens_d) for c in (xg, xga, xgd)) < 0.3 else ""))

finding("Finishing and goalkeeping",
        f"Goals minus xG went from {fin.mean_amorim:+.2f} to {fin.mean_carrick:+.2f} per game, "
        f"and goals conceded minus xG against from {keep.mean_amorim:+.2f} to "
        f"{keep.mean_carrick:+.2f}.",
        chance_note(fin, keep)
        + (" Both moved in Carrick's favour, which is how results improved more than the "
           "chances did. The data cannot say whether that was skill or luck."
           if results_up and fin.diff > 0 and keep.diff < 0 else ""))

finding("Playing style",
        (f"{len(style_clear)} of {len(style)} style stats changed clearly"
         + (f" ({', '.join(S._lower_first(x) for x in style_clear)})." if style_clear else ".")),
        "Most style stats look similar across the two spells."
        if len(style_clear) <= len(style) // 4 else
        "Several style stats differ between the two spells.")

finding("Schedule",
        f"Average opponent league position {sched.mean_amorim:.1f} under Amorim and "
        f"{sched.mean_carrick:.1f} under Carrick.",
        "Broadly comparable; the gap could be chance."
        if not S.is_reliable(sched.p_permutation, sched.diff_ci_low, sched.diff_ci_high) else
        "The schedules differed in strength, which matters when comparing results.")

st.divider()
st.subheader("Did Man Utd play better, or just get the results?")
played_better = chances_up or (xgd.diff > 0 and abs(xgd.cohens_d) >= 0.3)
st.markdown(ui.clean(
    ("**Mostly the results.** Points and wins went up, but the chances Man Utd created and "
     "allowed stayed about the same. The better scorelines came from finishing and "
     "goalkeeping running in Carrick's favour, not from a clear change in how matches were "
     "played." if results_up and not played_better else
     "**Both.** Results improved and so did the balance of chances, so the better scorelines "
     "came with better play on these measures." if results_up else
     "**Neither.** Results did not improve under Carrick.")
    + " With 20 and 17 games, and Carrick taking over at a different point in the season, "
      "this describes what happened. It cannot show what caused it."))

st.subheader("The open question")
st.markdown(ui.clean(
    "Did the Man Utd board make the change needed to save the season, or did it just start "
    "another cycle of managerial changes? If results held up while the underlying chances "
    "stayed flat, which would you expect next season?"))
st.info(ui.clean(
    "Spotted something in the data I missed, or read it differently? I would really "
    "appreciate hearing your findings."), icon=":material/forum:")
