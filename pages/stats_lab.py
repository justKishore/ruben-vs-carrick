"""Stats lab: test table, adjusted model, autocorrelation, permutation, W/D/L, goals."""

import numpy as np
import pandas as pd
import streamlit as st

from src import charts as C
from src import stats as S
from src import ui
from src.load import AMORIM, CARRICK

f = ui.current_filters()
tdf = ui.test_games(f)
label, kind = S.metric_label(f.metric), S.metric_kind(f.metric)

st.title("Stats lab")
counts = tdf["Manager"].value_counts()
ui.n_banner(int(counts.get(AMORIM, 0)), int(counts.get(CARRICK, 0)),
            "Tests always compare Amorim with Carrick; Fletcher is excluded. " + ui.filter_note(f))
if not ui.need_both(tdf):
    st.stop()
# --- The headline question ----------------------------------------------------------
ui.section("Did the managerial change improve performance?", "improvement")
imp = ui.run_f(f, "improvement_table")
wdl = ui.run_f(f, "wdl_test")
headline, why = S.improvement_summary(imp, wdl)
with st.container(border=True):
    st.markdown(f"### {ui.clean(headline)}")
    st.markdown(ui.clean(why))
imp_view = S.simple_summary(imp).merge(imp[["metric", "p_permutation_holm"]], on="metric")
ui.table(imp_view, "improvement_table",
         column_order=["Stat", "Amorim", "Carrick", "Change", "Plausible range (95%)",
                       "Effect size", "Better under", "p_permutation_holm", "Verdict"],
         column_config={"Stat": st.column_config.TextColumn("Key measure"),
                        "p_permutation_holm": st.column_config.NumberColumn(
                            "Holm p", format="%.3f",
                            help="Permutation p corrected for testing these 7 measures.")})
st.caption("Seven key measures chosen in advance (points, goals for and against, xG for and "
           "against, xG difference, big chance difference). Holm correction is applied across "
           "these seven only, which gives each one the best chance of showing a real change.")

ui.section("How to read the results", "how_to_read")
st.caption("Every test on this page asks one question: if the manager made no difference at all, "
           "how surprising would a gap this big be? Open Explain for each term in plain English.")

# --- All stats compared ------------------------------------------------------------
ui.section("All stats compared", "results_table")
sections = st.multiselect("Sections", ui.SECTIONS, default=ui.SECTIONS, key="lab_sections")
metrics = tuple(m for s in sections for m in S.metrics_by_section()[s])
table = ui.run_f(f, "comparison_table", metrics) if metrics else pd.DataFrame()

VERDICT_STYLE = {S.CLEAR: f"color: {C.TEXT}; font-weight: 600",
                 S.POSSIBLE: "color: #E0A43A",
                 S.NONE: f"color: {C.MUTED}"}
VERDICT_HELP = {
    S.CLEAR: "Unlikely to be chance, even allowing for testing many stats at once.",
    S.POSSIBLE: "Unusual on its own, but with this many stats tested it could easily be luck.",
    S.NONE: "The gap is within what chance alone regularly produces.",
}

if not table.empty:
    summary = S.simple_summary(table)
    tally = summary["Verdict"].value_counts()
    for col, v in zip(st.columns(3), (S.CLEAR, S.POSSIBLE, S.NONE)):
        col.metric(v, f"{int(tally.get(v, 0))} of {len(summary)}", help=VERDICT_HELP[v],
                   border=True)

    detail = st.toggle("Full statistical detail", key="lab_detail",
                       help="Every test, effect size, p-value and Holm correction.")

if not table.empty and not detail:
    show = st.segmented_control(
        "Show", ["All", S.CLEAR, S.POSSIBLE, S.NONE], default="All", key="lab_show",
        required=True)
    view = summary if show == "All" else summary[summary["Verdict"] == show]
    shown_cols = ["Stat", "Section", "Amorim", "Carrick", "Change", "Plausible range (95%)",
                  "Effect size", "Better under", "Verdict", "Games (A / C)"]
    styled = view.style.map(lambda v: VERDICT_STYLE.get(v, ""), subset=["Verdict"])
    event = ui.table(
        view, "results_table", display=styled, height=420, column_order=shown_cols,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Stat": st.column_config.TextColumn(pinned=True),
            "Change": st.column_config.TextColumn(help="Carrick's average minus Amorim's."),
            "Plausible range (95%)": st.column_config.TextColumn(
                help="95% bootstrap confidence interval for the change. If it spans zero, "
                     "no difference is still plausible."),
            "Effect size": st.column_config.TextColumn(
                help="Size of the gap relative to normal game-to-game swings (Cohen's d)."),
            "Better under": st.column_config.TextColumn(
                help="Which manager the direction favours. Blank for neutral stats. "
                     "Not a significance claim."),
            "Verdict": st.column_config.TextColumn(
                help="Clear: Holm-corrected permutation p < 0.05 and the range excludes zero. "
                     "Possible: only without the Holm correction. Otherwise no reliable difference."),
        },
    )
    rows = event.selection.rows if event and event.selection else []
    picked = view.iloc[rows[0]]["metric"] if rows else None
    if picked != st.session_state.get("_lab_last"):
        st.session_state["_lab_last"] = picked
        if picked and picked != f.metric:
            st.session_state[ui.PENDING_METRIC] = picked
            st.rerun()
    st.caption("Select a row to load that stat into the detailed sections below.")

if not table.empty and detail:
    show = table.assign(better=table["better"].map(lambda m: C.short(m) if m else "neutral or tie"))
    cols = ["label", "n_amorim", "n_carrick", "mean_amorim", "mean_carrick", "diff",
            "diff_ci_low", "diff_ci_high", "cohens_d", "d_ci_low", "d_ci_high", "cliffs_delta",
            "p_welch", "p_welch_holm", "p_mann_whitney", "p_mann_whitney_holm",
            "p_permutation", "p_permutation_holm", "better"]
    num = st.column_config.NumberColumn
    ui.table(
        show[cols].sort_values("p_permutation"), "test_table", height=420,
        column_config={
            "label": st.column_config.TextColumn("Stat", pinned=True),
            "n_amorim": num("n A"), "n_carrick": num("n C"),
            "mean_amorim": num("Mean Amorim", format="%.3f"),
            "mean_carrick": num("Mean Carrick", format="%.3f"),
            "diff": num("Diff (C minus A)", format="%.3f"),
            "diff_ci_low": num("Diff CI low", format="%.3f"),
            "diff_ci_high": num("Diff CI high", format="%.3f"),
            "cohens_d": num("Cohen's d", format="%.2f"),
            "d_ci_low": num("d CI low", format="%.2f"), "d_ci_high": num("d CI high", format="%.2f"),
            "cliffs_delta": num("Cliff's delta", format="%.2f"),
            "p_welch": num("Welch p", format="%.4f"), "p_welch_holm": num("Welch Holm", format="%.4f"),
            "p_mann_whitney": num("MW p", format="%.4f"),
            "p_mann_whitney_holm": num("MW Holm", format="%.4f"),
            "p_permutation": num("Perm p", format="%.4f"),
            "p_permutation_holm": num("Perm Holm", format="%.4f"),
            "better": st.column_config.TextColumn("Direction favours"),
        },
    )

if not table.empty:
    n_raw = int((table["p_permutation"] < S.ALPHA).sum())
    n_holm = int((table["p_permutation_holm"] < S.ALPHA).sum())
    expected = 0.05 * len(table)
    clear_names = ", ".join(table.loc[table["p_permutation_holm"] < S.ALPHA, "label"])
    ui.finding(
        f"{n_holm} of {len(table)} stats changed clearly between the two spells"
        + (f" ({clear_names})" if n_holm else "")
        + f". Another {n_raw - n_holm} look unusual on their own, but about {expected:.0f} would "
          "be expected to look that way by pure chance when testing this many stats, so most of "
          "them are probably luck.",
        f"{n_raw} raw permutation p-values below 0.05, {n_holm} after Holm correction.")

# --- Selected metric deep dive ------------------------------------------------------
ui.section(f"Selected stat: {label}", "selected_stat")
comp = ui.run_f(f, "compare", f.metric)
holm_p = None
if not table.empty and f.metric in set(table["metric"]):
    holm_p = float(table.loc[table["metric"] == f.metric, "p_permutation_holm"].iloc[0])
ui.n_banner(comp.n_amorim, comp.n_carrick)
ui.finding(S.plain_finding(comp, holm_p), S.describe_comparison(comp, holm_p))

ui.section("**Shuffle test (permutation)**", "permutation", level="markdown")
perm = ui.run_f(f, "permutation_for_metric", f.metric)
ui.chart(C.permutation_histogram(perm.null_distribution, perm.observed, label, kind),
         "permutation_hist",
         f"When the manager labels were shuffled at random, a gap at least as big as the real one "
         f"({S.fmt_value(perm.observed, kind, True)}) turned up in {perm.p_value:.1%} of 10,000 "
         "shuffles. "
         + ("That is rare, so chance alone is an unlikely explanation for this stat."
            if perm.p_value < S.ALPHA else
            "That is common enough that chance alone could explain the gap."),
         f"Two-sided permutation p={S.fmt_p(perm.p_value)}. Holm correction is not applied here; "
         "see the table above for the corrected verdict.")

# --- Adjusted model -------------------------------------------------------------------
ui.section("Adjusted model", "adjusted")
st.caption(f"{label} ~ manager + venue + opponent pre-match position + opponent form (PPG over "
           "last 5). GW1 has no pre-match table and is dropped. OLS uses HC3 robust errors.")
default_family = 1 if kind == "count" else 0
family = st.radio("Model", ["ols", "poisson", "negbin"], index=default_family, horizontal=True,
                  format_func={"ols": "Linear (OLS)", "poisson": "Poisson", "negbin": "NegBin"}.get,
                  key="lab_family",
                  help="Poisson and NegBin suit per-game counts (goals, big chances). They report "
                       "a rate ratio: Carrick's rate divided by Amorim's.")
model = ui.run_f(f, "adjusted_model", f.metric, family)
ui.n_banner(model.n_amorim, model.n_carrick)
if np.isfinite(model.estimate):
    null = 0.0 if family == "ols" else 1.0
    holds = (np.isfinite(model.p_value) and model.p_value < S.ALPHA
             and (model.ci_low > null or model.ci_high < null))
    ui.finding(
        ("Even after allowing for home or away, opponent strength and opponent form, the gap "
         f"in {label} between the managers remains and is unlikely to be chance."
         if holds else
         "Once home or away, opponent strength and opponent form are taken into account, the "
         f"gap in {label} between the managers is not reliably different from zero."),
        S.describe_model(model, label))
else:
    st.info(S.describe_model(model, label))
if not model.params.empty:
    ui.table(model.params.reset_index(names="term"), "adjusted_params",
             column_config={c: st.column_config.NumberColumn(format="%.3f")
                            for c in ("coef", "ci_low", "ci_high", "p")})
    if family != "ols":
        st.caption("Coefficients are on the log scale; exp(coef) is the rate ratio. "
                   + (f"Poisson dispersion {model.dispersion:.2f} (above 1.5 suggests NegBin)."
                      if family == "poisson" else f"NegBin alpha {model.alpha:.3f}."))

# --- Autocorrelation -------------------------------------------------------------------
ui.section("Autocorrelation", "autocorrelation")
ac = ui.run_f(f, "autocorrelation", f.metric)
if ac.ljung_box.empty:
    st.info("Not enough variation to compute autocorrelation.")
else:
    min_p = float(ac.ljung_box["lb_pvalue"].min())
    ui.chart(C.acf_chart(ac.acf, ac.conf_band), "acf",
             (f"{label} does not come in streaks: one game tells you little about the next, so "
              "treating games as independent (as the tests do) is reasonable."
              if min_p >= S.ALPHA else
              f"{label} shows some streakiness between nearby games, so the tests on this page "
              "may be somewhat overconfident for this stat."),
             f"Values demeaned within each manager (n={ac.n}). Smallest Ljung-Box p across lags 1 "
             f"to {len(ac.ljung_box)} is {S.fmt_p(min_p)}.")
    ui.table(ac.ljung_box, "ljung_box",
             column_config={"lb_stat": st.column_config.NumberColumn("Ljung-Box Q", format="%.3f"),
                            "lb_pvalue": st.column_config.NumberColumn("p", format="%.4f")})

# --- W/D/L --------------------------------------------------------------------------------
ui.section("Results: W/D/L (Fisher exact)", "wdl")
wdl = ui.run_f(f, "wdl_test")
c1, c2 = st.columns([1, 2])
c1.dataframe(wdl.table.rename(index=C.short).assign(n=wdl.table.sum(axis=1).to_numpy()))
with c2:
    ui.finding(
        f"Carrick won {wdl.win_rate_carrick:.0%} of games against {wdl.win_rate_amorim:.0%} for "
        "Amorim. "
        + ("Even a swing that big in wins, draws and losses can happen by chance over 20 and 17 "
           "games, so it is not statistically significant."
           if wdl.p_fisher >= S.ALPHA else
           "The change in the mix of results is larger than chance would usually produce."),
        f"Fisher exact (2 x 3) p={S.fmt_p(wdl.p_fisher)}, Cramer's V {wdl.cramers_v:.2f}. "
        f"Win rate difference {wdl.win_rate_diff * 100:+.0f} pp (95% CI "
        f"{wdl.win_rate_ci_low * 100:+.0f} to {wdl.win_rate_ci_high * 100:+.0f} pp), "
        f"n={wdl.n_carrick} vs {wdl.n_amorim}.")

# --- Goals models -----------------------------------------------------------------------
ui.section("Goals: Poisson and NegBin", "goals_models")
rows = []
for metric, name in (("MU goals", "Goals scored"), ("Opp goals", "Goals conceded")):
    for fam in ("poisson", "negbin"):
        for adjusted in (False, True):
            r = ui.run_f(f, "count_model", metric, fam, adjusted)
            rows.append({"Outcome": name, "Model": fam, "Adjusted": adjusted,
                         "Rate ratio (C / A)": r.estimate, "CI low": r.ci_low, "CI high": r.ci_high,
                         "p": r.p_value, "Bootstrap CI low": r.boot_ci_low,
                         "Bootstrap CI high": r.boot_ci_high, "n A": r.n_amorim, "n C": r.n_carrick,
                         "Poisson dispersion": r.dispersion, "NegBin alpha": r.alpha})
goals = pd.DataFrame(rows)
ui.table(goals, "goals_models", column_config={
    c: st.column_config.NumberColumn(format="%.3f") for c in goals.columns
    if goals[c].dtype.kind == "f"})
def _nb(outcome: str) -> pd.Series:
    sel = goals[(goals["Outcome"] == outcome) & (goals["Model"] == "negbin") & ~goals["Adjusted"]]
    return sel.iloc[0]


def _pct(rr: float) -> str:
    return f"{abs(rr - 1):.0%} {'more' if rr > 1 else 'fewer'}"


gs, gc = _nb("Goals scored"), _nb("Goals conceded")
spans = [n for n, r in (("scored", gs), ("conceded", gc)) if r["CI low"] <= 1 <= r["CI high"]]
ui.finding(
    f"Under Carrick, Man Utd scored {_pct(gs['Rate ratio (C / A)'])} goals per game and conceded "
    f"{_pct(gc['Rate ratio (C / A)'])}. "
    + ("Neither change is statistically reliable: both plausible ranges include no change."
       if len(spans) == 2 else
       "Only one of the changes is statistically reliable." if len(spans) == 1 else
       "Both changes are larger than chance would usually produce."),
    f"NegBin rate ratios (Carrick / Amorim): scored {gs['Rate ratio (C / A)']:.2f} (95% CI "
    f"{gs['CI low']:.2f} to {gs['CI high']:.2f}, p={S.fmt_p(gs['p'])}), conceded "
    f"{gc['Rate ratio (C / A)']:.2f} (95% CI {gc['CI low']:.2f} to {gc['CI high']:.2f}, "
    f"p={S.fmt_p(gc['p'])}). Adjusted rows drop GW1.")
