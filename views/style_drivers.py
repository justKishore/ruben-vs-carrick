"""Style and drivers (exploratory): did the playing style change under Carrick?"""

import numpy as np
import pandas as pd
import streamlit as st

from src import charts as C
from src import stats as S
from src import ui
from src.load import AMORIM, CARRICK

f = ui.current_filters()
tdf = ui.test_games(f)

st.title("Did the playing style change?")
st.markdown("This page looks only at **how** Man Utd played (attack, defence, passing and duels), "
            "not at results. It asks whether Carrick's games look different from Amorim's.")
st.warning(ui.SMALL_SAMPLE, icon=":material/warning:")
counts = tdf["Manager"].value_counts()
ui.n_banner(int(counts.get(AMORIM, 0)), int(counts.get(CARRICK, 0)), ui.filter_note(f))
if not ui.need_both(tdf, minimum=5):
    st.stop()

style = tuple(S.style_metrics())
table = ui.run_f(f, "comparison_table", style)
table["verdict"] = [S.verdict(r.p_permutation, r.p_permutation_holm, r.diff_ci_low, r.diff_ci_high)
                    for r in table.itertuples()]
table["area"] = table["metric"].map(S.area_of)
clf = ui.run_f(f, "style_classifier", style)
moved = table[table["verdict"] != S.NONE].sort_values("cohens_d", key=np.abs, ascending=False)


def change_words(rows: pd.DataFrame, limit: int = 4) -> str:
    return "; ".join(S.plain_change(r.metric, r.label, r.diff, r.mean_amorim, r.mean_carrick)
                     for r in rows.head(limit).itertuples())


# --- Short answer --------------------------------------------------------------------
ui.section("The short answer", "style_answer")
model_sees_it = clf.p_value < S.ALPHA
n_clear = int((table["verdict"] == S.CLEAR).sum())
if model_sees_it and len(moved):
    answer = "Some signs the games looked different, in a few specific ways."
elif model_sees_it:
    answer = "Some signs the games looked different, but no single stat changed clearly."
elif n_clear:
    answer = "A little: a few individual stats changed, but the overall style is hard to tell apart."
else:
    answer = "No clear change in playing style."
with st.container(border=True):
    st.markdown(f"### {answer}")
    st.markdown(ui.clean(
        f"A simple model given only style stats picked the right manager for "
        f"{clf.accuracy:.0%} of games, against {clf.baseline:.0%} from guessing "
        f"(p={S.fmt_p(clf.p_value)}). "
        + (f"The biggest changes in Carrick's games: {change_words(moved)}. " if len(moved) else "")
        + f"{n_clear} of {len(table)} style stats changed clearly after correcting for many tests. "
        "Because Carrick took over mid-season, part of any change may come from opponents, "
        "fixtures or the time of year rather than the manager."))

# --- Style shift by area ---------------------------------------------------------------
ui.section("What changed, area by area", "style_shift")
ui.chart(C.style_shift_chart(table), "style_shift",
         " ".join(f"**{area}:** {S.group_finding(table, ms)}"
                  for area, ms in S.STYLE_GROUPS.items()),
         f"Holm correction across all {len(table)} style stats. Filled dots are clear or possible "
         "changes; hollow dots could be chance. Dot side shows which manager's games had more.")

# --- Classifier -------------------------------------------------------------------------
ui.section("Can a model tell the two spells apart?", "classifier")
c1, c2, c3 = st.columns(3)
c1.metric("Model accuracy", f"{clf.accuracy:.0%}", border=True,
          help="Share of games assigned to the right manager on games the model had not seen.")
c2.metric("Guessing", f"{clf.baseline:.0%}", border=True,
          help="Accuracy from always guessing the manager with more games.")
c3.metric("Chance of a fluke", S.fmt_p(clf.p_value), border=True,
          help="Share of 1,000 label shuffles where the model did at least this well (p-value).")
top = clf.weights.head(6)


def _names(labels: pd.Series) -> str:
    items = [lab[0].lower() + lab[1:] if lab[1:2].islower() else lab for lab in labels]
    return ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1] if items else ""


amorim_tells = _names(top.loc[top["weight"] < 0, "label"])
carrick_tells = _names(top.loc[top["weight"] > 0, "label"])
tells = "; ".join(t for t in (
    f"games with more {amorim_tells} look like Amorim's" if amorim_tells else "",
    f"games with more {carrick_tells} look like Carrick's" if carrick_tells else "",
) if t)
ui.chart(C.null_histogram(clf.null_scores, clf.accuracy, "Accuracy with shuffled manager labels",
                          f"real {clf.accuracy:.0%}"), "style_classifier",
         ("The model tells the two spells apart much better than chance, so Carrick's games did "
          "look different. " if model_sees_it else
          "The model does no better than chance, so the style stats do not reliably separate the "
          "two spells. ")
         + f"What gives it away: {tells}.",
         f"Logistic regression with 5-fold cross-validation (n={clf.n_amorim} Amorim, "
         f"{clf.n_carrick} Carrick games), 1,000 label shuffles. Accuracy can also pick up anything "
         "else that changed with the season.")

# --- Style map ---------------------------------------------------------------------------
ui.section("Style map (PCA and clusters)", "pca")
cl = ui.run_f(f, "pca_kmeans", style, n_resamples=2000)
ui.chart(C.pca_chart(cl.scores, cl.explained_variance), "pca",
         ("Left to their own devices, the clusters line up with the managers more than chance, "
          "so the spells occupy different parts of the style map."
          if cl.p_adjusted_rand < S.ALPHA else
          "The two natural groups of games do not line up with the managers: Amorim and Carrick "
          "games are mixed together on the style map. Any change is subtle rather than a "
          "different way of playing.")
         + f" The map captures {cl.explained_variance[:2].sum():.0%} of the variation.",
         f"Adjusted Rand index {cl.adjusted_rand:.2f}, permutation p={S.fmt_p(cl.p_adjusted_rand)} "
         f"(2,000 shuffles). n={cl.n_amorim} vs {cl.n_carrick}.")
with st.expander("Games per cluster and what drives the map"):
    a, b = st.columns(2)
    a.dataframe(cl.crosstab.rename(index=C.short, columns=lambda c: f"Cluster {c + 1}"))
    loads = cl.loadings.assign(abs_pc1=cl.loadings["PC1"].abs()).nlargest(10, "abs_pc1")
    b.dataframe(loads[["PC1", "PC2"]].rename(index=S.metric_label).round(3))

# --- Secondary: drivers of points ----------------------------------------------------------------
st.divider()
st.subheader("Secondary: which stats go with winning points?")
if st.toggle("Show correlation and regression of points (exploratory)", key="show_drivers"):
    drivers = tuple(m for m in S.driver_metrics()
                    if m not in {"Distance covered (km)", "Sprints", "Opp Distance covered (km)",
                                 "Opp Sprints"})
    ui.section("Correlation with points", "correlation")
    cor = ui.run_f(f, "correlation_with_points", drivers)
    shown = cor.head(15)
    best = shown.iloc[0]
    ui.chart(C.interval_chart(shown, "rho", "ci_low", "ci_high",
                              "Spearman rho with points (95% bootstrap CI)"), "correlations",
             f"The stat most linked to points is {best['label']} (rho {best['rho']:.2f}). "
             f"{int((cor['p_holm'] < S.ALPHA).sum())} of {cor['rho'].notna().sum()} links survive "
             "correction for testing many stats. Links can run both ways: teams that are winning "
             "often sit back.",
             "Spearman correlation, bootstrap CIs, Holm correction.")
    ui.table(cor, "correlations_table",
             column_config={c: st.column_config.NumberColumn(format="%.3f")
                            for c in ("rho", "ci_low", "ci_high", "p", "p_holm")})
    ui.section("Regularised regression of points (Lasso)", "lasso")
    la = ui.run_f(f, "regularised_points_model", drivers)
    kept = la.coefficients[la.coefficients["coef"] != 0]
    if kept.empty:
        ui.finding("No stat predicts points reliably on unseen games once the model is penalised "
                   "for complexity.", f"Lasso kept no stats (n={la.n}).")
    else:
        ui.chart(C.coefficient_chart(la.coefficients), "lasso",
                 ("These stats predict points on unseen games only weakly, so treat them as hints."
                  if la.cv_r2 < 0.1 else "These stats carry some real predictive signal.")
                 + f" Strongest: {kept.iloc[0]['label']}.",
                 f"Cross-validated R squared {la.cv_r2:.2f}, n={la.n}.")
