"""Luck vs skill: finishing, chances conceded, expected points, big chances."""

import pandas as pd
import streamlit as st

from src import charts as C
from src import stats as S
from src import ui
from src.load import AMORIM, CARRICK

f = ui.current_filters()
df = ui.games(f)
tdf = ui.test_games(f)

st.title("Luck vs skill")
counts = tdf["Manager"].value_counts()
ui.n_banner(int(counts.get(AMORIM, 0)), int(counts.get(CARRICK, 0)), ui.filter_note(f))
st.info("Over 17 games, results metrics (goals, points) are much noisier than process metrics "
        "(xG, shots, big chances). A gap in results that is not matched by a gap in chances is "
        "more likely to include luck or finishing variance.", icon=":material/info:")
if not ui.need_both(tdf):
    st.stop()


def reliable(c: S.Comparison) -> bool:
    return S.is_reliable(c.p_permutation, c.diff_ci_low, c.diff_ci_high)


def sure(c: S.Comparison) -> str:
    return ("The gap is larger than chance would usually produce." if reliable(c) else
            "The gap between the managers could easily be chance.")


def gap(c: S.Comparison) -> str:
    return (f"Carrick {c.mean_carrick:+.2f} vs Amorim {c.mean_amorim:+.2f} per game "
            f"(n={c.n_carrick} vs {c.n_amorim}; difference 95% CI "
            f"{ui.fmt_ci(c.diff_ci_low, c.diff_ci_high)}, permutation p={S.fmt_p(c.p_permutation)})")


def over_under(value: float, unit: str = "goals") -> str:
    return f"{abs(value):.2f} {unit} per game {'above' if value > 0 else 'below'}"


# --- Goals vs xG -----------------------------------------------------------------------
ui.section("Goals vs xG", "goals_vs_xg")
c = ui.run_f(f, "compare", "Goals minus xG")
ui.chart(C.xy_scatter(df, "xG", "MU goals", "xG", "Goals scored",
                      means=ui.run_f(f, "xy_means", "xG", "MU goals")), "goals_vs_xg",
         f"Carrick's side scored {over_under(c.mean_carrick)} what its chances were worth; "
         f"Amorim's scored {over_under(c.mean_amorim)}. "
         + ("Carrick's side finished further above its chances. " if c.mean_carrick > c.mean_amorim
            else "Amorim's side finished further above its chances. ")
         + sure(c),
         f"Goals minus xG: {gap(c)}.")

# --- Goals conceded vs opponent xG ---------------------------------------------------------
ui.section("Goals conceded vs xG against", "conceded_vs_xga")
ga = ui.run_f(f, "compare", "Opp goals")
xga = ui.run_f(f, "compare", "Opp xG")
over = ui.run_f(f, "compare", "Conceded minus xG against")
if ga.diff < 0:
    conceded_text = (
        f"Goals conceded fell by {-ga.diff:.2f} per game under Carrick, while the quality of "
        f"chances allowed (xG against) fell by {max(-xga.diff, 0):.2f}. "
        + ("Most of the drop is not matched by fewer chances conceded, which points towards "
           "goalkeeping or opponent finishing, but with this few games the data cannot split the "
           "two reliably."
           if -xga.diff < 0.5 * -ga.diff else
           "Much of the drop is matched by fewer chances conceded, though with this few games "
           "neither gap is reliably different from zero."))
else:
    conceded_text = (f"Goals conceded did not fall under Carrick ({ga.mean_carrick:.2f} vs "
                     f"{ga.mean_amorim:.2f} per game).")
ui.chart(C.xy_scatter(df, "Opp xG", "Opp goals", "xG against", "Goals conceded",
                      means=ui.run_f(f, "xy_means", "Opp xG", "Opp goals")),
         "conceded_vs_xga", conceded_text,
         f"xG against difference {xga.diff:+.2f} (95% CI {ui.fmt_ci(xga.diff_ci_low, xga.diff_ci_high)}, "
         f"p={S.fmt_p(xga.p_permutation)}). Conceded minus xG against: {gap(over)}.")

# --- Actual vs expected points ---------------------------------------------------------------
ui.section("Actual points vs expected points", "xpts")
xp = df[df["Manager"].isin(f.managers)].dropna(subset=["xPts"]).sort_values("GW").copy()
xp["games_managed"] = xp.groupby("Manager").cumcount() + 1
xp["cum_points"] = xp.groupby("Manager")["Points"].cumsum()
xp["cum_xpts"] = xp.groupby("Manager")["xPts"].cumsum()
pc = ui.run_f(f, "compare", "Points minus xPts")
ui.chart(C.xpts_chart(xp), "points_vs_xpts",
         f"Given who they played and where, Carrick's side got {over_under(pc.mean_carrick, 'points')} "
         "what Man Utd's season-wide record against similar opponents at the same venue would "
         f"suggest, and Amorim's {over_under(pc.mean_amorim, 'points')}. "
         + ("Carrick's side beat its schedule by more than Amorim's did. " if pc.diff > 0 else
            "Amorim's side beat its schedule by more than Carrick's did. ")
         + sure(pc),
         f"Expected points use only venue and opponent pre-match position. Points minus xPts: "
         f"{gap(pc)}. GW1 has no pre-match position; the model is fitted on the same games it scores.")

# --- Pythagorean ------------------------------------------------------------------------------
ui.section("Pythagorean expected points", "pythagorean")
py = ui.run_f(f, "pythagorean_points").set_index("Manager")
lucky = [C.short(m) for m in py.index
         if py.at[m, "Actual PPG"] > py.at[m, "ci_high"]]
ui.chart(C.ppg_interval_chart(py.reset_index()), "pythagorean",
         "; ".join(f"{C.short(m)}'s goals for and against were worth about "
                   f"{py.at[m, 'Pythagorean PPG']:.2f} points per game and they got "
                   f"{py.at[m, 'Actual PPG']:.2f}" for m in py.index)
         + ". "
         + (f"{' and '.join(lucky)}'s points sit above the expected range: more points than "
            "those goal rates usually produce."
            if lucky else
            "Actual points sit inside the expected ranges, so results broadly match the goals."),
         "Expected points treat goals for and against per game as Poisson rates; ranges are 95% "
         "bootstrap CIs.")
ui.table(py.reset_index(), "pythagorean_table",
         column_config={c: st.column_config.NumberColumn(format="%.2f")
                        for c in py.columns if c != "n"})

# --- Finishing: xGOT vs goals -------------------------------------------------------------------
ui.section("Finishing: xG on target vs goals", "xgot")
c = ui.run_f(f, "compare", "Goals minus xGOT")
ui.chart(C.xy_scatter(df, "xG on target (xGOT)", "MU goals", "xG on target (xGOT)",
                      "Goals scored",
                      means=ui.run_f(f, "xy_means", "xG on target (xGOT)", "MU goals")), "goals_vs_xgot",
         f"Once shots were on target, Carrick's side scored {over_under(c.mean_carrick)} what the "
         f"shots were worth, Amorim's {over_under(c.mean_amorim)}. "
         + ("Carrick's side scored more than the quality of its shots on target suggested. "
            if c.mean_carrick > c.mean_amorim else "")
         + sure(c),
         f"Goals minus xGOT: {gap(c)}.")

# ===============================================================================================
st.header("Big chances")
FOOTNOTE = ("Sofascore's 'big chance' is a subjective event label applied by its analysts, not a "
            "model output. GW26 Man Utd big chances are 2, not the published 1 (one chance was "
            "ruled offside). Totals are small, so counts are shown alongside every percentage.")

# 1. Big chances missed and conversion
ui.section("1. Big chances missed and conversion", "bc_missed")
miss = ui.run_f(f, "compare", "Big chances missed")
miss_nb = ui.run_f(f, "count_model", "Big chances missed", "negbin")
conv = ui.run_f(f, "bc_conversion", "mu")
ui.chart(C.bar_with_dots(df, "Big chances missed", "Big chances missed per game"), "bc_missed",
         f"Carrick's side missed {miss.mean_carrick:.2f} big chances per game against "
         f"{miss.mean_amorim:.2f} under Amorim, and scored {conv.conversion_carrick:.0%} of its big "
         f"chances against {conv.conversion_amorim:.0%}. "
         + ("Carrick's side converted more of its best chances, but "
            if conv.conversion_carrick > conv.conversion_amorim else "")
         + ("the difference could easily be chance with this few chances."
            if conv.p_fisher >= S.ALPHA else "the difference is larger than chance would usually produce."),
         f"NegBin rate ratio for misses {miss_nb.estimate:.2f} (95% CI {miss_nb.ci_low:.2f} to "
         f"{miss_nb.ci_high:.2f}), permutation p={S.fmt_p(miss.p_permutation)}. Conversion difference "
         f"{conv.diff * 100:+.0f} pp, game-resampled 95% CI {conv.boot_ci_low * 100:+.0f} to "
         f"{conv.boot_ci_high * 100:+.0f} pp, Cohen's h {conv.cohens_h:.2f}, Fisher "
         f"p={S.fmt_p(conv.p_fisher)}, z-test p={S.fmt_p(conv.p_ztest)}.")
conv_df = pd.DataFrame({
    "Manager": ["Amorim", "Carrick"],
    "Games": [conv.n_games_amorim, conv.n_games_carrick],
    "Created": [conv.created_amorim, conv.created_carrick],
    "Scored": [conv.scored_amorim, conv.scored_carrick],
    "Missed": [conv.missed_amorim, conv.missed_carrick],
    "Conversion": [conv.conversion_amorim, conv.conversion_carrick],
})
ui.table(conv_df, "bc_conversion",
         column_config={"Conversion": st.column_config.NumberColumn(format="percent")})
st.caption("Missing fewer is not proof of better finishing with samples this small.")

# 2. Games lost vs big chances
ui.section("2. Defeats and big chances", "losses")
losses = S.loss_table(df)
if losses.empty:
    st.info("No defeats with these filters.")
else:
    tags = S.loss_tag_counts(df)
    ui.chart(C.loss_dot_plot(losses), "loss_dots",
             " ".join(f"{C.short(m)} lost {int(r.Losses)} of {int(r.Games)} games, and "
                      f"{int(r.Unlucky)} of those defeats came despite matching or beating the "
                      "opponent on big chances or xG." for m, r in tags.iterrows())
             + (" Most defeats were not one-sided."
                if tags["Unlucky"].sum() > tags["Deserved"].sum() else
                " Most defeats were deserved on the chances."),
             "'Deserved' means out-chanced on both big chances and xG. Single games are very noisy.")
    ui.table(losses.assign(Manager=losses["Manager"].map(C.short)), "loss_table",
             column_config={"xG": st.column_config.NumberColumn(format="%.2f"),
                            "xG against": st.column_config.NumberColumn(format="%.2f"),
                            "xG diff": st.column_config.NumberColumn(format="%.2f")})

# 3. PPG by BC diff bucket
ui.section("3. Points per game by big chance difference", "buckets")
buckets = S.ppg_by_bc_bucket(tdf)
b = buckets.set_index(["Manager", "bucket"])
parts = []
for mgr in (AMORIM, CARRICK):
    neg, pos = b.loc[(mgr, "Negative")], b.loc[(mgr, "Positive")]
    if neg["games"] and pos["games"]:
        parts.append(f"{C.short(mgr)} took {neg['PPG']:.2f} points per game when out-chanced "
                     f"(n={int(neg['games'])}) and {pos['PPG']:.2f} when creating more "
                     f"(n={int(pos['games'])})")
def _bucket_compare() -> str:
    if len(parts) < 2:
        return "Results followed chance quality in a similar way under both."
    pa, pc_ = b.loc[(AMORIM, "Positive")], b.loc[(CARRICK, "Positive")]
    overlap = pa["ci_low"] <= pc_["ci_high"] and pc_["ci_low"] <= pa["ci_high"]
    hedge = (" The ranges overlap, so this could easily be chance." if overlap else
             " The ranges do not overlap, but the cells are small.")
    if abs(pc_["PPG"] - pa["PPG"]) <= 0.25:
        return "Results followed chance quality in a similar way under both." + hedge
    lead, trail = (CARRICK, AMORIM) if pc_["PPG"] > pa["PPG"] else (AMORIM, CARRICK)
    return (f"When creating more big chances, {C.short(lead)}'s side took more points per game "
            f"({b.loc[(lead, 'Positive'), 'PPG']:.2f}) than {C.short(trail)}'s "
            f"({b.loc[(trail, 'Positive'), 'PPG']:.2f})." + hedge)


ui.chart(C.bucket_chart(buckets), "bc_buckets",
         "; ".join(parts) + ". " + _bucket_compare(),
         "Some bars rest on 2 to 4 games, so treat individual bars with caution. Whiskers are "
         "95% bootstrap CIs, resampling games within each bar.")

# 4. Defensive luck
ui.section("4. Defensive luck", "defensive_luck")
opp_bc = ui.run_f(f, "compare", "Opp Big chances")
opp_miss = ui.run_f(f, "compare", "Opp Big chances missed")
opp_conv = ui.run_f(f, "bc_conversion", "opp")
ui.chart(C.xy_scatter(df, "Opp Big chances", "Opp goals", "Opponent big chances",
                      "Goals conceded",
                      means=ui.run_f(f, "xy_means", "Opp Big chances", "Opp goals")), "defensive_luck",
         f"Opponents created {opp_bc.mean_carrick:.2f} big chances per game against Carrick's side "
         f"vs {opp_bc.mean_amorim:.2f} against Amorim's, and scored {opp_conv.conversion_carrick:.0%} "
         f"of them vs {opp_conv.conversion_amorim:.0%}. "
         + (("The drop in big chances conceded is larger than chance would usually produce. "
             if reliable(opp_bc) else
             "The drop in big chances conceded could easily be chance. ") if opp_bc.diff < 0 else "")
         + ("The lower opponent conversion depends mostly on opponent finishing and the "
            "goalkeeper rather than on how many chances were allowed."
            if opp_conv.diff < 0 else
            "Opponents converted no less often, so fewer goals conceded did not come from "
            "opponents missing more."),
         f"Opponent big chances difference {opp_bc.diff:+.2f} (permutation p="
         f"{S.fmt_p(opp_bc.p_permutation)}). Opponent conversion {opp_conv.scored_carrick} of "
         f"{opp_conv.created_carrick} vs {opp_conv.scored_amorim} of {opp_conv.created_amorim}, "
         f"Fisher p={S.fmt_p(opp_conv.p_fisher)}.")
ui.chart(C.bar_with_dots(df, "Opp Big chances missed", "Opponent big chances missed per game"),
         "opp_bc_missed",
         f"Opponents missed {opp_miss.mean_carrick:.2f} big chances per game against Carrick's side "
         f"and {opp_miss.mean_amorim:.2f} against Amorim's. {sure(opp_miss)}",
         f"Permutation p={S.fmt_p(opp_miss.p_permutation)}, n={opp_miss.n_carrick} vs "
         f"{opp_miss.n_amorim}.")
st.caption(FOOTNOTE)
