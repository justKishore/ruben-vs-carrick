"""Methods and caveats."""

import streamlit as st

st.title("Methods and caveats")

st.markdown("""
### Data
- **Source:** Sofascore match statistics for all 38 Man Utd Premier League games in 2025-26,
  cross-checked against Sofascore (last checked: 30/09/2026). The workbook is read only and never modified.
- **Managers by date:** Ruben Amorim GW1 to 20 (n=20), Darren Fletcher GW21 (n=1, interim),
  Michael Carrick GW22 to 38 (n=17). Fletcher appears only as a neutral grey marker and is never
  part of a test or merged into either manager.
- **Blank means missing, zero means zero.** Blanks are never filled. Sofascore hides stats that
  are 0 for both teams; the workbook records those as 0, which is treated as a real zero.
- **GW33** has no distance or sprint data. Those two stats use n=16 for Carrick; every other
  stat for GW33 is used.
- **GW1** has no pre-match table, so it is dropped from anything that needs opponent position
  or form: the adjusted model, expected points and schedule difficulty (Amorim n=19 there).
- **GW26 correction:** Sofascore lists 1 Man Utd big chance but 1 scored and 1 missed. One big
  chance was ruled offside, so the dashboard uses 2 created. This is applied in code, not in
  the workbook.
- **GW10:** Sofascore shows 0 Man Utd fouls at Forest. Kept as 0.
- **Dates** are as displayed in the source (Australian time), so some are a day after the UK date.

### Tests
- Every comparison is Carrick minus Amorim and shows n per manager.
- Default test set: Welch t, Mann-Whitney U and a two-sided permutation test on the difference
  in means (10,000 shuffles, fixed seed 2026).
- Effect sizes: Cohen's d (pooled SD) and Cliff's delta, with 95% percentile bootstrap CIs
  (10,000 resamples within each manager, fixed seed).
- When many stats are tested together, Holm correction is applied across that set; raw and
  adjusted p are both shown.
- Points, goals and results are not normal: W/D/L uses an exact Fisher-Freeman-Halton test,
  goals use Poisson and negative binomial models (rate ratio Carrick / Amorim, exposure of one
  game), alongside Mann-Whitney and permutation tests.
- **Adjusted model:** metric ~ manager + venue + opponent pre-match position + opponent form
  (points per game over up to the last 5). OLS uses HC3 robust standard errors.
- **Autocorrelation:** ACF and Ljung-Box on each manager's games after removing that manager's
  mean, to check whether games behave like independent draws.
- **"Better" colouring** is direction-aware and descriptive, not a significance claim. Lower is
  better for goals conceded, opponent xG and shots, fouls, cards, errors, dispossessions and big
  chances missed. Volume or style stats with no clear "better" (possession, distance, sprints,
  GK saves, clearances, tackles, interceptions, recoveries and similar) are neutral.
- Opponent stats that duplicate or mirror a Man Utd stat (possession, duel totals and duel
  percentages) and "Tackles" (identical to "Total tackles") are not offered as separate metrics,
  so they do not inflate the Holm correction.

### Luck vs skill
- **Expected points** come from two Poisson models (goals for, goals against) on venue and
  opponent pre-match position, fitted on all Amorim and Carrick games with no manager term. They
  describe the schedule, and they are fitted on the same games they score.
- **Pythagorean points** treat each manager's goals for and against per game as Poisson rates
  and compute the expected points of such a team, avoiding an exponent fitted on 37 games.
- **Big chances** are a subjective Sofascore event label. Conversion uses pooled counts, with a
  bootstrap CI that resamples games rather than chances to respect clustering.

### Did performance improve?
Seven key measures were fixed in advance (points, goals for and against, xG for and against,
xG difference, big chance difference) and Holm-corrected as their own family. A measure counts
as a clear improvement only if it moved in Man Utd's favour, its corrected permutation p is
below 0.05 and its 95% bootstrap CI excludes zero.

### Did the style change?
32 style stats in four areas (attack, defence, passing, duels), 8 each, are compared one by one
with Holm correction across the 32. As a combined check, a ridge-penalised logistic regression
tries to tell Amorim games from Carrick games using only those stats (5-fold stratified
cross-validation, scaling inside each fold), with a 1,000-shuffle permutation test of its
accuracy. Above-chance accuracy shows the spells differ, but cannot separate the manager from
anything else that changed mid-season.

### Exploratory tools
Change points (PELT), CUSUM, PCA, KMeans, correlations and Lasso are exploratory. With 37 games
and over 100 stats, some patterns will appear by chance.

### The big caveats
- **Small samples.** 20 vs 17 games. Most differences have CIs that include zero.
- **Confounding with season phase.** Carrick took over mid-season. Fixture congestion, injuries,
  opponents' form and the squad itself all changed. Nothing here can separate the manager from
  the time of year, so every result is an association, never a cause.
- **Results are noisier than process.** Goals and points swing on a few moments; xG, shots and
  big chances are more stable signals over 17 games.
""")
