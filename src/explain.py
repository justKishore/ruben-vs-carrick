"""Plain-English explanations shown behind each "Explain" button.

Each entry answers three things: what am I looking at, how do I read it, and
what should I be careful about. Keep the language simple, no em dashes.
"""

EXPLAIN: dict[str, str] = {
    # --- Overview -------------------------------------------------------------
    "kpis": """
**What it shows:** six headline numbers per game for Carrick, each with the change from Amorim.

- **PPG**: points per game (3 for a win, 1 for a draw).
- **GF / GA**: goals for and goals against per game.
- **xG / xGA**: expected goals for and against. xG scores every shot by how often a
  shot like it is scored, so it measures the quality of chances, not luck in finishing.
- **Win %**: share of games won.

**How to read it:** a green change means better for Man Utd, red means worse. For GA
and xGA, lower is better, so a fall shows green.

**Careful:** the colour only shows direction. It does not mean the gap is real rather
than chance. The Stats lab checks that.
""",
    "form": """
**What it shows:** every league game in order. Each square is one match: **green = win,
yellow = draw, red = loss**, with the letter inside as well.

**Who was in charge:** the shaded background shows the manager. Green tint is Amorim
(GW1 to 20), the grey strip marked F is Fletcher's one game (GW21), and the red tint is
Carrick (GW22 to 38). Hover a square for the opponent, score and the opponent's league
position before kick-off.

**The test underneath:** a Fisher exact test asks whether the mix of wins, draws and
losses is more different between the two managers than you would expect by chance.
""",
    "cumulative": """
**What it shows:** points piling up game by game for each manager, starting from their
first match in charge.

**How to read it:** a steeper line means more points per game. The thin grey line is a
2 points per game pace, roughly what a top four team needs. Lines above it are ahead of
that pace.

**Careful:** Carrick's line is shorter (17 games vs 20). A few results can bend a line
a lot over this many games.
""",
    # --- Trend ------------------------------------------------------------------
    "trend": """
**What it shows:** the selected stat for every game across the season.

- **Dots** are single games. Bigger dots mean a stronger opponent (higher in the table).
- **Solid lines** are rolling averages over the last few games (set the window in the
  sidebar). They are calculated within each manager's spell so the two eras do not blur.
- **Faint flat lines** are each manager's average for the whole spell.

**How to read it:** look at whether the flat lines differ and whether the rolling line
moves steadily or just bounces around.

**Tip:** drag across the chart to pick a range of gameweeks. The panel below then only
uses those games. Double-click the chart to clear it.

**Change points** (optional) mark where an algorithm thinks the average shifted. It is a
pattern-finder, not proof the manager caused the shift.
""",
    "trend_range": """
**What it shows:** the averages and the head-to-head test for the games currently shown
(the whole season, or the range you dragged on the chart).

**How to read it:** the sentence gives both averages, the gap, a range for the gap
(95% CI), and a p-value. If the range includes zero, the data cannot rule out "no real
difference".

See **Stats lab > How to read the results** for what p-values and CIs mean.
""",
    "cusum": """
**What it shows:** a running total of how far each game was above or below the average
of all games shown.

**How to read it:** when the line climbs steadily, games were above average for a
stretch; when it falls, below. A peak or trough is where a run of above-average games
turned into below-average ones, or the reverse.

**Careful:** this is descriptive. A turning point near GW22 is interesting but is not
evidence that the new manager caused it.
""",
    # --- Head to head ----------------------------------------------------------
    "radar": """
**What it shows:** four small radars, one per area of play (attack, defence, passing,
duels). Each has the same 8 stats every time, so you can compare like with like.

**How to read it:** each spoke is one stat on a common scale (a z-score: 0 is the average
game, +1 is one typical game-to-game swing above it). For stats where lower is better,
like shots against, the scale is flipped, so **further out is always better** for Man Utd.
Neutral style stats such as possession are not flipped. Where the red shape bulges past
the green one, Carrick's games were better (or higher) on those stats.

**Careful:** the card under each radar says which gaps are big enough to trust. Small gaps
on the radar are usually normal variation.
""",
    "forest": """
**What it shows:** the gap between the managers for every stat in the chosen area, on one
common scale, with an uncertainty range for each.

- The **dot** is the size of the gap (Cohen's d: the gap divided by the typical
  game-to-game spread, so 0.2 is small, 0.5 medium, 0.8 large).
- The **whisker** is the 95% range of plausible values.
- Red dots favour Carrick, green favour Amorim, grey are neutral style stats.

**How to read it:** if a whisker crosses the zero line, "no real difference" is still
plausible. Use the Area buttons to focus on results, attack, defence, passing or duels.

**Tip:** click a dot to open that stat in the Trend explorer.
""",
    "violin": """
**What it shows:** how the selected stat was spread across every game for each manager.

**How to read it:** each dot is a game. The shape is wider where games were bunched
together. The white line inside is the average.

**Why it helps:** two managers can have the same average but very different
consistency. This shows whether a gap comes from the typical game or from a few
extreme ones.
""",
    "venue_split": """
**What it shows:** the selected stat for each manager, split into home and away games.

**How to read it:** compare the home and away dots for each manager, and compare the
managers at the same venue. The table repeats this for the radar stats.

**Careful:** splitting by venue leaves only 8 to 10 games per group, so these numbers
move a lot from one result to the next.
""",
    "schedule": """
**What it shows:** how strong the opponents were. Each dot is one opponent's league
position before kick-off (1 = top of the table). The white bar is the average.

**How to read it:** a lower average means a harder run of fixtures. If one manager
faced much stronger teams, that matters when comparing their results.

**Careful:** league position is a rough measure of strength, especially early in the
season.
""",
    # --- Stats lab ---------------------------------------------------------------
    "how_to_read": """
**The question every test asks:** "If the manager made no difference at all, how
surprising would a gap this big be?"

- **n**: number of games behind each average. Small n means big uncertainty.
- **Difference**: Carrick's average minus Amorim's.
- **95% CI (confidence interval)**: a range of plausible values for the true gap. If it
  includes 0, "no real difference" is still plausible.
- **p-value**: how often a gap at least this large would turn up by chance if the
  manager made no difference. Below 0.05 is the usual bar for "probably not just chance".
- **Effect size**: how big the gap is compared with normal game-to-game swings.
  Cohen's d: about 0.2 small, 0.5 medium, 0.8 large. Cliff's delta: how often a random
  Carrick game beats a random Amorim game, from -1 to +1.
- **Holm correction**: test 100 stats and about 5 will look "significant" by pure luck.
  Holm raises the bar to allow for that. A result that survives Holm is far more
  trustworthy.
- **Three tests**: Welch t-test (compares averages), Mann-Whitney (compares typical
  values, less affected by outliers) and a permutation test (shuffles the manager labels
  10,000 times to see how often chance alone matches the real gap). When they agree, you
  can be more confident.
""",
    "results_table": """
**What it shows:** every stat compared between the two managers, one row per stat.

**How to read it:**
- **Amorim / Carrick**: average per game.
- **Change**: Carrick minus Amorim, with the 95% range of plausible values.
- **Effect size**: how big the gap is in plain words (negligible, small, medium, large).
- **Better under**: which manager the direction favours. "Neutral stat" means more is
  not clearly better or worse (for example possession, clearances or GK saves), so no
  manager is favoured.
- **Verdict**:
  - *Clear difference*: unlikely to be chance even after allowing for testing many stats.
  - *Possible difference*: looks unusual on its own, but with this many stats tested it
    could easily be luck.
  - *No reliable difference*: the gap is within what chance produces.

Switch on **Full statistical detail** to see every test, p-value and correction.
""",
    "selected_stat": """
**What it shows:** the full comparison for the stat picked in the sidebar.

**How to read it:** the sentence gives both averages, the gap, a 95% range for the gap,
and p-values. "Holm p" is the p-value after allowing for testing many stats at once.
See **How to read the results** above for each term.
""",
    "permutation": """
**What it shows:** what gaps look like when the manager labels are shuffled at random.

**How it works:** take all the games, randomly relabel which manager had which game,
and work out the gap. Do that 10,000 times. The grey bars show those chance gaps. The
white line is the real gap.

**How to read it:** if the white line sits inside the bulk of the grey bars, chance
alone often produces a gap that big. If it sits far out in the tail, chance rarely
does. The p-value is the share of shuffles at least as extreme as the real gap.
""",
    "adjusted": """
**What it shows:** the manager gap after allowing for things that differed between the
two spells: home or away, how high the opponent was in the table, and the opponent's
recent form.

**How to read it:** the sentence gives the gap that remains once those are accounted
for, with a 95% range. If the range includes zero (or 1 for a rate ratio), the gap is
not reliable.

- **Linear (OLS)**: the gap in the stat's own units per game.
- **Poisson / NegBin**: for counts like goals. Gives a **rate ratio**: 1.2 means 20%
  more per game under Carrick, 0.8 means 20% fewer.

**Careful:** it cannot account for everything (injuries, fixture congestion, time of
season). GW1 is left out because there was no league table yet.
""",
    "autocorrelation": """
**What it shows:** whether one game's value tends to predict the next game's.

**Why it matters:** the tests assume games are roughly independent. If good games come
in streaks, the tests can look more confident than they should.

**How to read it:** bars inside the grey band are normal. Bars sticking out mean games
a few matches apart are linked. The Ljung-Box test checks all lags at once: p below
0.05 suggests streakiness.

Each manager's average is removed first, so the change of manager itself does not
count as a streak.
""",
    "wdl": """
**What it shows:** wins, draws and losses for each manager, and whether the mix differs
by more than chance.

**How to read it:** the Fisher exact test gives the p-value. Cramer's V is the strength
of the difference from 0 (none) to 1 (complete). The win rate gap comes with a 95%
range.
""",
    "goals_models": """
**What it shows:** goals scored and conceded per game compared using models built for
counts, because goals are small whole numbers and do not follow a bell curve.

**How to read it:** the **rate ratio** is Carrick's rate divided by Amorim's. 0.7 for
goals conceded means 30% fewer conceded per game. If its range includes 1, there is no
reliable difference.

- **Poisson** assumes goals arrive at a steady random rate.
- **NegBin** allows more variation between games. If the two agree, either is fine.
- **Adjusted** rows also account for venue and opponent strength.
""",
    # --- Luck vs skill ------------------------------------------------------------
    "goals_vs_xg": """
**What it shows:** goals scored against xG for each game.

**How to read it:** the grey diagonal is "scored exactly what the chances were worth".
Dots above it are games where Man Utd finished better than expected; below, worse.

**Why it matters:** over a few months, scoring above xG often comes from hot finishing
streaks that tend not to last. A gap in xG is a sturdier sign of a real change.
""",
    "conceded_vs_xga": """
**What it shows:** goals conceded against the xG Man Utd allowed in each game.

**How to read it:** dots below the diagonal are games where Man Utd conceded fewer than
the chances suggested (good goalkeeping or wasteful opponents). Above it, the opposite.

**The question:** did goals conceded fall under Carrick because Man Utd allowed fewer
good chances (defending), or because opponents missed more (luck)? The note under the
chart splits the drop into those two parts.
""",
    "xpts": """
**What it shows:** actual points piling up (solid line) against expected points (dotted
line) for each manager.

**What "expected" means here:** points a side would typically get from that fixture,
based only on home or away and the opponent's league position. It is about the
schedule, not how the team played.

**How to read it:** solid above dotted means more points than the fixtures suggested.
""",
    "pythagorean": """
**What it shows:** the points per game a team "should" get given how many goals it
scores and concedes, compared with the points it actually got.

**How to read it:** the dot is the expected points per game, with its range. The
diamond is the actual figure. A diamond well to the right of the dot means results ran
ahead of the goals, often from winning close games, which tends not to last.
""",
    "xgot": """
**What it shows:** goals scored against xG on target (xGOT) for each game.

**What xGOT is:** like xG, but only for shots on target and taking into account where
in the goal the shot went. It shows how good the shots that reached the keeper were.

**How to read it:** above the diagonal, shots beat the goalkeeper more often than
expected; below, the keeper did better than expected.
""",
    "bc_missed": """
**What it shows:** big chances missed per game (bars are the average, dots are single
games), and the share of big chances that were scored.

**What a big chance is:** Sofascore's label for a chance a player would usually be
expected to score, often one on one or close range. It is a judgement call by analysts.

**How to read it:** missing more can simply mean creating more, so read it alongside
the conversion table, which shows how many were created and scored.
""",
    "losses": """
**What it shows:** every defeat, placed by big chance difference (across) and xG
difference (up and down).

**How to read it:** defeats in the shaded bottom-left box are "Deserved": the opponent
had more big chances and more xG. Anywhere else is "Unlucky": Man Utd matched or beat
the opponent on at least one of the two but still lost.

**Careful:** single games are very noisy, so this is a description, not a verdict.
""",
    "buckets": """
**What it shows:** points per game depending on whether Man Utd had fewer, the same or
more big chances than the opponent.

**How to read it:** if results follow chances, bars should rise from left to right for
both managers. If one manager earns many points even when out-chanced, that may be luck
or something the stats miss.

**Careful:** some bars rest on only 2 to 4 games (shown as n above each bar).
""",
    "defensive_luck": """
**What it shows:** opponent big chances against goals conceded per game, and how many
big chances opponents missed.

**How to read it:** conceding few big chances is good defending. Opponents missing a
lot of the chances they do get is mostly luck. If goals conceded fell mainly because
opponents missed more, expect it to bounce back.
""",
    # --- Style page and Stats lab headline ---------------------------------------
    "improvement": """
**The question:** after the change from Amorim to Carrick, did Man Utd perform better by
an amount that is unlikely to be luck?

**How it is answered:** seven key measures were chosen in advance: points, goals scored,
goals conceded, xG, xG against, xG difference and big chance difference. Each is tested
with a shuffle test, and the p-values are corrected for testing seven things at once
(Holm). A measure counts as a **clear** improvement only if it moved in Man Utd's favour,
the corrected p-value is below 0.05 and its plausible range excludes zero.

**Why so strict:** with 20 and 17 games, results swing a lot by chance. A team can look
much better over half a season without anything real having changed.

**What it cannot tell you:** even a clear improvement would not prove the manager caused
it, because the change also coincides with a different part of the season.
""",
    "style_answer": """
**What it shows:** a one-line answer to "did the style change?", based on two checks:

1. **Stat by stat:** for 32 style stats, is the gap between the managers bigger than
   chance, after correcting for testing many stats?
2. **All together:** can a simple model tell Carrick games from Amorim games using only
   style stats, better than guessing?

If both say yes, the style changed in measurable ways. If neither does, there is no clear
evidence it changed.
""",
    "style_shift": """
**What it shows:** how each of the 32 style stats moved under Carrick, grouped by area.

**How to read it:**
- Dots on the **right** (red): Carrick's games had more of this stat.
- Dots on the **left** (green): Amorim's games had more.
- **Filled** dots changed by more than chance would usually produce; **hollow** dots could
  be chance.
- The further from the middle, the bigger the change (0.5 medium, 0.8 large).

Here "more" is not better or worse: this chart is about how the team played.
""",
    "classifier": """
**What it shows:** whether the two spells can be told apart from style stats alone.

**How it works:** a simple model is trained on most games to guess the manager from the
style stats, then tested on games it has not seen. This is repeated five times so every
game is tested once.

- **Model accuracy:** how often it guessed right.
- **Guessing:** what you would get by always picking the manager with more games.
- **Chance of a fluke:** the same model is rerun 1,000 times with the manager labels
  shuffled at random. This is how often it did as well by luck.

**How to read it:** accuracy well above guessing with a small fluke chance means the
playing style really did differ between the spells. The sentence underneath lists what
gives each spell away.

**Careful:** the model can also pick up anything else that changed mid-season.
""",
    # --- Style and drivers -------------------------------------------------------
    "pca": """
**What it shows:** every game placed on a "style map" built from all 32 style stats at once.

**How it works:** PCA squeezes the style stats into two summary scores (the axes). Games
that were played in a similar way sit close together. KMeans then splits the games into
two natural groups (dot shapes) without being told who the manager was.

**How to read it:** if red and green dots sit in separate areas, and the shapes line up
with the colours, the two managers' teams played differently. If the colours are mixed
together, any change in style is subtle. The card underneath says which it is.
""",
    "correlation": """
**What it shows:** which stats tend to be higher in games where Man Utd got more points.

**How to read it:** rho runs from -1 to +1. Positive means higher values go with more
points. The whisker is the 95% range. Holm p corrects for testing many stats.

**Careful:** this does not mean the stat causes points. Teams that are winning often
sit back, which changes possession and shots, so cause and effect run both ways.
""",
    "lasso": """
**What it shows:** a model that tries to predict points from all the stats together,
built to keep only the few stats that matter most.

**How to read it:** each bar is how many extra points per game go with a typical
increase in that stat, holding the others fixed. Hover for how often the stat was kept
when the model was refitted on reshuffled data (higher is more stable).

**The key number:** cross-validated R squared, which tests the model on games it did not
see. Near or below zero means it cannot really predict points.
""",
    # --- Match explorer ---------------------------------------------------------
    "match_table": """
**What it shows:** every game in the current filters with its key numbers.

**How to use it:** click a column header to sort. Select a row to open the full match
card below. Download the table as CSV with the button underneath.
""",
    "match_card": """
**What it shows:** every stat for the selected game, side by side, like Sofascore.

**How to read it:** the home team is on the left. Each bar is that team's share of the
combined total, so a long bar means that side dominated the stat.
""",
}
