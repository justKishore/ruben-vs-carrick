# Amorim vs Carrick: Man Utd 2025-26

A Streamlit dashboard that compares Ruben Amorim and Michael Carrick managing the same Manchester United squad in the 2025-26 Premier League season (38 games). The aim is to be honest about the statistics, not just to make nice charts.

| Manager | Games | GW |
|---|---|---|
| Ruben Amorim | 20 | 1 to 20 |
| Darren Fletcher (interim) | 1 | 21 |
| Michael Carrick | 17 | 22 to 38 |

Fletcher appears only as a neutral marker. He is left out of every test.

## Key findings

Better results under Carrick (1.55 to 2.29 points per game, 40% to 71% wins), but not proven with 20 and 17 games. By xG the chances barely moved; by big chances they improved. Finishing and goalkeeping also ran in Carrick's favour. So it was mostly the results, with some sign of better chances, and none of these gaps stands out from normal game-to-game variation. The Conclusion page has the full summary.

## Pages

- **Overview**: KPI cards with both managers side by side, a "results vs process" chart (points and goals next to xG and big chances), a W/D/L form strip, and cumulative points by games managed.
- **Trend explorer**: per-game values with rolling and era means by gameweek, plus optional change-point and CUSUM overlays.
- **Head to head**: radars by area of play, a forest plot of standardised differences with bootstrap CIs, violins, a home/away split with CIs, and a schedule difficulty comparison.
- **Stats lab**: Welch t, Mann-Whitney and permutation tests with Cohen's d, Cliff's delta and Holm correction. Also an adjusted model (manager + venue + opponent position + opponent form) shown raw vs adjusted, ACF/Ljung-Box checks, Fisher exact on W/D/L, and Poisson/NegBin rate ratios for goals.
- **Luck vs skill**: goals vs xG and goals conceded vs xG against (with manager averages), actual vs expected points, Pythagorean points, finishing vs xGOT, and big chance analysis.
- **Style and drivers** (exploratory): style shift by area, a classifier test, PCA and KMeans, and what goes with winning points.
- **Match explorer**: a sortable match table and a per-match stat card.
- **Conclusion**: what we found in a few lines, whether Man Utd played better or just got the results, and an open question for readers.
- **Methods and caveats**

Built for non-experts first: every chart has a plain-English "What we found" card and an Explain button. Turn on **Show stats detail** in the sidebar to see p-values, effect sizes and test names. Sidebar filters are saved in the URL, so any view can be shared by link.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
pytest
```

Requires Python 3.11+.

## Data

The data is in `data/man_utd_2025_26_full_stats.xlsx`: per-match Man Utd and opponent stats, mostly from Sofascore. Blank cells mean missing, not zero. GW33 has no distance or sprint tracking, so averages of those stats leave it out.

## Project layout

`app.py` (navigation), `views/` (one file per page; named `views`, not `pages`, so Streamlit does not auto-load it as a legacy multipage app and break deep links), `src/` (`load.py`, `stats.py` and `charts.py` hold the logic and never import Streamlit; `ui.py` and `explain.py` hold the UI glue and help text), `tests/`.

## Feedback

Spotted something in the data I missed, or read it differently? Open an issue, I would really appreciate it.

## Caveats

- The samples are small (20 vs 17 games), so most differences are not reliably different from zero.
- The manager change happens at the same point as the change in season phase, so a difference between the two eras cannot be put down to the manager.
- Results metrics (points, goals) are noisier than process metrics (xG, shots) over this many games.
- Sofascore's "big chance" label is a subjective event definition.
