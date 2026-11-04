# Amorim vs Carrick: Man Utd 2025-26

A Streamlit dashboard that compares Ruben Amorim and Michael Carrick managing the same Manchester United squad in the 2025-26 Premier League season (38 games). The aim is to be honest about the statistics, not just to make nice charts.

| Manager | Games | GW |
|---|---|---|
| Ruben Amorim | 20 | 1 to 20 |
| Darren Fletcher (interim) | 1 | 21 |
| Michael Carrick | 17 | 22 to 38 |

Fletcher appears only as a neutral marker. He is left out of every test.

## Pages

- **Overview**: KPI cards that account for which direction is better, a W/D/L form strip, and cumulative points by games managed.
- **Trend explorer**: rolling and era means by gameweek, plus optional change-point and CUSUM overlays.
- **Head to head**: a z-scored radar, a forest plot of standardised differences with bootstrap CIs, violins, a home/away split and a schedule difficulty comparison.
- **Stats lab**: Welch t, Mann-Whitney and permutation tests with Cohen's d, Cliff's delta and Holm correction. Also an adjusted regression (manager + venue + opponent position + opponent form), ACF/Ljung-Box checks, Fisher exact on W/D/L, and Poisson/NegBin models for goals.
- **Luck vs skill**: goals vs xG, goals conceded vs opponent xG, actual vs expected points, Pythagorean points, and big chance analysis.
- **Style and drivers** (exploratory): PCA, KMeans and regularised regression.
- **Match explorer**: a sortable match table and a per-match stat card.
- **Methods and caveats**

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

## Caveats

- The samples are small (20 vs 17 games), so most differences are not reliably different from zero.
- The manager change happens at the same point as the change in season phase, so a difference between the two eras cannot be put down to the manager.
- Results metrics (points, goals) are noisier than process metrics (xG, shots) over this many games.
- Sofascore's "big chance" label is a subjective event definition.
