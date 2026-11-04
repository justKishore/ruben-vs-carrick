"""Match explorer: sortable table and a Sofascore-style match card."""

import pandas as pd
import streamlit as st

from src import charts as C
from src import stats as S
from src import ui

f = ui.current_filters()
df = ui.games(f)

st.title("Match explorer")
ui.section("All matches", "match_table")
st.caption("Sort by any column. Select a row to open its match card.")
if df.empty:
    st.warning("No games match these filters.")
    st.stop()

view = pd.DataFrame({
    "GW": df["GW"],
    "Date": df["Date"],
    "Manager": df["Manager"].map(C.short),
    "Opponent": df["Opponent"],
    "Venue": df["venue"],
    "Score": df["MU goals"].astype(str) + "-" + df["Opp goals"].astype(str),
    "Result": df["Result"],
    "Points": df["Points"],
    "xG": df["xG"],
    "xG against": df["Opp xG"],
    "Big chances": df["Big chances"],
    "Big chances against": df["Opp Big chances"],
    "Possession": df["Possession %"],
    "Opp position": df["opp_position"],
    "Expected points": df["xPts"],
}).reset_index(drop=True)
num = st.column_config.NumberColumn
event = ui.table(
    view, "matches", on_select="rerun", selection_mode="single-row", height=420,
    column_config={
        "Date": st.column_config.DateColumn(format="DD MMM YYYY"),
        "xG": num(format="%.2f"), "xG against": num(format="%.2f"),
        "Possession": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Opp position": num("Opp pre-match position", format="%d",
                            help="Blank for GW1: no table before the first game."),
        "Expected points": num(format="%.2f",
                               help="From venue and opponent position only."),
        "Points": num(format="%d"),
    },
)

swing = (df["Points"] - df["xPts"]).abs() if "xPts" in df else None
w, d, l = (int((df["Result"] == r).sum()) for r in ("W", "D", "L"))
outplayed_wins = df[(df["Result"] == "W") & (df["xG"] < df["Opp xG"])]
dominant_losses = df[(df["Result"] == "L") & (df["xG"] > df["Opp xG"])]
ui.finding(
    f"{len(df)} games shown: {w} wins, {d} draws, {l} defeats. Man Utd won {len(outplayed_wins)} "
    f"games despite lower xG than the opponent and lost {len(dominant_losses)} despite higher xG"
    + (f" (for example GW{int(dominant_losses.iloc[0]['GW'])} against "
       f"{dominant_losses.iloc[0]['Opponent']})" if len(dominant_losses) else "")
    + ".",
    "Sort by xG or xG against to find the most one-sided games.")

rows = event.selection.rows if event and event.selection else []
if not rows:
    st.info("Select a match in the table to see every stat side by side.")
    st.stop()

game = df.iloc[rows[0]]
mu_home = game["venue"] == "Home"
home, away = ("Man Utd", game["Opponent"]) if mu_home else (game["Opponent"], "Man Utd")
mu_colour = C.COLORS.get(game["Manager"], C.NEUTRAL)
home_colour, away_colour = (mu_colour, C.OPPONENT) if mu_home else (C.OPPONENT, mu_colour)
home_goals, away_goals = ((game["MU goals"], game["Opp goals"]) if mu_home
                          else (game["Opp goals"], game["MU goals"]))

ui.section(f"GW{game['GW']}: {home} {home_goals}-{away_goals} {away}", "match_card")
st.caption(f"{game['Date']:%d %B %Y}, {game['Stadium']}. Manager: {game['Manager']}. "
           f"Opponent pre-match position: "
           f"{'n/a' if pd.isna(game['opp_position']) else int(game['opp_position'])}.")

card_rows = []
for section, cols in S.SECTIONS.items():
    for col in cols:
        if col in S.DUPLICATE_COLS:
            continue
        mu_val, opp_val = game[col], game[f"Opp {col}"]
        left, right = (mu_val, opp_val) if mu_home else (opp_val, mu_val)
        card_rows.append({"label": col, "kind": S.metric_kind(col), "left": left, "right": right})
card = pd.DataFrame(card_rows)
xg_for, xg_against = game["xG"], game["Opp xG"]
bc_for, bc_against = game["Big chances"], game["Opp Big chances"]
result_word = {"W": "won", "D": "drew", "L": "lost"}[game["Result"]]
chance_edge = ("created the better chances" if xg_for > xg_against + 0.3 else
               "had the worse chances" if xg_for < xg_against - 0.3 else
               "were level on chance quality")
fit = ("so the result matched the chances" if
       (game["Result"] == "W" and xg_for > xg_against) or
       (game["Result"] == "L" and xg_for < xg_against) or
       (game["Result"] == "D" and abs(xg_for - xg_against) <= 0.3)
       else "so the result went against the run of the chances")
ui.chart(C.match_card(card, home, away, home_colour, away_colour), f"match_card_gw{game['GW']}",
         f"Man Utd {result_word} {game['MU goals']}-{game['Opp goals']} and {chance_edge} "
         f"(xG {xg_for:.2f} vs {xg_against:.2f}, big chances {bc_for} vs {bc_against}), {fit}. "
         f"Possession {game['Possession %']:.0%}.",
         "Home team on the left, as on Sofascore. Bar length is each side's share of the combined "
         "value. Blank tracking values (GW33) show as n/a.")
