"""Man Utd 2025-26: Amorim vs Carrick. Entry point: `streamlit run app.py`."""

import streamlit as st

from src import ui

st.set_page_config(
    page_title="Amorim vs Carrick | Man Utd 2025-26",
    page_icon=":material/sports_soccer:",
    layout="wide",
)

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/trend.py", title="Trend explorer", icon=":material/show_chart:"),
    st.Page("views/head_to_head.py", title="Head to head", icon=":material/compare_arrows:"),
    st.Page("views/stats_lab.py", title="Stats lab", icon=":material/science:"),
    st.Page("views/luck_vs_skill.py", title="Luck vs skill", icon=":material/casino:"),
    st.Page("views/style_drivers.py", title="Style and drivers", icon=":material/scatter_plot:"),
    st.Page("views/match_explorer.py", title="Match explorer", icon=":material/table_view:"),
    st.Page("views/conclusion.py", title="Conclusion", icon=":material/flag:"),
    st.Page("views/methods.py", title="Methods and caveats", icon=":material/menu_book:"),
]

nav = st.navigation(pages)
ui.sidebar()
nav.run()
