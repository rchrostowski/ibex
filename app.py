"""
IBEX — Root entry point
This file stays in the repo root as app.py.
It simply redirects to the Home landing page.
"""
import streamlit as st

st.set_page_config(
    page_title="IBEX — Precision Supplements for D1 Athletes",
    page_icon="assets/ibex_logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Immediately switch to the Home page
st.switch_page("pages/00_Home.py")
