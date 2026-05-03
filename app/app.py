from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for path in (APP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

st.set_page_config(page_title="XPrice", layout="wide", initial_sidebar_state="expanded")

pg = st.navigation([
    st.Page("pages/Overview.py", title="Overview"),
    st.Page("pages/1_ride_simulator.py", title="Ride Simulator"),
    st.Page("pages/2_operations_xai.py", title="Operations XAI"),
    st.Page("pages/3_feature_explorer.py", title="Feature Explorer"),
])
pg.run()
