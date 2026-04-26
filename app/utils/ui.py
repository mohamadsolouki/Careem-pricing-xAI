from __future__ import annotations

import streamlit as st


THEME = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(13,148,136,0.12), transparent 24%),
            radial-gradient(circle at top right, rgba(242,159,5,0.12), transparent 22%),
            linear-gradient(180deg, #f7f2e8 0%, #fffdfa 44%, #eef7f4 100%);
        color: #1f2937;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #133b33 0%, #0f4c43 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    .xprice-hero {
        background: linear-gradient(135deg, rgba(19,59,51,0.94) 0%, rgba(13,148,136,0.88) 100%);
        color: #f8fafc;
        padding: 1.4rem 1.5rem;
        border-radius: 22px;
        margin-bottom: 1rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.14);
    }
    .xprice-hero h1 {
        margin: 0.2rem 0 0.6rem 0;
        font-size: 2.1rem;
        line-height: 1.1;
    }
    .xprice-hero p {
        margin: 0;
        max-width: 60rem;
        color: rgba(248,250,252,0.9);
    }
    .xprice-eyebrow {
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.76rem;
        color: #fde68a;
        font-weight: 700;
    }
    .xprice-card {
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(15,76,67,0.10);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.9rem;
    }
    .xprice-card h3 {
        margin: 0 0 0.35rem 0;
        font-size: 1rem;
    }
    .xprice-card p {
        margin: 0;
        color: #475569;
        font-size: 0.95rem;
    }
</style>
"""


def apply_theme():
    st.markdown(THEME, unsafe_allow_html=True)


def hero(title: str, eyebrow: str, body: str):
    st.markdown(
        f"""
        <div class="xprice-hero">
            <div class="xprice-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str):
    st.markdown(
        f"""
        <div class="xprice-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )