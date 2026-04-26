from __future__ import annotations

import streamlit as st

try:
    import plotly.io as pio
    pio.templates.default = "plotly_white"
except Exception:
    pass

# ─── Design tokens ────────────────────────────────────────────────────────────
# Primary teal  #0d9488   teal-600
# Dark teal     #0f766e   teal-700
# Deeper teal   #134e4a   teal-900
# Amber accent  #f59e0b   amber-500
# App bg        #f8fafc   slate-50
# Surface       #ffffff
# Border        #e2e8f0   slate-200
# Text dark     #0f172a   slate-900
# Text mid      #475569   slate-600
# Text muted    #94a3b8   slate-400
# ─────────────────────────────────────────────────────────────────────────────

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Base ───────────────────────────────────── */
html, body, .stApp {
    background: #f8fafc !important;
    color: #0f172a !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
section.main {
    background: #f8fafc !important;
}
[data-testid="block-container"] {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

/* ── Top header bar ─────────────────────────── */
header[data-testid="stHeader"] {
    background: rgba(248,250,252,0.95) !important;
    backdrop-filter: blur(10px) !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

/* ── Sidebar ────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 20px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] * {
    color: #0f172a !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #0f766e !important;
    font-size: 0.76rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 700 !important;
    margin-top: 1.3rem !important;
    margin-bottom: 0.45rem !important;
    padding-bottom: 0.35rem !important;
    border-bottom: 2px solid #ccfbf1 !important;
}
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] p {
    color: #475569 !important;
    font-size: 0.88rem !important;
}

/* ── Headings ───────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3 {
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* ── Body text ──────────────────────────────── */
p, .stMarkdown p, li {
    color: #475569 !important;
}
strong, b, .stMarkdown strong {
    color: #0f172a !important;
}
a { color: #0d9488 !important; }
a:hover { color: #0f766e !important; text-decoration: underline; }

/* ── Metric containers ──────────────────────── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 1rem 1.3rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 6px 24px rgba(13,148,136,0.12) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    color: #0f172a !important;
    font-weight: 800 !important;
    font-size: 1.45rem !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

/* ── Buttons ────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.5rem 1.4rem !important;
    box-shadow: 0 2px 10px rgba(13,148,136,0.28) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0f766e 0%, #134e4a 100%) !important;
    box-shadow: 0 4px 18px rgba(13,148,136,0.42) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button p { color: #ffffff !important; }

/* ── Tabs ───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.1rem !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.15s ease !important;
    font-size: 0.9rem !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #0f766e !important;
    background: rgba(13,148,136,0.07) !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #0f766e !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.12) !important;
}
.stTabs [data-baseweb="tab"] p { color: inherit !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem !important; }

/* ── Form labels ────────────────────────────── */
label,
[data-testid="stWidgetLabel"] p,
.stSelectbox label, .stNumberInput label, .stTextInput label,
.stSlider label, .stDateInput label, .stTimeInput label,
.stRadio label, .stCheckbox label, .stToggle label,
.stMultiSelect label, .stTextArea label {
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
}

/* ── Select / text inputs ───────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
}
input, textarea, select {
    background: #ffffff !important;
    border-color: #e2e8f0 !important;
    color: #0f172a !important;
    border-radius: 10px !important;
}
input:focus, textarea:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,0.15) !important;
    outline: none !important;
}

/* ── Radio ──────────────────────────────────── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] span,
[data-testid="stRadio"] p {
    color: #374151 !important;
    font-weight: 500 !important;
}

/* ── Toggle ─────────────────────────────────── */
[data-testid="stToggle"] label,
[data-testid="stToggle"] p,
.stToggle label {
    color: #374151 !important;
    font-weight: 600 !important;
}
[data-testid="stToggle"] [role="switch"][aria-checked="true"] {
    background-color: #0d9488 !important;
}

/* ── Slider ─────────────────────────────────── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p,
[data-testid="stSlider"] span {
    color: #374151 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #0d9488 !important;
    border-color: #0d9488 !important;
}

/* ── Dataframes ─────────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    background: #ffffff !important;
}
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th {
    color: #0f172a !important;
}

/* ── Plotly charts ──────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    background: #ffffff !important;
}

/* ── Alert boxes ────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    background: #fffbeb !important;
    border: 1px solid #fcd34d !important;
    color: #92400e !important;
}
[data-testid="stAlert"] p { color: #92400e !important; }
div[data-baseweb="notification"][kind="info"] {
    background: #eff6ff !important;
    border-left: 4px solid #3b82f6 !important;
}
div[data-baseweb="notification"][kind="positive"] {
    background: #f0fdf4 !important;
    border-left: 4px solid #22c55e !important;
}

/* ── Caption / muted text ───────────────────── */
.stCaption, small,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
}

/* ── Divider ────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #e2e8f0 !important;
    margin: 1.5rem 0 !important;
}

/* ── Map iframes ────────────────────────────── */
[data-testid="stCustomComponentV1"] iframe {
    border-radius: 14px !important;
    border: 1px solid #e2e8f0 !important;
}

/* ── Scrollbar ──────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ══════════════════════════════════════════════
   XPrice custom components
   ══════════════════════════════════════════════ */

/* ── Hero banner ────────────────────────────── */
.xprice-hero {
    background: linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 55%, #fff7ed 100%);
    border: 1px solid #99f6e4;
    border-left: 5px solid #0d9488;
    border-radius: 20px;
    padding: 1.75rem 2.25rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 28px rgba(13,148,136,0.09);
    position: relative;
    overflow: hidden;
}
.xprice-hero::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(13,148,136,0.07), transparent 68%);
    pointer-events: none;
}
.xprice-eyebrow {
    display: inline-block;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-size: 0.7rem;
    color: #0d9488;
    font-weight: 700;
    background: rgba(13,148,136,0.1);
    padding: 0.22rem 0.75rem;
    border-radius: 99px;
    margin-bottom: 0.55rem;
}
.xprice-hero h1 {
    margin: 0 0 0.65rem 0 !important;
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    letter-spacing: -0.025em !important;
    line-height: 1.15 !important;
}
.xprice-hero p {
    margin: 0 !important;
    max-width: 66rem;
    color: #64748b !important;
    font-size: 0.97rem !important;
    line-height: 1.7 !important;
}

/* ── Card ───────────────────────────────────── */
.xprice-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.15rem 1.35rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.xprice-card:hover {
    box-shadow: 0 6px 28px rgba(13,148,136,0.1);
    transform: translateY(-1px);
}
.xprice-card h3 {
    margin: 0 0 0.45rem 0 !important;
    font-size: 0.97rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}
.xprice-card p {
    margin: 0 !important;
    color: #64748b !important;
    font-size: 0.93rem !important;
    line-height: 1.65 !important;
}

/* ── Section header ─────────────────────────── */
.xprice-section-header {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin: 1.3rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #f1f5f9;
}
.xprice-section-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #0d9488;
    flex-shrink: 0;
}
.xprice-section-header h3 {
    margin: 0 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    letter-spacing: -0.01em !important;
}

/* ── Fare result box ────────────────────────── */
.xprice-fare-box {
    background: linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 100%);
    border: 2px solid #0d9488;
    border-radius: 20px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(13,148,136,0.12);
}
.xprice-fare-box .fare-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #0f766e;
    margin-bottom: 0.3rem;
}
.xprice-fare-box .fare-value {
    font-size: 3.2rem;
    font-weight: 900;
    color: #0f172a;
    line-height: 1.05;
    letter-spacing: -0.04em;
}
.xprice-fare-box .fare-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.4rem;
    font-weight: 500;
}

/* ── Stat strip ─────────────────────────────── */
.xprice-stat-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin: 0.75rem 0;
}
.xprice-stat-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.5rem 0.9rem;
    min-width: 90px;
    flex: 1;
}
.xprice-stat-item .stat-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.15rem;
}
.xprice-stat-item .stat-value {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0f172a;
}

/* ── Tag pill ───────────────────────────────── */
.xprice-pill {
    display: inline-block;
    background: #f0fdfa;
    border: 1px solid #99f6e4;
    border-radius: 99px;
    padding: 0.22rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #0f766e;
    margin: 0.12rem;
}

/* ── Sidebar brand block ────────────────────── */
.xprice-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.6rem 0 0.9rem 0;
    margin-bottom: 0.4rem;
    border-bottom: 2px solid #f0fdfa;
}
.xprice-sidebar-brand .brand-icon {
    width: 34px; height: 34px;
    border-radius: 9px;
    background: linear-gradient(135deg, #0d9488, #0f766e);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-weight: 900;
    font-size: 1rem;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(13,148,136,0.35);
}
.xprice-sidebar-brand .brand-text .brand-name {
    font-weight: 800;
    font-size: 1.05rem;
    color: #0f172a;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.xprice-sidebar-brand .brand-text .brand-sub {
    font-size: 0.68rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── What-if result banner ──────────────────── */
.xprice-whatif-banner {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 4px solid #0d9488;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
.xprice-whatif-banner .wi-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 0.2rem;
}
.xprice-whatif-banner .wi-value {
    font-size: 2.4rem;
    font-weight: 900;
    color: #0f172a;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.xprice-whatif-banner .wi-ref {
    font-size: 0.83rem;
    color: #64748b;
    margin-top: 0.25rem;
}
</style>
"""


def apply_theme() -> None:
    st.markdown(THEME, unsafe_allow_html=True)


def hero(title: str, eyebrow: str, body: str) -> None:
    st.markdown(
        f"""<div class="xprice-hero">
            <div class="xprice-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{body}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    st.markdown(
        f"""<div class="xprice-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    st.markdown(
        f"""<div class="xprice-section-header">
            <div class="xprice-section-dot"></div>
            <h3>{title}</h3>
        </div>""",
        unsafe_allow_html=True,
    )


def fare_result(label: str, value_aed: float, sub: str = "") -> None:
    sub_html = f'<div class="fare-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""<div class="xprice-fare-box">
            <div class="fare-label">{label}</div>
            <div class="fare-value">AED {value_aed:,.2f}</div>
            {sub_html}
        </div>""",
        unsafe_allow_html=True,
    )


def whatif_result(predicted: float, reference: float) -> None:
    delta = predicted - reference
    sign = "+" if delta >= 0 else ""
    delta_color = "#059669" if delta <= 0 else "#dc2626"
    st.markdown(
        f"""<div class="xprice-whatif-banner">
            <div class="wi-label">What-if predicted fare</div>
            <div class="wi-value">AED {predicted:,.2f}</div>
            <div class="wi-ref" style="color:{delta_color};font-weight:700;">
                {sign}AED {delta:,.2f} vs engine reference (AED {reference:,.2f})
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.markdown(
        """<div class="xprice-sidebar-brand">
            <div class="brand-icon">X</div>
            <div class="brand-text">
                <div class="brand-name">XPrice</div>
                <div class="brand-sub">Explainable Pricing</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )