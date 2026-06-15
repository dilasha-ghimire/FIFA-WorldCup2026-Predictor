"""
app.py
────────────────────────────────────────────────────────────
Streamlit dashboard for the WC 2026 Match Predictor.

Shows:
  - Live accuracy metrics (3 levels)
  - Match predictions vs actual results (all 72 matches)
  - Accuracy chart (correct vs wrong vs upcoming)
  - Monte Carlo tournament win probabilities (top 16)

Auto-refreshes every 5 minutes to pick up new results.

Run:    streamlit run app.py
Deploy: share.streamlit.io
"""

import json
import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from fetch_results import fetch_results, update_predictions, calculate_accuracy

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title            = "WC 2026 Predictor",
    page_icon             = "🏆",
    layout                = "wide",
    initial_sidebar_state = "collapsed",
)

# ─────────────────────────────────────────────────────────────
# THEME STATE
# ─────────────────────────────────────────────────────────────

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

D = st.session_state.dark_mode

# ─────────────────────────────────────────────────────────────
# THEME TOKENS
# ─────────────────────────────────────────────────────────────

if D:
    BG         = "#0a0f1e"
    SURFACE    = "#141929"
    BORDER     = "#1e2d4a"
    TEXT       = "#e8eaf0"
    MUTED      = "#6b89b8"
    LABEL      = "#4a6fa5"
    ACCENT     = "#4a9eff"
    GREEN      = "#22c55e"
    RED        = "#ef4444"
    AMBER      = "#f59e0b"
    BTN_BG     = "#1e2d4a"
    BTN_TEXT   = "#e8eaf0"
    BTN_BORDER = "#4a6fa5"
    BAR_LOW    = "#1e3a5f"
    BAR_HIGH   = "#4a9eff"
    METRIC_BG  = "#141929"
else:
    BG         = "#f2ede4"
    SURFACE    = "#faf7f2"
    BORDER     = "#ddd5c2"
    TEXT       = "#1c1917"
    MUTED      = "#78716c"
    LABEL      = "#a8956e"
    ACCENT     = "#92622a"
    GREEN      = "#15803d"
    RED        = "#b91c1c"
    AMBER      = "#b45309"
    BTN_BG     = "#faf7f2"
    BTN_TEXT   = "#1c1917"
    BTN_BORDER = "#ddd5c2"
    BAR_LOW    = "#e8d9c0"
    BAR_HIGH   = "#92622a"
    METRIC_BG  = "#faf7f2"

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .stApp, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
    }}

    .stApp {{
        background-color: {BG} !important;
        color: {TEXT} !important;
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}

    /* Remove default streamlit padding top */
    .block-container {{ padding-top: 2rem !important; }}

    /* Metric cards — fully custom */
    div[data-testid="metric-container"] {{
        background: {METRIC_BG} !important;
        border: 1.5px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 20px 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }}
    div[data-testid="metric-container"] label,
    div[data-testid="stMetricLabel"] p {{
        color: {MUTED} !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }}
    div[data-testid="stMetricValue"] > div {{
        color: {TEXT} !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }}

    /* Section headers */
    .section-header {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {LABEL};
        margin: 36px 0 14px 0;
        padding-bottom: 10px;
        border-bottom: 1.5px solid {BORDER};
    }}

    /* Title */
    .main-title {{
        font-size: 32px;
        font-weight: 800;
        color: {TEXT};
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 6px;
    }}
    .main-subtitle {{
        font-size: 14px;
        color: {MUTED};
        font-weight: 400;
        margin-bottom: 0;
    }}
    .last-updated {{
        font-size: 11px;
        color: {LABEL};
        text-align: right;
        margin-top: 8px;
    }}

    /* Toggle button */
    div[data-testid="stButton"] > button {{
        background: {BTN_BG} !important;
        color: {BTN_TEXT} !important;
        border: 1.5px solid {BTN_BORDER} !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        transition: all 0.2s !important;
    }}
    div[data-testid="stButton"] > button:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
    }}

    /* Selectbox */
    div[data-testid="stSelectbox"] > div > div {{
        background: {METRIC_BG} !important;
        border: 1.5px solid {BORDER} !important;
        border-radius: 8px !important;
        color: {TEXT} !important;
    }}

    /* Radio — font color only, no highlight */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] p {{
        color: {TEXT} !important;
        font-size: 13px !important;
    }}
    div[data-testid="stRadio"] input[type="radio"]:checked ~ div p {{
        color: {ACCENT} !important;
        font-weight: 600 !important;
    }}

    /* Dataframe */
    div[data-testid="stDataFrame"] {{
        border: 1.5px solid {BORDER} !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }}

    /* Caption */
    .stCaption p {{
        color: {MUTED} !important;
        font-size: 12px !important;
    }}

    /* Divider */
    hr {{ border-color: {BORDER} !important; opacity: 0.5 !important; }}


    /* Spinner */
    .stSpinner > div {{ border-top-color: {ACCENT} !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    with open("predictions/wc2026_predictions.json") as f:
        predictions = json.load(f)
    results     = fetch_results()
    predictions = update_predictions(predictions, results)
    accuracy    = calculate_accuracy(predictions)
    with open("predictions/wc2026_simulation.json") as f:
        simulation = json.load(f)
    return predictions, accuracy, simulation

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

title_col, spacer, toggle_col = st.columns([4, 2, 1])

with title_col:
    st.markdown('<div class="main-title">FIFA World Cup 2026 Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Elo-based match predictions tracked against live results in real time</div>',
        unsafe_allow_html=True,
    )

with toggle_col:
    st.markdown("<br>", unsafe_allow_html=True)
    label = "☀️ Light" if D else "🌙 Dark"
    if st.button(label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Load data
with st.spinner("Fetching latest results..."):
    predictions, accuracy, simulation = load_data()

with toggle_col:
    st.markdown(
        f'<div class="last-updated">{accuracy["last_updated"]}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# ACCURACY METRICS
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Prediction Accuracy</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div style="background:{METRIC_BG};border:1.5px solid {BORDER};border-radius:10px;padding:14px 12px;">
    <div style="color:{MUTED};font-size:14px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Matches</div>
    <div style="color:{TEXT};font-size:32px;font-weight:800;">{accuracy['completed']}/{accuracy['total_matches']}</div>
    <div style="color:{MUTED};font-size:12px;margin-top:4px;">completed</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div style="background:{METRIC_BG};border:1.5px solid {BORDER};border-radius:10px;padding:14px 12px;">
    <div style="color:{MUTED};font-size:14px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Exact Accuracy</div>
    <div style="color:{TEXT};font-size:32px;font-weight:800;">{accuracy['exact_accuracy']}%</div>
    <div style="color:{MUTED};font-size:12px;margin-top:4px;">matched exactly</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div style="background:{METRIC_BG};border:1.5px solid {BORDER};border-radius:10px;padding:14px 12px;">
    <div style="color:{MUTED};font-size:14px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Team Held</div>
    <div style="color:{TEXT};font-size:32px;font-weight:800;">{accuracy['no_upset_accuracy']}%</div>
    <div style="color:{MUTED};font-size:12px;margin-top:4px;">won or drew</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div style="background:{METRIC_BG};border:1.5px solid {BORDER};border-radius:10px;padding:14px 12px;">
    <div style="color:{MUTED};font-size:14px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Genuinely Wrong</div>
    <div style="color:{TEXT};font-size:32px;font-weight:800;">{accuracy['genuinely_wrong_pct']}%</div>
    <div style="color:{MUTED};font-size:12px;margin-top:4px;">other team won</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div style="background:{METRIC_BG};border:1.5px solid {BORDER};border-radius:10px;padding:14px 12px;">
    <div style="color:{MUTED};font-size:14px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Upcoming</div>
    <div style="color:{TEXT};font-size:32px;font-weight:800;">{accuracy['upcoming']}</div>
    <div style="color:{MUTED};font-size:12px;margin-top:4px;">remaining</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────

left, right = st.columns([1, 1])

# Donut
with left:
    st.markdown('<div class="section-header">Results Breakdown</div>', unsafe_allow_html=True)
    correct  = accuracy["exact_correct"]
    wrong    = accuracy["genuinely_wrong"]
    draws    = accuracy["completed"] - correct - wrong
    upcoming = accuracy["upcoming"]

    fig_donut = go.Figure(go.Pie(
        labels=["Correct", "Draw (not wrong)", "Wrong", "Upcoming"],
        values=[correct, draws, wrong, upcoming],
        hole=0.62,
        marker=dict(colors=[GREEN, AMBER, RED, BORDER]),
        textinfo="label+value",
        textfont=dict(color=TEXT, size=12, family="Inter"),
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=40, b=10, l=10, r=10),
        height=380,
        font=dict(family="Inter", color=TEXT),
        annotations=[dict(
            text=f"<b>{accuracy['exact_accuracy']}%</b><br><span style='font-size:11px'>exact</span>",
            x=0.5, y=0.5,
            font=dict(size=20, color=TEXT, family="Inter"),
            showarrow=False,
        )]
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# Bar chart
with right:
    st.markdown('<div class="section-header">Tournament Win Probability</div>', unsafe_allow_html=True)
    top16 = simulation["results"][:16]
    teams = [r["team"] for r in top16]
    probs = [r["probability"] for r in top16]

    fig_bar = go.Figure(go.Bar(
        x=probs,
        y=teams,
        orientation="h",
        marker=dict(
            color=probs,
            colorscale=[[0, BAR_LOW], [1, BAR_HIGH]],
            showscale=False,
        ),
        text=[f"{p}%" for p in probs],
        textposition="outside",
        textfont=dict(color=MUTED, size=11, family="Inter"),
        hovertemplate="%{y}: %{x}%<extra></extra>",
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, max(probs) * 1.25]),
        yaxis=dict(
            autorange="reversed",
            color=MUTED,
            tickfont=dict(size=12, family="Inter", color=TEXT),
            ticklabelposition="outside left",
        ),
        margin=dict(t=10, b=10, l=130, r=70),
        height=430,
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(f"Based on {simulation['n_simulations']:,} Monte Carlo simulations")

# ─────────────────────────────────────────────────────────────
# MATCH TABLE
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">All Predictions</div>', unsafe_allow_html=True)

f1, f2 = st.columns([1, 3])
with f1:
    groups = ["All groups"] + sorted(set(p["group"] for p in predictions))
    selected_group = st.selectbox("Group", groups, label_visibility="collapsed")
with f2:
    status_filter = st.radio(
        "Status", ["All", "Completed", "Upcoming"],
        horizontal=True, label_visibility="collapsed"
    )

filtered = predictions
if selected_group != "All groups":
    filtered = [p for p in filtered if p["group"] == selected_group]
if status_filter == "Completed":
    filtered = [p for p in filtered if p["status"] == "completed"]
elif status_filter == "Upcoming":
    filtered = [p for p in filtered if p["status"] == "upcoming"]

rows = []
for p in filtered:
    if p["status"] == "completed":
        score = f"{p['actual_score_team1']} – {p['actual_score_team2']}"
        if p["correct"]:
            result = "✅ Correct"
        elif p["actual_outcome"] == "draw" and p["predicted_outcome"] != "draw":
            result = "〰️ Draw"
        else:
            result = "❌ Wrong"
    else:
        score  = "  VS"
        result = "⏳ Upcoming"

    rows.append({
        "Group":  p["group"],
        "ID":   p["match_id"],
        "Date": p["date"],
        "Team 1": p["team1"],
        "Score":  score,
        "Team 2": p["team2"],
        "Predicted": p["predicted_winner"],
        "W%": f"{p['prob_team1_win']}%",
        "D%": f"{p['prob_draw']}%",
        "L%": f"{p['prob_team2_win']}%",
        "Result": result,
    })

df = pd.DataFrame(rows)
df.insert(0, "Match No.", range(1, len(df) + 1))
df = df.set_index("Match No.")

def color_result(val):
    v = str(val)
    if "Correct"  in v: return f"color: {GREEN}; font-weight: 600"
    if "Wrong"    in v: return f"color: {RED};   font-weight: 600"
    if "Draw"     in v: return f"color: {AMBER}; font-weight: 600"
    return f"color: {MUTED}"

styled = (
    df.style
    .map(color_result, subset=["Result"])
    .set_properties(**{"background-color": METRIC_BG, "color": TEXT, "border-color": BORDER})
    .set_table_styles([{"selector": "th", "props": [
        ("background-color", BG),
        ("color", LABEL),
        ("font-size", "11px"),
        ("font-weight", "700"),
        ("letter-spacing", "0.08em"),
        ("text-transform", "uppercase"),
        ("border-bottom", f"1.5px solid {BORDER}"),
        ("font-family", "Inter, sans-serif"),
    ]}])
)

row_height = 35
header_height = 38
dynamic_height = min(header_height + row_height * len(filtered), 600)
st.dataframe(styled, use_container_width=True, height=dynamic_height)
st.caption(f"Showing {len(filtered)} of {len(predictions)} matches · Results update within 24hrs of each match")

# ─────────────────────────────────────────────────────────────
# FOOTER + AUTO REFRESH
# ─────────────────────────────────────────────────────────────

time.sleep(300)
st.rerun()