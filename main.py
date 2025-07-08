
import streamlit as st
import math
from scipy.stats import poisson
import pandas as pd

# ----------------------
# Set page config
# ----------------------
st.set_page_config(page_title="Pitcher ER Simulator", layout="wide")

# ----------------------
# Custom Background & Logo
# ----------------------
st.markdown("""
    <style>
    .stApp {
        background-image: url("images/background.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    #logo {
        display: flex;
        justify-content: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div id="logo"><img src="images/logo.png" width="160"></div>', unsafe_allow_html=True)

st.title("🎯 Pitcher Earned Runs Simulator")

# ----------------------
# User Inputs
# ----------------------
st.subheader("Input Stats")

col1, col2 = st.columns(2)

with col1:
    pitcher_name = st.text_input("Pitcher Name")
    era = st.number_input("ERA", value=3.50, step=0.01)
    total_ip = st.number_input("Total Innings Pitched", value=90.0, step=0.1)
    games_started = st.number_input("Games Started", value=15, step=1)
    last_3_ip = st.text_input("Last 3 Game IP (comma-separated, e.g. 5.2,6.1,5.0)")
    xera = st.number_input("xERA (optional, overrides ERA)", value=0.0, step=0.01)
    whip = st.number_input("WHIP (optional)", value=0.0, step=0.01)

with col2:
    opponent_ops = st.number_input("Opponent OPS", value=0.670, step=0.001)
    league_avg_ops = st.number_input("League Average OPS", value=0.715, step=0.001)
    ballpark = st.selectbox("Ballpark Factor", ["Neutral", "Pitcher-Friendly", "Hitter-Friendly"])
    under_odds = st.number_input("Sportsbook Odds (U2.5 ER)", value=-115)
    simulate_button = st.button("▶ Simulate Player")

# ----------------------
# Run Simulation
# ----------------------

if simulate_button:

    # Calculate baseline and trend IP
    try:
        ip_values = [float(i.strip()) for i in last_3_ip.split(",") if i.strip() != ""]
        trend_ip = sum(ip_values) / len(ip_values)
    except:
        st.error("⚠️ Please enter 3 valid IP values separated by commas (e.g. 5.2,6.1,5.0)")
        st.stop()

    base_ip = total_ip / games_started

    # Adjust for park factor
    if ballpark == "Pitcher-Friendly":
        park_adj = 0.2
    elif ballpark == "Hitter-Friendly":
        park_adj = -0.2
    else:
        park_adj = 0.0

    expected_ip = round(((base_ip + trend_ip) / 2) + park_adj, 2)

    # Use xERA if available
    used_era = xera if xera > 0 else era
    adjusted_era = round(used_era * (opponent_ops / league_avg_ops), 3)
    lambda_er = round(adjusted_era * (expected_ip / 9), 3)

    # Poisson probability for ER ≤ 2
    p0 = poisson.pmf(0, lambda_er)
    p1 = poisson.pmf(1, lambda_er)
    p2 = poisson.pmf(2, lambda_er)
    true_prob = round(p0 + p1 + p2, 4)

    # Implied probability
    if under_odds < 0:
        implied_prob = round(abs(under_odds) / (abs(under_odds) + 100), 4)
    else:
        implied_prob = round(100 / (under_odds + 100), 4)

    ev = round((true_prob - implied_prob) / implied_prob * 100, 2)

    # Difficulty Tier
    if true_prob >= 0.80:
        tier = "🟢 Elite"
    elif true_prob >= 0.70:
        tier = "🟡 Strong"
    elif true_prob >= 0.60:
        tier = "🟠 Moderate"
    else:
        tier = "🔴 Risky"

    # Warning logic for WHIP
    warning_msg = ""
    if whip > 1.45 and era < 3.20 and xera == 0:
        warning_msg = "⚠️ ERA may be misleading due to high WHIP. Consider using xERA or reducing confidence."

    # Display Results
    st.subheader("📊 Simulation Results")
    st.markdown(f"**Expected IP:** {expected_ip} innings")
    st.markdown(f"**Adjusted ERA vs Opponent:** {adjusted_era}")
    st.markdown(f"**Expected ER (λ):** {lambda_er}")
    st.markdown(f"**True Probability of Under 2.5 ER:** {true_prob*100}%")
    st.markdown(f"**Implied Probability (from Odds):** {implied_prob*100}%")
    st.markdown(f"**Expected Value (EV%):** {ev}%")
    st.markdown(f"**Difficulty Tier:** {tier}")
    if warning_msg:
        st.warning(warning_msg)

    # Player Result Board
    st.subheader("🧾 Player Board")
    df = pd.DataFrame({
        "Pitcher": [pitcher_name],
        "True Probability": [f"{true_prob*100:.1f}%"],
        "Implied Probability": [f"{implied_prob*100:.1f}%"],
        "EV %": [f"{ev:.1f}%"],
        "Tier": [tier.replace("🟢 ", "").replace("🟡 ", "").replace("🟠 ", "").replace("🔴 ", "")]
    })
    st.dataframe(df, use_container_width=True)
