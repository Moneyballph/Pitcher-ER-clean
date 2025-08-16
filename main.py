
import streamlit as st
import math
from scipy.stats import poisson
import pandas as pd

# =========================================================
#                   GLOBAL STYLES & STATE
# =========================================================
st.set_page_config(page_title="Pitcher ER & K Simulator", layout="wide")

# Background + logo (same as your original)
st.markdown("""
    <style>
    .stApp {
        background-image: url("images/stadium_background.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div id="logo"><img src="images/logo.png" width="160"></div>', unsafe_allow_html=True)

# Session state for parlay legs
if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# =========================================================
#                       HELPERS
# =========================================================
def american_to_prob(odds: float) -> float:
    """Implied probability from American odds."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def american_to_decimal(odds: float) -> float:
    """Decimal odds from American odds."""
    if odds >= 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))

def decimal_to_american(dec: float) -> int:
    """American odds from decimal odds."""
    if dec <= 1.0:
        return 0
    if dec >= 2.0:
        return round((dec - 1.0) * 100)
    return -round(100.0 / (dec - 1.0))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ---------- Binomial utilities for Strikeouts ----------
def binom_pmf(n, k, p):
    if k < 0 or k > n:
        return 0.0
    return math.exp(
        math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)
        + k*math.log(max(p,1e-12)) + (n-k)*math.log(max(1-p,1e-12))
    )

def binom_cdf(n, k, p):
    k = int(k)
    return sum(binom_pmf(n, i, p) for i in range(0, k+1))

def expected_bf(expected_ip, pa_per_inning=4.3):
    return max(1, int(round(expected_ip * pa_per_inning)))

def estimate_pK(pitcher_K_rate, opp_K_rate_vs_hand, park_factor=1.0, ump_factor=1.0, recent_factor=1.0):
    base = 0.6*pitcher_K_rate + 0.4*opp_K_rate_vs_hand
    adj = base * park_factor * ump_factor * recent_factor
    return clamp(adj, 0.10, 0.45)

def k_over_under_probs(line_ks, n_bf, pK):
    # integer line L.0 => Under = P(K <= L-1), Over = P(K >= L)
    # half line L.5 => Under = P(K <= floor(L)), Over = P(K >= floor(L)+1)
    if abs(line_ks - round(line_ks)) < 1e-9:
        k_under = int(line_ks) - 1
        k_over  = int(line_ks)
    else:
        k_under = math.floor(line_ks)
        k_over  = k_under + 1
    p_under = binom_cdf(n_bf, k_under, pK)
    p_over = 1.0 - binom_cdf(n_bf, k_over-1, pK)
    return p_over, p_under

# =========================================================
#                   PARLAY BUILDER LOGIC
# =========================================================
def add_leg_to_parlay(market:str, desc:str, side:str, line:str, odds:float, true_prob:float):
    """Append a leg to session parlay list."""
    st.session_state.parlay_legs.append({
        "Market": market,            # "ER" or "K"
        "Description": desc,         # e.g., "Cole U2.5 ER" or "Cole O6.5 K"
        "Side": side,                # "Under"/"Over"
        "Line": line,                # "2.5", "6.5", etc.
        "Odds": float(odds),         # American
        "True Prob": float(true_prob) # 0-1
    })

def parlay_summary(legs):
    """Compute combined decimal odds, true probability, implied prob, and EV.
    Assumes independence across legs (NO correlation adjustment)."""
    if not legs:
        return None

    # Combined decimal odds (product of each leg decimal)
    dec = 1.0
    for leg in legs:
        dec *= american_to_decimal(leg["Odds"])
    amer = decimal_to_american(dec)

    # True parlay probability (independent joint probability)
    true_p = 1.0
    for leg in legs:
        true_p *= leg["True Prob"]

    # Implied parlay probability from combined odds
    imp_p = american_to_prob(amer)

    # Edge (percentage points)
    edge_pp = (true_p - imp_p) * 100.0

    # ROI per $1 stake (expected value)
    payout = dec - 1.0  # decimal odds - 1
    true_ev = (true_p * payout) - (1 - true_p) * 1.0
    true_ev_pct = true_ev * 100.0

    return {
        "Decimal Odds": dec,
        "American Odds": amer,
        "True %": true_p * 100.0,
        "Implied %": imp_p * 100.0,
        "Edge (pp)": edge_pp,
        "True EV %": true_ev_pct
    }

# Simple tiering (same style as your ER section: based on TRUE probability)
def tier_from_true_prob(p: float) -> str:
    if p >= 0.80: return "🟢 Elite"
    if p >= 0.70: return "🟡 Strong"
    if p >= 0.60: return "🟠 Moderate"
    return "🔴 Risky"

# =========================================================
#                        UI LAYOUT
# =========================================================
st.title("🎯 Pitcher Earned Runs & Strikeouts Simulator")
tabs = st.tabs(["Earned Runs (U2.5)", "Strikeouts (K)", "Parlay Builder"])

# =========================================================
#               TAB 1: EARNED RUNS (UNCHANGED)
# =========================================================
with tabs[0]:
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
        simulate_button = st.button("▶ Simulate Player", key="simulate_er")

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

        # EV % (relative)
        ev = round((true_prob - implied_prob) / implied_prob * 100, 2)

        # True EV % (real ROI)
        if under_odds < 0:
            payout = 100 / abs(under_odds)
        else:
            payout = under_odds / 100
        true_ev = round((true_prob * payout) - ((1 - true_prob) * 1), 4)
        true_ev_percent = round(true_ev * 100, 2)

        # Tier logic (UNCHANGED)
        if true_prob >= 0.80:
            tier = "🟢 Elite"
        elif true_prob >= 0.70:
            tier = "🟡 Strong"
        elif true_prob >= 0.60:
            tier = "🟠 Moderate"
        else:
            tier = "🔴 Risky"

        # WHIP Warning
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
        st.markdown(f"**True Expected Value (ROI per $1):** {true_ev_percent}%")
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
            "True EV %": [f"{true_ev_percent:.1f}%"],
            "Tier": [tier.replace("🟢 ", "").replace("🟡 ", "").replace("🟠 ", "").replace("🔴 ", "")]
        })
        st.dataframe(df, use_container_width=True)

        # ---------- Add to Parlay (ER) ----------
        if st.button("➕ Add to Parlay: Under 2.5 ER", key="add_er"):
            add_leg_to_parlay(
                market="ER",
                desc=f"{pitcher_name or 'Pitcher'} U2.5 ER",
                side="Under",
                line="2.5",
                odds=float(under_odds),
                true_prob=float(true_prob)
            )
            st.success("Added to Parlay.")

# =========================================================
#               TAB 2: STRIKEOUTS (K) NEW
# =========================================================
with tabs[1]:
    st.subheader("Strikeouts (K)")

    c1, c2, c3 = st.columns(3)
    with c1:
        k_pitcher = st.text_input("Pitcher Name (K)", key="k_name")
        pitcher_k_pct = st.number_input("Pitcher K% (decimal)", value=0.27, min_value=0.0, max_value=0.9, step=0.01, format="%.2f")
        expected_ip_k = st.number_input("Expected IP", value=5.8, min_value=0.0, max_value=9.0, step=0.1, format="%.1f")
    with c2:
        opp_k_vs_hand = st.number_input("Opponent K% vs Hand (decimal)", value=0.24, min_value=0.0, max_value=0.9, step=0.01, format="%.2f")
        k_line = st.number_input("K Line (e.g., 5.5)", value=5.5, step=0.5, format="%.1f")
        odds_over = st.text_input("Over Odds (American)", value="-115")
    with c3:
        odds_under = st.text_input("Under Odds (American)", value="-105")
        pa_per_inning = st.number_input("PA per Inning", value=4.3, min_value=3.0, max_value=6.0, step=0.1, format="%.1f")
        with st.expander("Adjustments (optional)"):
            park_factor  = st.number_input("Park Factor (K)", value=1.00, min_value=0.90, max_value=1.10, step=0.01, format="%.2f")
            ump_factor   = st.number_input("Ump Factor (K)",  value=1.00, min_value=0.90, max_value=1.10, step=0.01, format="%.2f")
            recent_factor= st.number_input("Recent Form Factor", value=1.00, min_value=0.90, max_value=1.10, step=0.01, format="%.2f")

    if st.button("▶ Calculate Strikeouts"):
        # Parse odds
        try:
            odds_over_f  = float(odds_over)
            odds_under_f = float(odds_under)
        except ValueError:
            st.error("Please enter valid American odds (e.g., -115 or +120).")
            st.stop()

        # Model
        pK = estimate_pK(pitcher_k_pct, opp_k_vs_hand, park_factor, ump_factor, recent_factor)
        n_bf = expected_bf(expected_ip_k, pa_per_inning)
        p_over, p_under = k_over_under_probs(k_line, n_bf, pK)

        imp_over  = american_to_prob(odds_over_f)
        imp_under = american_to_prob(odds_under_f)

        # Tiers based on TRUE probability (consistent with ER)
        over_tier  = tier_from_true_prob(p_over)
        under_tier = tier_from_true_prob(p_under)

        st.markdown("---")
        st.markdown(f"### {k_pitcher or 'Pitcher'} — K Line {k_line}")
        st.write(f"**Expected Batters Faced (BF):** {n_bf}")
        st.write(f"**Per-PA Strikeout Probability (pK):** {pK:.3f}")

        cL, cR = st.columns(2)
        with cL:
            st.markdown("#### Over")
            st.write(f"**True Over %:** {p_over*100:.2f}%")
            st.write(f"**Implied Over %:** {imp_over*100:.2f}%")
            st.write(f"**Tier:** {over_tier}")
            if st.button("➕ Add to Parlay: Over K", key="add_k_over"):
                add_leg_to_parlay(
                    market="K",
                    desc=f"{k_pitcher or 'Pitcher'} O{k_line} K",
                    side="Over",
                    line=f"{k_line}",
                    odds=float(odds_over_f),
                    true_prob=float(p_over)
                )
                st.success("Added to Parlay.")
        with cR:
            st.markdown("#### Under")
            st.write(f"**True Under %:** {p_under*100:.2f}%")
            st.write(f"**Implied Under %:** {imp_under*100:.2f}%")
            st.write(f"**Tier:** {under_tier}")
            if st.button("➕ Add to Parlay: Under K", key="add_k_under"):
                add_leg_to_parlay(
                    market="K",
                    desc=f"{k_pitcher or 'Pitcher'} U{k_line} K",
                    side="Under",
                    line=f"{k_line}",
                    odds=float(odds_under_f),
                    true_prob=float(p_under)
                )
                st.success("Added to Parlay.")

# =========================================================
#               TAB 3: PARLAY BUILDER
# =========================================================
with tabs[2]:
    st.subheader("🧩 Parlay Builder")
    st.caption("Add legs from the Earned Runs or Strikeouts tabs, then review the combined edge and EV.")

    # Controls
    colA, colB = st.columns([3,1])
    with colA:
        if st.session_state.parlay_legs:
            df_parlay = pd.DataFrame([
                {
                    "Market": leg["Market"],
                    "Description": leg["Description"],
                    "Side": leg["Side"],
                    "Line": leg["Line"],
                    "Odds": int(leg["Odds"]),
                    "True %": f"{leg['True Prob']*100:.2f}%"
                } for leg in st.session_state.parlay_legs
            ])
            st.dataframe(df_parlay, use_container_width=True, hide_index=True)
        else:
            st.info("No legs added yet. Use the buttons in the ER/K tabs to add legs.")

    with colB:
        if st.button("🧹 Clear Parlay"):
            st.session_state.parlay_legs = []
            st.success("Parlay cleared.")

    # Summary
    if st.session_state.parlay_legs:
        summary = parlay_summary(st.session_state.parlay_legs)
        if summary:
            st.markdown("---")
            st.markdown("### 📈 Parlay Summary")
            st.write(f"**Combined American Odds:** {summary['American Odds']}")
            st.write(f"**Combined Decimal Odds:** {summary['Decimal Odds']:.3f}")
            st.write(f"**True Parlay %:** {summary['True %']:.2f}%")
            st.write(f"**Implied Parlay %:** {summary['Implied %']:.2f}%")
            st.write(f"**Edge (pp):** {summary['Edge (pp)']:.2f}")
            st.write(f"**True EV % (ROI per $1):** {summary['True EV %']:.2f}%")

            # Optional export row for your tracker (tab-separated, clean columns)
            st.markdown("#### Copy-ready Tracker Row")
            legs_text = " + ".join([leg["Description"] for leg in st.session_state.parlay_legs])
            tracker_row = f"{legs_text}\t{summary['American Odds']}\t{summary['True %']:.2f}%\t{summary['Implied %']:.2f}%\t{summary['Edge (pp)']:.2f}\t{summary['True EV %']:.2f}%"
            st.code(tracker_row, language="text")


