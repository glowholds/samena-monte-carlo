import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import base64
from io import BytesIO



# This should be one of the first things in your app
st.set_page_config(page_title="Your App Title", layout="wide")  # Optional

# Hide all Streamlit branding
hide_all = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
.stToolbar {visibility: hidden;}
</style>
"""
st.markdown(hide_all, unsafe_allow_html=True)


# ====================================
# CENTRALIZED DEFAULT VALUES
# ====================================
DEFAULT_VALUES = {
    # Financial Baseline
    "starting_reserves": 228299,  # Current cash reserves (FY2025)
    "monthly_expenses": 309343,  # Monthly operating expenses (FY2025: $5,193,534 / 12)
    "expense_growth_rate": 3.0,  # Annual expense growth rate (%)

    # Membership and Dues
    "initial_members": 1543,  # Number of member households (estimated from dues revenue)
    "monthly_dues": 114,  # Monthly dues per household (FY2025: $2,017,233 / 1543 / 12)
    "assessment_amount": 800,  # Special assessment amount

    # Member Behavior Predictions
    "assessment_payment_mean": 80,  # % who will pay assessment
    "assessment_payment_std": 10,  # Uncertainty range (±%)
    "retention_mean": 80,  # % who will stay
    "retention_std": 10,  # Uncertainty range (±%)
    "dues_increase_percent": 30,  # Dues increase %
    "dues_increase_std": 10,  # Implementation variance (±%)
    "new_members_per_month": 2,  # New members/month
    "new_members_std": 2,  # Variability

    # Assessment Payment Plan
    "payment_plan_months": 6,  # Number of months to spread assessment
    "payment_plan_adoption": 40,  # % who choose payment plan
    "payment_plan_uncertainty": 10,  # Uncertainty in adoption rate

    # Program Revenues and Expenses
    "youth_programs_monthly_revenue": 91345,  # Monthly youth programs revenue ($1,096,144 / 12)
    "youth_programs_monthly_expenses": 56964,  # Monthly youth programs expenses ($683,568 / 12)
    "aquatics_monthly_revenue": 110534,  # Monthly aquatics revenue ($1,326,415 / 12)
    "aquatics_monthly_expenses": 47102,  # Monthly aquatics expenses ($565,225 / 12)
    "fitness_monthly_revenue": 7102,  # Monthly fitness revenue ($85,233 / 12)
    "fitness_monthly_expenses": 15086,  # Monthly fitness expenses ($181,038 / 12)
    "rental_monthly_revenue": 8808,  # Monthly rental revenue ($105,695 / 12)
    "retail_monthly_revenue": 2951,  # Monthly retail revenue ($35,416 / 12)
    "retail_monthly_expenses": 2601,  # Monthly retail expenses ($31,215 / 12)
    "special_events_monthly_revenue": 2572,  # Monthly special events revenue ($30,873 / 12)
    "special_events_monthly_expenses": 1699,  # Monthly special events expenses ($20,391 / 12)
    "program_growth_rate": 2.0,  # Annual program growth rate (%)
    "program_uncertainty": 15.0,  # Program revenue/expense uncertainty (%)

    # Lump Sum Gift
    "lump_sum_gift_amount": 0,  # One-time gift amount
    "lump_sum_gift_uncertainty": 20.0,  # Gift uncertainty (%)

    # Simulation Settings
    "months": 24,  # Months to project
    "n_simulations": 10000,  # Simulation runs
    "danger_threshold": 250000,  # Minimum safe reserve level

    # Display Settings
    "current_cash": 228299,  # Cash from FY2025 Statement of Financial Position
    "current_savings": 0,  # No separate savings shown in FY2025
    "annual_expenses": 4832058,  # Total annual expenses FY2025
    "annual_dues_revenue": 2017233,  # Annual dues revenue FY2025
}

# Set page config
st.set_page_config(
    page_title="Samena Club Financial Projections",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure wider sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 400px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2a5298;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 5px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1e3c72;
        transform: translateY(-2px);
    }
    .insight-box {
        background: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #2a5298;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .help-text {
        font-size: 0.85rem;
        color: #666;
        font-style: italic;
        margin-top: 0.25rem;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def create_logo_placeholder():
    """Create a placeholder logo for Samena Club"""
    fig, ax = plt.subplots(figsize=(3, 1))
    ax.text(0.5, 0.5, 'SAMENA CLUB', fontsize=20, weight='bold',
            ha='center', va='center', color='#2a5298')
    ax.axis('off')

    # Save to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    plt.close()

    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_str}"


def format_currency(value):
    """Format number as currency with appropriate styling"""
    if value < 0:
        return f"<span style='color: red;'>-${abs(value):,.0f}</span>"
    return f"${value:,.0f}"


def load_samena_logo():
    try:
        with open("samena.svg", "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/svg+xml;base64,{data}"
    except FileNotFoundError:
        fig, ax = plt.subplots(figsize=(3, 1))
        ax.text(0.5, 0.5, 'SAMENA CLUB', fontsize=20, weight='bold',
                ha='center', va='center', color='#2a5298')
        ax.axis('off')
        buf = BytesIO();
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0);
        plt.close()
        img_str = base64.b64encode(buf.read()).decode()
        return f"data:image/png;base64,{img_str}"


def calculate_risk_score(ending_reserves, danger_threshold):
    """Calculate a risk score based on ending reserves"""
    below_threshold = sum(1 for r in ending_reserves if r < danger_threshold)
    risk_percentage = (below_threshold / len(ending_reserves)) * 100

    if risk_percentage < 5:
        return "Low Risk", "🟢"
    elif risk_percentage < 20:
        return "Moderate Risk", "🟡"
    else:
        return "High Risk", "🔴"


def find_optimal_dues_increase(base_members=1543, base_dues=114, elasticity=0.6):
    """
    Find the optimal dues increase based on elasticity
    Returns a DataFrame with revenue projections for different increase levels
    """
    results = []

    for dues_increase in range(0, 101, 5):  # 0% to 100% in 5% steps
        # Estimate retention based on elasticity
        # Elasticity means: X% price increase leads to (X * elasticity)% member loss
        retention_rate = max(0.5, 1 - (dues_increase * elasticity / 100))

        new_dues = base_dues * (1 + dues_increase / 100)
        retained_members = int(base_members * retention_rate)
        monthly_revenue = retained_members * new_dues

        results.append({
            'Dues Increase %': dues_increase,
            'Retention %': retention_rate * 100,
            'Members Retained': retained_members,
            'New Monthly Dues': f"${new_dues:.0f}",
            'Monthly Revenue': f"${monthly_revenue:,.0f}",
            'Revenue vs Current': f"{((monthly_revenue / (base_members * base_dues)) - 1) * 100:+.1f}%"
        })

    return pd.DataFrame(results)


def run_simulation(
        starting_reserves,
        monthly_expenses,
        initial_members,
        assessment_amount,
        monthly_dues,
        assessment_payment_mean,
        assessment_payment_std,
        retention_mean,
        retention_std,
        dues_increase_mean,
        dues_increase_std,
        new_members_per_month,
        new_members_std,
        months,
        n_simulations,
        expense_growth_rate=0.0,
        program_revenues=None,
        program_expenses=None,
        program_growth_rate=0.0,
        program_uncertainty=0.0,
        lump_sum_gift_amount=0,
        lump_sum_gift_uncertainty=0.0,
        payment_plan_months=0,
        payment_plan_adoption=0.0,
        payment_plan_uncertainty=0.0
):
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random variables with proper bounds
    assessment_payment_rate = np.clip(
        np.random.normal(assessment_payment_mean, assessment_payment_std, n_simulations),
        0.4, 1.0
    )
    membership_retention_rate = np.clip(
        np.random.normal(retention_mean, retention_std, n_simulations),
        0.5, 1.0
    )
    dues_increase_factor = np.clip(
        np.random.normal(dues_increase_mean, dues_increase_std, n_simulations),
        1.0, 2.0
    )

    # Payment plan adoption rate
    if payment_plan_months > 0:
        payment_plan_rate = np.clip(
            np.random.normal(payment_plan_adoption, payment_plan_uncertainty, n_simulations),
            0.0, 1.0
        )
    else:
        payment_plan_rate = np.zeros(n_simulations)

    # New members with variability
    new_members = np.maximum(
        np.random.normal(new_members_per_month, new_members_std, size=(n_simulations, months)),
        0
    ).astype(int)

    # Program revenue variability factors
    program_variability = np.clip(
        np.random.normal(1.0, program_uncertainty, n_simulations),
        0.5, 1.5
    )

    # Lump sum gift variability (with possibility of not receiving it)
    if lump_sum_gift_amount > 0:
        # Generate gift amounts with uncertainty
        gift_amounts = np.random.normal(lump_sum_gift_amount, lump_sum_gift_amount * lump_sum_gift_uncertainty,
                                        n_simulations)
        # Add probability of not receiving gift at all (higher uncertainty = higher chance of no gift)
        gift_probability = np.random.random(n_simulations) > (
                lump_sum_gift_uncertainty * 0.5)  # Max 50% chance of no gift
        gift_amounts = gift_amounts * gift_probability
        gift_amounts = np.maximum(gift_amounts, 0)  # No negative gifts
    else:
        gift_amounts = np.zeros(n_simulations)

    ending_reserves = []
    reserve_trajectories = []
    member_trajectories = []
    program_revenue_trajectories = []

    for i in range(n_simulations):
        reserves = starting_reserves
        paying_members = int(initial_members * membership_retention_rate[i])

        # Calculate assessment revenue
        total_assessment = assessment_amount * (initial_members * assessment_payment_rate[i])

        # Split between immediate and payment plan
        if payment_plan_months > 0:
            members_on_plan = payment_plan_rate[i]
            immediate_assessment = total_assessment * (1 - members_on_plan)
            monthly_assessment = (total_assessment * members_on_plan) / payment_plan_months
        else:
            immediate_assessment = total_assessment
            monthly_assessment = 0

        # Add immediate assessment revenue
        reserves += immediate_assessment

        # One-time lump sum gift
        reserves += gift_amounts[i]

        # Adjusted monthly dues
        monthly_dues_adjusted = monthly_dues * dues_increase_factor[i]

        # Track monthly data
        monthly_reserves = [reserves]
        monthly_members = [paying_members]
        monthly_program_revenue = []
        current_expenses = monthly_expenses

        for month in range(months):
            # Add new members
            new_members_count = new_members[i][month]
            paying_members += new_members_count

            # Apply expense growth
            current_expenses = monthly_expenses * (1 + expense_growth_rate) ** (month / 12)

            # Calculate monthly financials
            monthly_income = paying_members * monthly_dues_adjusted

            # Add monthly assessment payment if within payment plan period
            if month < payment_plan_months and payment_plan_months > 0:
                monthly_income += monthly_assessment

            # Add program revenues and expenses with uncertainty
            if program_revenues and program_expenses:
                program_growth_factor = (1 + program_growth_rate) ** (month / 12)
                monthly_program_net = 0
                for prog_rev, prog_exp in zip(program_revenues, program_expenses):
                    # Apply both growth and uncertainty
                    adjusted_rev = prog_rev * program_growth_factor * program_variability[i]
                    adjusted_exp = prog_exp * program_growth_factor * program_variability[i]
                    monthly_program_net += (adjusted_rev - adjusted_exp)
                monthly_income += monthly_program_net
                monthly_program_revenue.append(monthly_program_net)

            reserves += monthly_income - current_expenses

            # Track data
            monthly_reserves.append(reserves)
            monthly_members.append(paying_members)

        ending_reserves.append(reserves)
        reserve_trajectories.append(monthly_reserves)
        member_trajectories.append(monthly_members)
        program_revenue_trajectories.append(monthly_program_revenue)

    return ending_reserves, reserve_trajectories, member_trajectories


def run_simulation_gradual(
        starting_reserves,
        monthly_expenses,
        initial_members,
        assessment_amount,
        monthly_dues,
        assessment_payment_mean,
        assessment_payment_std,
        retention_mean,
        retention_std,
        dues_increase_mean,
        dues_increase_std,
        new_members_per_month,
        new_members_std,
        months,
        n_simulations,
        expense_growth_rate=0.0,
        program_revenues=None,
        program_expenses=None,
        program_growth_rate=0.0,
        program_uncertainty=0.0,
        lump_sum_gift_amount=0,
        lump_sum_gift_uncertainty=0.0,
        payment_plan_months=0,
        payment_plan_adoption=0.0,
        payment_plan_uncertainty=0.0
):
    """
    Gradual transition model where:
    - Assessment is paid immediately or over payment plan
    - Members who don't pay assessment continue paying old dues until their renewal
    - Members who pay assessment pay increased dues at their renewal
    - Member losses happen gradually over 12 months
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random variables with proper bounds
    assessment_payment_rate = np.clip(
        np.random.normal(assessment_payment_mean, assessment_payment_std, n_simulations),
        0.4, 1.0
    )

    # Generate initial retention rates
    raw_retention_rate = np.random.normal(retention_mean, retention_std, n_simulations)

    # IMPORTANT: Ensure retention rate is never higher than assessment payment rate
    # If they don't pay assessment, they can't retain membership
    membership_retention_rate = np.minimum(
        np.clip(raw_retention_rate, 0.5, 1.0),
        assessment_payment_rate
    )

    dues_increase_factor = np.clip(
        np.random.normal(dues_increase_mean, dues_increase_std, n_simulations),
        1.0, 2.0
    )

    # Payment plan adoption rate
    if payment_plan_months > 0:
        payment_plan_rate = np.clip(
            np.random.normal(payment_plan_adoption, payment_plan_uncertainty, n_simulations),
            0.0, 1.0
        )
    else:
        payment_plan_rate = np.zeros(n_simulations)

    # New members with variability
    new_members = np.maximum(
        np.random.normal(new_members_per_month, new_members_std, size=(n_simulations, months)),
        0
    ).astype(int)

    # Program revenue variability factors
    program_variability = np.clip(
        np.random.normal(1.0, program_uncertainty, n_simulations),
        0.5, 1.5
    )

    # Lump sum gift variability (with possibility of not receiving it)
    if lump_sum_gift_amount > 0:
        # Generate gift amounts with uncertainty
        gift_amounts = np.random.normal(lump_sum_gift_amount, lump_sum_gift_amount * lump_sum_gift_uncertainty,
                                        n_simulations)
        # Add probability of not receiving gift at all (higher uncertainty = higher chance of no gift)
        gift_probability = np.random.random(n_simulations) > (
                lump_sum_gift_uncertainty * 0.5)  # Max 50% chance of no gift
        gift_amounts = gift_amounts * gift_probability
        gift_amounts = np.maximum(gift_amounts, 0)  # No negative gifts
    else:
        gift_amounts = np.zeros(n_simulations)

    ending_reserves = []
    reserve_trajectories = []
    member_trajectories = []
    program_revenue_trajectories = []

    for i in range(n_simulations):
        reserves = starting_reserves

        # Calculate final membership (those who will stay long-term)
        final_members = int(initial_members * membership_retention_rate[i])

        # Calculate assessment revenue
        total_assessment = assessment_amount * (initial_members * assessment_payment_rate[i])

        # Split between immediate and payment plan
        if payment_plan_months > 0:
            members_on_plan = payment_plan_rate[i]
            immediate_assessment = total_assessment * (1 - members_on_plan)
            monthly_assessment = (total_assessment * members_on_plan) / payment_plan_months
        else:
            immediate_assessment = total_assessment
            monthly_assessment = 0

        # Add immediate assessment revenue
        reserves += immediate_assessment

        # One-time lump sum gift
        reserves += gift_amounts[i]

        # Adjusted monthly dues
        monthly_dues_adjusted = monthly_dues * dues_increase_factor[i]

        # Track monthly data
        monthly_reserves = [reserves]
        monthly_members = [initial_members]  # Start with all members
        monthly_program_revenue = []
        current_expenses = monthly_expenses

        for month in range(months):
            # During first 12 months: gradual transition
            if month < 12:
                # Each month, 1/12 of memberships come up for renewal
                renewal_proportion = (month + 1) / 12.0

                # Key insight: We need to track who is still paying
                # 1. Not yet renewed: Everyone who hasn't reached renewal (they ALL still pay old dues)
                members_not_yet_renewed = int(initial_members * (1 - renewal_proportion))

                # 2. Those who renewed and stayed (now paying new dues)
                members_renewed_and_stayed = int(final_members * renewal_proportion)

                # Total currently paying members
                current_members = members_not_yet_renewed + members_renewed_and_stayed

                # Dues calculation is correct:
                # - Not yet renewed pay OLD dues
                # - Renewed and stayed pay NEW dues
                monthly_dues_income = (members_not_yet_renewed * monthly_dues +
                                       members_renewed_and_stayed * monthly_dues_adjusted)

            else:
                # After 12 months: only retained members at new dues
                current_members = final_members
                monthly_dues_income = current_members * monthly_dues_adjusted

            # Add new members (they pay new dues)
            new_members_count = new_members[i][month]
            current_members += new_members_count
            monthly_dues_income += new_members_count * monthly_dues_adjusted

            # Add monthly assessment payment if within payment plan period
            if month < payment_plan_months and payment_plan_months > 0:
                monthly_dues_income += monthly_assessment

            # Apply expense growth
            current_expenses = monthly_expenses * (1 + expense_growth_rate) ** (month / 12)

            # Add program revenues and expenses with uncertainty
            if program_revenues and program_expenses:
                program_growth_factor = (1 + program_growth_rate) ** (month / 12)
                monthly_program_net = 0
                for prog_rev, prog_exp in zip(program_revenues, program_expenses):
                    # Apply both growth and uncertainty
                    adjusted_rev = prog_rev * program_growth_factor * program_variability[i]
                    adjusted_exp = prog_exp * program_growth_factor * program_variability[i]
                    monthly_program_net += (adjusted_rev - adjusted_exp)
                monthly_dues_income += monthly_program_net
                monthly_program_revenue.append(monthly_program_net)

            reserves += monthly_dues_income - current_expenses

            # Track data
            monthly_reserves.append(reserves)
            monthly_members.append(current_members)

        ending_reserves.append(reserves)
        reserve_trajectories.append(monthly_reserves)
        member_trajectories.append(monthly_members)
        program_revenue_trajectories.append(monthly_program_revenue)

    return ending_reserves, reserve_trajectories, member_trajectories


def create_plotly_distribution(ending_reserves, danger_threshold):
    """Create an interactive Plotly distribution chart"""
    fig = go.Figure()

    # Add histogram
    fig.add_trace(go.Histogram(
        x=ending_reserves,
        nbinsx=50,
        name='Simulations',
        marker_color='skyblue',
        opacity=0.7
    ))

    # Add percentile lines
    p5 = np.percentile(ending_reserves, 5)
    p50 = np.percentile(ending_reserves, 50)
    p95 = np.percentile(ending_reserves, 95)

    fig.add_vline(x=p5, line_dash="dash", line_color="red",
                  annotation_text="5th %ile (Pessimistic)")
    fig.add_vline(x=p50, line_dash="solid", line_color="blue",
                  annotation_text="Median")
    fig.add_vline(x=p95, line_dash="dash", line_color="green",
                  annotation_text="95th %ile (Optimistic)")
    fig.add_vline(x=danger_threshold, line_dash="dot", line_color="orange",
                  annotation_text="Danger Threshold")

    fig.update_layout(
        title="Distribution of Ending Reserve Balances",
        xaxis_title="Ending Reserve Balance ($)",
        yaxis_title="Number of Simulations",
        showlegend=False,
        height=500
    )

    return fig


def create_trajectory_plot(reserve_trajectories, months, danger_threshold):
    """Create an interactive trajectory plot using Plotly"""
    df = pd.DataFrame(reserve_trajectories).T

    # Calculate percentiles
    percentiles = df.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T

    fig = go.Figure()

    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=list(range(months + 1)) * 2,
        y=list(percentiles[0.05]) + list(percentiles[0.95])[::-1],
        fill='toself',
        fillcolor='rgba(135, 206, 235, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='90% Confidence'
    ))

    fig.add_trace(go.Scatter(
        x=list(range(months + 1)) * 2,
        y=list(percentiles[0.25]) + list(percentiles[0.75])[::-1],
        fill='toself',
        fillcolor='rgba(135, 206, 235, 0.4)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='50% Confidence'
    ))

    # Add median line
    fig.add_trace(go.Scatter(
        x=list(range(months + 1)),
        y=percentiles[0.5],
        mode='lines',
        name='Median Path',
        line=dict(color='blue', width=3)
    ))

    # Add danger threshold
    fig.add_hline(y=danger_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Danger Threshold (${danger_threshold:,})")

    fig.update_layout(
        title="Projected Reserve Balance Over Time",
        xaxis_title="Month",
        yaxis_title="Reserve Balance ($)",
        hovermode='x unified',
        height=500
    )

    return fig


def main():
    # Header with logo placeholder
    logo_base64 = load_samena_logo()

    st.markdown(f"""
    <div class="main-header">
        <img src="{logo_base64}" style="height: 60px; margin-bottom: 1rem;">
        <h1 style="margin: 0;">Financial Projection Simulator</h1>
        <p style="margin: 0; opacity: 0.9;">Plan for a Sustainable Future</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Financial Projections", "📈 Elasticity Analysis", "ℹ️ Help"])

    with main_tab1:
        # Sidebar with inputs
        with st.sidebar:
            st.header("📊 Scenario Builder")
            st.markdown("""
            <div class="help-text">
            Adjust the settings below to see how different scenarios might affect the club's finances.
            Current data is based on your FY2025 financial statements.
            </div>
            """, unsafe_allow_html=True)

            # Financial Baseline
            with st.expander("💰 Current Financial Situation", expanded=False):
                st.markdown("**What's in the bank right now?**")
                starting_reserves = st.number_input(
                    "Current Cash Reserves",
                    value=DEFAULT_VALUES["starting_reserves"],
                    step=10000,
                    help=f"Based on FY2025: Cash ${DEFAULT_VALUES['current_cash']:,}",
                    format="%d"
                )

                st.markdown("**Overhead (non-program) monthly expenses**")
                monthly_expenses = st.number_input(
                    "Monthly Overhead Expenses",
                    value=DEFAULT_VALUES["monthly_expenses"],
                    step=5000,
                    help=f"FY2025 total monthly expense (432,795) minus sum of program-specific monthly costs (123,452) = 309,343",
                    format="%d"
                )

                st.markdown("**Expected cost increases**")
                expense_growth_rate = st.slider(
                    "Annual Expense Growth Rate (%)",
                    0.0, 10.0, DEFAULT_VALUES["expense_growth_rate"],
                    help="How much costs typically increase each year (inflation, wage increases, etc.)"
                ) / 100

            # Membership and Dues
            with st.expander("👥 Members and Monthly Dues", expanded=False):
                st.markdown("**Current membership**")
                initial_members = st.number_input(
                    "Number of Member Households",
                    value=DEFAULT_VALUES["initial_members"],
                    step=10,
                    help=f"Based on FY2025: Estimated from dues revenue",
                )

                st.markdown("**Current monthly dues**")
                monthly_dues = st.number_input(
                    "Monthly Dues per Household",
                    value=DEFAULT_VALUES["monthly_dues"],
                    step=25,
                    help=f"Based on FY2025: ${DEFAULT_VALUES['annual_dues_revenue'] / 1e6:.1f}M annual dues ÷ {DEFAULT_VALUES['initial_members']} members ÷ 12 months",
                    format="%d"
                )

                st.markdown("**Proposed one-time assessment**")
                assessment_amount = st.number_input(
                    "Special Assessment Amount",
                    value=DEFAULT_VALUES["assessment_amount"],
                    step=100,
                    help="One-time payment requested from each member household",
                    format="%d"
                )

            # Assessment Payment Plan
            with st.expander("💳 Assessment Payment Plan", expanded=False):
                st.markdown("""
                <div class="help-text">
                Allow members to pay the assessment over multiple months instead of all at once.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Payment plan duration**")
                payment_plan_months = st.slider(
                    "Number of months to spread payment",
                    0, 12, DEFAULT_VALUES["payment_plan_months"],
                    help="0 = no payment plan (all pay upfront)"
                )

                if payment_plan_months > 0:
                    st.markdown("**Payment plan adoption**")
                    col1, col2 = st.columns(2)

                    with col1:
                        payment_plan_adoption = st.slider(
                            "% who choose payment plan",
                            0, 100, DEFAULT_VALUES["payment_plan_adoption"], 5,
                            help="What percentage will choose to pay over time?"
                        )
                        payment_plan_adoption_decimal = payment_plan_adoption / 100

                    with col2:
                        payment_plan_uncertainty = st.slider(
                            "Uncertainty (±%)",
                            0, 30, DEFAULT_VALUES["payment_plan_uncertainty"], 5,
                            help="How confident are you about adoption rate?",
                            key="payment_plan_uncertainty"
                        )
                        payment_plan_uncertainty = payment_plan_uncertainty / 100

                    # Show payment breakdown
                    if assessment_amount > 0:
                        monthly_payment = assessment_amount / payment_plan_months
                        st.info(f"""
                        💡 **Payment Plan Details:**
                        - Monthly payment: ${monthly_payment:.0f}
                        - Total over {payment_plan_months} months: ${assessment_amount:,}
                        - Expected on plan: ~{int(initial_members * payment_plan_adoption_decimal)} members
                        """)
                else:
                    payment_plan_adoption_decimal = 0
                    payment_plan_uncertainty = 0

            # Lump Sum Gift
            with st.expander("🎁 One-Time Gift/Donation", expanded=False):
                st.markdown("""
                <div class="help-text">
                Model a potential one-time donation from a generous member or supporter to help the club through its current situation.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Potential donation amount**")
                lump_sum_gift_amount = st.number_input(
                    "Lump Sum Gift Amount",
                    value=DEFAULT_VALUES["lump_sum_gift_amount"],
                    step=10000,
                    help="One-time donation amount (e.g., from a major donor wanting to help)",
                    format="%d"
                )

                st.markdown("**Donation uncertainty**")
                lump_sum_gift_uncertainty = st.slider(
                    "Gift Uncertainty (±%)",
                    0, 50, int(DEFAULT_VALUES["lump_sum_gift_uncertainty"]), 5,
                    help="How certain are you about receiving this gift? Higher = less certain",
                    key="gift_uncertainty"
                )
                lump_sum_gift_uncertainty = lump_sum_gift_uncertainty / 100

            # Program Revenues and Expenses
            with st.expander("🏊 Program Revenues & Expenses", expanded=False):
                st.markdown("""
                <div class="help-text">
                Enter monthly revenues and expenses for each program. These will grow at the program growth rate.
                Note: Rental revenue has no associated expenses in the financials.
                </div>
                """, unsafe_allow_html=True)

                # Youth Programs
                st.markdown("**👦 Youth Programs**")
                col1, col2 = st.columns(2)
                with col1:
                    youth_programs_revenue = st.number_input(
                        "Monthly Revenue",
                        value=DEFAULT_VALUES["youth_programs_monthly_revenue"],
                        step=5000,
                        help="Average monthly revenue from youth programs",
                        format="%d",
                        key="youth_rev"
                    )
                with col2:
                    youth_programs_expenses = st.number_input(
                        "Monthly Expenses",
                        value=DEFAULT_VALUES["youth_programs_monthly_expenses"],
                        step=5000,
                        help="Average monthly expenses for youth programs",
                        format="%d",
                        key="youth_exp"
                    )

                # Aquatics
                st.markdown("**🏊 Aquatics**")
                col1, col2 = st.columns(2)
                with col1:
                    aquatics_revenue = st.number_input(
                        "Monthly Revenue",
                        value=DEFAULT_VALUES["aquatics_monthly_revenue"],
                        step=5000,
                        help="Average monthly revenue from aquatics",
                        format="%d",
                        key="aquatics_rev"
                    )
                with col2:
                    aquatics_expenses = st.number_input(
                        "Monthly Expenses",
                        value=DEFAULT_VALUES["aquatics_monthly_expenses"],
                        step=5000,
                        help="Average monthly expenses for aquatics",
                        format="%d",
                        key="aquatics_exp"
                    )

                # Fitness
                st.markdown("**💪 Fitness**")
                col1, col2 = st.columns(2)
                with col1:
                    fitness_revenue = st.number_input(
                        "Monthly Revenue",
                        value=DEFAULT_VALUES["fitness_monthly_revenue"],
                        step=5000,
                        help="Average monthly revenue from fitness",
                        format="%d",
                        key="fitness_rev"
                    )
                with col2:
                    fitness_expenses = st.number_input(
                        "Monthly Expenses",
                        value=DEFAULT_VALUES["fitness_monthly_expenses"],
                        step=5000,
                        help="Average monthly expenses for fitness",
                        format="%d",
                        key="fitness_exp"
                    )

                # Rental (Revenue only)
                st.markdown("**🏠 Rental**")
                rental_revenue = st.number_input(
                    "Monthly Revenue",
                    value=DEFAULT_VALUES["rental_monthly_revenue"],
                    step=1000,
                    help="Average monthly revenue from facility rentals",
                    format="%d",
                    key="rental_rev"
                )

                # Retail
                st.markdown("**🛍️ Retail**")
                col1, col2 = st.columns(2)
                with col1:
                    retail_revenue = st.number_input(
                        "Monthly Revenue",
                        value=DEFAULT_VALUES["retail_monthly_revenue"],
                        step=1000,
                        help="Average monthly revenue from retail",
                        format="%d",
                        key="retail_rev"
                    )
                with col2:
                    retail_expenses = st.number_input(
                        "Monthly Expenses",
                        value=DEFAULT_VALUES["retail_monthly_expenses"],
                        step=1000,
                        help="Average monthly expenses for retail",
                        format="%d",
                        key="retail_exp"
                    )

                # Special Events
                st.markdown("**🎉 Special Events**")
                col1, col2 = st.columns(2)
                with col1:
                    special_events_revenue = st.number_input(
                        "Monthly Revenue",
                        value=DEFAULT_VALUES["special_events_monthly_revenue"],
                        step=1000,
                        help="Average monthly revenue from special events",
                        format="%d",
                        key="events_rev"
                    )
                with col2:
                    special_events_expenses = st.number_input(
                        "Monthly Expenses",
                        value=DEFAULT_VALUES["special_events_monthly_expenses"],
                        step=1000,
                        help="Average monthly expenses for special events",
                        format="%d",
                        key="events_exp"
                    )

                # Program growth rate
                st.markdown("**📈 Program Growth**")
                col1, col2 = st.columns(2)
                with col1:
                    program_growth_rate = st.slider(
                        "Annual Program Growth Rate (%)",
                        -5.0, 10.0, DEFAULT_VALUES["program_growth_rate"],
                        help="Expected annual growth in program revenues and expenses"
                    )
                    program_growth_rate = program_growth_rate / 100

                with col2:
                    program_uncertainty = st.slider(
                        "Program Uncertainty (±%)",
                        0, 30, int(DEFAULT_VALUES["program_uncertainty"]), 5,
                        help="How much program revenues/expenses might vary from expectations",
                        key="program_uncertainty"
                    )
                    program_uncertainty = program_uncertainty / 100

            # Behavior Assumptions
            with st.expander("📈 Member Behavior Predictions", expanded=False):
                st.markdown("""
                <div class="warning-box">
                <strong>⚠️ These are your best guesses</strong><br>
                The accuracy of the simulation depends on how well you estimate member behavior.
                Consider being conservative in your estimates.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Assessment payment expectations**")
                col1, col2 = st.columns(2)

                with col1:
                    assessment_payment_mean = st.slider(
                        "% Who Will Pay Assessment",
                        40, 100, DEFAULT_VALUES["assessment_payment_mean"], 5,
                        help="What percentage of members will actually pay the special assessment?"
                    )
                    assessment_payment_mean_decimal = assessment_payment_mean / 100

                with col2:
                    assessment_payment_std = st.slider(
                        "Uncertainty Range (±%)",
                        0, 30, DEFAULT_VALUES["assessment_payment_std"], 5,
                        help="How confident are you? Lower = more certain",
                        key="assessment_uncertainty"
                    )
                    assessment_payment_std = assessment_payment_std / 100

                st.markdown("**Member retention after changes**")
                col1, col2 = st.columns(2)

                with col1:
                    # Ensure retention can't exceed assessment payment
                    max_retention = min(assessment_payment_mean, DEFAULT_VALUES["retention_mean"])
                    retention_mean = st.slider(
                        "% Who Will Stay",
                        50, assessment_payment_mean, max_retention, 5,
                        help="What percentage of members will remain after assessment & dues increase? Note: This cannot exceed the assessment payment rate since non-payment results in membership loss."
                    )
                    retention_mean_decimal = retention_mean / 100

                with col2:
                    retention_std = st.slider(
                        "Uncertainty Range (±%)",
                        0, 30, DEFAULT_VALUES["retention_std"], 5,
                        help="How confident are you? Lower = more certain",
                        key="retention_uncertainty"
                    )
                    retention_std = retention_std / 100

                st.markdown("**Planned dues increase**")
                col1, col2 = st.columns(2)

                with col1:
                    dues_increase_percent = st.slider(
                        "Dues Increase %",
                        0, 100, DEFAULT_VALUES["dues_increase_percent"], 10,
                        help="Percentage increase in monthly dues"
                    )
                    dues_increase_mean = 1 + (dues_increase_percent / 100)

                with col2:
                    dues_increase_std = st.slider(
                        "Implementation Variance (±%)",
                        0, 20, DEFAULT_VALUES["dues_increase_std"], 5,
                        help="Variation in actual increase applied",
                        key="dues_uncertainty"
                    )
                    dues_increase_std = dues_increase_std / 100

                st.markdown("**New member growth**")
                col1, col2 = st.columns(2)

                with col1:
                    new_members_per_month = st.slider(
                        "New Members/Month",
                        0, 20, DEFAULT_VALUES["new_members_per_month"],
                        help="Average new households joining monthly"
                    )

                with col2:
                    new_members_std = st.slider(
                        "Variability",
                        0, 10, DEFAULT_VALUES["new_members_std"],
                        help="How much this varies month to month"
                    )

            # Simulation Settings
            with st.expander("⚙️ Simulation Settings", expanded=False):
                use_gradual_transition = st.checkbox(
                    "Use Gradual Member Transition",
                    value=True,
                    help="If checked, members who don't pay assessment continue paying old dues until their renewal date. If unchecked, all changes happen immediately."
                )

                months = st.slider(
                    "Months to Project",
                    6, 36, DEFAULT_VALUES["months"],
                    help="How far into the future to simulate"
                )

                n_simulations = st.select_slider(
                    "Simulation Runs",
                    options=[1000, 5000, 10000, 20000],
                    value=DEFAULT_VALUES["n_simulations"],
                    help="More runs = more accurate results (but slower)"
                )

                danger_threshold = st.number_input(
                    "Minimum Safe Reserve Level",
                    value=DEFAULT_VALUES["danger_threshold"],
                    step=50000,
                    help="The minimum cash balance needed for safe operations",
                    format="%d"
                )

            # ─── Quick Overview ───
            st.markdown("---")
            st.subheader("📊 Quick Overview")

            # Calculate key figures
            one_time_income = int(initial_members * assessment_payment_mean_decimal * assessment_amount)
            if payment_plan_months > 0:
                immediate_income = int(one_time_income * (1 - payment_plan_adoption_decimal))
                deferred_income = one_time_income - immediate_income
            else:
                immediate_income = one_time_income
                deferred_income = 0

            expected_gift = int(lump_sum_gift_amount * (1 - lump_sum_gift_uncertainty * 0.5))
            immediate_income += expected_gift if lump_sum_gift_amount > 0 else 0

            monthly_dues_revenue = int(initial_members * retention_mean_decimal * monthly_dues * dues_increase_mean)

            program_gross_revenue = (
                    youth_programs_revenue + aquatics_revenue + fitness_revenue
                    + rental_revenue + retail_revenue + special_events_revenue
            )
            program_expenses = (
                    youth_programs_expenses + aquatics_expenses + fitness_expenses
                    + retail_expenses + special_events_expenses
            )

            overhead_expenses = monthly_expenses

            monthly_surplus = monthly_dues_revenue + (program_gross_revenue - program_expenses) - overhead_expenses

            # Display metrics in two columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Immediate Income", f"${immediate_income:,}")
                if deferred_income > 0:
                    st.metric("Deferred Income (Payment Plan)", f"${deferred_income:,}")
                st.metric("Monthly Dues Revenue", f"${monthly_dues_revenue:,}")
                st.metric("Program Gross Revenue", f"${program_gross_revenue:,}")
            with col2:
                st.metric("Program Expenses", f"${program_expenses:,}")
                st.metric("Overhead Expenses", f"${overhead_expenses:,}")
                st.metric("Monthly Surplus/Deficit", f"${monthly_surplus:,}", delta=f"${monthly_surplus:,}")

            # Run projection
            run_simulation_btn = st.button("🚀 Run Financial Projection", type="primary")

        # Main content area - continuing from run_simulation_btn
        if run_simulation_btn:
            with st.spinner("Running simulations..."):
                # Prepare program data
                program_revenues = [youth_programs_revenue, aquatics_revenue, fitness_revenue,
                                    rental_revenue, retail_revenue, special_events_revenue]
                program_expenses = [youth_programs_expenses, aquatics_expenses, fitness_expenses,
                                    0,  # Rental has no expenses
                                    retail_expenses, special_events_expenses]

                # Choose which simulation to run
                if use_gradual_transition:
                    ending_reserves, reserve_trajectories, member_trajectories = run_simulation_gradual(
                        starting_reserves,
                        monthly_expenses,
                        initial_members,
                        assessment_amount,
                        monthly_dues,
                        assessment_payment_mean_decimal,
                        assessment_payment_std,
                        retention_mean_decimal,
                        retention_std,
                        dues_increase_mean,
                        dues_increase_std,
                        new_members_per_month,
                        new_members_std,
                        months,
                        n_simulations,
                        expense_growth_rate,
                        program_revenues,
                        program_expenses,
                        program_growth_rate,
                        program_uncertainty,
                        lump_sum_gift_amount,
                        lump_sum_gift_uncertainty,
                        payment_plan_months,
                        payment_plan_adoption_decimal,
                        payment_plan_uncertainty
                    )
                else:
                    ending_reserves, reserve_trajectories, member_trajectories = run_simulation(
                        starting_reserves,
                        monthly_expenses,
                        initial_members,
                        assessment_amount,
                        monthly_dues,
                        assessment_payment_mean_decimal,
                        assessment_payment_std,
                        retention_mean_decimal,
                        retention_std,
                        dues_increase_mean,
                        dues_increase_std,
                        new_members_per_month,
                        new_members_std,
                        months,
                        n_simulations,
                        expense_growth_rate,
                        program_revenues,
                        program_expenses,
                        program_growth_rate,
                        program_uncertainty,
                        lump_sum_gift_amount,
                        lump_sum_gift_uncertainty,
                        payment_plan_months,
                        payment_plan_adoption_decimal,
                        payment_plan_uncertainty
                    )

            # Calculate statistics
            p5 = np.percentile(ending_reserves, 5)
            p25 = np.percentile(ending_reserves, 25)
            p50 = np.percentile(ending_reserves, 50)
            p75 = np.percentile(ending_reserves, 75)
            p95 = np.percentile(ending_reserves, 95)

            risk_level, risk_icon = calculate_risk_score(ending_reserves, danger_threshold)

            # Display results
            st.header("📊 Simulation Results")

            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Median Outcome",
                    f"${p50:,.0f}",
                    f"{((p50 - starting_reserves) / starting_reserves * 100):.1f}%"
                )

            with col2:
                st.metric(
                    "Worst Case (5%)",
                    f"${p5:,.0f}",
                    f"{((p5 - starting_reserves) / starting_reserves * 100):.1f}%"
                )

            with col3:
                st.metric(
                    "Best Case (95%)",
                    f"${p95:,.0f}",
                    f"{((p95 - starting_reserves) / starting_reserves * 100):.1f}%"
                )

            with col4:
                st.metric(
                    "Risk Level",
                    risk_level,
                    risk_icon
                )

            # Insights box
            program_net = (youth_programs_revenue - youth_programs_expenses +
                           aquatics_revenue - aquatics_expenses +
                           fitness_revenue - fitness_expenses +
                           rental_revenue +  # Rental has no expenses
                           retail_revenue - retail_expenses +
                           special_events_revenue - special_events_expenses)

            # Recompute these so expected_assessment exists here:
            expected_assessment = int(initial_members * assessment_payment_mean_decimal * assessment_amount)
            expected_gift = int(lump_sum_gift_amount * (1 - lump_sum_gift_uncertainty * 0.5))

            # Build the one-time income sources section
            one_time_sources = f'<li>Assessment revenue: <strong>${expected_assessment:,}</strong>'
            if payment_plan_months > 0:
                immediate_assess = int(expected_assessment * (1 - payment_plan_adoption_decimal))
                deferred_assess = expected_assessment - immediate_assess
                one_time_sources += f' (${immediate_assess:,} immediate, ${deferred_assess:,} over {payment_plan_months} months)'
            one_time_sources += '</li>'

            if lump_sum_gift_amount > 0:
                one_time_sources += f'<li>Gift/donation: <strong>${expected_gift:,}</strong> ({100 - lump_sum_gift_uncertainty * 50:.0f}% probability)</li>'

            st.markdown(f"""
            <div class="insight-box">
                <h4>💡 What This Means</h4>
                <ul>
                    <li><strong>{(sum(1 for r in ending_reserves if r > danger_threshold) / len(ending_reserves) * 100):.1f}%</strong> chance 
                        of maintaining safe reserve levels (above ${danger_threshold:,})</li>
                    <li>If the median scenario plays out, you'll {'gain' if p50 > starting_reserves else 'lose'} 
                        <strong>${abs(p50 - starting_reserves):,.0f}</strong> over {months} months</li>
                    <li>In the worst 5% of scenarios, reserves could drop to <strong>${p5:,.0f}</strong></li>
                    <li>One-time income sources:
                        <ul>
                            {one_time_sources}
                        </ul>
                    </li>
                    <li>Monthly revenue breakdown:
                        <ul>
                            <li>Dues revenue: <strong>${(initial_members * retention_mean_decimal * monthly_dues * dues_increase_mean):,.0f}</strong></li>
                            <li>Program net revenue: <strong>${program_net:,.0f}</strong></li>
                            <li>Total revenue: <strong>${(initial_members * retention_mean_decimal * monthly_dues * dues_increase_mean + program_net):,.0f}</strong></li>
                        </ul>
                    </li>
                    <li>Monthly expenses: <strong>${monthly_expenses:,.0f}</strong></li>
                    <li>Transition model: <strong>{'Gradual (12 months)' if use_gradual_transition else 'Immediate'}</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Visualizations
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📊 Distribution", "📈 Trajectories", "🏊 Program Analysis", "📋 Detailed Analysis"])

            with tab1:
                st.plotly_chart(create_plotly_distribution(ending_reserves, danger_threshold),
                                use_container_width=True)

                # Quick stats
                st.markdown("### Distribution Statistics")
                stats_df = pd.DataFrame({
                    'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
                    'Reserve Balance': [f"${x:,.0f}" for x in [p5, p25, p50, p75, p95]],
                    'Change from Start': [f"{((x - starting_reserves) / starting_reserves * 100):+.1f}%"
                                          for x in [p5, p25, p50, p75, p95]]
                })
                st.dataframe(stats_df, hide_index=True)

            with tab2:
                st.plotly_chart(create_trajectory_plot(reserve_trajectories, months, danger_threshold),
                                use_container_width=True)

                # Member trajectory insight
                member_df = pd.DataFrame(member_trajectories).T
                final_members_median = member_df.iloc[-1].median()

                st.info(f"""
                📊 **Membership Projection**: Starting with {initial_members} members, 
                the median projection shows {int(final_members_median)} members after {months} months 
                (a {'gain' if final_members_median > initial_members else 'loss'} of 
                {abs(int(final_members_median - initial_members))} members).
                """)

                # Add membership trajectory chart
                st.markdown("### Membership Over Time")

                # Calculate percentiles for membership
                member_percentiles = member_df.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T

                fig_members = go.Figure()

                # Add confidence bands
                fig_members.add_trace(go.Scatter(
                    x=list(range(months + 1)) * 2,
                    y=list(member_percentiles[0.05]) + list(member_percentiles[0.95])[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 182, 193, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name='90% Confidence'
                ))

                # Add median line
                fig_members.add_trace(go.Scatter(
                    x=list(range(months + 1)),
                    y=member_percentiles[0.5],
                    mode='lines',
                    name='Median Membership',
                    line=dict(color='darkred', width=3)
                ))

                # Add initial membership reference line
                fig_members.add_hline(y=initial_members, line_dash="dot", line_color="gray",
                                      annotation_text=f"Starting: {initial_members}")

                fig_members.update_layout(
                    title="Projected Membership Over Time",
                    xaxis_title="Month",
                    yaxis_title="Number of Members",
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig_members, use_container_width=True)

                # Add monthly revenue analysis
                st.markdown("### Monthly Revenue Analysis")

                # Calculate monthly revenues from trajectories
                monthly_revenues = []
                for i in range(n_simulations):
                    sim_revenues = []
                    for month in range(months + 1):
                        if month == 0:
                            # Month 0 includes assessment
                            revenue = expected_assessment / n_simulations  # Rough approximation
                        else:
                            # Calculate from reserve changes
                            if month < len(reserve_trajectories[i]):
                                reserve_change = reserve_trajectories[i][month] - reserve_trajectories[i][month - 1]
                                # Revenue = reserve change + expenses
                                monthly_expense = monthly_expenses * (1 + expense_growth_rate) ** ((month - 1) / 12)
                                revenue = reserve_change + monthly_expense
                                sim_revenues.append(revenue)
                    monthly_revenues.append(sim_revenues)

                # Convert to DataFrame for percentile calculations
                revenue_df = pd.DataFrame(monthly_revenues).T
                revenue_percentiles = revenue_df.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T

                fig_revenue = go.Figure()

                # Add confidence bands
                fig_revenue.add_trace(go.Scatter(
                    x=list(range(1, months + 1)) * 2,
                    y=list(revenue_percentiles[0.05]) + list(revenue_percentiles[0.95])[::-1],
                    fill='toself',
                    fillcolor='rgba(144, 238, 144, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name='90% Confidence'
                ))

                # Add median line
                fig_revenue.add_trace(go.Scatter(
                    x=list(range(1, months + 1)),
                    y=revenue_percentiles[0.5],
                    mode='lines',
                    name='Median Revenue',
                    line=dict(color='darkgreen', width=3)
                ))

                # Add expense line
                expense_line = [monthly_expenses * (1 + expense_growth_rate) ** (m / 12) for m in range(months)]
                fig_revenue.add_trace(go.Scatter(
                    x=list(range(1, months + 1)),
                    y=expense_line,
                    mode='lines',
                    name='Monthly Expenses',
                    line=dict(color='red', width=2, dash='dash')
                ))

                fig_revenue.update_layout(
                    title="Monthly Revenue vs Expenses",
                    xaxis_title="Month",
                    yaxis_title="Amount ($)",
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig_revenue, use_container_width=True)

                # Add transition comparison if gradual model is used
                if use_gradual_transition:
                    st.markdown("### Transition Period Analysis (First 12 Months)")

                    # Create comparison metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        avg_members_yr1 = member_percentiles[0.5][:13].mean()
                        st.metric(
                            "Avg Members (Year 1)",
                            f"{int(avg_members_yr1):,}",
                            f"{int(avg_members_yr1 - initial_members):,}"
                        )

                    with col2:
                        avg_revenue_yr1 = revenue_percentiles[0.5][:12].mean()
                        st.metric(
                            "Avg Monthly Revenue (Year 1)",
                            f"${int(avg_revenue_yr1):,}",
                            f"${int(avg_revenue_yr1 - monthly_expenses):,}"
                        )

                    with col3:
                        transition_benefit = (initial_members - int(
                            initial_members * retention_mean_decimal)) * monthly_dues * 6
                        st.metric(
                            "Transition Benefit",
                            f"${int(transition_benefit):,}",
                            "From gradual loss"
                        )

                    st.info("""
                    📊 **Gradual Transition Analysis**: During the 12-month transition period, members who don't pay 
                    the assessment continue paying old dues until their renewal date. This generates additional revenue 
                    compared to immediate termination, but may be offset by the delayed dues increase for retained members.
                    """)

                # Add detailed month-by-month breakdown
                with st.expander("📊 Show Month-by-Month Breakdown"):
                    # Create detailed breakdown for first 12 months
                    breakdown_data = []
                    # Use median values for the example
                    example_final_members = int(initial_members * retention_mean_decimal)
                    example_new_dues = int(monthly_dues * dues_increase_mean)

                    for month in range(min(12, months)):
                        if use_gradual_transition and month < 12:
                            renewal_prop = (month + 1) / 12.0
                            not_renewed = int(initial_members * (1 - renewal_prop))
                            renewed_stayed = int(example_final_members * renewal_prop)

                            breakdown_data.append({
                                'Month': month + 1,
                                'Not Yet Renewed': not_renewed,
                                'Renewed & Stayed': renewed_stayed,
                                'Total Members': not_renewed + renewed_stayed,
                                'Revenue (Old Dues)': f"${not_renewed * monthly_dues:,}",
                                'Revenue (New Dues)': f"${renewed_stayed * example_new_dues:,}",
                                'Total Dues Revenue': f"${not_renewed * monthly_dues + renewed_stayed * example_new_dues:,}"
                            })
                        else:
                            # Immediate model or after transition
                            members = example_final_members
                            revenue = members * example_new_dues
                            breakdown_data.append({
                                'Month': month + 1,
                                'Not Yet Renewed': 0,
                                'Renewed & Stayed': members,
                                'Total Members': members,
                                'Revenue (Old Dues)': "$0",
                                'Revenue (New Dues)': f"${revenue:,}",
                                'Total Dues Revenue': f"${revenue:,}"
                            })

                    breakdown_df = pd.DataFrame(breakdown_data)
                    st.dataframe(breakdown_df, hide_index=True)

            with tab3:
                st.markdown("### Program Revenue & Expense Analysis")

                # Program summary table
                program_df = pd.DataFrame({
                    'Program': ['Youth Programs', 'Aquatics', 'Fitness', 'Rental', 'Retail', 'Special Events'],
                    'Monthly Revenue': [f"${r:,}" for r in [youth_programs_revenue, aquatics_revenue,
                                                            fitness_revenue, rental_revenue,
                                                            retail_revenue, special_events_revenue]],
                    'Monthly Expenses': [f"${e:,}" for e in [youth_programs_expenses, aquatics_expenses,
                                                             fitness_expenses, 0,
                                                             retail_expenses, special_events_expenses]],
                    'Net Income': [f"${r - e:,}" for r, e in [(youth_programs_revenue, youth_programs_expenses),
                                                              (aquatics_revenue, aquatics_expenses),
                                                              (fitness_revenue, fitness_expenses),
                                                              (rental_revenue, 0),
                                                              (retail_revenue, retail_expenses),
                                                              (special_events_revenue, special_events_expenses)]],
                    'Margin %': [f"{((r - e) / r * 100):.1f}%" if r > 0 else "0.0%"
                                 for r, e in [(youth_programs_revenue, youth_programs_expenses),
                                              (aquatics_revenue, aquatics_expenses),
                                              (fitness_revenue, fitness_expenses),
                                              (rental_revenue, 0),
                                              (retail_revenue, retail_expenses),
                                              (special_events_revenue, special_events_expenses)]]
                })

                st.dataframe(program_df, hide_index=True)

                # Program contribution chart
                program_names = ['Youth Programs', 'Aquatics', 'Fitness', 'Rental', 'Retail', 'Special Events']
                program_nets = [youth_programs_revenue - youth_programs_expenses,
                                aquatics_revenue - aquatics_expenses,
                                fitness_revenue - fitness_expenses,
                                rental_revenue,  # No expenses
                                retail_revenue - retail_expenses,
                                special_events_revenue - special_events_expenses]

                fig_programs = go.Figure()
                fig_programs.add_trace(go.Bar(
                    x=program_names,
                    y=program_nets,
                    text=[f"${v:,.0f}" for v in program_nets],
                    textposition='auto',
                    marker_color=['green' if v > 0 else 'red' for v in program_nets]
                ))

                fig_programs.update_layout(
                    title="Program Net Contribution",
                    xaxis_title="Program",
                    yaxis_title="Monthly Net Income ($)",
                    showlegend=False,
                    height=400
                )

                st.plotly_chart(fig_programs, use_container_width=True)

                # Program growth projection
                st.markdown("### Program Revenue Growth Projection")

                months_range = list(range(months + 1))
                program_growth_projection = [program_net * (1 + program_growth_rate) ** (m / 12) for m in months_range]

                fig_growth = go.Figure()
                fig_growth.add_trace(go.Scatter(
                    x=months_range,
                    y=program_growth_projection,
                    mode='lines',
                    name='Program Net Revenue',
                    line=dict(color='purple', width=3)
                ))

                fig_growth.update_layout(
                    title=f"Projected Program Net Revenue ({program_growth_rate * 100:.1f}% Annual Growth)",
                    xaxis_title="Month",
                    yaxis_title="Monthly Net Revenue ($)",
                    height=400
                )

                st.plotly_chart(fig_growth, use_container_width=True)

                st.info(f"""
                📊 **Program Summary**: The six programs combined currently generate **${program_net:,}** in monthly net revenue.

                With {program_growth_rate * 100:.1f}% annual growth, this could reach **${program_net * (1 + program_growth_rate) ** 2:,.0f}** in 24 months.

                The {program_uncertainty * 100:.0f}% uncertainty range means actual results could vary significantly from projections.
                """)

            with tab4:
                st.markdown("### Detailed Scenario Analysis")

                # Scenario comparison
                scenarios = {
                    'Pessimistic (5th percentile)': p5,
                    'Conservative (25th percentile)': p25,
                    'Expected (Median)': p50,
                    'Optimistic (75th percentile)': p75,
                    'Best Case (95th percentile)': p95
                }

                scenario_df = pd.DataFrame({
                    'Scenario': scenarios.keys(),
                    'Ending Reserves': [f"${v:,.0f}" for v in scenarios.values()],
                    'Monthly Burn Rate': [f"${(starting_reserves - v) / months:,.0f}"
                                          if v < starting_reserves else "Positive"
                                          for v in scenarios.values()],
                    'Months Until Depletion': [f"{int(v / monthly_expenses)}" if v > 0 else "Already depleted"
                                               for v in scenarios.values()]
                })

                st.dataframe(scenario_df, hide_index=True)

                # Risk factors
                st.markdown("### Risk Factor Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Positive Factors:**")
                    st.markdown(f"""
                    - Assessment collection: ${assessment_amount * initial_members * assessment_payment_mean_decimal:,.0f} expected
                    {f'- Lump sum gift: ${expected_gift:,.0f} expected' if lump_sum_gift_amount > 0 else ''}
                    - Dues increase: {(dues_increase_mean - 1) * 100:.0f}% boost to revenue
                    - New member growth: {new_members_per_month * months} expected over {months} months
                    - Program net revenue: ${program_net:,.0f}/month
                    - Program growth: {program_growth_rate * 100:.0f}% annual increase
                    """)

                with col2:
                    st.markdown("**Risk Factors:**")
                    st.markdown(f"""
                    - Member attrition: {(1 - retention_mean_decimal) * 100:.0f}% expected loss
                    - Expense growth: {expense_growth_rate * 100:.0f}% annual increase
                    - Collection uncertainty: ±{assessment_payment_std * 100:.0f}% variation
                    {f'- Gift uncertainty: ±{lump_sum_gift_uncertainty * 100:.0f}% variation' if lump_sum_gift_amount > 0 else ''}
                    - Program uncertainty: ±{program_uncertainty * 100:.0f}% variation
                    """)

                # Export options
                st.markdown("### Export Results")
                results_summary = {
                    'simulation_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'parameters': {
                        'starting_reserves': starting_reserves,
                        'monthly_expenses': monthly_expenses,
                        'initial_members': initial_members,
                        'assessment_amount': assessment_amount,
                        'monthly_dues': monthly_dues,
                        'lump_sum_gift': lump_sum_gift_amount,
                        'months_simulated': months,
                        'simulations_run': n_simulations,
                        'transition_model': 'gradual' if use_gradual_transition else 'immediate',
                        'payment_plan_months': payment_plan_months,
                        'payment_plan_adoption': payment_plan_adoption_decimal
                    },
                    'results': {
                        'median_outcome': p50,
                        'percentile_5': p5,
                        'percentile_95': p95,
                        'risk_level': risk_level,
                        'probability_above_threshold': sum(1 for r in ending_reserves if r > danger_threshold) / len(
                            ending_reserves)
                    }
                }

                st.download_button(
                    "📥 Download Results (JSON)",
                    data=pd.Series(results_summary).to_json(indent=2),
                    file_name=f"samena_simulation_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

        else:
            # Welcome message when no simulation has been run
            st.markdown("""
            <div class="main-header" style="text-align: left; padding: 2rem;">
                <h2>Welcome to Your Financial Planning Tool</h2>
                <p>This simulator helps you understand how different decisions might affect Samena Club's financial future.</p>
            </div>
            """, unsafe_allow_html=True)

            # Current situation summary
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Current Situation</h3>
                    <p><strong>Cash on hand:</strong> ${DEFAULT_VALUES['starting_reserves']:,}</p>
                    <p><strong>Monthly expenses:</strong> ~${DEFAULT_VALUES['monthly_expenses']:,}</p>
                    <p><strong>Runway:</strong> ~{DEFAULT_VALUES['starting_reserves'] / DEFAULT_VALUES['monthly_expenses']:.1f} months</p>
                    <p class="help-text">Based on FY2025 financials</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>👥 Membership</h3>
                    <p><strong>Current members:</strong> {DEFAULT_VALUES['initial_members']} households</p>
                    <p><strong>Monthly dues:</strong> ${DEFAULT_VALUES['monthly_dues']}/household</p>
                    <p><strong>Annual dues revenue:</strong> ${DEFAULT_VALUES['annual_dues_revenue'] / 1e6:.1f}M</p>
                    <p class="help-text">Covers ~{DEFAULT_VALUES['annual_dues_revenue'] / DEFAULT_VALUES['annual_expenses'] * 100:.0f}% of expenses</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>🎯 What You Can Test</h3>
                    <p>• Special assessments</p>
                    <p>• Dues increases</p>
                    <p>• Lump sum gifts</p>
                    <p>• Member retention impacts</p>
                    <p>• Growth scenarios</p>
                    <p>• Payment plans</p>
                </div>
                """, unsafe_allow_html=True)

            # Quick start guide
            with st.expander("📖 How to Use This Tool", expanded=True):
                st.markdown("""
                ### Getting Started

                1. **Review the current situation** in the sidebar (we've pre-filled it based on your 990)
                2. **Set your proposed changes**:
                   - How much is the special assessment?
                   - What percentage dues increase are you considering?
                   - Is there a potential major gift/donation?
                3. **Consider payment plans** to improve member satisfaction and cash flow
                4. **Estimate member reactions**:
                   - What percentage will pay the assessment?
                   - How many members might leave?
                5. **Run the simulation** to see thousands of possible outcomes
                6. **Review the results** to understand the risks and opportunities

                ### Understanding the Results

                The simulator runs your scenario thousands of times with slight variations to show you:
                - **Median outcome**: The most likely result
                - **Best/worst cases**: What could happen in extreme scenarios  
                - **Risk levels**: How likely you are to run low on reserves
                - **Trajectory charts**: How finances might evolve over time

                ### Tips for Better Predictions

                - **Be conservative** with your estimates - it's better to be pleasantly surprised
                - **Consider surveying members** before finalizing assessment amounts
                - **Look at the 5th percentile** (worst case) to understand your risks
                - **Test multiple scenarios** to find the sweet spot
                - **Use the Elasticity Analysis tab** to optimize dues increases
                """)

            st.info("👈 **Adjust the settings in the sidebar and click 'Run Financial Projection' to begin**")

    with main_tab2:
        st.header("📈 Elasticity Analysis - Find Your Sweet Spot")

        st.markdown("""
        This tool helps you find the optimal balance between dues increases and member retention.
        It calculates expected revenue based on how sensitive your members are to price changes.
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            elasticity = st.slider(
                "Price Elasticity",
                0.2, 1.2, 0.6, 0.1,
                help="How sensitive are members to price? Lower = less sensitive"
            )

        with col2:
            base_members_elast = st.number_input(
                "Current Members",
                value=DEFAULT_VALUES["initial_members"],
                step=10,
                help="Number of member households"
            )

        with col3:
            base_dues_elast = st.number_input(
                "Current Monthly Dues",
                value=DEFAULT_VALUES["monthly_dues"],
                step=5,
                help="Monthly dues per household"
            )

        # Generate elasticity table
        elasticity_df = find_optimal_dues_increase(base_members_elast, base_dues_elast, elasticity)

        # Find the optimal increase
        # Convert revenue strings back to numbers for finding max
        elasticity_df['Revenue_Numeric'] = elasticity_df['Monthly Revenue'].str.replace('$', '').str.replace(',', '').astype(float)

        optimal_idx = elasticity_df['Revenue_Numeric'].idxmax()
        optimal_increase = elasticity_df.loc[optimal_idx, 'Dues Increase %']
        optimal_revenue = elasticity_df.loc[optimal_idx, 'Monthly Revenue']

        # Display optimal point
        st.success(f"""
        🎯 **Optimal Dues Increase: {optimal_increase}%**
        - Expected Revenue: {optimal_revenue}
        - Member Retention: {elasticity_df.loc[optimal_idx, 'Retention %']:.1f}%
        - Members Retained: {elasticity_df.loc[optimal_idx, 'Members Retained']:,}
        """)

        # Create visualization
        fig_elasticity = go.Figure()

        # Add revenue line
        fig_elasticity.add_trace(go.Scatter(
            x=elasticity_df['Dues Increase %'],
            y=elasticity_df['Revenue_Numeric'],
            mode='lines+markers',
            name='Monthly Revenue',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))

        # Add optimal point
        fig_elasticity.add_trace(go.Scatter(
            x=[optimal_increase],
            y=[elasticity_df.loc[optimal_idx, 'Revenue_Numeric']],
            mode='markers',
            name='Optimal Point',
            marker=dict(size=15, color='red', symbol='star')
        ))

        # Add current revenue line
        current_revenue = base_members_elast * base_dues_elast
        fig_elasticity.add_hline(y=current_revenue, line_dash="dash", line_color="gray",
                                 annotation_text=f"Current Revenue: ${current_revenue:,}")

        fig_elasticity.update_layout(
            title=f"Revenue vs Dues Increase (Elasticity = {elasticity})",
            xaxis_title="Dues Increase (%)",
            yaxis_title="Monthly Revenue ($)",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig_elasticity, use_container_width=True)

        # Show detailed table
        st.markdown("### Detailed Analysis")

        # Format table for display (without numeric column)
        display_df = elasticity_df.drop('Revenue_Numeric', axis=1)

        # Highlight optimal row

        def highlight_optimal(row):
            if row['Dues Increase %'] == optimal_increase:
                return ['background-color: #d4edda'] * len(row)
            return [''] * len(row)

        styled_df = display_df.style.apply(highlight_optimal, axis=1)
        st.dataframe(styled_df, hide_index=True)

        # Elasticity explanation
        with st.expander("📚 Understanding Price Elasticity"):
            st.markdown("""
            ### What is Price Elasticity?

            Price elasticity measures how sensitive your members are to dues changes:

            - **Low Elasticity (0.2-0.5)**: Members are not very price-sensitive
              - Premium clubs with unique offerings
              - Limited alternatives available
              - Strong member loyalty

            - **Medium Elasticity (0.5-0.8)**: Moderate price sensitivity
              - Typical for most clubs
              - Some alternatives exist
              - Mixed member demographics

            - **High Elasticity (0.8-1.2)**: Members are very price-sensitive
              - Many alternatives available
              - Price-conscious membership
              - Commoditized offerings

            ### How to Estimate Your Club's Elasticity

            1. **Look at Past Increases**: How did members react to previous dues changes?
            2. **Survey Members**: Ask about price sensitivity and alternatives
            3. **Analyze Competition**: How do your prices compare to alternatives?
            4. **Consider Demographics**: Higher income areas typically have lower elasticity

            ### Using This Analysis

            The optimal increase shown maximizes revenue, but consider:
            - **Member Relations**: A smaller increase might be worth lower revenue
            - **Long-term Effects**: High increases might hurt future growth
            - **Implementation**: Gradual increases might be better received
            """)

    with main_tab3:
        st.header("ℹ️ Help & Documentation")

        st.markdown("""
        ### Quick Tips

        1. **Start Conservative**: It's better to underestimate revenue and overestimate member loss
        2. **Use the 5th Percentile**: Plan based on pessimistic scenarios, not median outcomes
        3. **Test Multiple Scenarios**: Try different combinations to find what works
        4. **Consider Payment Plans**: They can improve cash flow and member satisfaction

        ### Key Features

        #### 🔄 Transition Models
        - **Immediate**: All changes happen at once
        - **Gradual**: Members transition as contracts renew over 12 months

        #### 💳 Payment Plans
        - Spread assessment payments over multiple months
        - Reduces immediate financial burden on members
        - May increase overall payment rates

        #### 📈 Elasticity Analysis
        - Find the optimal balance between price and retention
        - Based on economic modeling of member behavior
        - Helps maximize revenue while minimizing member loss

        ### Understanding Your Results

        - **Median (50th percentile)**: The most likely outcome
        - **5th Percentile**: Only 5% of scenarios are worse
        - **95th Percentile**: Only 5% of scenarios are better

        ### Need Help?

        Contact your board treasurer or finance committee for assistance with:
        - Estimating member behavior
        - Understanding financial projections
        - Implementing recommendations
        """)


if __name__ == "__main__":
    main()
