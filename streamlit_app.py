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

# Hide the menu
hide_menu_style = """
<style>
/* Hide hamburger menu */
#MainMenu {visibility: hidden;}

/* Hide footer */
footer {visibility: hidden;}

/* Hide the deploy button (rocket icon) */
.stDeployButton {display: none;}

/* Optional: Hide the GitHub icon if present */
.stGitHubButton {display: none;}

/* Optional: Hide the entire toolbar */
.stToolbar {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# ====================================
# CENTRALIZED DEFAULT VALUES
# ====================================
DEFAULT_VALUES = {
    # Financial Baseline
    "starting_reserves": 250000,  # Current cash reserves
    "monthly_expenses": 400000,  # Monthly operating expenses
    "expense_growth_rate": 3.0,  # Annual expense growth rate (%)

    # Membership and Dues
    "initial_members": 1500,  # Number of member households
    "monthly_dues": 130,  # Monthly dues per household
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

    # Program Revenues and Expenses
    "swim_lessons_monthly_revenue": 50000,  # Monthly swim lesson revenue
    "swim_lessons_monthly_expenses": 35000,  # Monthly swim lesson expenses
    "summer_camp_monthly_revenue": 80000,  # Monthly summer camp revenue (averaged)
    "summer_camp_monthly_expenses": 60000,  # Monthly summer camp expenses (averaged)
    "after_school_monthly_revenue": 40000,  # Monthly after school revenue
    "after_school_monthly_expenses": 30000,  # Monthly after school expenses
    "preschool_monthly_revenue": 60000,  # Monthly preschool revenue
    "preschool_monthly_expenses": 45000,  # Monthly preschool expenses
    "program_growth_rate": 2.0,  # Annual program growth rate (%)
    "program_uncertainty": 15.0,  # Program revenue/expense uncertainty (%)

    # Simulation Settings
    "months": 24,  # Months to project
    "n_simulations": 10000,  # Simulation runs
    "danger_threshold": 250000,  # Minimum safe reserve level

    # Display Settings
    "current_cash": 720251,  # Cash from Form 990
    "current_savings": 26113,  # Savings from Form 990
    "annual_expenses": 4800000,  # Total annual expenses
    "annual_dues_revenue": 2000000,  # Annual dues revenue
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
        program_uncertainty=0.0
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

    ending_reserves = []
    reserve_trajectories = []
    member_trajectories = []
    program_revenue_trajectories = []

    for i in range(n_simulations):
        reserves = starting_reserves
        paying_members = int(initial_members * membership_retention_rate[i])

        # One-time assessment revenue
        assessment_revenue = assessment_amount * (initial_members * assessment_payment_rate[i])
        reserves += assessment_revenue

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

    # Sidebar with inputs
    with st.sidebar:
        st.header("📊 Scenario Builder")
        st.markdown("""
        <div class="help-text">
        Adjust the settings below to see how different scenarios might affect the club's finances.
        Current data is based on your 2023 Form 990.
        </div>
        """, unsafe_allow_html=True)

        # Financial Baseline
        with st.expander("💰 Current Financial Situation", expanded=True):
            st.markdown("**What's in the bank right now?**")
            starting_reserves = st.number_input(
                "Current Cash Reserves",
                value=DEFAULT_VALUES["starting_reserves"],
                step=10000,
                help=f"Based on Form 990: Cash (${DEFAULT_VALUES['current_cash']:,}) + Savings (${DEFAULT_VALUES['current_savings']:,})",
                format="%d"
            )

            st.markdown("**How much does it cost to run the club each month?**")
            monthly_expenses = st.number_input(
                "Monthly Operating Expenses",
                value=DEFAULT_VALUES["monthly_expenses"],
                step=5000,
                help=f"Based on Form 990: Total annual expenses (${DEFAULT_VALUES['annual_expenses'] / 1e6:.1f}M) ÷ 12",
                format="%d"
            )

            st.markdown("**Expected cost increases**")
            expense_growth_rate = st.slider(
                "Annual Expense Growth Rate (%)",
                0.0, 10.0, DEFAULT_VALUES["expense_growth_rate"],
                help="How much costs typically increase each year (inflation, wage increases, etc.)"
            ) / 100

        # Membership and Dues
        with st.expander("👥 Members and Monthly Dues", expanded=True):
            st.markdown("**Current membership**")
            initial_members = st.number_input(
                "Number of Member Households",
                value=DEFAULT_VALUES["initial_members"],
                step=10,
                help=f"Based on Form 990: {DEFAULT_VALUES['initial_members']} members reported"
            )

            st.markdown("**Current monthly dues**")
            monthly_dues = st.number_input(
                "Monthly Dues per Household",
                value=DEFAULT_VALUES["monthly_dues"],
                step=25,
                help=f"Based on Form 990: ${DEFAULT_VALUES['annual_dues_revenue'] / 1e6:.1f}M annual dues ÷ {DEFAULT_VALUES['initial_members']} members ÷ 12 months",
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

        # Program Revenues and Expenses
        with st.expander("🏊 Program Revenues & Expenses", expanded=True):
            st.markdown("""
            <div class="help-text">
            Enter monthly revenues and expenses for each program. These will grow at the program growth rate.
            </div>
            """, unsafe_allow_html=True)

            # Swim Lessons
            st.markdown("**🏊 Swim Lessons**")
            col1, col2 = st.columns(2)
            with col1:
                swim_lessons_revenue = st.number_input(
                    "Monthly Revenue",
                    value=DEFAULT_VALUES["swim_lessons_monthly_revenue"],
                    step=5000,
                    help="Average monthly revenue from swim lessons",
                    format="%d",
                    key="swim_rev"
                )
            with col2:
                swim_lessons_expenses = st.number_input(
                    "Monthly Expenses",
                    value=DEFAULT_VALUES["swim_lessons_monthly_expenses"],
                    step=5000,
                    help="Average monthly expenses for swim lessons",
                    format="%d",
                    key="swim_exp"
                )

            # Summer Camp
            st.markdown("**☀️ Summer Camp**")
            col1, col2 = st.columns(2)
            with col1:
                summer_camp_revenue = st.number_input(
                    "Monthly Revenue (averaged)",
                    value=DEFAULT_VALUES["summer_camp_monthly_revenue"],
                    step=5000,
                    help="Average monthly revenue from summer camp (seasonal revenue averaged over 12 months)",
                    format="%d",
                    key="camp_rev"
                )
            with col2:
                summer_camp_expenses = st.number_input(
                    "Monthly Expenses (averaged)",
                    value=DEFAULT_VALUES["summer_camp_monthly_expenses"],
                    step=5000,
                    help="Average monthly expenses for summer camp",
                    format="%d",
                    key="camp_exp"
                )

            # After School
            st.markdown("**📚 After School Program**")
            col1, col2 = st.columns(2)
            with col1:
                after_school_revenue = st.number_input(
                    "Monthly Revenue",
                    value=DEFAULT_VALUES["after_school_monthly_revenue"],
                    step=5000,
                    help="Average monthly revenue from after school program",
                    format="%d",
                    key="after_rev"
                )
            with col2:
                after_school_expenses = st.number_input(
                    "Monthly Expenses",
                    value=DEFAULT_VALUES["after_school_monthly_expenses"],
                    step=5000,
                    help="Average monthly expenses for after school program",
                    format="%d",
                    key="after_exp"
                )

            # Preschool
            st.markdown("**👶 Preschool**")
            col1, col2 = st.columns(2)
            with col1:
                preschool_revenue = st.number_input(
                    "Monthly Revenue",
                    value=DEFAULT_VALUES["preschool_monthly_revenue"],
                    step=5000,
                    help="Average monthly revenue from preschool",
                    format="%d",
                    key="pre_rev"
                )
            with col2:
                preschool_expenses = st.number_input(
                    "Monthly Expenses",
                    value=DEFAULT_VALUES["preschool_monthly_expenses"],
                    step=5000,
                    help="Average monthly expenses for preschool",
                    format="%d",
                    key="pre_exp"
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
        with st.expander("📈 Member Behavior Predictions", expanded=True):
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
                assessment_payment_mean = assessment_payment_mean / 100

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
                retention_mean = st.slider(
                    "% Who Will Stay",
                    50, 100, DEFAULT_VALUES["retention_mean"], 5,
                    help="What percentage of members will remain after assessment & dues increase?"
                )
                retention_mean = retention_mean / 100

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

        # Quick calculations display
        st.markdown("---")
        st.markdown("### 📊 Quick Math")
        expected_assessment = int(initial_members * assessment_payment_mean * assessment_amount)
        new_monthly_revenue = int(initial_members * retention_mean * monthly_dues * dues_increase_mean)

        # Calculate program net revenue
        program_net = (swim_lessons_revenue - swim_lessons_expenses +
                       summer_camp_revenue - summer_camp_expenses +
                       after_school_revenue - after_school_expenses +
                       preschool_revenue - preschool_expenses)

        total_monthly_revenue = new_monthly_revenue + program_net
        monthly_difference = total_monthly_revenue - monthly_expenses

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Expected from assessment:**  \n${expected_assessment:,}")
            st.markdown(f"**New monthly dues revenue:**  \n${new_monthly_revenue:,}")
            st.markdown(f"**Monthly program net revenue:**  \n${program_net:,}")

        with col2:
            st.markdown(f"**Total monthly revenue:**  \n${total_monthly_revenue:,}")
            st.markdown(f"**Current monthly expenses:**  \n${monthly_expenses:,}")
            deficit_color = "green" if monthly_difference >= 0 else "red"
            st.markdown(
                f"**Monthly surplus/deficit:**  \n<span style='color: {deficit_color}; font-weight: bold;'>${monthly_difference:,}</span>",
                unsafe_allow_html=True)

        # Run simulation button
        run_simulation_btn = st.button("🚀 Run Financial Projection", type="primary")

    # Main content area
    if run_simulation_btn:
        with st.spinner("Running simulations..."):
            # Prepare program data
            program_revenues = [swim_lessons_revenue, summer_camp_revenue,
                                after_school_revenue, preschool_revenue]
            program_expenses = [swim_lessons_expenses, summer_camp_expenses,
                                after_school_expenses, preschool_expenses]

            ending_reserves, reserve_trajectories, member_trajectories = run_simulation(
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
                expense_growth_rate,
                program_revenues,
                program_expenses,
                program_growth_rate,
                program_uncertainty
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
        program_net = (swim_lessons_revenue - swim_lessons_expenses +
                       summer_camp_revenue - summer_camp_expenses +
                       after_school_revenue - after_school_expenses +
                       preschool_revenue - preschool_expenses)

        st.markdown(f"""
        <div class="insight-box">
            <h4>💡 What This Means</h4>
            <ul>
                <li><strong>{(sum(1 for r in ending_reserves if r > danger_threshold) / len(ending_reserves) * 100):.1f}%</strong> chance 
                    of maintaining safe reserve levels (above ${danger_threshold:,})</li>
                <li>If the median scenario plays out, you'll {'gain' if p50 > starting_reserves else 'lose'} 
                    <strong>${abs(p50 - starting_reserves):,}</strong> over {months} months</li>
                <li>In the worst 5% of scenarios, reserves could drop to <strong>${p5:,}</strong></li>
                <li>Monthly revenue breakdown:
                    <ul>
                        <li>Dues revenue: <strong>${(initial_members * retention_mean * monthly_dues * dues_increase_mean):,.0f}</strong></li>
                        <li>Program net revenue: <strong>${program_net:,.0f}</strong></li>
                        <li>Total revenue: <strong>${(initial_members * retention_mean * monthly_dues * dues_increase_mean + program_net):,.0f}</strong></li>
                    </ul>
                </li>
                <li>Monthly expenses: <strong>${monthly_expenses:,.0f}</strong></li>
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

        with tab3:
            st.markdown("### Program Revenue & Expense Analysis")

            # Program summary table
            program_df = pd.DataFrame({
                'Program': ['Swim Lessons', 'Summer Camp', 'After School', 'Preschool'],
                'Monthly Revenue': [f"${r:,}" for r in [swim_lessons_revenue, summer_camp_revenue,
                                                        after_school_revenue, preschool_revenue]],
                'Monthly Expenses': [f"${e:,}" for e in [swim_lessons_expenses, summer_camp_expenses,
                                                         after_school_expenses, preschool_expenses]],
                'Net Income': [f"${r - e:,}" for r, e in [(swim_lessons_revenue, swim_lessons_expenses),
                                                          (summer_camp_revenue, summer_camp_expenses),
                                                          (after_school_revenue, after_school_expenses),
                                                          (preschool_revenue, preschool_expenses)]],
                'Margin %': [f"{((r - e) / r * 100):.1f}%" if r > 0 else "0.0%"
                             for r, e in [(swim_lessons_revenue, swim_lessons_expenses),
                                          (summer_camp_revenue, summer_camp_expenses),
                                          (after_school_revenue, after_school_expenses),
                                          (preschool_revenue, preschool_expenses)]]
            })

            st.dataframe(program_df, hide_index=True)

            # Program contribution chart
            program_names = ['Swim Lessons', 'Summer Camp', 'After School', 'Preschool']
            program_nets = [swim_lessons_revenue - swim_lessons_expenses,
                            summer_camp_revenue - summer_camp_expenses,
                            after_school_revenue - after_school_expenses,
                            preschool_revenue - preschool_expenses]

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
            📊 **Program Summary**: The four programs combined currently generate **${program_net:,}** in monthly net revenue.

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
                - Assessment collection: ${assessment_amount * initial_members * assessment_payment_mean:,.0f} expected
                - Dues increase: {(dues_increase_mean - 1) * 100:.0f}% boost to revenue
                - New member growth: {new_members_per_month * months} expected over {months} months
                - Program net revenue: ${program_net:,.0f}/month
                - Program growth: {program_growth_rate * 100:.0f}% annual increase
                """)

            with col2:
                st.markdown("**Risk Factors:**")
                st.markdown(f"""
                - Member attrition: {(1 - retention_mean) * 100:.0f}% expected loss
                - Expense growth: {expense_growth_rate * 100:.0f}% annual increase
                - Collection uncertainty: ±{assessment_payment_std * 100:.0f}% variation
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
                    'months_simulated': months,
                    'simulations_run': n_simulations
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
                <p class="help-text">Based on 2023 Form 990</p>
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
                <p>• Member retention impacts</p>
                <p>• Growth scenarios</p>
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
            3. **Estimate member reactions**:
               - What percentage will pay the assessment?
               - How many members might leave?
            4. **Run the simulation** to see thousands of possible outcomes
            5. **Review the results** to understand the risks and opportunities

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
            """)

        st.info("👈 **Adjust the settings in the sidebar and click 'Run Financial Projection' to begin**")


if __name__ == "__main__":
    main()