import os
import pandas as pd
import streamlit as st
from modules.logger import get_logger
logger = get_logger(__name__)
from data.supply_chain_data import generate_demand_data, generate_inventory_data, SUPPLIER_SCENARIOS
from modules.ml_demand_forecast import train_and_forecast
from modules.inventory_optimizer import optimize_inventory, results_to_dataframe, compute_savings
from modules.llm_supplier_risk import analyze_supplier_risk

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="Supply Chain AI Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_data(show_spinner="Generating realistic synthetic supply chain data...")
def load_data():
    demand_df = generate_demand_data(days=730)
    inventory_df = generate_inventory_data()
    logger.info(f"Data generation complete. Demand shape: {demand_df.shape}, Inventory shape: {inventory_df.shape}")
    return demand_df, inventory_df

demand_df, inventory_df = load_data()
SKU_LIST = demand_df["sku"].unique().tolist()


st.markdown("""
<style>
.change-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid #86efac;
    border-left: 4px solid #16a34a;
    border-radius: 10px;
    padding: 18px 22px;
    margin-top: 4px;
}
.change-card-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #15803d;
    margin-bottom: 6px;
}
.change-card-text {
    font-size: 0.97rem;
    color: #166534;
    line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Enterprise Supply Chain AI")
st.markdown(
    "Demonstrating the synergy of **Classical ML (XGBoost)** · "
    "**Operations Research (EOQ)** · **Agentic LLM (Grok)**."
)

tab1, tab2, tab3 = st.tabs([
    "📊 1. Demand Forecasting",
    "📦 2. Inventory Optimization",
    "⚠️ 3. Supplier Risk",
])

# Classical ML – Demand Forecasting
with tab1:
    st.subheader("Time-Series Demand Prediction")

    selected_sku = st.selectbox("Select SKU", SKU_LIST)
    forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)
    run_forecast = st.button("▶ Run Forecast", type="primary", key="btn_forecast")

    if run_forecast:
        logger.info(f"UI Trigger: Running XGBoost forecast for SKU: {selected_sku}, Horizon: {forecast_days} days")
        with st.spinner(f"Training XGBoost model on {selected_sku}..."):
            hist_df, forecast_df, metrics = train_and_forecast(demand_df, selected_sku, forecast_days)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Holdout MAE", f"{metrics['mae']} units")
        m2.metric("Holdout MAPE", f"{metrics['mape']}%")
        m3.metric("Best Estimators", metrics["best_n_estimators"])
        m4.metric("Avg Forecasted Demand", f"{metrics['avg_forecast']} units/day")

        hist_plot = hist_df.rename(columns={"demand": "Historical Actuals"}).set_index("date")[["Historical Actuals"]]
        fcast_plot = forecast_df.rename(columns={"forecast": "XGBoost Forecast"}).set_index("date")[["XGBoost Forecast"]]
        chart_df = hist_plot.join(fcast_plot, how="outer")
        st.line_chart(chart_df)
        st.caption("Displaying last 90 days of history + recursive forecast. MAE/MAPE measured strictly on a 30-day holdout to prevent data leakage.")

#  – Inventory Optimization
with tab2:
    st.subheader("Operations Research")

    if st.button("▶ Run Inventory Optimization", type="primary", key="btn_inv"):
        logger.info("UI Trigger: Running Inventory Optimization (EOQ/ROP).")
        with st.spinner("Computing mathematical EOQ and reorder limits..."):
            results = optimize_inventory(inventory_df, demand_df)
            savings = compute_savings(results)

        s1, s2, s3 = st.columns(3)
        s1.metric(
            "EOQ Annual Cost",
            f"₹{savings['eoq_annual_cost']:,.0f}",
            delta=f"−₹{savings['savings_inr']:,.0f} vs monthly ordering",
            delta_color="normal",
        )
        s2.metric("vs Naive Monthly Policy", f"₹{savings['naive_annual_cost']:,.0f}")
        s3.metric("Cost Reduction", f"{savings['savings_pct']}%")

        st.markdown("#### SKU-Level Results")
        results_df = results_to_dataframe(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("#### Detailed Action Plan")
        for r in results:
            status_icon = {"OK": "🟢", "REORDER_NOW": "🟡", "CRITICAL": "🔴"}[r.status]
            with st.expander(f"{status_icon} {r.sku} — {r.status}"):
                d1, d2, d3 = st.columns(3)
                d1.metric("EOQ (Optimal Order)", f"{r.eoq:,} units")
                d1.metric("Order Cycle", f"{r.order_cycle_days} days")
                d2.metric("Safety Stock", f"{r.safety_stock:,} units")
                d2.metric("Reorder Point", f"{r.reorder_point:,} units")
                d3.metric("Current On-Hand", f"{r.current_stock:,} units")
                d3.metric("Days of Supply", f"{r.days_of_supply} days")

                if r.status == "CRITICAL":
                    st.error(f"⚠️ Current stock ({r.current_stock:,}) is critically below safety stock ({r.safety_stock:,}). Emergency order required immediately.")
                elif r.status == "REORDER_NOW":
                    st.warning(f"📋 Stock ({r.current_stock:,}) has fallen below the reorder point ({r.reorder_point:,}). Trigger purchase order for {r.eoq:,} units.")

# LLM – Supplier Risk Agent
with tab3:
    st.subheader("Non-Trivial Supplier Risk Judgment")

    scenario_labels = [s["label"] for s in SUPPLIER_SCENARIOS]
    selected_label = st.selectbox("Choose a scenario or paste your own:", scenario_labels)
    selected_scenario = next(s for s in SUPPLIER_SCENARIOS if s["label"] == selected_label)

    user_input = st.text_area(
        "Incoming Supplier Email / Intelligence Report:",
        value=selected_scenario["text"],
        height=140,
    )

    if st.button("▶ Run Risk Agent Assessment", type="primary", key="btn_llm"):
        if not user_input.strip():
            logger.warning("UI Trigger: Risk assessment aborted — empty input.")
            st.error("Please enter a supplier update to assess.")
        else:
            logger.info(f"UI Trigger: Running LLM Supplier Risk Agent on scenario: {selected_label}")
            with st.spinner("Grok Agent evaluating compounding risk factors..."):
                result = analyze_supplier_risk(user_input)

            tier_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(result.risk_tier, "⚪")

            c1, c2, c3 = st.columns(3)
            if result.risk_tier == "CRITICAL":
                c1.error(f"{tier_color} **Risk Tier:** {result.risk_tier}")
            elif result.risk_tier == "HIGH":
                c1.warning(f"{tier_color} **Risk Tier:** {result.risk_tier}")
            else:
                c1.success(f"{tier_color} **Risk Tier:** {result.risk_tier}")

            c2.info(f"⚡ **Strategic Action:** `{result.strategic_action}`")
            c3.metric("Confidence Score", f"{result.confidence_score}/100")

            st.markdown("#### 🔗 Compounding Factors Identified")
            for i, factor in enumerate(result.compounding_factors, 1):
                st.markdown(f"**{i}.** {factor}")

            st.markdown("#### 🧠 Agent Rationale")
            st.info(result.rationale)

            st.markdown("#### 🔄 What Would Change This Assessment?")
            st.markdown(
                f"""<div class="change-card">
                    <div class="change-card-label">Pivot Condition</div>
                    <div class="change-card-text">{result.what_would_change_assessment}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            with st.expander("📋 Full Structured Output (JSON)"):
                st.json(result.model_dump())