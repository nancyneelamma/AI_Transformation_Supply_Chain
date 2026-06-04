import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass
from modules.logger import get_logger
logger = get_logger(__name__)
@dataclass
class InventoryResult:
    sku: str
    avg_daily_demand: float
    daily_demand_std: float
    unit_cost: float
    ordering_cost: float
    holding_rate: float
    lead_time_days: float
    service_level: float
    current_stock: int
    annual_demand: int
    eoq: int = 0
    safety_stock: int = 0
    reorder_point: int = 0
    order_cycle_days: float = 0.0
    total_annual_cost: float = 0.0
    days_of_supply: float = 0.0
    status: str = "OK"

def optimize_inventory(inventory_df: pd.DataFrame, demand_df: pd.DataFrame) -> list[InventoryResult]:
    # Vectorized Demand Aggregation
    logger.info("Starting inventory optimization (EOQ, Safety Stock, Reorder Point).")
    demand_stats = demand_df.groupby('sku')['demand'].agg(
        avg_daily_demand='mean', daily_demand_std='std'
    ).reset_index()

    # Merge Inventory and Demand
    df = inventory_df.merge(demand_stats, on='sku')

    # Vectorized Math Operations (Using pure Pandas/Numpy instead of manual loops)
    holding_cost_per_unit = df['holding_rate'] * df['unit_cost']

    # EOQ = sqrt(2DS/H)
    df['eoq'] = np.maximum(1, np.sqrt((2 * df['annual_demand'] * df['ordering_cost']) / holding_cost_per_unit)).astype(int)

    # Safety Stock = Z * std * sqrt(L)
    z_scores = norm.ppf(df['service_level'])
    df['safety_stock'] = np.maximum(0, z_scores * np.sqrt(df['lead_time_days'] * df['daily_demand_std']**2 +
                         df['avg_daily_demand']**2 * df['lead_time_std']**2)).astype(int)

    # Reorder Point
    df['reorder_point'] = (df['avg_daily_demand'] * df['lead_time_days']).astype(int) + df['safety_stock']

    # Cycle Metrics
    df['order_cycle_days'] = np.where(df['avg_daily_demand'] > 0, (df['eoq'] / df['avg_daily_demand']).round(1), 0.0)
    df['days_of_supply'] = np.where(df['avg_daily_demand'] > 0, (df['current_stock'] / df['avg_daily_demand']).round(1), 0.0)

    # Cost Calculations
    order_cost_pa = (df['annual_demand'] / df['eoq']) * df['ordering_cost']
    holding_cost_pa = ((df['eoq'] / 2) + df['safety_stock']) * holding_cost_per_unit
    df['total_annual_cost'] = (order_cost_pa + holding_cost_pa).round(0)

    # Status Logic
    conditions = [
        df['current_stock'] <= df['safety_stock'],
        df['current_stock'] <= df['reorder_point']
    ]
    df['status'] = np.select(conditions, ["CRITICAL", "REORDER_NOW"], default="OK")

    # Sort Priority
    df['priority'] = df['status'].map({"CRITICAL": 0, "REORDER_NOW": 1, "OK": 2})
    df = df.sort_values('priority')
    logger.info(f"Inventory optimization complete for {len(df)} SKUs.")
    # Convert back to dataclass objects to ensure the Streamlit UI (app.py) doesn't break
    return [InventoryResult(**row) for row in df.drop(columns=['priority', 'lead_time_std']).to_dict(orient='records')]

def results_to_dataframe(results: list[InventoryResult]) -> pd.DataFrame:
    rows = []
    status_emoji = {"OK": "🟢", "REORDER_NOW": "🟡", "CRITICAL": "🔴"}
    for r in results:
        rows.append({
            "Status": f"{status_emoji[r.status]} {r.status}",
            "SKU": r.sku,
            "Avg Daily Demand": f"{r.avg_daily_demand:.0f} units",
            "EOQ (Order Qty)": f"{r.eoq:,} units",
            "Safety Stock": f"{r.safety_stock:,} units",
            "Reorder Point": f"{r.reorder_point:,} units",
            "Current Stock": f"{r.current_stock:,} units",
            "Days of Supply": f"{r.days_of_supply} days",
            "Order Cycle": f"{r.order_cycle_days} days",
            "Annual Cost (₹)": f"₹{r.total_annual_cost:,.0f}",
        })
    return pd.DataFrame(rows)

def compute_savings(results: list[InventoryResult]) -> dict:
    logger.debug("Computing financial savings vs naive monthly policy.")
    eoq_total = sum(r.total_annual_cost for r in results)
    naive_total = 0.0
    for r in results:
        naive_q = max(1, int(r.annual_demand / 12))  # order once a month naive policy
        holding = (naive_q / 2 + r.safety_stock) * r.holding_rate * r.unit_cost
        ordering = (r.annual_demand / naive_q) * r.ordering_cost
        naive_total += holding + ordering

    savings_pct = (naive_total - eoq_total) / naive_total * 100
    savings_abs = naive_total - eoq_total
    logger.info(f"Savings computed: ₹{savings_abs:,.0f} ({savings_pct:.1f}%)")
    return {
        "eoq_annual_cost": round(eoq_total, 0),
        "naive_annual_cost": round(naive_total, 0),
        "savings_pct": round(savings_pct, 1),
        "savings_inr": round(savings_abs, 0),
    }