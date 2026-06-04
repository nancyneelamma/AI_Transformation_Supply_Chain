import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from modules.logger import get_logger
logger = get_logger(__name__)

# Demand Forecasting Parameters (used by ML module for demand forecasting and by OR module for inventory optimization)

SKU_PROFILES = {
    "SKU-RICE-5KG": {
        "base_demand": 320,
        "trend_per_day": 0.08,
        "seasonality_amp": 60,
        "weekend_boost": 1.35,
        "noise_std": 20,
    },
    "SKU-OIL-1L": {
        "base_demand": 210,
        "trend_per_day": 0.05,
        "seasonality_amp": 40,
        "weekend_boost": 1.20,
        "noise_std": 15,
    },
    "SKU-SUGAR-1KG": {
        "base_demand": 175,
        "trend_per_day": 0.03,
        "seasonality_amp": 30,
        "weekend_boost": 1.15,
        "noise_std": 12,
    },
}

def generate_demand_data(days: int = 730) -> pd.DataFrame:
    logger.debug(f"Generating {days} days of demand data for {len(SKU_PROFILES)} SKUs.")
    rng = np.random.default_rng(42)
    start = datetime(2023, 1, 1)
    records = []

    for sku, p in SKU_PROFILES.items():
        for i in range(days):
            d = start + timedelta(days=i)
            base = p["base_demand"] + i * p["trend_per_day"]
            seasonality = p["seasonality_amp"] * math.sin(i * (2 * math.pi / 365))
            weekend = p["weekend_boost"] if d.weekday() >= 5 else 1.0
            noise = rng.normal(0, p["noise_std"])
            demand = max(0, int((base + seasonality) * weekend + noise))
            records.append({"date": d, "sku": sku, "demand": demand})

    return pd.DataFrame(records)

# Inventory Optimization Parameters (used by OR module for optimization and by ML module for scenario-based forecasting)
def generate_inventory_data() -> pd.DataFrame:
    logger.debug("Generating inventory parameters and current stock levels.")
    rng = np.random.default_rng(42)
    demand_df = generate_demand_data()
    avg_daily = demand_df.groupby("sku")["demand"].mean().rename("avg_daily")

    rows = []
    meta = {
        "SKU-RICE-5KG":  {"unit_cost": 280, "ordering_cost": 1800, "holding_rate": 0.22, "lead_time_days": 7, "lead_time_std": 1.5, "service_level": 0.95},
        "SKU-OIL-1L":    {"unit_cost": 145, "ordering_cost": 1200, "holding_rate": 0.20, "lead_time_days": 5, "lead_time_std": 1.0, "service_level": 0.97},
        "SKU-SUGAR-1KG": {"unit_cost": 48,  "ordering_cost": 900,  "holding_rate": 0.18, "lead_time_days": 4, "lead_time_std": 0.8, "service_level": 0.90},
    }

    for sku, m in meta.items():
        annual_demand = int(avg_daily[sku] * 365)
        current_stock = int(rng.integers(int(annual_demand * 0.03), int(annual_demand * 0.08)))
        rows.append({
            "sku": sku,
            "annual_demand": annual_demand,
            "unit_cost": m["unit_cost"],
            "ordering_cost": m["ordering_cost"],
            "holding_rate": m["holding_rate"],
            "lead_time_days": m["lead_time_days"],
            "lead_time_std": m["lead_time_std"],
            "service_level": m["service_level"],
            "current_stock": current_stock,
        })

    return pd.DataFrame(rows)

# Supplier Risk Scenarios (used by OR module for risk simulation and by ML module for scenario-based forecasting)
SUPPLIER_SCENARIOS = [
    {
        "label": "🔴 Port Strike + Sole-Source + Financial Distress",
        "text": "URGENT: Port of LA Strike & Financial Update.\n\nDue to an unexpected dockworker strike, our shipments of microcontrollers are stalled for at least 14 days. Additionally, our Q3 filing shows our debt-to-equity ratio has hit 3.5 and we are pausing all factory upgrades. We are your sole supplier for this component. Please advise."
    },
    {
        "label": "🟠 Single-Region Flood + Low Inventory Buffer",
        "text": "Update from Warehouse Manager — Chennai Facility.\n\nCyclone-related flooding has shut our Chennai plant for an estimated 10–12 days. We hold approximately 8 days of finished-goods inventory at your nearest distribution centre. No alternate production site is currently qualified for this SKU family."
    },
    {
        "label": "🟡 Mild Delay — Alternative Supplier Available",
        "text": "Routine Update — Q4 Production Schedule.\n\nA planned maintenance window at our Pune facility will delay the next shipment by 4 days (from Nov 12 to Nov 16). Our secondary production line in Ahmedabad can partially cover demand. Current stock at your end appears sufficient per last week's report."
    },
    {
        "label": "✏️ Custom — paste your own supplier email",
        "text": "",
    }
]