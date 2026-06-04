import sys
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

@pytest.fixture(scope="session")
def demand_df():
    from data.supply_chain_data import generate_demand_data
    return generate_demand_data(days=730)

@pytest.fixture(scope="session")
def inventory_df():
    from data.supply_chain_data import generate_inventory_data
    return generate_inventory_data()

@pytest.fixture(scope="session")
def optimization_results(demand_df, inventory_df):
    from modules.inventory_optimizer import optimize_inventory
    return optimize_inventory(inventory_df, demand_df)

class TestSupplyChainData:
    EXPECTED_SKUS = {"SKU-RICE-5KG", "SKU-OIL-1L", "SKU-SUGAR-1KG"}
    
    def test_demand_data_shape_and_quality(self, demand_df):
        assert demand_df.shape == (2190, 3)
        assert demand_df.isnull().sum().sum() == 0
        assert (demand_df["demand"] >= 0).all()
        assert set(demand_df["sku"].unique()) == self.EXPECTED_SKUS

    def test_demand_is_deterministic(self):
        from data.supply_chain_data import generate_demand_data
        df1 = generate_demand_data(days=730)
        df2 = generate_demand_data(days=730)
        pd.testing.assert_frame_equal(df1, df2)

    def test_inventory_data_validity(self, inventory_df):
        for col in ["unit_cost", "ordering_cost", "annual_demand", "current_stock"]:
            assert (inventory_df[col] > 0).all()

class TestMLDemandForecast:
    @pytest.fixture(scope="class")
    def rice_result(self, demand_df):
        from modules.ml_demand_forecast import train_and_forecast
        return train_and_forecast(demand_df, "SKU-RICE-5KG", forecast_days=30)

    def test_forecast_shapes_and_no_nulls(self, rice_result):
        hist_df, forecast_df, metrics = rice_result
        assert hist_df.shape == (90, 2)
        assert forecast_df.shape == (30, 2)
        assert forecast_df.isnull().sum().sum() == 0
        assert (forecast_df["forecast"] >= 0).all()

    def test_metrics_validity(self, rice_result):
        _, _, metrics = rice_result
        assert metrics["mae"] > 0
        assert 0 < metrics["mape"] < 20

    def test_deterministic_output(self, demand_df):
        from modules.ml_demand_forecast import train_and_forecast
        _, f1, m1 = train_and_forecast(demand_df, "SKU-RICE-5KG", 30)
        _, f2, m2 = train_and_forecast(demand_df, "SKU-RICE-5KG", 30)
        pd.testing.assert_frame_equal(f1, f2)

class TestInventoryOptimizer:
    def test_eoq_formula_correctness(self, optimization_results, inventory_df):
        for r in optimization_results:
            inv = inventory_df[inventory_df["sku"] == r.sku].iloc[0]
            H = inv["holding_rate"] * inv["unit_cost"]
            expected_eoq = int(np.sqrt(2 * inv["annual_demand"] * inv["ordering_cost"] / H))
            assert abs(r.eoq - expected_eoq) <= 1

    def test_reorder_point_greater_than_safety_stock(self, optimization_results):
        for r in optimization_results:
            assert r.reorder_point >= r.safety_stock

    def test_status_logic(self, inventory_df, demand_df):
        from modules.inventory_optimizer import optimize_inventory
        df_mod = inventory_df.copy()
        df_mod["current_stock"] = 0  # Force critical
        results = optimize_inventory(df_mod, demand_df)
        assert all(r.status == "CRITICAL" for r in results)

    def test_compute_savings_eoq_beats_naive(self, optimization_results):
        from modules.inventory_optimizer import compute_savings
        savings = compute_savings(optimization_results)
        assert savings["eoq_annual_cost"] < savings["naive_annual_cost"]
        assert savings["savings_pct"] > 0

class TestLLMSupplierRisk:
    def test_demo_path_returns_pydantic_model(self):
        """With no API key, _build_demo_response() should return a valid SupplierRiskAssessment."""
        from modules.llm_supplier_risk import analyze_supplier_risk, SupplierRiskAssessment
        with patch.dict(os.environ, {}, clear=True):
            result = analyze_supplier_risk("test supplier input")
            assert isinstance(result, SupplierRiskAssessment)
            assert result.risk_tier in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_demo_response_reflects_input(self):
        """Demo response should change based on the actual input text, not be hardcoded."""
        from modules.llm_supplier_risk import analyze_supplier_risk
        with patch.dict(os.environ, {}, clear=True):
            critical_result = analyze_supplier_risk(
                "sole-source supplier has declared bankruptcy and the port is on strike"
            )
            low_result = analyze_supplier_risk(
                "minor delay of 2 days, alternative supplier available, buffer stock sufficient"
            )
            tier_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
            assert tier_rank[critical_result.risk_tier] > tier_rank[low_result.risk_tier]

    def test_demo_confidence_is_low(self):
        """Demo mode should honestly report low confidence, not claim 82/100."""
        from modules.llm_supplier_risk import analyze_supplier_risk
        with patch.dict(os.environ, {}, clear=True):
            result = analyze_supplier_risk("some supplier update text")
            assert result.confidence_score <= 40, (
                f"Demo mode confidence should be <= 40, got {result.confidence_score}. "
                "High confidence without an API key is misleading."
            )

    def test_api_error_falls_back_gracefully(self):
        """On API error, should fall back to keyword-based response with error info in rationale."""
        from modules.llm_supplier_risk import analyze_supplier_risk, SupplierRiskAssessment
        with patch.dict(os.environ, {"XAI_API_KEY": "fake-key"}):
            result = analyze_supplier_risk("sole-source supplier struck by flood, no alternatives")
            assert isinstance(result, SupplierRiskAssessment)
            assert result.risk_tier in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
            assert any(
                phrase in result.rationale
                for phrase in ["API error", "Demo mode", "demo", "error", "fallback"]
            )

    def test_all_fields_populated(self):
        """Every field in SupplierRiskAssessment should be non-empty in demo mode."""
        from modules.llm_supplier_risk import analyze_supplier_risk
        with patch.dict(os.environ, {}, clear=True):
            result = analyze_supplier_risk("supplier update: port strike affecting shipments")
            assert result.risk_tier
            assert len(result.compounding_factors) > 0
            assert result.strategic_action
            assert result.rationale
            assert isinstance(result.confidence_score, int)
            assert result.what_would_change_assessment

class TestIntegration:
    def test_forecast_avg_close_to_historical_avg(self, demand_df):
        from modules.ml_demand_forecast import train_and_forecast
        for sku in demand_df["sku"].unique():
            hist_avg = demand_df[demand_df["sku"] == sku]["demand"].mean()
            _, forecast_df, _ = train_and_forecast(demand_df, sku, 30)
            ratio = forecast_df["forecast"].mean() / hist_avg
            assert 0.5 < ratio < 2.0