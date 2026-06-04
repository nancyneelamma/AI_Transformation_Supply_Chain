# Enterprise Supply Chain AI

Built for the Theseus AI Lab CXO Assessment. Three tabs, three AI approaches, one supply chain problem.

## Video Link

https://www.loom.com/share/f8dfdd755ef743cebe05ebd60260bd2e

---

## What it does

**Tab 1 : Demand Forecasting**
Trains an XGBoost model on 2 years of synthetic demand data and forecasts the next 7–60 days per SKU. Validated on a 30-day holdout it never saw during training. SKU-RICE-5KG hits 5.3% MAPE.

**Tab 2 : Inventory Optimization**
Runs EOQ math to find the optimal order quantity, safety stock, and reorder point for each SKU. Cuts annual inventory cost by 40.7% (₹2,49,575) compared to a naive monthly ordering policy.

**Tab 3 : Supplier Risk Agent**
Paste any supplier email. Grok reads it and returns a structured risk assessment — tier, compounding factors, a recommended action, and the exact piece of information that would change the call. No API key? It falls back to a keyword-based analyser that at least reads your actual input rather than showing a hardcoded example.

---

## Data

No external datasets needed. Everything is generated at startup by `data/supply_chain_data.py`.

It creates 730 days of daily demand across 3 SKUs (Rice 5KG, Oil 1L, Sugar 1KG) with trend, annual seasonality, weekend spikes, and realistic noise baked in. Inventory parameters — unit costs, lead times, service levels — are hardcoded per SKU to reflect plausible FMCG values. The same random seed is used every time so results are deterministic and tests are reproducible.

## Running it

```bash
git clone <your-repo-link>
cd supply-chain-analytics
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```

Tabs 1 and 2 work with no API key. For Tab 3, create a `.env` file:
```
XAI_API_KEY=your_key_here
```

---

## Tests

```bash
cd test && pytest test_supply_chain.py -v
```

13 tests, all passing — covers data quality, EOQ formula correctness, ML accuracy, and LLM fallback behaviour.

---

## Stack

Streamlit · XGBoost · NumPy/SciPy · Grok (xAI) · Pydantic · pytest
