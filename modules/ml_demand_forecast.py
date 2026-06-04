import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from modules.logger import get_logger
logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

LAG_DAYS = [7, 14, 21, 28]
ROLL_WINDOWS = [7, 14]

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Lag features
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df["demand"].shift(lag)

    # Rolling statistics
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = df["demand"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["demand"].shift(1).rolling(w).std()

    # Calendar and trend features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_index"] = (df["date"] - df["date"].min()).dt.days

    return df.dropna()

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("date", "demand", "sku")]

def train_and_forecast(df: pd.DataFrame, sku: str, forecast_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    logger.info(f"Starting ML demand forecast for SKU: {sku}, Horizon: {forecast_days} days.")
    sku_df = df[df["sku"] == sku][["date", "demand"]].copy()
    feat_df = _build_features(sku_df)
    feature_cols = _get_feature_cols(feat_df)
    logger.debug(f"Feature engineering complete. Generated {len(feature_cols)} features.")
    # Time-series train/validation split (Holdout last 30 days)
    cutoff = feat_df["date"].max() - timedelta(days=30)
    train = feat_df[feat_df["date"] <= cutoff]
    val = feat_df[feat_df["date"] > cutoff]

    X_train, y_train = train[feature_cols], train["demand"]
    X_val, y_val = val[feature_cols], val["demand"]

    model = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        early_stopping_rounds=30, random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Validation metrics
    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    mape = mean_absolute_percentage_error(y_val, val_preds) * 100
    logger.info(f"Holdout Validation complete - MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    logger.debug("Retraining XGBoost on full dataset for future predictions.")
    # Retrain on full data for future forecasting
    model_full = XGBRegressor(
        n_estimators=model.best_iteration + 1, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=42, verbosity=0,
    )
    model_full.fit(feat_df[feature_cols], feat_df["demand"])

    # Recursive multi-step forecast
    history_demands = list(sku_df["demand"].values)
    last_date = sku_df["date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    preds = []

    for fd in future_dates:
        tmp = pd.DataFrame({
            "date": list(sku_df["date"]) + [last_date + timedelta(days=k) for k in range(1, len(preds) + 1)],
            "demand": history_demands,
        })
        tmp_feat = _build_features(tmp)
        
        if tmp_feat.empty:
            preds.append(int(np.mean(history_demands[-14:])))
            history_demands.append(preds[-1])
            continue

        last_row = tmp_feat.iloc[[-1]][feature_cols].copy()
        last_row["day_of_week"] = fd.weekday()
        last_row["month"] = fd.month
        last_row["day_of_month"] = fd.day
        last_row["is_weekend"] = int(fd.weekday() >= 5)
        last_row["day_index"] = (fd - sku_df["date"].min()).days

        p = max(0, int(model_full.predict(last_row)[0]))
        preds.append(p)
        history_demands.append(p)
    logger.info(f"Forecast generation complete. Average predicted daily demand: {int(np.mean(preds))}")
    forecast_df = pd.DataFrame({"date": future_dates, "forecast": preds})
    hist_df = sku_df.tail(90).copy()

    metrics = {
        "mae": round(mae, 1),
        "mape": round(mape, 1),
        "best_n_estimators": model.best_iteration + 1,
        "avg_forecast": int(np.mean(preds)),
    }

    return hist_df, forecast_df, metrics