import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from google.cloud import bigquery
import google.auth

df = pd.read_csv("data/kiosk_demand_history.csv", parse_dates=["date"])
print(f"Loaded {len(df)} rows, {df['kiosk_id'].nunique()} kiosks")

FEATURES = [
    "foot_traffic_index", "is_promo", "temp_f", "precip_in",
    "dow", "month", "week_of_year",
    "rolling_7d", "rolling_28d", "lag_1d", "lag_7d",
    "inventory_on_hand", "restock_qty",
]
TARGET = "samples_dispensed"
df["is_promo"] = df["is_promo"].astype(int)
df_train = df[df["lag_7d"] > 0].copy()
X = df_train[FEATURES]
y = df_train[TARGET]

cutoff = df_train["date"].max() - pd.Timedelta(days=30)
train_mask = df_train["date"] <= cutoff
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

preds = model.predict(X_test)
print(f"\nMAE:  {mean_absolute_error(y_test, preds):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")

with open("models/xgboost_demand.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to models/xgboost_demand.pkl")

df_forecast = df_train[~train_mask].copy()
df_forecast["yhat_demand"] = model.predict(X_test).clip(0).round(0).astype(int)

depletion_results = []
for kid, group in df_forecast.groupby("kiosk_id"):
    group = group.sort_values("date")
    last = group["date"].max()
    avg_demand = int(group["yhat_demand"].mean())
    last_inv = group.iloc[-1]["inventory_on_hand"]
    inv = last_inv
    dep_date = None
    for d in range(1, 31):
        future_date = last + timedelta(days=d)
        inv -= avg_demand
        if inv <= 0 and dep_date is None:
            dep_date = future_date
            inv = 0
    days = (dep_date - last).days if dep_date else None
    depletion_results.append({
        "kiosk_id": kid,
        "location_name": group.iloc[0].get("location_name", ""),
        "as_of_date": last.strftime("%Y-%m-%d"),
        "predicted_depletion_date": dep_date.strftime("%Y-%m-%d") if dep_date else "No depletion",
        "days_until_depletion": days,
        "needs_refill_within_7d": (days <= 7) if days is not None else False,
        "needs_refill_within_14d": (days <= 14) if days is not None else False,
        "final_inventory_sim": max(0, int(inv)),
    })

depletion_df = pd.DataFrame(depletion_results)
depletion_df.to_csv("data/kiosk_depletion_forecast.csv", index=False)

fc_save = df_forecast[["kiosk_id","location_name","date","samples_dispensed","yhat_demand","inventory_on_hand"]].copy()
fc_save.to_csv("data/kiosk_demand_forecast.csv", index=False)

# ── PROPHET MODEL ───────────────────────────────────────────
print("\nTraining Prophet model...")
from prophet import Prophet

prophet_results = []
for kid, group in df_train.groupby("kiosk_id"):
    grp = group[["date", "samples_dispensed", "is_promo", "temp_f", "foot_traffic_index"]].copy()
    grp = grp.rename(columns={"date": "ds", "samples_dispensed": "y"})

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_regressor("is_promo")
    m.add_regressor("temp_f")
    m.add_regressor("foot_traffic_index")
    m.fit(grp[grp["ds"] <= cutoff])

    future = grp[grp["ds"] > cutoff][["ds", "is_promo", "temp_f", "foot_traffic_index"]].copy()
    if len(future) > 0:
        pred = m.predict(future)
        for _, row in pred.iterrows():
            prophet_results.append({
                "kiosk_id": kid,
                "date": row["ds"].strftime("%Y-%m-%d"),
                "prophet_demand": max(0, int(round(row["yhat"]))),
                "prophet_lower": max(0, int(round(row["yhat_lower"]))),
                "prophet_upper": max(0, int(round(row["yhat_upper"]))),
            })

prophet_df = pd.DataFrame(prophet_results)
prophet_df.to_csv("data/kiosk_prophet_forecast.csv", index=False)
print(f"Prophet forecast saved: {len(prophet_df)} rows")

# Merge with XGBoost forecast
fc_save["date"] = fc_save["date"].astype(str)
prophet_df["date"] = prophet_df["date"].astype(str)
merged = fc_save.merge(prophet_df, on=["kiosk_id", "date"], how="left")
merged.to_csv("data/kiosk_combined_forecast.csv", index=False)
print("Combined forecast saved to data/kiosk_combined_forecast.csv")

print("\nUploading forecasts to BigQuery...")
credentials, project = google.auth.default()
bq = bigquery.Client(credentials=credentials, project=project)
job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)

bq.load_table_from_dataframe(depletion_df, f"{project}.kiosk_data.depletion_forecast", job_config=job_config).result()

fc_up = fc_save.copy()
fc_up["date"] = fc_up["date"].astype(str)
bq.load_table_from_dataframe(fc_up, f"{project}.kiosk_data.demand_forecast", job_config=job_config).result()

print("✓ DONE — Model trained, forecasts uploaded!")
print(depletion_df)