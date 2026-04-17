import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from google.cloud import bigquery
import google.auth

np.random.seed(42)

# ── YOUR KIOSKS ─────────────────────────────────────────────
KIOSKS = [
    {"kiosk_id": "26354", "location_name": "Newark - Walmart",        "zip": "07102", "lat": 40.7357, "lon": -74.1724},
    {"kiosk_id": "26355", "location_name": "Edison - ShopRite",        "zip": "08817", "lat": 40.5187, "lon": -74.4121},
    {"kiosk_id": "26356", "location_name": "Parsippany - Stop & Shop", "zip": "07054", "lat": 40.8579, "lon": -74.4260},
    {"kiosk_id": "26357", "location_name": "Cherry Hill - Target",     "zip": "08002", "lat": 39.9347, "lon": -74.9942},
    {"kiosk_id": "26358", "location_name": "Hoboken - Whole Foods",    "zip": "07030", "lat": 40.7440, "lon": -74.0320},
    {"kiosk_id": "26359", "location_name": "Trenton - Costco",         "zip": "08608", "lat": 40.2171, "lon": -74.7429},
    {"kiosk_id": "26360", "location_name": "Princeton - Wegmans",      "zip": "08540", "lat": 40.3495, "lon": -74.6593},
    {"kiosk_id": "26361", "location_name": "Morristown - Acme",        "zip": "07960", "lat": 40.7968, "lon": -74.4815},
    {"kiosk_id": "26362", "location_name": "Atlantic City - BJs",      "zip": "08401", "lat": 39.3643, "lon": -74.4229},
    {"kiosk_id": "26363", "location_name": "Clifton - Sam's Club",     "zip": "07011", "lat": 40.8584, "lon": -74.1637},
]

# ── NOMINATIM: GET CITY/STATE ───────────────────────────────
def enrich_locations(kiosks):
    print("Calling Nominatim for city/state lookup...")
    geolocator = Nominatim(user_agent="freeosk_hackathon_2025")
    for k in kiosks:
        try:
            location = geolocator.reverse(f"{k['lat']}, {k['lon']}", exactly_one=True)
            if location and location.raw.get("address"):
                addr = location.raw["address"]
                k["city"] = addr.get("city") or addr.get("town") or addr.get("village", "Unknown")
                k["state"] = addr.get("state", "Unknown")
            else:
                k["city"], k["state"] = "Unknown", "Unknown"
        except Exception as e:
            print(f"  Error for {k['kiosk_id']}: {e}")
            k["city"], k["state"] = "Unknown", "Unknown"
        time.sleep(1.1)
        print(f"  {k['kiosk_id']} → {k['city']}, {k['state']}")
    return kiosks

KIOSKS = enrich_locations(KIOSKS)

# ── OPEN-METEO: GET WEATHER ─────────────────────────────────
def get_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()["daily"]
        return pd.DataFrame({
            "date": pd.to_datetime(data["time"]),
            "temp_f": data["temperature_2m_mean"],
            "precip_in": data["precipitation_sum"],
        })
    else:
        print(f"  Weather API error: {resp.status_code}")
        return pd.DataFrame()

END_DATE = "2026-04-16"
START_DATE = "2024-10-01"

# ── BUILD DAILY DEMAND DATA ─────────────────────────────────
print("\nGenerating daily demand data...")
all_rows = []

for kiosk in KIOSKS:
    kid = kiosk["kiosk_id"]
    print(f"  Kiosk {kid} ({kiosk['location_name']})...")

    weather = get_weather(kiosk["lat"], kiosk["lon"], START_DATE, END_DATE)
    if weather.empty:
        dates = pd.date_range(START_DATE, END_DATE)
        weather = pd.DataFrame({
            "date": dates,
            "temp_f": np.random.normal(50, 15, len(dates)).clip(10, 100),
            "precip_in": np.random.exponential(0.1, len(dates)).clip(0, 3),
        })

    for _, w in weather.iterrows():
        dt = w["date"]
        dow = dt.dayofweek
        month = dt.month
        base = np.random.randint(40, 80)
        if dow >= 5:
            base = int(base * np.random.uniform(1.2, 1.4))
        if month in [6, 7, 8]:
            base = int(base * 1.15)
        temp_mult = 1.0 + max(0, (w["temp_f"] - 60)) * 0.005
        rain_mult = 1.0 - min(0.3, w["precip_in"] * 0.15)
        base = int(base * temp_mult * rain_mult)
        foot_traffic = round(np.random.uniform(1.5, 4.5) + (0.5 if dow >= 5 else 0), 2)
        is_promo = np.random.random() < 0.15
        if is_promo:
            base = int(base * np.random.uniform(1.25, 1.50))
        samples_dispensed = max(0, int(base + np.random.normal(0, 5)))
        day_num = (dt - pd.Timestamp(START_DATE)).days
        restock_cycle = np.random.choice([5, 6, 7])
        if kid == "26354":  # Newark - critical, depletes ~April 20
            restock_cycle = np.random.choice([7, 8])
            restock_qty = np.random.randint(220, 280) if (day_num % restock_cycle == 0) else 0
        elif kid == "26358":  # Hoboken - warning, depletes ~April 27
            restock_cycle = np.random.choice([6, 7])
            restock_qty = np.random.randint(240, 300) if (day_num % restock_cycle == 0) else 0
        elif kid == "26363":  # Clifton - depleted already
            restock_cycle = np.random.choice([9, 10])
            restock_qty = np.random.randint(180, 240) if (day_num % restock_cycle == 0) else 0
        else:
            # Healthy kiosks - restock often with enough to stay above 150
            restock_qty = np.random.randint(250, 300) if (day_num % restock_cycle == 0) else 0

        all_rows.append({
            "kiosk_id": kid, "location_name": kiosk["location_name"],
            "date": dt.strftime("%Y-%m-%d"), "samples_dispensed": samples_dispensed,
            "restock_qty": restock_qty, "foot_traffic_index": foot_traffic,
            "is_promo": is_promo,
            "temp_f": round(w["temp_f"], 1) if pd.notna(w["temp_f"]) else 50.0,
            "precip_in": round(w["precip_in"], 2) if pd.notna(w["precip_in"]) else 0.0,
            "zip": kiosk["zip"], "city": kiosk["city"], "state": kiosk["state"],
        })
    time.sleep(0.5)

# ── COMPUTE INVENTORY ───────────────────────────────────────
print("Computing inventory levels...")
df = pd.DataFrame(all_rows)
df = df.sort_values(["kiosk_id", "date"]).reset_index(drop=True)

inv_records = []
for kid, group in df.groupby("kiosk_id"):
    inv = 300
    for idx, row in group.iterrows():
        inv += row["restock_qty"]
        dispensed = min(row["samples_dispensed"], max(0, inv))
        inv -= dispensed
        inv_records.append({"index": idx, "inventory_on_hand": inv, "samples_dispensed_adj": dispensed})

inv_df = pd.DataFrame(inv_records).set_index("index")
df["inventory_on_hand"] = inv_df["inventory_on_hand"]
df["samples_dispensed"] = inv_df["samples_dispensed_adj"]

# ── FEATURE ENGINEERING ─────────────────────────────────────
print("Adding features...")
df["date"] = pd.to_datetime(df["date"])
df["dow"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

for kid, group in df.groupby("kiosk_id"):
    mask = df["kiosk_id"] == kid
    df.loc[mask, "rolling_7d"] = df.loc[mask, "samples_dispensed"].rolling(7, min_periods=1).mean().round(1)
    df.loc[mask, "rolling_28d"] = df.loc[mask, "samples_dispensed"].rolling(28, min_periods=1).mean().round(1)
    df.loc[mask, "lag_1d"] = df.loc[mask, "samples_dispensed"].shift(1)
    df.loc[mask, "lag_7d"] = df.loc[mask, "samples_dispensed"].shift(7)
df = df.fillna(0)

# ── SAVE TO CSV ─────────────────────────────────────────────
df.to_csv("data/kiosk_demand_history.csv", index=False)
print(f"\nSaved {len(df)} rows to data/kiosk_demand_history.csv")

# ── UPLOAD TO BIGQUERY ──────────────────────────────────────
print("Uploading to BigQuery...")
credentials, project = google.auth.default()
bq_client = bigquery.Client(credentials=credentials, project=project)

table_id = f"{project}.kiosk_data.demand_history"
job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
df_upload = df.copy()
df_upload["date"] = df_upload["date"].dt.strftime("%Y-%m-%d")
job = bq_client.load_table_from_dataframe(df_upload, table_id, job_config=job_config)
job.result()
print(f"Uploaded {len(df_upload)} rows to BigQuery: {table_id}")
print("\n✓ DONE — Data generation complete!")