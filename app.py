import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from google.oauth2 import service_account
from openai import OpenAI
import google.auth

ORANGE = "#F37021"
BLUE = "#0072BA"
YELLOW = "#FFC20E"
SKY = "#52BFEE"
GREY = "#76777A"

KIOSK_COORDS = {
    "26354": (40.7357, -74.1724), "26355": (40.5187, -74.4121),
    "26356": (40.8579, -74.4260), "26357": (39.9347, -74.9942),
    "26358": (40.7440, -74.0320), "26359": (40.2171, -74.7429),
    "26360": (40.3495, -74.6593), "26361": (40.7968, -74.4815),
    "26362": (39.3643, -74.4229), "26363": (40.8584, -74.1637),
}
WEATHER_CODES = {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 71: "Light snow",
    73: "Snow", 75: "Heavy snow", 80: "Light showers", 81: "Showers", 82: "Heavy showers",
    95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Heavy thunderstorm"}

st.set_page_config(page_title="Freeosk Kiosk Intelligence", page_icon="🧃", layout="wide")

@st.cache_resource
def get_bq():
    try:
        creds = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
        return bigquery.Client(credentials=creds)
    except Exception:
        credentials, project = google.auth.default()
        return bigquery.Client(credentials=credentials, project=project)

@st.cache_resource
def get_hf():
    return OpenAI(base_url="https://router.huggingface.co/v1", api_key=st.secrets["huggingface"]["hf_token"])

bq = get_bq()
hf = get_hf()

@st.cache_data(ttl=600)
def load_data():
    try:
        h = bq.query("SELECT * FROM `freeosk-hackathon.kiosk_data.demand_history` ORDER BY kiosk_id, date").to_dataframe()
        f = bq.query("SELECT * FROM `freeosk-hackathon.kiosk_data.demand_forecast` ORDER BY kiosk_id, date").to_dataframe()
        d = bq.query("SELECT * FROM `freeosk-hackathon.kiosk_data.depletion_forecast` ORDER BY kiosk_id").to_dataframe()
        return h, f, d
    except Exception:
        h = pd.read_csv("data/kiosk_demand_history.csv", parse_dates=["date"])
        f = pd.read_csv("data/kiosk_demand_forecast.csv")
        d = pd.read_csv("data/kiosk_depletion_forecast.csv")
        return h, f, d

@st.cache_data(ttl=600)
def load_combined():
    try:
        return pd.read_csv("data/kiosk_combined_forecast.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,precipitation,weather_code&temperature_unit=fahrenheit&precipitation_unit=inch&timezone=America/New_York"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("current", {})
    except Exception:
        pass
    return {}

history, forecast, depletion = load_data()
combined = load_combined()

st.title("🧃 Freeosk Kiosk Intelligence")
st.caption("AI-Powered Sample Depletion Forecasting · GCP BigQuery · Hugging Face LLM · Open-Meteo Weather")

with st.sidebar:
    st.header("🏪 Select Kiosk")
    kiosk_opts = history[["kiosk_id", "location_name"]].drop_duplicates()
    kiosk_opts["label"] = kiosk_opts["kiosk_id"] + " — " + kiosk_opts["location_name"]
    selected_label = st.selectbox("Kiosk", kiosk_opts["label"].tolist(), label_visibility="collapsed")
    selected_kiosk = selected_label.split(" — ")[0]
    st.divider()
    st.header("Fleet Totals")
    st.metric("Total Samples Dispensed", f"{int(history['samples_dispensed'].sum()):,}")
    st.metric("Avg Daily (All Kiosks)", f"{int(history.groupby('date')['samples_dispensed'].sum().mean()):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Executive Summary", "🔍 Kiosk Deep Dive", "🎯 Model Comparison", "🌤️ Live Weather", "💬 AI Assistant"])


# ══════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════
with tab1:
    n7 = depletion[depletion["needs_refill_within_7d"] == True]
    n14 = depletion[depletion["needs_refill_within_14d"] == True]
    healthy = len(depletion) - len(n14)
    avg_inv = int(depletion["final_inventory_sim"].mean())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔴 Critical (≤7 Days)", len(n7))
    k2.metric("🟡 Warning (≤14 Days)", len(n14))
    k3.metric("✅ Healthy Kiosks", healthy)
    k4.metric("📦 Avg Remaining Stock", f"{avg_inv:,}")

    # Show depleted kiosks
    depleted = depletion[(depletion["final_inventory_sim"] <= 0) & (depletion["predicted_depletion_date"] != "No depletion")]
    for _, row in depleted.iterrows():
        st.error(f"**DEPLETED:** {row['location_name']} — Ran out on **{row['predicted_depletion_date']}** · Needs immediate restock")

    # Show critical (future depletion within 7 days, still has stock)
    critical = depletion[(depletion["needs_refill_within_7d"] == True) & (depletion["final_inventory_sim"] > 0)]
    for _, row in critical.iterrows():
        st.error(f"**CRITICAL:** {row['location_name']} — Depletes **{row['predicted_depletion_date']}** · Remaining: **{int(row['final_inventory_sim'])}** units")

    # Show warning (future depletion within 14 days, still has stock, not already critical)
    warning = depletion[(depletion["needs_refill_within_14d"] == True) & (depletion["final_inventory_sim"] > 0) & (depletion["needs_refill_within_7d"] == False)]
    for _, row in warning.iterrows():
        st.warning(f"**WARNING:** {row['location_name']} — Depletes **{row['predicted_depletion_date']}** · Remaining: **{int(row['final_inventory_sim'])}** units")

    if len(depleted) == 0 and len(critical) == 0 and len(warning) == 0:
        st.success("All kiosks have sufficient stock. No alerts.")

    st.subheader("📋 Fleet Depletion Overview")
    dep_show = depletion.rename(columns={"kiosk_id": "Kiosk ID", "location_name": "Location", "predicted_depletion_date": "Depletion Date", "days_until_depletion": "Days Left", "needs_refill_within_7d": "≤7d Risk", "needs_refill_within_14d": "≤14d Risk", "final_inventory_sim": "Remaining Stock"})
    st.dataframe(dep_show[["Kiosk ID", "Location", "Depletion Date", "Days Left", "≤7d Risk", "≤14d Risk", "Remaining Stock"]], hide_index=True)

    st.subheader("📦 Fleet Inventory Snapshot")
    inv_sorted = depletion.sort_values("final_inventory_sim")
    bar_colors = [ORANGE if row["needs_refill_within_7d"] else (YELLOW if row["needs_refill_within_14d"] else BLUE) for _, row in inv_sorted.iterrows()]
    fig_fleet = go.Figure(go.Bar(x=inv_sorted["final_inventory_sim"], y=inv_sorted["location_name"], orientation="h", marker_color=bar_colors, text=inv_sorted["final_inventory_sim"], textposition="outside"))
    fig_fleet.update_layout(height=400, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=40, t=10, b=40))
    st.plotly_chart(fig_fleet, use_container_width=True)

    st.subheader("📅 Demand Heatmap — Kiosk × Day of Week")
    hm = history.copy()
    hm["day_name"] = hm["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
    pivot = hm.pivot_table(values="samples_dispensed", index="location_name", columns="day_name", aggfunc="mean")
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])
    z_vals = [[float(v) if pd.notna(v) else 0.0 for v in row] for row in pivot.values]
    t_vals = [[str(int(float(v))) if pd.notna(v) else "0" for v in row] for row in pivot.values]
    fig_heat = go.Figure(go.Heatmap(z=z_vals, x=list(pivot.columns), y=list(pivot.index), colorscale=[[0, "#FFFFFF"], [0.3, SKY], [0.6, YELLOW], [1, ORANGE]], text=t_vals, texttemplate="%{text}", textfont=dict(size=12), showscale=False))
    fig_heat.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=40))
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2: KIOSK DEEP DIVE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"🏪 {selected_label}")
    kh = history[history["kiosk_id"].astype(str) == str(selected_kiosk)].sort_values("date")
    this_k = depletion[depletion["kiosk_id"].astype(str) == str(selected_kiosk)]

    if len(kh) == 0:
        st.warning("No data for this kiosk.")
    else:
        avg_demand = int(kh["samples_dispensed"].mean())
        peak = int(kh["samples_dispensed"].max())
        promo_days_df = kh[kh["is_promo"] == True]
        non_promo_df = kh[kh["is_promo"] == False]
        promo_lift = 0
        if len(promo_days_df) > 0 and len(non_promo_df) > 0:
            promo_lift = int(((promo_days_df["samples_dispensed"].mean() / non_promo_df["samples_dispensed"].mean()) - 1) * 100)
        dep_date = "Healthy"
        if not this_k.empty:
            dep_date = this_k.iloc[0]["predicted_depletion_date"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Daily Demand", avg_demand)
        m2.metric("Peak Single Day", peak)
        m3.metric("Promo Lift", f"+{promo_lift}%")
        m4.metric("Depletion Forecast", dep_date)

        st.subheader("📈 Demand Trend")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=kh["date"], y=kh["samples_dispensed"], mode="lines", name="Daily Demand", line=dict(color=SKY, width=1.5), fill="tozeroy", fillcolor="rgba(82,191,238,0.1)"))
        fig1.add_trace(go.Scatter(x=kh["date"], y=kh["rolling_7d"], mode="lines", name="7-Day Avg", line=dict(color=BLUE, width=2.5)))
        fig1.add_trace(go.Scatter(x=kh["date"], y=kh["rolling_28d"], mode="lines", name="28-Day Avg", line=dict(color=GREY, width=1.5, dash="dot")))
        promo_pts = kh[kh["is_promo"] == True]
        if len(promo_pts) > 0:
            fig1.add_trace(go.Scatter(x=promo_pts["date"], y=promo_pts["samples_dispensed"], mode="markers", name="Promo Day", marker=dict(color=ORANGE, size=10, symbol="star")))
        fig1.update_layout(height=420, legend=dict(orientation="h", y=1.05), margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📦 Inventory Level")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=kh["date"], y=kh["inventory_on_hand"], mode="lines", name="Inventory", line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(0,114,186,0.08)"))
        fig2.add_hline(y=100, line_dash="dash", line_color=ORANGE, annotation_text="Danger Zone (100)")
        fig2.add_hline(y=0, line_dash="dot", line_color="#E53E3E", annotation_text="Stockout")
        restocks = kh[kh["restock_qty"] > 0]
        if len(restocks) > 0:
            fig2.add_trace(go.Scatter(x=restocks["date"], y=restocks["inventory_on_hand"], mode="markers", name="Restock", marker=dict(color="#38A169", size=8, symbol="triangle-up")))
        fig2.update_layout(height=380, legend=dict(orientation="h", y=1.05), margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🌡️ Demand Drivers")
        d1, d2 = st.columns(2)
        with d1:
            fig_temp = px.scatter(kh, x="temp_f", y="samples_dispensed", color="is_promo", color_discrete_map={True: ORANGE, False: SKY}, trendline="ols", labels={"temp_f": "Temp (°F)", "samples_dispensed": "Samples", "is_promo": "Promo"})
            fig_temp.update_layout(title="Temperature vs Demand", height=350)
            st.plotly_chart(fig_temp, use_container_width=True)
        with d2:
            fig_traffic = px.scatter(kh, x="foot_traffic_index", y="samples_dispensed", color="is_promo", color_discrete_map={True: ORANGE, False: SKY}, trendline="ols", labels={"foot_traffic_index": "Traffic", "samples_dispensed": "Samples", "is_promo": "Promo"})
            fig_traffic.update_layout(title="Foot Traffic vs Demand", height=350)
            st.plotly_chart(fig_traffic, use_container_width=True)
        d3, d4 = st.columns(2)
        with d3:
            fig_rain = px.scatter(kh, x="precip_in", y="samples_dispensed", color_discrete_sequence=[BLUE], trendline="ols", labels={"precip_in": "Rain (in)", "samples_dispensed": "Samples"})
            fig_rain.update_layout(title="Rain vs Demand", height=350)
            st.plotly_chart(fig_rain, use_container_width=True)
        with d4:
            dow_k = kh.groupby("dow")["samples_dispensed"].mean().reset_index()
            dow_k["day"] = dow_k["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
            fig_dow = go.Figure(go.Bar(x=dow_k["day"], y=dow_k["samples_dispensed"], marker_color=[BLUE]*5 + [ORANGE]*2, text=dow_k["samples_dispensed"].round(0).astype(int), textposition="outside"))
            fig_dow.update_layout(title="Demand by Day of Week", height=350, yaxis_title="Avg Samples")
            st.plotly_chart(fig_dow, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: MODEL COMPARISON (XGBoost vs Prophet)
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎯 Model Comparison: XGBoost vs Prophet")

    kf = forecast[forecast["kiosk_id"].astype(str) == str(selected_kiosk)].sort_values("date")
    kc = combined[combined["kiosk_id"].astype(str) == str(selected_kiosk)].copy() if not combined.empty else pd.DataFrame()

    if not kf.empty:
        mae_xgb = abs(kf["samples_dispensed"] - kf["yhat_demand"]).mean()
        avg_actual = kf["samples_dispensed"].mean()
        accuracy_xgb = max(0, (1 - mae_xgb / avg_actual) * 100) if avg_actual > 0 else 0

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("#### XGBoost")
            st.metric("Accuracy", f"{accuracy_xgb:.1f}%")
            st.metric("MAE", f"{mae_xgb:.1f} samples/day")
            direction_xgb = ((kf["yhat_demand"].diff() > 0) == (kf["samples_dispensed"].diff() > 0)).mean() * 100
            st.metric("Direction Accuracy", f"{direction_xgb:.0f}%")

        with a2:
            if not kc.empty and "prophet_demand" in kc.columns:
                kc_valid = kc.dropna(subset=["prophet_demand"])
                if not kc_valid.empty:
                    mae_prophet = abs(kc_valid["samples_dispensed"] - kc_valid["prophet_demand"]).mean()
                    avg_actual_p = kc_valid["samples_dispensed"].mean()
                    accuracy_prophet = max(0, (1 - mae_prophet / avg_actual_p) * 100) if avg_actual_p > 0 else 0
                    direction_prophet = ((kc_valid["prophet_demand"].diff() > 0) == (kc_valid["samples_dispensed"].diff() > 0)).mean() * 100
                    st.markdown("#### Prophet")
                    st.metric("Accuracy", f"{accuracy_prophet:.1f}%")
                    st.metric("MAE", f"{mae_prophet:.1f} samples/day")
                    st.metric("Direction Accuracy", f"{direction_prophet:.0f}%")
                else:
                    st.markdown("#### Prophet")
                    st.warning("No Prophet data for this kiosk")
            else:
                st.markdown("#### Prophet")
                st.warning("Prophet forecast not available")

        # Winner callout
        if not kc.empty and "prophet_demand" in kc.columns:
            kc_valid = kc.dropna(subset=["prophet_demand"])
            if not kc_valid.empty:
                mae_prophet = abs(kc_valid["samples_dispensed"] - kc_valid["prophet_demand"]).mean()
                if mae_xgb < mae_prophet:
                    st.success(f"**XGBoost wins** with MAE {mae_xgb:.1f} vs Prophet's {mae_prophet:.1f} — {((mae_prophet - mae_xgb) / mae_prophet * 100):.0f}% more accurate.")
                else:
                    st.success(f"**Prophet wins** with MAE {mae_prophet:.1f} vs XGBoost's {mae_xgb:.1f} — {((mae_xgb - mae_prophet) / mae_xgb * 100):.0f}% more accurate.")

        # Combined chart
        st.subheader("📈 Forecast Comparison")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(x=kf["date"], y=kf["samples_dispensed"], name="Actual", marker_color=SKY, opacity=0.5))
        fig_comp.add_trace(go.Scatter(x=kf["date"], y=kf["yhat_demand"], mode="lines+markers", name="XGBoost", line=dict(color=ORANGE, width=2.5), marker=dict(size=5)))

        if not kc.empty and "prophet_demand" in kc.columns:
            fig_comp.add_trace(go.Scatter(x=kc["date"], y=kc["prophet_demand"], mode="lines+markers", name="Prophet", line=dict(color=BLUE, width=2.5), marker=dict(size=5)))
            if "prophet_upper" in kc.columns:
                fig_comp.add_trace(go.Scatter(x=kc["date"], y=kc["prophet_upper"], mode="lines", name="Prophet Upper", line=dict(color=BLUE, width=0.5, dash="dot"), showlegend=False))
                fig_comp.add_trace(go.Scatter(x=kc["date"], y=kc["prophet_lower"], mode="lines", name="Prophet Lower", line=dict(color=BLUE, width=0.5, dash="dot"), fill="tonexty", fillcolor="rgba(0,114,186,0.1)", showlegend=False))

        fig_comp.update_layout(height=450, barmode="overlay", legend=dict(orientation="h", y=1.05), margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_comp, use_container_width=True)

        # Residuals comparison
        st.subheader("📉 Residual Comparison")
        kf_res = kf.copy()
        kf_res["xgb_residual"] = kf_res["samples_dispensed"] - kf_res["yhat_demand"]
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=kf_res["date"], y=kf_res["xgb_residual"], name="XGBoost Error", marker_color=ORANGE, opacity=0.7))

        if not kc.empty and "prophet_demand" in kc.columns:
            kc_res = kc.dropna(subset=["prophet_demand"]).copy()
            if not kc_res.empty:
                kc_res["prophet_residual"] = kc_res["samples_dispensed"] - kc_res["prophet_demand"]
                fig_res.add_trace(go.Bar(x=kc_res["date"], y=kc_res["prophet_residual"], name="Prophet Error", marker_color=BLUE, opacity=0.7))

        fig_res.add_hline(y=0, line_color=GREY, line_width=1)
        fig_res.update_layout(height=350, barmode="group", yaxis_title="Actual - Predicted", legend=dict(orientation="h", y=1.05), margin=dict(l=40, r=20, t=20, b=40))
        st.plotly_chart(fig_res, use_container_width=True)
        st.info("**Reading residuals:** Bars above zero = model underestimated. Below zero = overestimated. Smaller bars = better.")
    else:
        st.warning("No forecast data for this kiosk.")


# ══════════════════════════════════════════════════════════════
# TAB 4: LIVE WEATHER
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"🌤️ Live Weather — {selected_label}")

    coords = KIOSK_COORDS.get(str(selected_kiosk))
    if coords:
        weather_now = get_live_weather(coords[0], coords[1])
        if weather_now:
            temp_now = weather_now.get("temperature_2m", None)
            precip_now = weather_now.get("precipitation", 0)
            code = weather_now.get("weather_code", 0)
            condition = WEATHER_CODES.get(code, "Unknown")

            w1, w2, w3 = st.columns(3)
            w1.metric("🌡️ Current Temperature", f"{temp_now}°F" if temp_now else "N/A")
            w2.metric("🌧️ Precipitation", f"{precip_now} in")
            w3.metric("☁️ Conditions", condition)

            # Demand impact estimate
            st.subheader("📊 Estimated Demand Impact")
            kh = history[history["kiosk_id"].astype(str) == str(selected_kiosk)]
            avg_demand = int(kh["samples_dispensed"].mean()) if len(kh) > 0 else 50

            impact_pct = 0
            impact_reason = ""
            if temp_now and temp_now > 80:
                impact_pct = 15
                impact_reason = f"Hot weather ({temp_now}°F) typically boosts demand"
            elif temp_now and temp_now > 70:
                impact_pct = 8
                impact_reason = f"Warm weather ({temp_now}°F) slightly boosts demand"
            elif temp_now and temp_now < 35:
                impact_pct = -15
                impact_reason = f"Cold weather ({temp_now}°F) typically reduces demand"

            if precip_now > 0.5:
                impact_pct -= 20
                impact_reason += " + heavy rain suppresses foot traffic"
            elif precip_now > 0.1:
                impact_pct -= 10
                impact_reason += " + rain reduces foot traffic"

            estimated_today = int(avg_demand * (1 + impact_pct / 100))

            e1, e2, e3 = st.columns(3)
            e1.metric("Avg Daily Demand", avg_demand)
            e2.metric("Estimated Today", estimated_today, delta=f"{impact_pct:+d}%")
            e3.metric("Weather Impact", f"{impact_pct:+d}%")

            if impact_reason:
                if impact_pct > 0:
                    st.info(f"🔥 **Expect higher demand today:** {impact_reason}")
                elif impact_pct < 0:
                    st.warning(f"🌧️ **Expect lower demand today:** {impact_reason}")
                else:
                    st.success("☀️ **Normal demand expected today**")

            # Fleet-wide weather
            st.subheader("🗺️ Fleet Weather Overview")
            fleet_weather = []
            for kid, (lat, lon) in KIOSK_COORDS.items():
                loc_name = kiosk_opts[kiosk_opts["kiosk_id"].astype(str) == kid]["location_name"].values
                name = loc_name[0] if len(loc_name) > 0 else kid
                w = get_live_weather(lat, lon)
                if w:
                    fleet_weather.append({
                        "Kiosk": name,
                        "Temp (°F)": w.get("temperature_2m", "N/A"),
                        "Rain (in)": w.get("precipitation", 0),
                        "Conditions": WEATHER_CODES.get(w.get("weather_code", 0), "Unknown"),
                    })
            if fleet_weather:
                st.dataframe(pd.DataFrame(fleet_weather), hide_index=True)
        else:
            st.error("Could not fetch live weather data. Open-Meteo may be temporarily unavailable.")
    else:
        st.warning("No coordinates available for this kiosk.")


# ══════════════════════════════════════════════════════════════
# TAB 5: AI ASSISTANT
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("💬 Freeosk AI Assistant")
    st.info("**Try:** 'Which kiosks need refill this week?' · 'What drives demand at Newark?' · 'Give me an executive summary' · 'Compare XGBoost vs Prophet' · 'How does weather affect demand?'")

    ctx = depletion.to_string(index=False)
    kiosk_summary = history.groupby("location_name").agg(
        avg_demand=("samples_dispensed", "mean"),
        total=("samples_dispensed", "sum"),
        promo_days=("is_promo", "sum"),
        avg_traffic=("foot_traffic_index", "mean"),
    ).round(1).to_string()

    # Get current weather for context
    weather_ctx = ""
    coords = KIOSK_COORDS.get(str(selected_kiosk))
    if coords:
        w = get_live_weather(coords[0], coords[1])
        if w:
            weather_ctx = f"\nCURRENT WEATHER at selected kiosk: {w.get('temperature_2m', 'N/A')}°F, {WEATHER_CODES.get(w.get('weather_code', 0), 'Unknown')}, precipitation: {w.get('precipitation', 0)} in"

    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    user_q = st.chat_input("Ask about kiosk performance...")
    if user_q:
        st.session_state.msgs.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    resp = hf.chat.completions.create(
                        model="meta-llama/Llama-3.1-8B-Instruct",
                        messages=[
                            {"role": "system", "content": f"""You are a senior kiosk ops analyst for Freeosk. Today is April 17, 2026. Be concise, use specific numbers from the data, and end every answer with a clear recommendation.

DEPLETION FORECAST:
{ctx}

KIOSK PERFORMANCE:
{kiosk_summary}
{weather_ctx}

Selected kiosk: {selected_label}

You can analyze demand patterns, compare kiosk performance, explain weather impacts on demand, and recommend operational actions. When discussing models, XGBoost and Prophet are both available. When asked about SQL or data queries, explain what query would answer the question and what the expected result would look like."""},
                            *st.session_state.msgs,
                        ],
                        max_tokens=600,
                    )
                    reply = resp.choices[0].message.content
                except Exception as e:
                    reply = f"Error: {e}. Check your HF token."
            st.write(reply)
            st.session_state.msgs.append({"role": "assistant", "content": reply})

st.divider()
st.caption("Freeosk AI Hackathon 2026 · Synthetic Data · XGBoost + Prophet ML · GCP BigQuery · Hugging Face LLM · Open-Meteo Live Weather · Nominatim Geocoding")