import streamlit as st
import pandas as pd
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
def load_history():
    return bq.query("SELECT * FROM `freeosk-hackathon.kiosk_data.demand_history` ORDER BY kiosk_id, date").to_dataframe()

@st.cache_data(ttl=600)
def load_forecast():
    return bq.query("SELECT * FROM `freeosk-hackathon.kiosk_data.demand_forecast` ORDER BY kiosk_id, date").to_dataframe()

@st.cache_data(ttl=600)
def load_depletion():
    return bq.query("SELECT * FROM `freeosk-hackathon.kiosk_data.depletion_forecast` ORDER BY kiosk_id").to_dataframe()

history = load_history()
forecast = load_forecast()
depletion = load_depletion()

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

tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Summary", "🔍 Kiosk Deep Dive", "🎯 Model Performance", "💬 AI Assistant"])

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

    if len(n7) > 0:
        for _, row in n7.iterrows():
            st.error(f"**CRITICAL:** {row['location_name']} — Depletes **{row['predicted_depletion_date']}** · Remaining: **{int(row['final_inventory_sim'])}** units")
    if len(n14) > 0:
        for _, row in n14.iterrows():
            if row["kiosk_id"] not in n7["kiosk_id"].values:
                st.warning(f"**WARNING:** {row['location_name']} — Depletes **{row['predicted_depletion_date']}** · Remaining: **{int(row['final_inventory_sim'])}** units")
    if len(n7) == 0 and len(n14) == 0:
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
    pivot = pivot[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
    z_vals = [[float(v) if pd.notna(v) else 0.0 for v in row] for row in pivot.values]
    t_vals = [[str(int(float(v))) if pd.notna(v) else "0" for v in row] for row in pivot.values]
    fig_heat = go.Figure(go.Heatmap(z=z_vals, x=list(pivot.columns), y=list(pivot.index), colorscale=[[0, "#FFFFFF"], [0.3, SKY], [0.6, YELLOW], [1, ORANGE]], text=t_vals, texttemplate="%{text}", textfont=dict(size=12), showscale=False))
    fig_heat.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=40))
    st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    st.subheader(f"🏪 {selected_label}")
    kh = history[history["kiosk_id"].astype(str) == str(selected_kiosk)].sort_values("date")
    this_k = depletion[depletion["kiosk_id"].astype(str) == str(selected_kiosk)]
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
    fig1.add_trace(go.Scatter(x=promo_pts["date"], y=promo_pts["samples_dispensed"], mode="markers", name="Promo Day", marker=dict(color=ORANGE, size=10, symbol="star")))
    fig1.update_layout(height=420, legend=dict(orientation="h", y=1.05), margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📦 Inventory Level")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=kh["date"], y=kh["inventory_on_hand"], mode="lines", name="Inventory", line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(0,114,186,0.08)"))
    fig2.add_hline(y=100, line_dash="dash", line_color=ORANGE, annotation_text="Danger Zone (100)")
    fig2.add_hline(y=0, line_dash="dot", line_color="#E53E3E", annotation_text="Stockout")
    restocks = kh[kh["restock_qty"] > 0]
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

with tab3:
    st.subheader("🎯 Forecast Accuracy")
    kf = forecast[forecast["kiosk_id"].astype(str) == str(selected_kiosk)].sort_values("date")
    if not kf.empty:
        mae = abs(kf["samples_dispensed"] - kf["yhat_demand"]).mean()
        mape = (abs(kf["samples_dispensed"] - kf["yhat_demand"]) / kf["samples_dispensed"].clip(1)).mean() * 100
        accuracy = 100 - mape
        a1, a2, a3 = st.columns(3)
        a1.metric("Model Accuracy", f"{accuracy:.1f}%")
        a2.metric("MAE", f"{mae:.1f} samples/day")
        direction_correct = ((kf["yhat_demand"].diff() > 0) == (kf["samples_dispensed"].diff() > 0)).mean() * 100
        a3.metric("Direction Accuracy", f"{direction_correct:.0f}%")
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Bar(x=kf["date"], y=kf["samples_dispensed"], name="Actual", marker_color=SKY, opacity=0.6))
        fig_fc.add_trace(go.Scatter(x=kf["date"], y=kf["yhat_demand"], mode="lines+markers", name="Predicted", line=dict(color=ORANGE, width=2.5), marker=dict(size=6)))
        fig_fc.update_layout(height=420, barmode="overlay", legend=dict(orientation="h", y=1.05), margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_fc, use_container_width=True)
        st.subheader("📉 Residual Analysis")
        kf_res = kf.copy()
        kf_res["residual"] = kf_res["samples_dispensed"] - kf_res["yhat_demand"]
        colors_res = [BLUE if r >= 0 else ORANGE for r in kf_res["residual"]]
        fig_res = go.Figure(go.Bar(x=kf_res["date"], y=kf_res["residual"], marker_color=colors_res))
        fig_res.add_hline(y=0, line_color=GREY, line_width=1)
        fig_res.update_layout(height=300, yaxis_title="Actual - Predicted", margin=dict(l=40, r=20, t=20, b=40))
        st.plotly_chart(fig_res, use_container_width=True)
        st.info("**Reading residuals:** Blue = model underestimated. Orange = model overestimated. Good models have residuals scattered randomly around zero.")
    else:
        st.warning("No forecast data for this kiosk.")

with tab4:
    st.subheader("💬 Freeosk AI Assistant")
    st.info("**Try:** 'Which kiosks need refill this week?' · 'What drives demand at Newark?' · 'Give me an executive summary'")
    ctx = depletion.to_string(index=False)
    kiosk_summary = history.groupby("location_name").agg(avg_demand=("samples_dispensed", "mean"), total=("samples_dispensed", "sum"), promo_days=("is_promo", "sum"), avg_traffic=("foot_traffic_index", "mean")).round(1).to_string()
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
                    resp = hf.chat.completions.create(model="meta-llama/Llama-3.1-8B-Instruct", messages=[{"role": "system", "content": f"You are a senior kiosk ops analyst for Freeosk. Be concise, use numbers, end with a recommendation.\n\nDEPLETION DATA:\n{ctx}\n\nKIOSK SUMMARY:\n{kiosk_summary}\n\nSelected: {selected_label}"}, *st.session_state.msgs], max_tokens=600)
                    reply = resp.choices[0].message.content
                except Exception as e:
                    reply = f"Error: {e}. Check your HF token."
            st.write(reply)
            st.session_state.msgs.append({"role": "assistant", "content": reply})

st.divider()
st.caption("Freeosk AI Hackathon 2026 · Synthetic Data · XGBoost ML · GCP BigQuery · Hugging Face LLM · Open-Meteo · Nominatim")