import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import warnings
warnings.filterwarnings("ignore")

st.title("🌦️ ClimateScope Dashboard - Milestone 2 (Core Analysis & Visualization Design)")

# -----------------------------------------------------
# Step 1: Load Cleaned Dataset from Milestone 1
# -----------------------------------------------------
if os.path.exists("data/weather_cleaned.csv"):
    df = pd.read_csv("data/weather_cleaned.csv")
    st.success("✅ Loaded cleaned dataset from Milestone 1")
else:
    fl = st.file_uploader("📁 Upload cleaned CSV file", type=["csv"])
    if fl:
        df = pd.read_csv(fl)
        st.success(f"✅ Uploaded file: {fl.name}")
    else:
        st.stop()

st.write("**Dataset Shape:**", df.shape)
st.dataframe(df.head())

# -----------------------------------------------------
# Step 2: Statistical Summary
# -----------------------------------------------------
st.subheader("📊 Descriptive Statistics")
st.dataframe(df.describe())

# -----------------------------------------------------
# Step 3: Correlation Matrix
# -----------------------------------------------------
st.subheader("🔗 Correlation Analysis")
num_cols = df.select_dtypes(include="number").columns.tolist()
corr = df[num_cols].corr()
st.dataframe(corr)

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Correlation Heatmap"
)
st.plotly_chart(fig_corr, use_container_width=True)

corr.to_csv("data/correlation_matrix.csv", index=True)
st.info("Correlation matrix saved as data/correlation_matrix.csv")

# -----------------------------------------------------
# Step 4: Seasonal Trends (Monthly Averages)
# -----------------------------------------------------
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.groupby(df["date"].dt.to_period("M")).mean(numeric_only=True)
    monthly.index = monthly.index.to_timestamp()

    st.subheader("📈 Temperature Trend Over Time")
    temp_col = [c for c in df.columns if "temp" in c.lower()]
    if temp_col:
        fig = px.line(monthly, x=monthly.index, y=temp_col[0],
                      title="Average Monthly Temperature")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Temperature column not found.")

    st.subheader("💧 Precipitation Trend Over Time")
    precip_col = [c for c in df.columns if "precip" in c.lower()]
    if precip_col:
        fig2 = px.line(monthly, x=monthly.index, y=precip_col[0],
                       title="Average Monthly Precipitation")
        st.plotly_chart(fig2, use_container_width=True)

    monthly.to_csv("data/monthly_trends.csv")
    st.info("Monthly trend data saved as data/monthly_trends.csv")

# -----------------------------------------------------
# Step 5: Identify Extreme Events (Top/Bottom 1%)
# -----------------------------------------------------
st.subheader("⚠️ Extreme Weather Events")
if "temperature_celsius" in df.columns:
    high_thresh = df["temperature_celsius"].quantile(0.99)
    low_thresh = df["temperature_celsius"].quantile(0.01)

    extreme_heat = df[df["temperature_celsius"] >= high_thresh]
    extreme_cold = df[df["temperature_celsius"] <= low_thresh]

    st.write(f"🌡️ Extreme Heat Events: {len(extreme_heat)} rows")
    st.write(f"🥶 Extreme Cold Events: {len(extreme_cold)} rows")

    extreme_heat.to_csv("data/extreme_heat_events.csv", index=False)
    extreme_cold.to_csv("data/extreme_cold_events.csv", index=False)
    st.success("Extreme event datasets saved to /data/")
else:
    st.warning("Temperature column not found!")

# -----------------------------------------------------
# Step 6: Regional Comparison
# -----------------------------------------------------
st.subheader("🌍 Regional / Country Comparison")
region_cols = [c for c in df.columns if c.lower() in ["region", "country", "continent"]]
if region_cols:
    region_col = region_cols[0]
    cols_to_use = [c for c in ["temperature_celsius", "humidity", "precipitation"] if c in df.columns]
    if cols_to_use:
        group_stats = df.groupby(region_col)[cols_to_use].mean(numeric_only=True).reset_index()
        st.dataframe(group_stats.head())

        fig_region = px.bar(
            group_stats,
            x=region_col,
            y=cols_to_use[0],
            title=f"Average {cols_to_use[0].replace('_',' ').title()} by Region",
            color=cols_to_use[0],
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_region, use_container_width=True)

        group_stats.to_csv("data/region_summary.csv", index=False)
        st.info("Regional summary saved as data/region_summary.csv")
    else:
        st.warning("No valid columns found for regional comparison.")
else:
    st.warning("No region or country column found!")

# -----------------------------------------------------
# Step 7: Dashboard Wireframe Preview (Concept)
# -----------------------------------------------------
st.subheader("🧩 Planned Dashboard Layout (Wireframe Preview)")
st.markdown("""
**Sections planned for final dashboard (Milestone 3):**
1. 🌡️ *Global Temperature Trend* — Line chart of average monthly temperature  
2. 🌍 *Choropleth Map* — Avg temperature by country/region  
3. 💧 *Precipitation Heatmap* — Month vs. precipitation intensity  
4. 🌪️ *Scatterplot* — Wind speed vs. temperature correlation  
5. ⚠️ *Extreme Events Table* — Top 1% and bottom 1% temperature events  
""")

# -----------------------------------------------------
# Step 8: Summary
# -----------------------------------------------------
st.subheader("📜 Milestone 2 Summary")
summary2 = {
    "Total Records": len(df),
    "Numeric Columns": len(num_cols),
    "Correlation File": "data/correlation_matrix.csv",
    "Extreme Event Files": ["data/extreme_heat_events.csv", "data/extreme_cold_events.csv"],
    "Regional Summary": "data/region_summary.csv",
}
st.json(summary2)
