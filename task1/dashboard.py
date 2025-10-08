import streamlit as st
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

st.title("🌍 ClimateScope Dashboard - Milestone 1 (Cleaned Dataset)")

# -----------------------------
# Step 1: Load dataset
# -----------------------------
fl = st.file_uploader("📁 Upload a CSV file", type=["csv", "txt", "xlsx", "xls"])
if fl is not None:
    df = pd.read_csv(fl, encoding="ISO-8859-1")
    st.success(f"✅ Uploaded file: {fl.name}")
else:
    df = pd.read_csv("task1/GlobalWeatherRepository.csv", encoding="ISO-8859-1")
    st.info("Using default dataset: task1/GlobalWeatherRepository.csv")

# -----------------------------
# Step 2: Inspect dataset
# -----------------------------
st.subheader("Dataset Info")
import io
buffer = io.StringIO()
df.info(buf=buffer)
info = buffer.getvalue()
st.text(info)

st.subheader("First 5 Rows")
st.dataframe(df.head())

st.subheader("Missing Values")
st.dataframe(df.isnull().sum())

# -----------------------------
# Step 3: Handle Missing Values
# -----------------------------
for col in df.select_dtypes(include="number").columns:
    df[col].fillna(df[col].mean(), inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

st.subheader("Missing Values After Cleaning")
st.dataframe(df.isnull().sum())

# -----------------------------
# Step 4: Convert Units / Normalize
# -----------------------------
if "temperature_kelvin" in df.columns:
    df["temperature_celsius"] = df["temperature_kelvin"] - 273.15

if "wind_speed" in df.columns:
    df["wind_speed_norm"] = (df["wind_speed"] - df["wind_speed"].min()) / (df["wind_speed"].max() - df["wind_speed"].min())

# -----------------------------
# Step 5: Aggregate Data (Daily → Monthly)
# -----------------------------
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    monthly_df = df.resample("M", on="date").mean()
else:
    monthly_df = df.copy()

st.subheader("Monthly Aggregated Data (First 5 Rows)")
st.dataframe(monthly_df.head())

# -----------------------------
# Step 6: Save Cleaned Dataset & Summary
# -----------------------------
os.makedirs("data", exist_ok=True)

clean_path = "data/weather_cleaned.csv"
monthly_df.to_csv(clean_path, index=False)
st.success(f"✅ Cleaned dataset saved as {clean_path}")

summary = {
    "Total Rows": df.shape[0],
    "Total Columns": df.shape[1],
    "Columns": df.columns.tolist(),
    "Missing Values": df.isnull().sum().to_dict()
}

summary_path = "data/summary.txt"
with open(summary_path, "w") as f:
    f.write("=== Dataset Summary ===\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")
st.success(f"✅ Summary saved as {summary_path}")

# -----------------------------
# Step 7: Visualizations
# -----------------------------
st.subheader("Temperature Over Time")
if "temperature_celsius" in monthly_df.columns:
    st.line_chart(monthly_df["temperature_celsius"])
else:
    st.warning("Temperature column not found!")

st.subheader("Wind Speed Distribution")
if "wind_speed_norm" in monthly_df.columns:
    st.line_chart(monthly_df["wind_speed_norm"])
else:
    st.warning("Wind speed column not found!")

# -----------------------------
# Step 8: Summary Display
# -----------------------------
st.subheader("Dataset Summary")
st.write(summary)
