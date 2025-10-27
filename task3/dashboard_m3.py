# ===============================================
# 🌦️ ClimateScope Dashboard - Milestone 3 (Enhanced UI)
# Modern Elegant Design for Professional Presentation
# ===============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
import os

warnings.filterwarnings("ignore")

# -----------------------------------------------
# Page Setup
# -----------------------------------------------
st.set_page_config(
    page_title="🌍 ClimateScope Dashboard - Milestone 3",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 💅 Custom CSS for Elegant Look
st.markdown("""
    <style>
        /* Overall Background */
        body {
            background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
        }

        /* Header Title */
        .stTitle {
            font-size: 36px !important;
            font-weight: 800 !important;
            text-align: center;
            color: #1E88E5;
            background: -webkit-linear-gradient(#1976D2, #42A5F5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        /* Section headers */
        .stSubheader {
            font-size: 22px !important;
            font-weight: 600 !important;
            color: #0D47A1;
            margin-top: 25px !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2196F3, #64B5F6);
            color: white;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
            color: white !important;
        }

        /* Cards / Containers */
        .stPlotlyChart, .stMarkdown {
            background: rgba(255,255,255,0.9);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }

        /* Info Boxes */
        .stAlert {
            border-radius: 12px !important;
        }

        /* Buttons & Inputs */
        .stButton>button {
            background-color: #1976D2 !important;
            color: white !important;
            border-radius: 10px !important;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1565C0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# Page Header
# -----------------------------------------------
st.title("🌍 ClimateScope Dashboard - Milestone 3 (Interactive Visualizations)")
st.markdown("#### Explore global weather trends, patterns, and extreme events interactively.")

# -----------------------------------------------
# Step 1: Load Cleaned Dataset
# -----------------------------------------------
DATA_PATH = "weather_cleaned.csv"  # Same folder

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.success("✅ Cleaned dataset loaded successfully!")
else:
    st.error("❌ Dataset not found! Please make sure 'weather_cleaned.csv' is in the same folder.")
    st.stop()

# -----------------------------------------------
# Step 2: Clean Columns & Detect Date
# -----------------------------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

if "last_updated" in df.columns:
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    date_col = "last_updated"
elif "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    date_col = "date"
else:
    date_col = None

# -----------------------------------------------
# Step 3: Sidebar Filters
# -----------------------------------------------
st.sidebar.header("🌎 Filter Options")

# Country Filter
country_col = None
for col in df.columns:
    if "country" in col.lower():
        country_col = col
        break

if country_col:
    countries = df[country_col].dropna().unique().tolist()
    selected_country = st.sidebar.selectbox("Select Country", ["All"] + sorted(countries))
else:
    selected_country = "All"

# Date Range
if date_col:
    min_date, max_date = df[date_col].min(), df[date_col].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
else:
    date_range = None

# Apply Filters
filtered_df = df.copy()
if selected_country != "All" and country_col:
    filtered_df = filtered_df[filtered_df[country_col] == selected_country]
if date_range and len(date_range) == 2 and date_col:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df[date_col] >= start) & (filtered_df[date_col] <= end)]

# -----------------------------------------------
# Step 4: Temperature Trends
# -----------------------------------------------
st.subheader("🌡️ Temperature Trends Over Time")

if date_col and "temperature_celsius" in filtered_df.columns:
    fig1 = px.line(
        filtered_df,
        x=date_col,
        y="temperature_celsius",
        color=country_col if country_col else None,
        title="Temperature Variation Over Time",
        markers=True,
    )
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("Temperature or date column not available!")

# -----------------------------------------------
# Step 5: Correlation Scatter Plot
# -----------------------------------------------
st.subheader("📊 Correlation Between Variables")

numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()
if len(numeric_cols) >= 2:
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select X-axis Feature", numeric_cols)
    with col2:
        y_feature = st.selectbox("Select Y-axis Feature", numeric_cols, index=min(1, len(numeric_cols) - 1))

    fig2 = px.scatter(
        filtered_df,
        x=x_feature,
        y=y_feature,
        color=country_col if country_col else None,
        trendline="ols",
        title=f"Correlation between {x_feature} and {y_feature}",
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Not enough numeric columns for correlation visualization!")

# -----------------------------------------------
# Step 6: Regional Comparison
# -----------------------------------------------
st.subheader("🌍 Average Temperature by Country")

if country_col and "temperature_celsius" in df.columns:
    region_avg = df.groupby(country_col)["temperature_celsius"].mean().reset_index()
    fig3 = px.bar(
        region_avg,
        x=country_col,
        y="temperature_celsius",
        color="temperature_celsius",
        color_continuous_scale="Blues",
        title="Average Temperature by Country",
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("Country or temperature data not found!")

# -----------------------------------------------
# Step 7: Insights Section
# -----------------------------------------------
st.subheader("💡 Key Insights & Observations")

if "temperature_celsius" in filtered_df.columns:
    avg_temp = round(filtered_df["temperature_celsius"].mean(), 2)
    max_temp = round(filtered_df["temperature_celsius"].max(), 2)
    min_temp = round(filtered_df["temperature_celsius"].min(), 2)

    st.markdown(f"""
    <div style='background:rgba(227,242,253,0.8);padding:15px;border-radius:10px'>
    - 🌡️ <b>Average Temperature:</b> {avg_temp} °C  
    - 🔥 <b>Highest Recorded Temperature:</b> {max_temp} °C  
    - ❄️ <b>Lowest Recorded Temperature:</b> {min_temp} °C  
    </div>
    """, unsafe_allow_html=True)

    st.info("Use the sidebar filters to explore specific countries and time periods interactively.")
else:
    st.warning("Temperature column not found for insights calculation.")

# -----------------------------------------------
# End
# -----------------------------------------------
st.success("✅ Milestone 3 dashboard loaded successfully!")
