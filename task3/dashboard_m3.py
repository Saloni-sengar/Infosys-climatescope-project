import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import numpy as np
# NEW Imports for forecasting
from prophet import Prophet
from prophet.plot import plot_plotly
# NEW Import for the button navigation
from streamlit_option_menu import option_menu
# NEW Import for ISO-3 country codes
import pycountry

# --- Page Configuration ---
# Set the layout to "wide" to use the full screen width
st.set_page_config(
    page_title="ClimateScope Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NEW: Attractive "Frosted Glass" LIGHT THEME CSS ---
st.markdown("""
<style>
    /* --- 1. Import Professional Font --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* --- 2. Global Styling (Frosted Light Theme) --- */
    body, .main {
        font-family: 'Inter', sans-serif;
        color: #1f2937; /* Default text color (dark slate) */
    }

    /* Main app background (Subtle Gradient) */
    .main .block-container {
        background: linear-gradient(180deg, #e0f2fe 0%, #f3f4f6 100%); /* Light Sky Blue to Light Gray */
        padding: 2rem 3rem; /* Adjusted padding */
        border-radius: 12px;
    }

    /* --- 3. Typography (Enhanced Contrast) --- */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        color: #111827; /* Darkest Gray */
        text-align: center;
        padding-bottom: 1.5rem;
        text-shadow: none;
    }
    h2 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #334155; /* Darker Slate */
        border-bottom: 1px solid #d1d5db;
        padding-bottom: 0.6rem;
        margin-top: 2rem;
    }
    h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        margin-top: 1.5rem;
        color: #475569; /* Medium Slate */
    }

    /* Secondary text */
    .stCaption {
        color: #6b7280;
        font-style: italic;
    }

    /* --- 4. Sidebar Styling (Refined Light Purple) --- */
    .st-emotion-cache-16txtl3 { /* Sidebar */
        background-color: #f5f3ff; /* Light Violet */
        border-right: 1px solid #c4b5fd; /* Darker Violet border */
    }
    .st-emotion-cache-16txtl3 h1,
    .st-emotion-cache-16txtl3 h2,
    .st-emotion-cache-16txtl3 h3,
    .st-emotion-cache-16txtl3 label {
        color: #7c3aed; /* Stronger Purple Text */
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar expander (closed) */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #f5f3ff;
        border: 1px solid #ddd6fe;
        margin-bottom: 5px;
        border-radius: 8px;
    }
    /* Sidebar expander (open) */
    [data-testid="stSidebar"] [data-testid="stExpander"][aria-expanded="true"] {
        background-color: #ede9fe;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderHeader"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #7c3aed; /* Stronger Purple Text */
    }

    /* --- 5. Slider Styling (Vibrant Accent) --- */
    [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div {
         background-color: #ddd6fe; /* Lighter violet */
         height: 6px;
         border-radius: 3px;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(2) > div {
        background: linear-gradient(90deg, #4f46e5 0%, #a855f7 100%); /* Indigo to Purple Gradient */
        height: 6px;
        border-radius: 3px;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(3) {
        border: 2px solid #ffffff; /* White border on thumb */
        background-color: #4f46e5; /* Indigo */
        height: 20px;
        width: 20px;
        box-shadow: 0 0 8px rgba(79, 70, 229, 0.5); /* Indigo glow */
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(3):hover {
        box-shadow: 0 0 12px rgba(79, 70, 229, 0.7); /* Stronger indigo glow */
    }
    .st-emotion-cache-13m0kn0 { /* Slider value label */
        font-family: 'monospace';
        color: #374151; /* Dark text */
    }

    /* --- 6. UNIFIED CARD STYLING ("Frosted" White Cards) --- */
    .st-emotion-cache-1vze3mj, /* Metric boxes */
    [data-testid="stPlotlyChart"], /* Plotly charts */
    [data-testid="stDataFrame"], /* Dataframes */
    .main [data-testid="stExpander"] /* Expanders in main body */
    {
        background-color: rgba(255, 255, 255, 0.9); /* Slightly more opaque white */
        backdrop-filter: blur(8px); /* Subtle Frosted effect */
        border: 1px solid rgba(230, 230, 230, 0.7); /* Lighter border */
        border-radius: 16px; /* Softer corners */
        padding: 1.5rem;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08); /* Refined shadow */
        margin-bottom: 1.5rem; /* Add space below cards */
    }
    .main [data-testid="stExpander"] { /* Ensure expanders in main body have margin */
         margin-bottom: 1.5rem;
    }

    /* --- 7. Specific Card Tweaks --- */

    /* Metric boxes */
    .st-emotion-cache-1vze3mj {
        padding: 1.5rem !important;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .st-emotion-cache-1vze3mj:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 18px rgba(0, 0, 0, 0.1);
    }
    /* Metric text colors */
    .st-emotion-cache-1vze3mj .stMetricLabel,
    .st-emotion-cache-1vze3mj .stMetricValue,
    .st-emotion-cache-1vze3mj .stMetricDelta [data-testid="stMetricDelta"] {
         color: #1f2937;
    }
    /* Delta colors */
    .st-emotion-cache-1vze3mj .stMetricDelta [data-testid="stMetricDelta"] > div:first-child { color: #dc2626; }
    .st-emotion-cache-1vze3mj .stMetricDelta [data-testid="stMetricDelta"] > div:last-child { color: #16a34a; }

    /* Plotly charts */
    [data-testid="stPlotlyChart"] {
        padding: 0 !important;
        overflow: hidden;
        border: 1px solid rgba(230, 230, 230, 0.7) !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        padding: 0 !important;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] .col-header { /* Dataframe header */
        background-color: #f9fafb;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        color: #374151;
        border-bottom: 1px solid #e5e7eb;
    }
    [data-testid="stDataFrame"] .data-row {
         border-bottom: 1px solid #e5e7eb;
    }
    [data-testid="stDataFrame"] .data-row:hover {
        background-color: #f0f9ff;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #a855f7 100%);
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.1rem;
        border: none;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(79, 70, 229, 0.3);
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(79, 70, 229, 0.4);
    }

</style>
""", unsafe_allow_html=True)


# --- NEW: Helper Functions for Categorization ---
def categorize_uv(uv):
    """Maps UV index to a risk category."""
    if pd.isna(uv):
        return 'Unknown'
    if uv <= 2:
        return 'Low'
    elif uv <= 5:
        return 'Moderate'
    elif uv <= 7:
        return 'High'
    elif uv <= 10:
        return 'Very High'
    else:
        return 'Extreme'

def categorize_visibility(vis):
    """Maps visibility in km to a category."""
    if pd.isna(vis):
        return 'Unknown'
    if vis > 10:
        return 'Good'
    elif vis > 5: # 5.1 to 10
        return 'Moderate'
    elif vis > 1: # 1.1 to 5
        return 'Poor'
    else: # 0 to 1
        return 'Very Poor'


# --- Data Loading and Caching ---
@st.cache_data  # Cache the data loading for performance
def load_data(filepath):
    """
    Loads and preprocesses the weather data from a CSV file.
    """
    try:
        # Check if running in Streamlit cloud
        if 'streamlit' in os.environ.get('HOSTNAME', ''):
            csv_path = filepath
        else:
            # Construct an absolute path to the CSV file locally
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(script_dir, filepath)
            except NameError:
                # Fallback for simple local run (where __file__ might fail)
                csv_path = filepath

            # Fallback for simple local run (where __file__ might fail)
            if not os.path.exists(csv_path):
                csv_path = filepath

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at {csv_path}")

        data = pd.read_csv(csv_path)

        # --- Data Preprocessing ---
        data['last_updated_date'] = pd.to_datetime(data['last_updated'], format='%Y-%m-%d %H:%M', errors='coerce')

        original_count = len(data)
        data = data.dropna(subset=['last_updated_date'])
        dropped_count = original_count - len(data)

        if dropped_count > 0:
            print(f"Dropped {dropped_count} rows with invalid dates.")

        data['year'] = data['last_updated_date'].dt.year
        data['month_name'] = data['last_updated_date'].dt.month_name()
        
        # --- NEW TIME FEATURES ---
        data['hour'] = data['last_updated_date'].dt.hour
        data['day_of_week'] = data['last_updated_date'].dt.day_name()
        # --- END NEW FEATURES ---

        # Map AQI index to human-readable names
        data['aqi_label'] = data['air_quality_us-epa-index'].map({
            1: '1 - Good',
            2: '2 - Moderate',
            3: '3 - Unhealthy (SG)',
            4: '4 - Unhealthy',
            5: '5 - Very Unhealthy',
            6: '6 - Hazardous'
        }).fillna('Unknown')

        # NEW: Categorize UV and Visibility
        data['uv_category'] = data['uv_index'].apply(categorize_uv)
        data['visibility_category'] = data['visibility_km'].apply(categorize_visibility)

        return data
    except FileNotFoundError:
        st.error(f"Error: The data file '{filepath}' was not found.")
        st.info(f"The script tried to find the file here: `{os.path.abspath(filepath)}`")
        st.warning("Please make sure 'weather_cleaned.csv' is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading or processing the data: {e}")
        return None

# --- NEW: Helper Function for ISO-3 Code Conversion ---
@st.cache_data
def get_iso_alpha_3(country_name):
    """Converts a country name to its 3-letter ISO code."""
    try:
        return pycountry.countries.get(name=country_name).alpha_3
    except AttributeError:
        # Handle cases where the name might be slightly off or not found
        try:
            return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
        except LookupError:
            return None # Return None if country code can't be found

# --- Plotting Functions ---

def create_wind_rose(df):
    """Creates a Wind Rose chart from the dataframe."""
    # Bin wind direction (16 directions)
    direction_bins = [0] + list(np.arange(11.25, 360, 22.5)) + [360]
    direction_labels_raw = ["N_first", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N_last"]

    df_rose_chart = df.copy()
    df_rose_chart['wind_direction_binned'] = pd.cut(df_rose_chart['wind_degree'] % 360, bins=direction_bins, labels=direction_labels_raw, right=False, include_lowest=True)
    df_rose_chart['wind_direction_binned'] = df_rose_chart['wind_direction_binned'].replace({"N_first": "N", "N_last": "N"})

    ordered_labels = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    df_rose_chart['wind_direction_binned'] = pd.Categorical(df_rose_chart['wind_direction_binned'], categories=ordered_labels, ordered=True)

    # Bin wind speed
    speed_bins = [0, 5, 10, 15, 20, 30, 40, np.inf]
    speed_labels = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-40", "40+"]
    df_rose_chart['wind_speed_binned'] = pd.cut(df_rose_chart['wind_kph'], bins=speed_bins, labels=speed_labels, right=False)

    df_rose = df_rose_chart.groupby(['wind_direction_binned', 'wind_speed_binned'], observed=True).size().reset_index(name='frequency')
    df_rose_pivot = df_rose.pivot(index='wind_direction_binned', columns='wind_speed_binned', values='frequency').fillna(0)
    df_rose_pivot = df_rose_pivot.reindex(ordered_labels).fillna(0)

    total_observations = df_rose_pivot.sum().sum()
    if total_observations == 0:
        total_observations = 1 # Avoid division by zero

    df_rose_pivot_percent = (df_rose_pivot / total_observations) * 100

    fig = go.Figure()

    colors = px.colors.sequential.Viridis_r # Keep dark color scale for contrast
    for i, speed in enumerate(speed_labels):
        if speed in df_rose_pivot_percent.columns:
            fig.add_trace(go.Barpolar(
                r=df_rose_pivot_percent[speed],
                theta=df_rose_pivot_percent.index,
                name=f'{speed} kph',
                marker_color=colors[min(i, len(colors)-1)]
            ))

    fig.update_layout(
        template='plotly_white',
        polar=dict(radialaxis=dict(visible=True, ticksuffix='%', gridcolor='#d1d5db'),
                   angularaxis=dict(direction="clockwise", period=360, gridcolor='#d1d5db')),
        legend_title="Wind Speed (kph)",
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#1f2937' # Dark font
    )
    return fig

# --- NEW: Standalone Plotting Functions for Dynamic Tabs ---

def plot_temp_trend(df):
    """Plots Temperature Trend Line Chart"""
    st.markdown(f"**Temperature Trend**")
    try:
        fig = px.line(df, x='last_updated_date', y='temperature_celsius', template='plotly_white')
        fig.update_layout(xaxis_title='Date', yaxis_title='Temp (°C)', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    except Exception as e: 
        st.error(f"Temp Trend Error: {e}")

def plot_humidity_trend(df):
    """Plots Humidity Trend Line Chart"""
    st.markdown(f"**Humidity Trend**")
    try:
        fig = px.line(df, x='last_updated_date', y='humidity', template='plotly_white')
        fig.update_layout(xaxis_title='Date', yaxis_title='Humidity (%)', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_traces(line_color='#3b82f6') # Brighter Blue
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    except Exception as e: 
        st.error(f"Humidity Trend Error: {e}")

def plot_pressure_trend(df):
    """Plots Pressure Trend Line Chart"""
    st.markdown(f"**Pressure Trend**")
    try:
        fig = px.line(df, x='last_updated_date', y='pressure_mb', template='plotly_white')
        fig.update_layout(xaxis_title='Date', yaxis_title='Pressure (mb)', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_traces(line_color='#ef4444') # Brighter Red
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    except Exception as e: 
        st.error(f"Pressure Trend Error: {e}")

def plot_precip_bar_by_location(df):
    """Plots Avg. Precipitation by Location Bar Chart"""
    st.markdown(f"**Avg. Precipitation by Location (Top 20)**")
    try:
        precip_data = df.groupby('location_name')['precip_mm'].mean().nlargest(20).reset_index()
        fig = px.bar(precip_data, x='location_name', y='precip_mm', template='plotly_white')
        fig.update_layout(yaxis_title='Precipitation (mm)', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    except Exception as e: 
        st.error(f"Precipitation Bar Error: {e}")

def plot_wind_histogram(df):
    """Plots Wind Speed Distribution Histogram"""
    st.markdown(f"**Wind Speed Distribution**")
    try:
        fig = px.histogram(df, x='wind_kph', template='plotly_white')
        fig.update_layout(xaxis_title='Wind Speed (kph)', yaxis_title='Frequency', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    except Exception as e: 
        st.error(f"Wind Histogram Error: {e}")

def plot_condition_bar(df):
    """Plots Top 15 Weather Conditions Bar Chart"""
    st.markdown(f"**Top 15 Weather Conditions**")
    try:
        condition_counts = df['condition_text'].value_counts().nlargest(15).reset_index()
        fig = px.bar(condition_counts, y='condition_text', x='count', orientation='h', template='plotly_white')
        fig.update_layout(yaxis_title=None, xaxis_title='Count', yaxis={'categoryorder':'total ascending'}, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    except Exception as e: 
        st.error(f"Conditions Bar Error: {e}")

# --- NEW VISUAL FUNCTION ---
def plot_hourly_heatmap(df):
    """Plots an Hourly vs. Day of Week Heatmap for Temperature"""
    st.markdown(f"**Hourly Average Temperature Heatmap**")
    if 'hour' not in df.columns or 'day_of_week' not in df.columns:
        st.error("Missing 'hour' or 'day_of_week' columns. Please update load_data().")
        return
    try:
        heatmap_data = df.pivot_table(values='temperature_celsius', index='day_of_week', columns='hour', aggfunc='mean')
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        fig = px.imshow(heatmap_data, aspect='auto', color_continuous_scale=px.colors.sequential.YlOrRd,
                          title="Average Temperature (°C) by Hour and Day", template='plotly_white')
        fig.update_layout(xaxis_title='Hour of Day', yaxis_title=None, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    except Exception as e:
        st.error(f"Could not display hourly heatmap: {e}")


# --- Helper Function for CSV Download ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for download."""
    return df.to_csv(index=False).encode('utf-8')

# --- Tab Rendering Functions ---

# --- UPDATED: render_global_overview_tab (WITH ISO-3 FIX & ANIMATIONS) ---
def render_global_overview_tab(filtered_df, global_avg_metrics, filter_message, selected_countries, map_zoom):
    """Renders the content for the Executive Dashboard tab."""
    st.header("🛰️ Executive Dashboard")
    st.markdown(filter_message)

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Global KPIs with Delta
        st.subheader("High-Level Metrics")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        # Calculate filtered metrics
        filtered_avg_temp = filtered_df['temperature_celsius'].mean()
        filtered_avg_humidity = filtered_df['humidity'].mean()
        filtered_avg_wind = filtered_df['wind_kph'].mean()
        total_reports = len(filtered_df)

        # Calculate deltas (only if a single country is selected)
        delta_temp_str, delta_hum_str, delta_wind_str = None, None, None
        if len(selected_countries) == 1:
            delta_temp = filtered_avg_temp - global_avg_metrics['temp']
            delta_temp_str = f"{delta_temp:+.1f} °C vs. global avg"

            delta_hum = filtered_avg_humidity - global_avg_metrics['humidity']
            delta_hum_str = f"{delta_hum:+.1f} % vs. global avg"

            delta_wind = filtered_avg_wind - global_avg_metrics['wind']
            delta_wind_str = f"{delta_wind:+.1f} kph vs. global avg"

        kpi_col1.metric("Avg. Temp", f"{filtered_avg_temp:.1f} °C", delta=delta_temp_str)
        kpi_col2.metric("Avg. Humidity", f"{filtered_avg_humidity:.1f} %", delta=delta_hum_str)
        kpi_col3.metric("Avg. Wind Speed", f"{filtered_avg_wind:.1f} kph", delta=delta_wind_str)
        kpi_col4.metric("Total Data Points", f"{total_reports:,}")

        # Smarter Map
        st.subheader("Interactive Map (Animated by Year)")
        try:
            # Updated Logic: Show Choropleth ONLY if NO countries are selected (Global)
            if not selected_countries:
                st.markdown("Global Average Temperature by Country")
                # Group by year AND country for animation
                country_avg = filtered_df.groupby(['country', 'year'])['temperature_celsius'].mean().reset_index()
                
                # --- THIS IS THE FIX ---
                # 1. Convert country names to ISO-3 codes
                country_avg['iso_alpha_3'] = country_avg['country'].apply(get_iso_alpha_3)
                
                # 2. Drop rows where conversion failed
                country_avg = country_avg.dropna(subset=['iso_alpha_3'])
                
                fig_choro = px.choropleth(
                    country_avg, 
                    locations='iso_alpha_3',  # <-- UPDATED
                    locationmode='ISO-3',     # <-- UPDATED
                    color='temperature_celsius', 
                    hover_name='country',
                    hover_data={'temperature_celsius': ':.1f'},
                    color_continuous_scale=px.colors.sequential.YlOrRd, 
                    template='plotly_white',
                    animation_frame='year'
                )
                # --- END OF FIX ---
                
                fig_choro.update_geos(bgcolor='rgba(0,0,0,0)', landcolor='#e5e7eb', subunitcolor='#d1d5db', showcountries=True)
                fig_choro.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_colorbar_title='Avg Temp (°C)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_choro, use_container_width=True, theme="streamlit")
            else:
                # Show scatter map if one or more countries are selected
                st.markdown("Live Temperature by Location")
                map_center = {"lat": filtered_df['latitude'].mean(), "lon": filtered_df['longitude'].mean()} if not filtered_df.empty else {"lat": 0, "lon": 0}
                
                # Sort by year for animation
                map_df = filtered_df.sort_values('year')
                
                fig_map = px.scatter_geo(
                    map_df, lat='latitude', lon='longitude', color='temperature_celsius',
                    hover_name='location_name', hover_data={"condition_text": True, "temperature_celsius": ":.1f", "latitude": False, "longitude": False},
                    template='plotly_white', projection='natural earth',
                    animation_frame='year',
                    animation_group='location_name'
                )
                fig_map.update_geos(bgcolor='rgba(0,0,0,0)', landcolor='#e5e7eb', subunitcolor='#d1d5db')
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_colorbar_title='Temp (°C)', geo_center=map_center, geo_projection_scale=map_zoom, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_map, use_container_width=True, theme="streamlit")
        except Exception as e:
            st.error(f"Could not display map: {e}")

        # Comparative Charts
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.subheader("📊 Avg. Precipitation by Country (Top 20)")
            try:
                if not selected_countries:
                    # Group by year AND country for animation
                    precip_data = filtered_df.groupby(['country', 'year'])['precip_mm'].mean().reset_index()
                    # Get Top 20 based on overall average
                    top_20_countries = filtered_df.groupby('country')['precip_mm'].mean().nlargest(20).index
                    precip_data = precip_data[precip_data['country'].isin(top_20_countries)]
                    
                    fig_precip = px.bar(
                        precip_data, x='country', y='precip_mm', 
                        template='plotly_white',
                        animation_frame='year'
                    )
                    fig_precip.update_layout(yaxis_title='Precipitation (mm)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_precip, use_container_width=True, theme="streamlit")
                else:
                    st.info("Global comparison requires 'All Countries' (no selection) in the sidebar.")
            except Exception as e:
                st.error(f"Could not display precipitation chart: {e}")

        with comp_col2:
            st.subheader("💨 Avg. Wind Speed by Country (Top 20)")
            try:
                if not selected_countries:
                    # Group by year AND country for animation
                    wind_data = filtered_df.groupby(['country', 'year'])['wind_kph'].mean().reset_index()
                    # Get Top 20 based on overall average
                    top_20_countries = filtered_df.groupby('country')['wind_kph'].mean().nlargest(20).index
                    wind_data = wind_data[wind_data['country'].isin(top_20_countries)]
                    
                    fig_wind_bar = px.bar(
                        wind_data, x='country', y='wind_kph', 
                        template='plotly_white',
                        animation_frame='year'
                    )
                    fig_wind_bar.update_layout(yaxis_title='Wind Speed (kph)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_wind_bar, use_container_width=True, theme="streamlit")
                else:
                    st.info("Global comparison requires 'All Countries' (no selection) in the sidebar.")
            except Exception as e:
                st.error(f"Could not display wind chart: {e}")

def render_location_deep_dive_tab(filtered_df, global_avg_metrics, filter_title, selected_countries, selected_location):
    """Renders the content for the Location Deep-Dive tab."""
    st.header(f"📍 Location Deep-Dive: {filter_title}")

    # Check for single country selection (or multi-country with "All Locations")
    if not selected_countries: # No country selected
         st.info("Please select one or more countries from the sidebar to see a detailed deep-dive.")
         return
    elif selected_countries and selected_location == "All Locations":
         st.markdown("Showing aggregated data for all selected locations.")
    
    if filtered_df.empty:
        st.warning("No data available for the selected location and date range.")
        return

    # --- DYNAMIC CHARTING LOGIC ---
    
    # 1. Define all available plotting functions
    chart_options = {
        "Temperature Trend": plot_temp_trend,
        "Humidity Trend": plot_humidity_trend,
        "Pressure Trend": plot_pressure_trend,
        "Hourly Temp Heatmap": plot_hourly_heatmap,
        "Wind Speed Distribution": plot_wind_histogram,
        "Top 15 Conditions": plot_condition_bar,
        # Only add this chart if we are in a single country view
    }
    
    if len(selected_countries) == 1:
         chart_options["Precipitation by Location"] = plot_precip_bar_by_location

    # --- End Charting Logic ---

    # Location KPIs (This section remains mostly the same)
    st.subheader("Key Metrics (vs. Global Avg)")
    kpi_col1_c, kpi_col2_c, kpi_col3_c, kpi_col4_c, kpi_col5_c = st.columns(5)

    try:
        loc_avg_temp = filtered_df['temperature_celsius'].mean()
        loc_max_wind = filtered_df['wind_kph'].max()
        loc_avg_precip = filtered_df['precip_mm'].mean()
        loc_avg_uv = filtered_df['uv_index'].mean()
        loc_avg_vis = filtered_df['visibility_km'].mean()
        
        delta_temp = loc_avg_temp - global_avg_metrics['temp']
        delta_precip = loc_avg_precip - global_avg_metrics['precip']
        delta_uv = loc_avg_uv - global_avg_metrics['uv']
        delta_vis = loc_avg_vis - global_avg_metrics['vis']

        kpi_col1_c.metric("Avg. Temp", f"{loc_avg_temp:.1f} °C", delta=f"{delta_temp:+.1f} °C")
        kpi_col2_c.metric("Max Wind", f"{loc_max_wind:.1f} kph")
        kpi_col3_c.metric("Avg. Precip", f"{loc_avg_precip:.2f} mm", delta=f"{delta_precip:+.2f} mm")
        kpi_col4_c.metric("Avg. UV Index", f"{loc_avg_uv:.1f}", delta=f"{delta_uv:+.1f}")
        kpi_col5_c.metric("Avg. Visibility", f"{loc_avg_vis:.1f} km", delta=f"{delta_vis:+.1f} km")
    except Exception as e:
        st.warning(f"Could not calculate all KPIs: {e}")


    # --- UPDATED: Dynamic Time-Series Charts ---
    st.subheader("📈 Dynamic Chart Explorer")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Dropdown to select the chart for the first column
        chart_choice_1 = st.selectbox(
            "Select Chart 1",
            options=list(chart_options.keys()),
            index=0 # Default to 'Temperature Trend'
        )
        # Call the selected function
        selected_function_1 = chart_options[chart_choice_1]
        selected_function_1(filtered_df)

    with chart_col2:
        # Dropdown to select the chart for the second column
        chart_choice_2 = st.selectbox(
            "Select Chart 2",
            options=list(chart_options.keys()),
            index=1 # Default to 'Humidity Trend'
        )
        # Call the selected function
        selected_function_2 = chart_options[chart_choice_2]
        selected_function_2(filtered_df)

    # (The original "Condition Distributions" section is now part of the dynamic charts)
    
    # Category Distributions (UV, Visibility, AQI) - This can remain
    st.subheader("📊 Category Distributions")
    cat_col1, cat_col2, cat_col3 = st.columns(3)

    with cat_col1:
        st.markdown(f"**UV Risk Categories**")
        try:
            uv_order = ['Low', 'Moderate', 'High', 'Very High', 'Extreme', 'Unknown']
            uv_counts = filtered_df['uv_category'].value_counts().reindex(uv_order).dropna().reset_index()

            if uv_counts.empty or uv_counts['count'].sum() == 0:
                st.info("No UV data.")
            else:
                fig_uv_bar = px.bar(uv_counts, x='uv_category', y='count',
                                    title="UV Risk Categories", template='plotly_white')
                fig_uv_bar.update_layout(xaxis_title=None, yaxis_title='Count', margin=dict(t=40, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_uv_bar, use_container_width=True, theme="streamlit")
        except Exception as e: st.error(f"UV Bar Chart Error: {e}")

    with cat_col2:
        st.markdown(f"**Visibility Categories**")
        try:
            vis_counts = filtered_df['visibility_category'].value_counts().reset_index()

            if vis_counts.empty or vis_counts['count'].sum() == 0:
                st.info("No Visibility data.")
            else:
                fig_vis_pie = px.pie(vis_counts, names='visibility_category', values='count',
                                     title="Visibility Categories", template='plotly_white', hole=0.4)
                fig_vis_pie.update_traces(textposition='inside', textinfo='percent+label', sort=True)
                fig_vis_pie.update_layout(margin=dict(t=40, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_vis_pie, use_container_width=True, theme="streamlit")
        except Exception as e: st.error(f"Visibility Pie Error: {e}")

    with cat_col3:
        st.markdown(f"**Air Quality**")
        try:
            aqi_counts = filtered_df['aqi_label'].value_counts().reset_index()
            if aqi_counts.empty or aqi_counts['count'].sum() == 0:
                st.info("No AQI data.")
            else:
                fig_aqi = px.pie(aqi_counts, names='aqi_label', values='count',
                                 title=f"Air Quality Index (US EPA)", template='plotly_white', hole=0.4)
                fig_aqi.update_traces(textposition='inside', textinfo='percent+label', sort=True)
                fig_aqi.update_layout(margin=dict(t=40, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_aqi, use_container_width=True, theme="streamlit")
        except Exception as e: st.error(f"AQI Pie Error: {e}")


# --- UPDATED: render_humidity_tab (WITH ANIMATION) ---
def render_humidity_tab(filtered_df, filter_message):
    """Renders the content for the new Humidity Analysis tab."""
    st.header("💧 Humidity Analysis")
    st.markdown("Comprehensive humidity patterns and moisture analysis.")
    st.markdown(filter_message) # Show the active filters

    if filtered_df.empty or filtered_df['humidity'].isnull().all():
        st.warning("No humidity data available for the selected filters.")
    else:
        # KPIs from screenshot
        st.subheader("Humidity Metrics")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        avg_hum = filtered_df['humidity'].mean()
        max_hum = filtered_df['humidity'].max()
        min_hum = filtered_df['humidity'].min()
        std_hum = filtered_df['humidity'].std()

        kpi_col1.metric("Average Humidity", f"{avg_hum:.1f} %")
        kpi_col2.metric("Maximum", f"{max_hum:.1f} %")
        kpi_col3.metric("Minimum", f"{min_hum:.1f} %")
        kpi_col4.metric("Std Dev", f"{std_hum:.1f} %")

        # Charts from screenshot
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Humidity Distribution")
            st.markdown("Distribution with Box Plot")
            try:
                fig_hist = px.histogram(filtered_df, x='humidity', marginal='box',
                                        template='plotly_white', color_discrete_sequence=['#3b82f6']) # Brighter blue
                fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0))
                st.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Humidity Histogram Error: {e}")

        with chart_col2:
            st.subheader("Humidity vs Temperature (Animated)")
            st.markdown("Relationship between humidity and temperature over time.")
            try:
                # --- MODIFICATION FOR ANIMATION ---
                # 1. Sort data by the animation frame (year)
                scatter_df = filtered_df.sort_values('year')
                # 2. Sample *after* sorting if dataset is large
                if len(scatter_df) > 2000:
                    scatter_df = scatter_df.sample(2000)
                
                fig_scatter = px.scatter(
                    scatter_df, 
                    x='temperature_celsius', 
                    y='humidity',
                    color='temperature_celsius', 
                    color_continuous_scale=px.colors.sequential.YlOrRd,
                    template='plotly_white',
                    # --- NEW ANIMATION LINES ---
                    animation_frame='year',       # Creates the play button/slider
                    animation_group='location_name', # Tells dots how to transition
                    hover_name='location_name'    # Show location name on hover
                )
                fig_scatter.update_layout(
                    xaxis_title='Temp (°C)', 
                    yaxis_title='Humidity (%)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    margin=dict(t=40, b=0)
                )
                st.plotly_chart(fig_scatter, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Humidity Scatter Error: {e}")

def render_comparison_tab(df, all_countries_list):
    """Renders the content for the 1-vs-1 Comparison tab."""
    st.header("🆚 Country Comparison")
    st.markdown("Select two *or more* countries to compare their key metrics and temperature trends side-by-side.")

    countries_to_select = [c for c in all_countries_list if c != "All Countries"]
    if not countries_to_select:
         st.warning("No country data available.")
         return

    # Use multiselect for comparison
    compared_countries = st.multiselect(
        "Select Countries to Compare", 
        countries_to_select,
        default=countries_to_select[0:2] # Default to the first two
    )

    if not compared_countries:
        st.info("Please select at least one country to see metrics.")
        return

    # Create dynamic columns for each selected country
    cols = st.columns(len(compared_countries))
    
    all_data_for_plot = []

    for i, country in enumerate(compared_countries):
        with cols[i]:
            df_country = df[df['country'] == country]
            
            if df_country.empty:
                st.warning(f"No data for {country}")
                continue

            st.subheader(f"Metrics for {country}")
            avg_temp = df_country['temperature_celsius'].mean()
            avg_hum = df_country['humidity'].mean()
            avg_wind = df_country['wind_kph'].mean()

            st.metric("Avg. Temp", f"{avg_temp:.1f} °C")
            st.metric("Avg. Humidity", f"{avg_hum:.1f} %")
            st.metric("Avg. Wind", f"{avg_wind:.1f} kph")
            
            # Prepare data for combined plot
            df_country_resampled = df_country.set_index('last_updated_date').resample('D')['temperature_celsius'].mean().reset_index()
            df_country_resampled['country'] = country
            all_data_for_plot.append(df_country_resampled)

    # Create one combined chart at the bottom
    if all_data_for_plot:
        combined_df = pd.concat(all_data_for_plot)
        st.subheader("Comparative Temperature Trend (Daily Avg)")
        fig_combined = px.line(
            combined_df, 
            x='last_updated_date', 
            y='temperature_celsius', 
            color='country', # Use color to differentiate
            template='plotly_white'
        )
        fig_combined.update_layout(xaxis_title='Date', yaxis_title='Temp (°C)', margin=dict(t=20, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_combined, use_container_width=True, theme="streamlit")


# --- UPDATED: render_trends_tab (WITH ANIMATIONS) ---
def render_trends_tab(filtered_df, filter_message):
    """Renders the content for the Trends & Correlations tab."""
    st.header("📈 Correlations")
    st.markdown(filter_message)

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.subheader("Correlation Heatmap")
        st.markdown("Shows the statistical correlation between different weather metrics.")
        try:
            numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
            cols_to_correlate = [col for col in numeric_cols if col not in ['latitude', 'longitude', 'year', 'last_updated_epoch', 'air_quality_us-epa-index', 'air_quality_gb-defra-index', 'hour']]

            if len(cols_to_correlate) > 1:
                corr = filtered_df[cols_to_correlate].corr()
                fig_corr_heat = px.imshow(corr, text_auto=".2f", aspect="auto",
                                          color_continuous_scale=px.colors.diverging.RdBu, color_continuous_midpoint=0,
                                          template='plotly_white')
                fig_corr_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_corr_heat, use_container_width=True, theme="streamlit")
                st.caption("Insight: Notice the negative correlation between Temperature and Humidity, and positive between Temperature and UV Index.")
            else:
                st.info("Not enough numeric data for correlation.")
        except Exception as e:
            st.error(f"Could not display correlation heatmap: {e}")

        st.subheader("Dynamic Correlation Explorer (Animated)")
        st.markdown("Select any two numeric variables to see their relationship animate over time.")
        try:
            numeric_cols_list = filtered_df.select_dtypes(include=np.number).columns.tolist()
            cols_to_plot = [col for col in numeric_cols_list if col not in ['latitude', 'longitude', 'year', 'last_updated_epoch']]

            default_x = cols_to_plot.index('temperature_celsius') if 'temperature_celsius' in cols_to_plot else 0
            default_y = cols_to_plot.index('humidity') if 'humidity' in cols_to_plot else 1

            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                x_axis = st.selectbox("Select X-Axis", cols_to_plot, index=default_x)
            with exp_col2:
                y_axis = st.selectbox("Select Y-Axis", cols_to_plot, index=default_y)
            
            # Sort data for animation
            scatter_df_dynamic = filtered_df.sort_values('year')
            if len(scatter_df_dynamic) > 2000:
                scatter_df_dynamic = scatter_df_dynamic.sample(2000)

            fig_dynamic = px.scatter(
                scatter_df_dynamic, 
                x=x_axis, 
                y=y_axis,
                color='temperature_celsius',
                color_continuous_scale=px.colors.sequential.YlOrRd,
                title=f"{x_axis} vs. {y_axis}",
                template='plotly_white',
                # Animation lines
                animation_frame='year',
                animation_group='location_name',
                hover_name='location_name'
            )
            fig_dynamic.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dynamic, use_container_width=True, theme="streamlit")

        except Exception as e:
            st.error(f"Could not generate dynamic scatter plot: {e}")

        # --- Scatter plots (NOW BOTH ANIMATED) ---
        corr_col1, corr_col2 = st.columns(2)
        with corr_col1:
            st.subheader("🌡️ Temp vs. Humidity (Animated)")
            try:
                # Sort data for animation
                scatter_df = filtered_df.sort_values('year')
                if len(scatter_df) > 2000:
                    scatter_df = scatter_df.sample(2000)
                    
                fig_scatter = px.scatter(
                    scatter_df, 
                    x='temperature_celsius', 
                    y='humidity',
                    color='temperature_celsius', 
                    color_continuous_scale=px.colors.sequential.YlOrRd,
                    title="Temperature vs. Humidity", 
                    template='plotly_white',
                    # --- NEW ANIMATION LINES ---
                    animation_frame='year',
                    animation_group='location_name',
                    hover_name='location_name'
                )
                fig_scatter.update_layout(xaxis_title='Temp (°C)', yaxis_title='Humidity (%)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_scatter, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Temp/Humidity Scatter Error: {e}")

        with corr_col2:
            st.subheader("☀️ Temp vs. UV Index (Animated)")
            try:
                # Sort data for animation
                scatter_df_uv = filtered_df.sort_values('year')
                if len(scatter_df_uv) > 2000:
                    scatter_df_uv = scatter_df_uv.sample(2000)
                
                fig_scatter_uv = px.scatter(
                    scatter_df_uv, 
                    x='temperature_celsius', 
                    y='uv_index',
                    color='uv_index', 
                    color_continuous_scale=px.colors.sequential.Agsunset,
                    title="Temperature vs. UV Index", 
                    template='plotly_white',
                    # --- NEW ANIMATION LINES ---
                    animation_frame='year',
                    animation_group='location_name',
                    hover_name='location_name'
                )
                fig_scatter_uv.update_layout(xaxis_title='Temp (°C)', yaxis_title='UV Index', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_scatter_uv, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Temp/UV Scatter Error: {e}")

        st.subheader("📅 Average Monthly Temperature Heatmap")
        try:
            heatmap_data = filtered_df.pivot_table(values='temperature_celsius', index='year', columns='month_name', aggfunc='mean')
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            heatmap_data = heatmap_data.reindex(columns=month_order)
            fig_heat = px.imshow(heatmap_data, aspect='auto', color_continuous_scale=px.colors.sequential.YlOrRd,
                                 title="Average Monthly Temperature (°C)", template='plotly_white')
            fig_heat.update_layout(xaxis_title='Month', yaxis_title='Year', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_heat, use_container_width=True, theme="streamlit")
            st.caption("Insight: Visualizes seasonal temperature variations year over year.")
        except Exception as e:
            st.error(f"Could not display monthly heatmap: {e}")

# --- UPDATED: render_extremes_tab (WITH KeyError FIX) ---
def render_extremes_tab(filtered_df, filter_message):
    """Renders the content for the Extreme Events & Wind tab."""
    st.header("🌪️ Extreme Events")
    st.markdown(filter_message)

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.subheader("Wind Rose")
        st.markdown("Frequency of wind speed and direction.")
        try:
            fig_rose = create_wind_rose(filtered_df)
            st.plotly_chart(fig_rose, use_container_width=True, theme="streamlit")
        except Exception as e:
            st.error(f"Could not display Wind Rose: {e}")
            st.exception(e)

        st.subheader("Extreme Event Data Tables")
        st.markdown("Top 10 most extreme events within the filtered data.")
        
        # --- THIS IS THE FIX ---
        # Changed 'last_updated' to 'last_updated_date'
        # Also, ensure all columns exist, especially after aggregation
        extreme_cols_to_show = ['location_name', 'country', 'last_updated_date', 'temperature_celsius', 'wind_kph', 'precip_mm']
        
        # Filter list to only columns that actually exist in the final_df
        safe_cols = [col for col in extreme_cols_to_show if col in filtered_df.columns]
        
        # --- END OF FIX ---

        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            st.subheader("🔥 Top 10 Hottest Events")
            st.dataframe(filtered_df.nlargest(10, 'temperature_celsius')[safe_cols], use_container_width=True)

            st.subheader("💨 Top 10 Windiest Events")
            st.dataframe(filtered_df.nlargest(10, 'wind_kph')[safe_cols], use_container_width=True)

        with ex_col2:
            st.subheader("❄️ Top 10 Coldest Events")
            st.dataframe(filtered_df.nsmallest(10, 'temperature_celsius')[safe_cols], use_container_width=True)

            st.subheader("🌧️ Top 10 Rainiest Events")
            st.dataframe(filtered_df.nlargest(10, 'precip_mm')[safe_cols], use_container_width=True)
        st.caption("Insight: Identify locations and times prone to specific extreme conditions.")

def render_data_explorer_tab(filtered_df, filter_message):
    """Renders the content for the Data Explorer tab."""
    st.header("🗂️ Data Explorer")
    st.markdown(filter_message)

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.subheader("Download Filtered Data")
        try:
            csv_data = convert_df_to_csv(filtered_df)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv_data,
                file_name=f"climatescope_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Could not prepare data for download: {e}")

        with st.expander("Show Statistical Summary"):
            st.markdown("Statistical overview of numeric data for selected filters.")
            try:
                numeric_cols_for_desc = filtered_df.select_dtypes(include=np.number).columns
                st.dataframe(filtered_df[numeric_cols_for_desc].describe(), use_container_width=True)
            except Exception as e: st.error(f"Could not display data summary: {e}")

        st.subheader("Filtered Raw Data")
        st.markdown("Raw data based on filters. Sort by clicking headers.")
        try:
            cols_to_display = ['location_name', 'country', 'last_updated_date', 'temperature_celsius', 'condition_text', 'wind_kph', 'precip_mm', 'humidity', 'uv_index', 'aqi_label']
            cols_to_display_safe = [col for col in cols_to_display if col in filtered_df.columns]
            st.dataframe(filtered_df[cols_to_display_safe], use_container_width=True)

            if st.checkbox("Show All Columns"):
                st.dataframe(filtered_df, use_container_width=True)
        except Exception as e: st.error(f"Could not display raw data table: {e}")

# --- UPDATED: render_about_tab with GUIDELINES ---
# --- UPDATED: render_about_tab with FULL DESCRIPTIONS ---
def render_about_tab():
    """Renders the content for the About Project tab, including usage guidelines."""
    st.header("ℹ️ About & How to Use")

    # How to Use Guide
    with st.expander("🚀 How to Use This Dashboard", expanded=True):
        st.markdown("""
        Welcome to **ClimateScope**! This dashboard helps you explore global weather trends.

        **1. Using the Sidebar Filters:**
        * **Select Countries:** Choose one or multiple countries. Leave blank to see global data.
        * **Select Location:** If you select *exactly one* country, you can then filter by a specific location within it.
        * **Select Date Range:** Use the calendar to pick a start and end date for the analysis period.
        * **Select Time Aggregation:** **(New!)** Choose to view 'Raw Data' or see 'Daily', 'Weekly', or 'Monthly' averages. This is great for spotting long-term trends!
        * **Metric Range Sliders:** Open the expanders (like "Temperature Range") to filter the data based on specific value ranges (e.g., show only data where temperature was between 10°C and 25°C).
        * **Filters Apply Globally:** All filters you set in the sidebar will update the data shown across *all* tabs in the dashboard.

        **2. Navigating the Tabs:**
        Use the buttons at the top to switch between different analysis views:
        * **Executive Dashboard:** **(Animated!)** High-level overview, global map, and country comparisons that animate by year.
        * **Location Deep-Dive:** **(Dynamic!)** Use the dropdowns to select from multiple charts like trends, distributions, and heatmaps.
        * **Humidity Analysis:** **(Animated!)** Specific focus on humidity and its animated relationship with temperature.
        * **Country Comparison:** **(Dynamic!)** Select two or more countries to compare their metrics side-by-side.
        * **Correlations:** **(Animated!)** Explore statistical relationships between variables with animated scatter plots.
        * **Extreme Events:** Identify the hottest, coldest, windiest, and rainiest events, plus wind patterns (Wind Rose).
        * **Forecasting:** Generate a future temperature forecast for a *specific selected location* using the Prophet model.
        * **Data Explorer:** View the filtered raw data, get a statistical summary, and download the data as a CSV file.
        * **About Project:** Information about the project's goals, data source, and technology used (you are here!).

        Explore the data and discover weather patterns around the world! 🌍
        """)

    with st.expander("Project Objective"):
        st.markdown("""
        The objective of this project is to transform the 'Global Weather Repository' dataset into a dynamic, interactive dashboard named **ClimateScope**. The goal is to provide a comprehensive tool for users to explore, analyze, and understand complex global weather patterns. 
        
        This includes:
        * Identifying long-term temperature trends.
        * Understanding correlations between different weather metrics (like humidity and temperature).
        * Tracking and isolating extreme weather events.
        * Comparing the climates of different countries and locations.
        * Generating predictive forecasts for specific locations.
        """)

    with st.expander("Data Visualization Design"):
        st.markdown("""
        - **Suitable Visualization Types:** A strategic selection of charts was used to best represent the data:
            - **Geospatial Maps (Choropleth & Scatter):** To visualize the geographic distribution of metrics like temperature and animate these changes over time.
            - **Time-Series Line Charts:** To clearly display trends in temperature, humidity, and pressure over the selected date range.
            - **Bar Charts & Histograms:** For comparing categorical data (like weather conditions) and understanding the distribution of numeric data (like wind speed).
            - **Heatmaps (Temporal & Correlation):** Used to show complex relationships, such as average monthly temperatures over years or the statistical correlation between all numeric variables.
            - **Pie/Donut Charts:** To effectively show the percentage-based composition of categorical data like Air Quality Index (AQI) and UV categories.
            - **Wind Rose:** A specialized polar chart to visualize the frequency and strength of wind from different directions.
            - **Prophet Forecast Plots:** To visualize predictive trends and their components (e.g., weekly, yearly seasonality).
        """)

    with st.expander("Visualization Development"):
        st.markdown("""
        - **Tools:** The dashboard is built entirely in **Python**.
        - **Streamlit:** Serves as the core web application framework, providing the interactive widgets (sliders, dropdowns), page layout, and caching mechanisms for performance.
        - **Plotly Express & Graph Objects:** Used as the primary plotting library to create rich, interactive, and aesthetically pleasing visualizations that allow for zooming, panning, and hover-over details.
        - **Pandas:** The backbone for all data manipulation, filtering, aggregation (e.g., daily/weekly/monthly resampling), and preprocessing.
        - **Prophet:** Integrated for the forecasting tab to perform robust time-series predictions on demand.
        - **CSS:** Custom CSS was injected to create the modern 'frosted glass' theme, improve typography, and ensure a professional, polished user interface.
        """)

    with st.expander("Insights Generation"):
        st.markdown("""
        - **Notable Findings:** The dashboard facilitates the discovery of several key insights:
            - **Correlation Confirmed:** The correlation heatmap and scatter plots consistently show a strong negative correlation between temperature and humidity, and a positive correlation between temperature and UV index.
            - **Seasonal & Diurnal Patterns:** The 'Hourly Average Temperature Heatmap' clearly visualizes the diurnal (daily) warming and cooling cycle, while the 'Monthly Average' heatmap reveals distinct seasonal shifts, especially in non-equatorial regions.
            - **Extreme Event Identification:** The 'Extreme Events' tab allows for the quick identification of specific locations and dates that experienced anomalous weather, such as extreme heatwaves or high-precipitation events.
            - **Trend Analysis:** By aggregating data monthly and using the animated maps, users can visually observe year-over-year warming trends in many regions.
        """)

    with st.expander("Tech Stack Summary"):
        st.markdown("""
        - **Language:** Python 3
        - **Libraries:** Streamlit, Pandas, Plotly, NumPy, **Prophet**
        - **Data Source:** [Kaggle Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/data)
        """)


# --- NEW: FORECASTING TAB FUNCTION ---
@st.cache_data
def run_forecast(data, periods):
    """Fits Prophet model and returns forecast."""
    m = Prophet()
    m.fit(data)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return m, forecast

def render_forecasting_tab(filtered_df, selected_countries, selected_location):
    """Renders the content for the Forecasting tab."""
    st.header("🔮 Forecasting")

    if len(selected_countries) != 1 or selected_location == "All Locations":
        st.info("Please select a specific Country **and** Location from the sidebar to generate a forecast.")
        return

    if filtered_df.empty:
        st.warning("No data available for the selected location to create a forecast.")
        return

    st.markdown(f"Generating forecast for **{selected_location}, {selected_countries[0]}**.")
    st.markdown("Uses the Prophet model to predict future temperature trends (illustrative only).")

    periods_to_forecast = st.slider("Select forecast period (days):", 30, 365, 90)

    try:
        with st.spinner(f"Generating {periods_to_forecast}-day forecast..."):
            # Prepare data: Must be daily data ('D') for Prophet
            forecast_df = filtered_df.set_index('last_updated_date').resample('D')['temperature_celsius'].mean().reset_index()
            forecast_df = forecast_df.rename(columns={'last_updated_date': 'ds', 'temperature_celsius': 'y'})
            forecast_df = forecast_df.dropna()

            if len(forecast_df) < 30:
                st.error("Not enough data points (< 30 days) for a reliable forecast.")
                return

            model, forecast = run_forecast(forecast_df, periods_to_forecast)

            st.subheader("Forecast Plot")
            fig_forecast = plot_plotly(model, forecast)
            fig_forecast.update_layout(
                template='plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Date",
                yaxis_title="Temperature (°C)"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.subheader("Forecast Components")
            fig_components = model.plot_components(forecast)
            st.pyplot(fig_components)

            with st.expander("Show Forecast Data"):
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_to_forecast), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
        st.exception(e)


# --- Load Data ---
df = load_data('weather_cleaned.csv')

# --- Main Application ---
if df is not None:

    st.title("🌍 ClimateScope: Visualizing Global Weather Trends")

    # --- Calculate global averages ONCE for KPI deltas ---
    global_avg_metrics = {
        'temp': df['temperature_celsius'].mean(),
        'humidity': df['humidity'].mean(),
        'wind': df['wind_kph'].mean(),
        'precip': df['precip_mm'].mean(),
        'uv': df['uv_index'].mean(),
        'vis': df['visibility_km'].mean()
    }

    # --- Calculate min/max for sliders ONCE ---
    try:
        min_temp = float(df['temperature_celsius'].dropna().min())
        max_temp = float(df['temperature_celsius'].dropna().max())
        min_hum = float(df['humidity'].dropna().min())
        max_hum = float(df['humidity'].dropna().max())
        min_wind = float(df['wind_kph'].dropna().min())
        max_wind = float(df['wind_kph'].dropna().max())
        min_pressure = float(df['pressure_mb'].dropna().min())
        max_pressure = float(df['pressure_mb'].dropna().max())
        min_uv = float(df['uv_index'].dropna().min())
        max_uv = float(df['uv_index'].dropna().max())
        min_vis = float(df['visibility_km'].dropna().min())
        max_vis = float(df['visibility_km'].dropna().max())
    except Exception as e:
        st.error(f"Error setting slider bounds: {e}. Using defaults.")
        # Set default ranges if calculation fails
        min_temp, max_temp = -50.0, 50.0
        min_hum, max_hum = 0.0, 100.0
        min_wind, max_wind = 0.0, 100.0
        min_pressure, max_pressure = 900.0, 1100.0
        min_uv, max_uv = 0.0, 11.0
        min_vis, max_vis = 0.0, 20.0


    # --- Sidebar Filters ---
    st.sidebar.title("🌍 ClimateScope Filters")
    st.sidebar.markdown("Use the filters below to explore the data.")

    # --- UPDATED: Country Filter (Multi-select) ---
    countries = sorted(df['country'].unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        countries,
        default=[]  # Start with none selected (which will mean "All")
    )

    # --- UPDATED: Dependent Location Filter ---
    locations = ["All Locations"]
    # Only show locations IF exactly ONE country is selected
    if len(selected_countries) == 1:
        country_df = df[df['country'] == selected_countries[0]]
        unique_locations = country_df['location_name'].dropna().unique()
        if len(unique_locations) > 0:
            sorted_locations = sorted([str(loc) for loc in unique_locations])
            locations.extend(sorted_locations)
        
        help_text = "Select a location for a deep-dive."
    else:
        help_text = "Select a *single* country to filter by location."

    selected_location = st.sidebar.selectbox(
        "Select a Location (Optional)",
        locations,
        index=0,
        help=help_text,
        disabled=len(selected_countries) != 1 # Disable if not exactly one country
    )

    # Date Range Filter
    min_date = df['last_updated_date'].min().date()
    max_date = df['last_updated_date'].max().date()
    selected_date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    
    # --- NEW: Time Aggregation Filter ---
    selected_agg = st.sidebar.selectbox(
        "Select Time Aggregation",
        ["Raw Data", "Daily", "Weekly", "Monthly"],
        index=0, # Default to "Raw Data"
        help="Group data by a time period. 'Raw Data' shows every record."
    )

    # Metric Range Sliders
    st.sidebar.markdown("---") # Separator
    st.sidebar.markdown("**Filter by Metric Ranges:**")

    with st.sidebar.expander("Temperature Range", expanded=False):
        selected_temp_range = st.slider("°C", min_temp, max_temp, (min_temp, max_temp), label_visibility="collapsed")

    with st.sidebar.expander("Humidity Range", expanded=False):
        selected_hum_range = st.slider("%", min_hum, max_hum, (min_hum, max_hum), label_visibility="collapsed")

    with st.sidebar.expander("Wind Speed Range", expanded=True):
        selected_wind_range = st.slider("kph", min_wind, max_wind, (min_wind, max_wind), label_visibility="collapsed")

    with st.sidebar.expander("Pressure Range", expanded=False):
        selected_pressure_range = st.slider("mb", min_pressure, max_pressure, (min_pressure, max_pressure), label_visibility="collapsed")

    with st.sidebar.expander("UV Index Range", expanded=False):
        selected_uv_range = st.slider("UV Index", min_uv, max_uv, (min_uv, max_uv), label_visibility="collapsed")

    with st.sidebar.expander("Visibility Range", expanded=False):
        selected_vis_range = st.slider("km", min_vis, max_vis, (min_vis, max_vis), label_visibility="collapsed")

    # --- [START] COMPLETELY REVISED: Data Filtering Logic ---
    filtered_df = df.copy()
    map_zoom = 1
    filter_title = ""

    # 1. Filter by Country
    if selected_countries: # If the list is not empty
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
        filter_title = ", ".join(selected_countries)
        if len(selected_countries) == 1:
            map_zoom = 4
        else:
            map_zoom = 1 # Zoom out if multiple countries
    else:
        filter_title = "Global"
        map_zoom = 1

    # 2. Filter by Location (Only if one country is selected AND a location is chosen)
    if len(selected_countries) == 1 and selected_location != "All Locations":
        filtered_df = filtered_df[filtered_df['location_name'] == selected_location]
        filter_title += f" / {selected_location}"
        map_zoom = 8

    # 3. Filter by Date Range
    filter_message = f"Displaying data for: **{filter_title}**"
    if len(selected_date_range) == 2:
        try:
            start_date = pd.to_datetime(selected_date_range[0])
            # Add one day to end_date to make it inclusive
            end_date = pd.to_datetime(selected_date_range[1]) + pd.Timedelta(days=1)
            filtered_df = filtered_df[
                (filtered_df['last_updated_date'] >= start_date) &
                (filtered_df['last_updated_date'] < end_date) # Use < for end_date
            ]
            filter_message += f" | Date Range: **{selected_date_range[0]}** to **{selected_date_range[1]}**"
        except Exception as e:
            st.error(f"Invalid date range: {e}")
            filtered_df = pd.DataFrame(columns=df.columns) # Empty df if error

    # 4. Filter by Metric Sliders
    try:
        # Use boolean masks and combine them
        masks = []
        if selected_temp_range != (min_temp, max_temp):
            masks.append((filtered_df['temperature_celsius'] >= selected_temp_range[0]) & (filtered_df['temperature_celsius'] <= selected_temp_range[1]))
        if selected_hum_range != (min_hum, max_hum):
             masks.append((filtered_df['humidity'] >= selected_hum_range[0]) & (filtered_df['humidity'] <= selected_hum_range[1]))
        if selected_wind_range != (min_wind, max_wind):
            masks.append((filtered_df['wind_kph'] >= selected_wind_range[0]) & (filtered_df['wind_kph'] <= selected_wind_range[1]))
        if selected_pressure_range != (min_pressure, max_pressure):
            masks.append((filtered_df['pressure_mb'] >= selected_pressure_range[0]) & (filtered_df['pressure_mb'] <= selected_pressure_range[1]))
        if selected_uv_range != (min_uv, max_uv):
             masks.append((filtered_df['uv_index'] >= selected_uv_range[0]) & (filtered_df['uv_index'] <= selected_uv_range[1]))
        if selected_vis_range != (min_vis, max_vis):
            masks.append((filtered_df['visibility_km'] >= selected_vis_range[0]) & (filtered_df['visibility_km'] <= selected_vis_range[1]))

        # Apply combined mask, handling NaNs by keeping them (fillna(True))
        if masks:
            combined_mask = pd.concat(masks, axis=1).all(axis=1)
            filtered_df = filtered_df[combined_mask.fillna(True)]
            filter_message += " | **Metric ranges applied**"

    except Exception as e:
        st.error(f"Error applying metric filters: {e}")


    # --- 5. NEW: Apply Time Aggregation ---
    
    # This is the final, filtered DataFrame that will be passed to the tabs
    final_df = filtered_df.copy()

    if selected_agg != "Raw Data" and not filtered_df.empty:
        try:
            agg_rule = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}[selected_agg]
            
            # Define how to aggregate different columns
            # We'll .mean() numbers and take the .first() of categories
            
            numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
            # Explicitly list categorical columns used in plots
            categorical_cols = ['condition_text', 'aqi_label', 'uv_category', 'visibility_category']
            
            # Group by location so we don't average different cities together
            grouping_cols = ['country', 'location_name']
            
            # Columns to average
            cols_to_mean = [col for col in numeric_cols if col not in grouping_cols]
            # Columns to take first value
            cols_to_first = [col for col in categorical_cols if col in filtered_df.columns]
            
            # Create aggregation dictionary
            agg_dict = {}
            for col in cols_to_mean:
                agg_dict[col] = 'mean'
            for col in cols_to_first:
                agg_dict[col] = 'first'
                
            # Perform the groupby + resample operation
            
            # Create a list to store resampled dataframes
            resampled_dfs = []
            
            # Group by each location and resample
            for (country, location), group in filtered_df.groupby(grouping_cols):
                # Set date as index for resampling
                group = group.set_index('last_updated_date')
                
                # Apply resampling and aggregation
                resampled_group = group.resample(rule=agg_rule).agg(agg_dict)
                
                # Add back the grouping columns
                resampled_group['country'] = country
                resampled_group['location_name'] = location
                
                resampled_dfs.append(resampled_group)

            if resampled_dfs:
                final_df = pd.concat(resampled_dfs).reset_index()
                # Re-create 'year' column as it might be needed and was averaged
                final_df['year'] = final_df['last_updated_date'].dt.year
            else:
                # If no data after grouping (unlikely), create empty df
                final_df = pd.DataFrame(columns=df.columns)

            filter_message += f" | Aggregation: **{selected_agg}**"
            
        except Exception as e:
            st.error(f"Error applying time aggregation: {e}")
            final_df = pd.DataFrame(columns=df.columns) # Empty df if error
    
    # Update filter message for Raw Data
    elif selected_agg == "Raw Data":
        filter_message += f" | Aggregation: **Raw Data**"

    # --- [END] COMPLETELY REVISED: Data Filtering Logic ---


    # --- Create Tabs using streamlit-option-menu ---
    selected_tab = option_menu(
        menu_title=None,
        options=[
            "Executive Dashboard", "Location Deep-Dive", "Humidity Analysis", "Country Comparison",
            "Correlations", "Extreme Events", "Forecasting", "Data Explorer", "About Project"
        ],
        icons=[
            "bi-bar-chart-line-fill", "bi-pin-map-fill", "bi-moisture", "bi-files",
            "bi-pie-chart-fill", "bi-tornado", "bi-graph-up-arrow", "bi-database-fill", "bi-info-circle-fill"
        ],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
             "container": {"padding": "5px!important", "background": "linear-gradient(180deg, #e0f2fe 0%, #f3f4f6 100%)"},
             "icon": {"color": "#4f46e5", "font-size": "16px"},
             "nav-link": {
                 "font-family": "Inter",
                 "font-size": "14px",
                 "font-weight": "600",
                 "text-align": "center",
                 "margin":"0px 4px",
                 "padding":"10px 15px",
                 "color": "#4f46e5",
                 "background-color": "rgba(255, 255, 255, 0.7)",
                 "border-radius": "10px",
                 "border": "1px solid rgba(255, 255, 255, 0.3)",
                 "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.05)",
                 "--hover-color": "#ffffff"
             },
             "nav-link-selected": {
                 "font-family": "Inter",
                 "background": "linear-gradient(90deg, #4f46e5 0%, #a855f7 100%)",
                 "color": "#ffffff",
                 "font-weight": "700",
                 "border": "1px solid #4f46e5",
                 "box-shadow": "0 4px 8px rgba(79, 70, 229, 0.3)"
             },
         }
    )

    # --- Render Selected Tab (Using final_df) ---
    if selected_tab == "Executive Dashboard":
        render_global_overview_tab(final_df, global_avg_metrics, filter_message, selected_countries, map_zoom)
    elif selected_tab == "Location Deep-Dive":
        render_location_deep_dive_tab(final_df, global_avg_metrics, filter_title, selected_countries, selected_location)
    elif selected_tab == "Humidity Analysis":
        render_humidity_tab(final_df, filter_message)
    elif selected_tab == "Country Comparison":
        render_comparison_tab(df, countries) # Pass original df and full country list
    elif selected_tab == "Correlations":
        render_trends_tab(final_df, filter_message)
    elif selected_tab == "Extreme Events":
        render_extremes_tab(final_df, filter_message)
    elif selected_tab == "Forecasting":
        # Forecasting tab needs the *un-aggregated* data for the selected location
        render_forecasting_tab(filtered_df, selected_countries, selected_location)
    elif selected_tab == "Data Explorer":
        render_data_explorer_tab(final_df, filter_message)
    elif selected_tab == "About Project":
        render_about_tab()

else:
    st.error("Fatal Error: Data could not be loaded. Dashboard cannot be displayed.")
    st.warning("Ensure 'weather_cleaned.csv' is in the same directory and not corrupted.")