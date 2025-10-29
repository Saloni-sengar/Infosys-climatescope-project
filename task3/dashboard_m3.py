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

# --- Page Configuration ---
# Set the layout to "wide" to use the full screen width
st.set_page_config(
    page_title="ClimateScope Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a more polished "website-like" feel ---
st.markdown("""
<style>
    /* Main app background with subtle gradient */
    .main .block-container {
        /* background-color: #0E1117; */
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); /* slate-800 to slate-900 */
        padding-top: 2rem; /* Add padding at the top */
        padding-bottom: 2rem;
        border-radius: 10px; /* Optional: adds rounded corners to main area */
    }

    /* Sidebar styling */
    .st-emotion-cache-16txtl3 { /* Specific selector for sidebar */
        background-color: #1a1c24; /* Slightly darker gray */
        border-right: 1px solid #3a3c44;
    }
    .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3, .st-emotion-cache-16txtl3 label {
        color: #e2e8f0; /* Lighter text for sidebar */
    }

    /* Metric box styling with enhanced appearance */
    .st-emotion-cache-1vze3mj { /* Selector for metric boxes */
        background-color: #1f2937; /* slate-800 */
        border: 1px solid #334155; /* slate-700 */
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease-in-out;
    }
    .st-emotion-cache-1vze3mj:hover {
        transform: translateY(-3px); /* Subtle lift on hover */
    }
    .st-emotion-cache-1vze3mj .stMetricLabel, .st-emotion-cache-1vze3mj .stMetricValue {
         color: #f1f5f9; /* slate-100 */
    }
    /* Style the delta (up = green, down = red) */
    .st-emotion-cache-1vze3mj .stMetricDelta [data-testid="stMetricDelta"] {
         color: #f1f5f9; /* Default delta color */
    }
    .st-emotion-cache-1vze3mj .stMetricDelta [data-testid="stMetricDelta"] > div:first-child {
         color: #f87171; /* Red for negative delta */
    }
    .st-emotion-cache-1vze3mj .stMetricDelta [data-testid="stMetricDelta"] > div:last-child {
         color: #4ade80; /* Green for positive delta */
    }


    /* Tab styling - more distinct */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; /* Increase gap between tabs */
        border-bottom: 2px solid #334155; /* Add a bottom border to the tab list */
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; /* Make inactive tabs transparent */
        border-radius: 8px 8px 0 0;
        margin-right: 0px;
        padding: 10px 15px;
        color: #94a3b8; /* slate-400 - inactive tab color */
        border: none;
        border-bottom: 2px solid transparent; /* Prepare for active state border */
        transition: all 0.2s ease-in-out;
    }
     .stTabs [data-baseweb="tab"]:hover {
        background-color: #334155; /* slate-700 - Hover background */
        color: #e2e8f0; /* slate-200 - Hover text color */
     }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1f2937; /* slate-800 - Active tab background */
        color: #ffffff;
        font-weight: bold;
        border: 2px solid #334155; /* Border matching the tab list */
        border-bottom: 2px solid #1f2937; /* Bottom border blends with background */
        margin-bottom: -2px; /* Pull tab up slightly */
    }

    /* Headers */
    h1, h2, h3 {
        color: #cbd5e1; /* slate-300 */
        font-weight: 600; /* Slightly bolder */
    }
    h1 {
        color: #f1f5f9; /* slate-100 */
        text-align: center; /* Center main title */
        padding-bottom: 1rem;
        /* NEW: Added shadow for depth */
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    h2 {
         border-bottom: 1px solid #475569; /* slate-600 */
         padding-bottom: 0.5rem;
         margin-top: 1.5rem;
    }
    h3 {
         margin-top: 1rem;
         color: #e2e8f0; /* slate-200 */
    }

    /* Style markdown links */
    a {
        color: #60a5fa; /* blue-400 */
    }
    a:hover {
        color: #93c5fd; /* blue-300 */
    }

    /* Style captions */
    .stCaption {
        color: #94a3b8; /* slate-400 */
        font-style: italic;
    }
    
    /* NEW: Style expanders to match theme */
    [data-testid="stExpander"] {
        background-color: #1f293788; /* slate-800 with transparency */
        border-radius: 10px;
        border: 1px solid #334155; /* slate-700 */
    }
    [data-testid="stExpander"] [data-testid="stExpanderHeader"] {
        color: #e2e8f0; /* slate-200 */
        font-weight: 600;
    }

    /* NEW: Style Plotly charts with rounded corners */
    [data-testid="stPlotlyChart"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #334155; /* slate-700 */
    }
    
    /* NEW: Style download button */
    .stDownloadButton > button {
        background-color: #2563eb; /* blue-600 */
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: background-color 0.2s ease-in-out;
    }
    .stDownloadButton > button:hover {
        background-color: #3b82f6; /* blue-500 */
    }

</style>
""", unsafe_allow_html=True)

# --- Data Loading and Caching ---
@st.cache_data  # Cache the data loading for performance
def load_data(filepath):
    """
    Loads and preprocesses the weather data from a CSV file.
    """
    try:
        # Check if running in Streamlit cloud
        if 'streamlit' in os.environ.get('HOSTNAME', ''):
             # Assume file is in the root
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

        # Map AQI index to human-readable names
        data['aqi_label'] = data['air_quality_us-epa-index'].map({
            1: '1 - Good',
            2: '2 - Moderate',
            3: '3 - Unhealthy (SG)',
            4: '4 - Unhealthy',
            5: '5 - Very Unhealthy',
            6: '6 - Hazardous'
        }).fillna('Unknown')

        return data
    except FileNotFoundError:
        st.error(f"Error: The data file '{filepath}' was not found.")
        st.info(f"The script tried to find the file here: `{os.path.abspath(filepath)}`")
        st.warning("Please make sure 'weather_cleaned.csv' is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading or processing the data: {e}")
        return None

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

    colors = px.colors.sequential.Viridis_r # Reversed Viridis for better contrast on dark
    for i, speed in enumerate(speed_labels):
        if speed in df_rose_pivot_percent.columns:
            fig.add_trace(go.Barpolar(
                r=df_rose_pivot_percent[speed],
                theta=df_rose_pivot_percent.index,
                name=f'{speed} kph',
                marker_color=colors[min(i, len(colors)-1)]
            ))

    fig.update_layout(
        #title='Wind Rose (Frequency % by Speed and Direction)',
        template='plotly_dark',
        polar=dict(radialaxis=dict(visible=True, ticksuffix='%', gridcolor='#475569'), # slate-600
                   angularaxis=dict(direction="clockwise", period=360, gridcolor='#475569')), # slate-600
        legend_title="Wind Speed (kph)",
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0' # slate-200
    )
    return fig

# --- Helper Function for CSV Download ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for download."""
    return df.to_csv(index=False).encode('utf-8')

# --- Tab Rendering Functions ---

def render_global_overview_tab(filtered_df, global_avg_metrics, filter_message, selected_country, map_zoom):
    """Renders the content for the Global Overview tab."""
    st.header("🛰️ Global Weather Snapshot")
    st.markdown(filter_message)

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Global KPIs with Delta
        st.subheader("High-Level Metrics (Filtered Data)")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        # Calculate filtered metrics
        filtered_avg_temp = filtered_df['temperature_celsius'].mean()
        filtered_avg_humidity = filtered_df['humidity'].mean()
        filtered_avg_wind = filtered_df['wind_kph'].mean()
        total_reports = len(filtered_df)

        # Calculate deltas (only if a country is selected)
        delta_temp_str, delta_hum_str, delta_wind_str = None, None, None
        if selected_country != "All Countries":
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
        st.subheader("Interactive Map")
        try:
            # Updated Logic: Show Choropleth ONLY if "All Countries" is selected
            if selected_country == "All Countries":
                st.markdown("Global Average Temperature by Country")
                country_avg = filtered_df.groupby('country')['temperature_celsius'].mean().reset_index()
                fig_choro = px.choropleth(
                    country_avg, locations='country', locationmode='country names',
                    color='temperature_celsius', hover_name='country',
                    hover_data={'temperature_celsius': ':.1f'},
                    color_continuous_scale=px.colors.sequential.YlOrRd, template='plotly_dark')
                fig_choro.update_geos(bgcolor='rgba(0,0,0,0)', landcolor='#334155', subunitcolor='#475569', showcountries=True) # Adjusted colors
                fig_choro.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_colorbar_title='Avg Temp (°C)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_choro, use_container_width=True, theme="streamlit")
            else:
                # Show scatter map if a country or location is selected
                st.markdown("Live Temperature by Location")
                map_center = {"lat": filtered_df['latitude'].mean(), "lon": filtered_df['longitude'].mean()} if not filtered_df.empty else {"lat": 0, "lon": 0}
                fig_map = px.scatter_geo(
                    filtered_df, lat='latitude', lon='longitude', color='temperature_celsius',
                    hover_name='location_name', hover_data={"condition_text": True, "temperature_celsius": ":.1f", "latitude": False, "longitude": False},
                    template='plotly_dark', projection='natural earth')
                fig_map.update_geos(bgcolor='rgba(0,0,0,0)', landcolor='#334155', subunitcolor='#475569') # Adjusted colors
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_colorbar_title='Temp (°C)', geo_center=map_center, geo_projection_scale=map_zoom, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_map, use_container_width=True, theme="streamlit")
        except Exception as e:
            st.error(f"Could not display map: {e}")

        # Comparative Charts
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.subheader("📊 Avg. Precipitation by Country (Top 20)")
            try:
                # Only show if 'All Countries' is selected
                if selected_country == "All Countries":
                    precip_data = filtered_df.groupby('country')['precip_mm'].mean().nlargest(20).reset_index()
                    fig_precip = px.bar(precip_data, x='country', y='precip_mm', template='plotly_dark')
                    fig_precip.update_layout(yaxis_title='Precipitation (mm)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_precip, use_container_width=True, theme="streamlit")
                else:
                    st.info("Global comparison requires 'All Countries' selection.")
            except Exception as e:
                st.error(f"Could not display precipitation chart: {e}")

        with comp_col2:
            st.subheader("💨 Avg. Wind Speed by Country (Top 20)")
            try:
                # Only show if 'All Countries' is selected
                if selected_country == "All Countries":
                    wind_data = filtered_df.groupby('country')['wind_kph'].mean().nlargest(20).reset_index()
                    fig_wind_bar = px.bar(wind_data, x='country', y='wind_kph', template='plotly_dark')
                    fig_wind_bar.update_layout(yaxis_title='Wind Speed (kph)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_wind_bar, use_container_width=True, theme="streamlit")
                else:
                    st.info("Global comparison requires 'All Countries' selection.")
            except Exception as e:
                st.error(f"Could not display wind chart: {e}")

def render_location_deep_dive_tab(filtered_df, global_avg_metrics, filter_title, selected_country):
    """Renders the content for the Location Deep-Dive tab."""
    st.header(f"📍 Location Deep-Dive: {filter_title}") # Title uses updated filter_title

    # Check if a specific country is selected (not 'All Countries')
    if selected_country == "All Countries":
        st.info("Please select a single country from the sidebar to see a detailed deep-dive.")
    elif filtered_df.empty:
        st.warning("No data available for the selected location and date range.")
    else:
        # Location KPIs with Delta
        st.subheader("Key Metrics for this Location (vs. Global Avg)")
        kpi_col1_c, kpi_col2_c, kpi_col3_c, kpi_col4_c, kpi_col5_c = st.columns(5)
        
        # Calculate metrics
        loc_avg_temp = filtered_df['temperature_celsius'].mean()
        loc_max_wind = filtered_df['wind_kph'].max()
        loc_avg_precip = filtered_df['precip_mm'].mean()
        loc_avg_uv = filtered_df['uv_index'].mean()
        loc_avg_vis = filtered_df['visibility_km'].mean()

        # Calculate deltas
        delta_temp = loc_avg_temp - global_avg_metrics['temp']
        delta_precip = loc_avg_precip - global_avg_metrics['precip']
        delta_uv = loc_avg_uv - global_avg_metrics['uv']
        delta_vis = loc_avg_vis - global_avg_metrics['vis']

        kpi_col1_c.metric("Avg. Temp", f"{loc_avg_temp:.1f} °C", delta=f"{delta_temp:+.1f} °C")
        kpi_col2_c.metric("Max Wind", f"{loc_max_wind:.1f} kph")
        kpi_col3_c.metric("Avg. Precip", f"{loc_avg_precip:.2f} mm", delta=f"{delta_precip:+.2f} mm")
        kpi_col4_c.metric("Avg. UV Index", f"{loc_avg_uv:.1f}", delta=f"{delta_uv:+.1f}")
        kpi_col5_c.metric("Avg. Visibility", f"{loc_avg_vis:.1f} km", delta=f"{delta_vis:+.1f} km")

        # Time-Series Charts
        st.subheader("📈 Time-Series Trends")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown(f"**Temperature Trend**")
            try:
                fig_trend = px.line(filtered_df, x='last_updated_date', y='temperature_celsius', template='plotly_dark')
                fig_trend.update_layout(xaxis_title='Date', yaxis_title='Temp (°C)', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_trend, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Temp Trend Error: {e}")

            st.markdown(f"**Pressure Trend**")
            try:
                fig_pres_trend = px.line(filtered_df, x='last_updated_date', y='pressure_mb', template='plotly_dark')
                fig_pres_trend.update_layout(xaxis_title='Date', yaxis_title='Pressure (mb)', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                fig_pres_trend.update_traces(line_color='#f87171') # Red
                st.plotly_chart(fig_pres_trend, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Pressure Trend Error: {e}")

        with chart_col2:
            st.markdown(f"**Humidity Trend**")
            try:
                fig_hum_trend = px.line(filtered_df, x='last_updated_date', y='humidity', template='plotly_dark')
                fig_hum_trend.update_layout(xaxis_title='Date', yaxis_title='Humidity (%)', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                fig_hum_trend.update_traces(line_color='#60a5fa') # Blue
                st.plotly_chart(fig_hum_trend, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Humidity Trend Error: {e}")

            st.markdown(f"**Avg. Precipitation by Location (Top 20)**")
            try:
                # This chart is more useful if we are looking at a whole country
                # If a single location is selected, it will just show that one location
                precip_data = filtered_df.groupby('location_name')['precip_mm'].mean().nlargest(20).reset_index()
                fig_precip_loc = px.bar(precip_data, x='location_name', y='precip_mm', template='plotly_dark')
                fig_precip_loc.update_layout(yaxis_title='Precipitation (mm)', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_precip_loc, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Precipitation Bar Error: {e}")

        # Condition Distributions
        st.subheader("📊 Condition Distributions")
        chart_col3, chart_col4, chart_col5 = st.columns(3)
        with chart_col3:
            st.markdown(f"**Top 15 Weather Conditions**")
            try:
                condition_counts = filtered_df['condition_text'].value_counts().nlargest(15).reset_index()
                fig_cond = px.bar(condition_counts, y='condition_text', x='count', orientation='h', template='plotly_dark')
                fig_cond.update_layout(yaxis_title=None, xaxis_title='Count', yaxis={'categoryorder':'total ascending'}, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_cond, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Conditions Bar Error: {e}")

        with chart_col4:
            st.markdown(f"**Wind Speed Distribution**")
            try:
                fig_wind = px.histogram(filtered_df, x='wind_kph', template='plotly_dark')
                fig_wind.update_layout(xaxis_title='Wind Speed (kph)', yaxis_title='Frequency', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_wind, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Wind Histogram Error: {e}")

        with chart_col5:
            st.markdown(f"**Cloud Cover Distribution**")
            try:
                fig_cloud = px.histogram(filtered_df, x='cloud', template='plotly_dark')
                fig_cloud.update_layout(xaxis_title='Cloud Cover (%)', yaxis_title='Frequency', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                fig_cloud.update_traces(marker_color='#d1d5db') # Gray
                st.plotly_chart(fig_cloud, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Cloud Histogram Error: {e}")

        # Air Quality Pie Chart
        st.subheader("🌬️ Air Quality")
        try:
            aqi_counts = filtered_df['aqi_label'].value_counts().reset_index()
            if aqi_counts.empty or aqi_counts['count'].sum() == 0:
                st.info("No Air Quality data available for this selection.")
            else:
                fig_aqi = px.pie(aqi_counts, names='aqi_label', values='count',
                                 title=f"Air Quality Index (US EPA)", template='plotly_dark', hole=0.4)
                fig_aqi.update_traces(textposition='inside', textinfo='percent+label')
                fig_aqi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_aqi, use_container_width=True, theme="streamlit")
        except Exception as e: st.error(f"AQI Pie Error: {e}")

def render_trends_tab(filtered_df, filter_message):
    """Renders the content for the Trends & Correlations tab."""
    st.header("📈 Trends & Correlation Analysis")
    st.markdown(filter_message)

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.subheader("Correlation Heatmap")
        st.markdown("Shows the statistical correlation between different weather metrics (1.0 = perfect positive, -1.0 = perfect negative).")
        try:
            numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
            # Remove lat/lon/year as they aren't direct weather metrics for correlation
            cols_to_correlate = [col for col in numeric_cols if col not in ['latitude', 'longitude', 'year', 'last_updated_epoch', 'air_quality_us-epa-index', 'air_quality_gb-defra-index']]
            
            # Ensure there are enough columns to correlate
            if len(cols_to_correlate) > 1:
                corr = filtered_df[cols_to_correlate].corr()
                fig_corr_heat = px.imshow(corr, text_auto=".2f", aspect="auto", # Format to 2 decimal places
                                          color_continuous_scale=px.colors.diverging.RdBu, color_continuous_midpoint=0,
                                          template='plotly_dark')
                fig_corr_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_corr_heat, use_container_width=True, theme="streamlit")
                st.caption("Insight: Notice the expected negative correlation between Temperature and Humidity in many regions, and positive correlation between Temperature and UV Index.")
            else:
                st.info("Not enough numeric data to generate a correlation heatmap for this selection.")
        except Exception as e:
            st.error(f"Could not display correlation heatmap: {e}")
        
        # --- NEW: Dynamic Correlation Explorer ---
        st.subheader("Dynamic Correlation Explorer")
        st.markdown("Select any two numeric variables to see their relationship on a scatter plot.")
        
        try:
            numeric_cols_list = filtered_df.select_dtypes(include=np.number).columns.tolist()
            cols_to_plot = [col for col in numeric_cols_list if col not in ['latitude', 'longitude', 'year', 'last_updated_epoch']]
            
            # Default selections
            try:
                default_x = cols_to_plot.index('temperature_celsius')
            except ValueError:
                default_x = 0
                
            try:
                default_y = cols_to_plot.index('humidity')
            except ValueError:
                default_y = 1
            
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                x_axis = st.selectbox("Select X-Axis", cols_to_plot, index=default_x)
            with exp_col2:
                y_axis = st.selectbox("Select Y-Axis", cols_to_plot, index=default_y)
            
            # Sample for performance
            scatter_df_dynamic = filtered_df.sample(min(len(filtered_df), 2000))
            
            fig_dynamic = px.scatter(scatter_df_dynamic, x=x_axis, y=y_axis,
                                     color='temperature_celsius', 
                                     color_continuous_scale=px.colors.sequential.YlOrRd,
                                     title=f"{x_axis} vs. {y_axis}",
                                     template='plotly_dark')
            fig_dynamic.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dynamic, use_container_width=True, theme="streamlit")
            
        except Exception as e:
            st.error(f"Could not generate dynamic scatter plot: {e}")

        # --- End of New Section ---

        corr_col1, corr_col2 = st.columns(2)
        with corr_col1:
            st.subheader("🌡️ Temperature vs. Humidity")
            try:
                # Sample data for performance if dataframe is large
                scatter_df = filtered_df.sample(min(len(filtered_df), 2000)) 
                fig_scatter = px.scatter(scatter_df, x='temperature_celsius', y='humidity',
                                         color='temperature_celsius', color_continuous_scale=px.colors.sequential.YlOrRd,
                                         title="Temperature vs. Humidity (Sampled)", template='plotly_dark')
                fig_scatter.update_layout(xaxis_title='Temp (°C)', yaxis_title='Humidity (%)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_scatter, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Temp/Humidity Scatter Error: {e}")

        with corr_col2:
            st.subheader("☀️ Temperature vs. UV Index")
            try:
                scatter_df_uv = filtered_df.sample(min(len(filtered_df), 2000))
                fig_scatter_uv = px.scatter(scatter_df_uv, x='temperature_celsius', y='uv_index',
                                              color='uv_index', color_continuous_scale=px.colors.sequential.Agsunset,
                                              title="Temperature vs. UV Index (Sampled)", template='plotly_dark')
                fig_scatter_uv.update_layout(xaxis_title='Temp (°C)', yaxis_title='UV Index', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_scatter_uv, use_container_width=True, theme="streamlit")
            except Exception as e: st.error(f"Temp/UV Scatter Error: {e}")

        st.subheader("📅 Average Monthly Temperature Heatmap")
        try:
            heatmap_data = filtered_df.pivot_table(values='temperature_celsius', index='year', columns='month_name', aggfunc='mean')
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            heatmap_data = heatmap_data.reindex(columns=month_order) # Ensure correct month order
            fig_heat = px.imshow(heatmap_data, aspect='auto', color_continuous_scale=px.colors.sequential.YlOrRd,
                                 title="Average Monthly Temperature (°C)", template='plotly_dark')
            fig_heat.update_layout(xaxis_title='Month', yaxis_title='Year', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_heat, use_container_width=True, theme="streamlit")
            st.caption("Insight: This heatmap clearly visualizes seasonal temperature variations year over year.")
        except Exception as e:
            st.error(f"Could not display monthly heatmap: {e}")

def render_extremes_tab(filtered_df, filter_message):
    """Renders the content for the Extreme Events & Wind tab."""
    st.header("🌪️ Extreme Event & Wind Analysis")
    st.markdown(filter_message)

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.subheader("Wind Rose")
        st.markdown("Shows the frequency of wind from different directions, colored by speed (kph).")
        try:
            fig_rose = create_wind_rose(filtered_df) # Use function
            st.plotly_chart(fig_rose, use_container_width=True, theme="streamlit")
        except Exception as e:
            st.error(f"Could not display Wind Rose: {e}")
            st.exception(e) # Show more details

        st.subheader("Extreme Event Data Tables")
        st.markdown("Top 10 most extreme weather events recorded within the filtered data.")
        extreme_cols_to_show = ['location_name', 'country', 'last_updated', 'temperature_celsius', 'wind_kph', 'precip_mm']

        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            st.subheader("🔥 Top 10 Hottest Events")
            st.dataframe(filtered_df.nlargest(10, 'temperature_celsius')[extreme_cols_to_show], use_container_width=True)

            st.subheader("💨 Top 10 Windiest Events")
            st.dataframe(filtered_df.nlargest(10, 'wind_kph')[extreme_cols_to_show], use_container_width=True)

        with ex_col2:
            st.subheader("❄️ Top 10 Coldest Events")
            st.dataframe(filtered_df.nsmallest(10, 'temperature_celsius')[extreme_cols_to_show], use_container_width=True)

            st.subheader("🌧️ Top 10 Rainiest Events")
            st.dataframe(filtered_df.nlargest(10, 'precip_mm')[extreme_cols_to_show], use_container_width=True)
        st.caption("Insight: These tables help identify locations and times prone to specific extreme weather conditions like heatwaves or high winds.")

def render_data_explorer_tab(filtered_df, filter_message):
    """Renders the content for the Data Explorer tab."""
    st.header("🗂️ Data Explorer")
    st.markdown(filter_message)

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # NEW: Download Button
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
        
        # NEW: Put summary in an expander
        with st.expander("Show Statistical Summary"):
            st.markdown("A statistical overview of the numeric data for the selected filters.")
            try:
                numeric_cols_for_desc = filtered_df.select_dtypes(include=np.number).columns
                st.dataframe(filtered_df[numeric_cols_for_desc].describe(), use_container_width=True)
            except Exception as e: st.error(f"Could not display data summary: {e}")

        st.subheader("Filtered Raw Data")
        st.markdown("The complete raw data set based on your current filter selection. You can sort by clicking on column headers.")
        try:
            # Display subset of columns for better readability initially
            cols_to_display = ['location_name', 'country', 'last_updated_date', 'temperature_celsius', 'condition_text', 'wind_kph', 'precip_mm', 'humidity', 'uv_index', 'aqi_label']
            # Ensure all columns exist before trying to display them
            cols_to_display_safe = [col for col in cols_to_display if col in filtered_df.columns]
            st.dataframe(filtered_df[cols_to_display_safe], use_container_width=True)
            
            # Option to show all columns
            if st.checkbox("Show All Columns"):
                st.dataframe(filtered_df, use_container_width=True)
        except Exception as e: st.error(f"Could not display raw data table: {e}")

def render_about_tab():
    """Renders the content for the About Project tab."""
    st.header("ℹ️ About This Project")

    # NEW: Use expanders to make the page cleaner
    with st.expander("Project Objective", expanded=True):
        st.markdown("""
        The objective of ClimateScope is to analyze and visually represent global weather patterns using the Global Weather Repository dataset. This project aims to uncover seasonal trends, regional variations, and extreme weather events through interactive and insightful visualizations. By leveraging daily-updated, worldwide weather data, the project enables users to explore climate behavior over time, compare conditions across regions, and identify anomalies. The ultimate goal is to provide an accessible, data-driven platform that supports climate awareness, decision-making, and further research into global weather dynamics.
        """)

    with st.expander("Data Visualization Design"):
        st.markdown("""
        - **Suitable Visualization Types:**
            - **Choropleth maps:** Used on the Global Overview tab to show average temperature patterns across countries. 
            - **Line charts:** Employed extensively in the Location Deep-Dive tab for visualizing time-series trends (Temperature, Humidity, Pressure).
            - **Scatterplots:** Utilized in the Trends & Correlations tab to explore relationships between variables (Temp vs. Humidity, Temp vs. UV Index).
            - **Heatmaps:** Implemented for showing Correlation Analysis and Average Monthly Temperature variations.
            - **Bar Charts:** Used for comparing discrete categories like Top Conditions, Average Precipitation/Wind by Country/Location.
            - **Histograms:** Applied to show distributions of continuous variables like Wind Speed and Cloud Cover.
            - **Pie Chart (Donut):** Used for visualizing the proportion of Air Quality Index levels.
            - **Wind Rose:** A specialized polar bar chart on the Extreme Events & Wind tab to show wind speed/direction frequency. 
        - **Interactive Dashboard Layout:** A multi-tab layout was designed using Streamlit Tabs for logical grouping of information. Interactive filters (Country dropdown, Date range slider) are placed in the sidebar for global control. A consistent dark theme with custom CSS enhances aesthetics.
        """)

    with st.expander("Visualization Development"):
        st.markdown("""
        - **Tools:** Built using Python with Streamlit for the web application framework and Plotly (Plotly Express & Graph Objects) for creating interactive visualizations. Pandas was used for data manipulation.
        - **Interactivity:** Integrated filters (Country dropdown, Date range slider) that dynamically update all visualizations. Plotly charts inherently offer hover-over details, zooming, and panning.
        """)

    with st.expander("Insights Generation"):
        st.markdown("""
        - **Notable Findings:** Captions and context are provided alongside key charts to highlight insights. For instance:
            - The **Delta KPIs** instantly show how a region compares to the global average.
            - The **Correlation Heatmap** identifies expected and sometimes unexpected relationships between weather variables.
            - The **Extreme Events** tables directly pinpoint specific instances of high/low temperatures, wind, and precipitation.
            - The **Monthly Temperature Heatmap** clearly shows seasonal cycles.
            - The **Wind Rose** illustrates dominant wind patterns for a selected location.
        - **Trend Summarization:** Time-series charts on the Location Deep-Dive tab allow users to visually identify warming/cooling trends, humidity changes, etc., within the selected date range. Regional trends can be inferred by comparing different countries or locations using the filters.
        """)

    with st.expander("Tech Stack Summary"):
        st.markdown("""
        - **Language:** Python 3
        - **Libraries:** Streamlit, Pandas, Plotly, NumPy, **Prophet**
        - **Data Source:** [Kaggle Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/data)
        """)

# --- NEW: COMPARISON TAB FUNCTION ---
def render_comparison_tab(df, all_countries_list):
    """Renders the content for the 1-vs-1 Comparison tab."""
    st.header("🆚 Country Comparison")
    st.markdown("Select two countries to compare their key metrics and temperature trends side-by-side.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ensure 'All Countries' is not an option here
        countries_to_select = [c for c in all_countries_list if c != "All Countries"]
        if not countries_to_select:
             st.warning("No country data available.")
             return
             
        country_a = st.selectbox("Select Country A", countries_to_select, index=0)
        df_a = df[df['country'] == country_a]
        
        if df_a.empty:
            st.warning(f"No data for {country_a}")
        else:
            st.subheader(f"Metrics for {country_a}")
            avg_temp_a = df_a['temperature_celsius'].mean()
            avg_hum_a = df_a['humidity'].mean()
            avg_wind_a = df_a['wind_kph'].mean()
            
            st.metric("Avg. Temp", f"{avg_temp_a:.1f} °C")
            st.metric("Avg. Humidity", f"{avg_hum_a:.1f} %")
            st.metric("Avg. Wind", f"{avg_wind_a:.1f} kph")
            
            st.markdown(f"**Temperature Trend for {country_a}**")
            fig_a = px.line(df_a, x='last_updated_date', y='temperature_celsius', template='plotly_dark')
            fig_a.update_layout(xaxis_title='Date', yaxis_title='Temp (°C)', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_a, use_container_width=True, theme="streamlit")

    with col2:
        countries_to_select_b = [c for c in all_countries_list if c != "All Countries"]
        if not countries_to_select_b:
             st.warning("No country data available.")
             return
             
        # Default to the second country in the list
        default_index_b = 1 if len(countries_to_select_b) > 1 else 0
        country_b = st.selectbox("Select Country B", countries_to_select_b, index=default_index_b)
        df_b = df[df['country'] == country_b]
        
        if df_b.empty:
            st.warning(f"No data for {country_b}")
        else:
            st.subheader(f"Metrics for {country_b}")
            avg_temp_b = df_b['temperature_celsius'].mean()
            avg_hum_b = df_b['humidity'].mean()
            avg_wind_b = df_b['wind_kph'].mean()
            
            st.metric("Avg. Temp", f"{avg_temp_b:.1f} °C")
            st.metric("Avg. Humidity", f"{avg_hum_b:.1f} %")
            st.metric("Avg. Wind", f"{avg_wind_b:.1f} kph")
            
            st.markdown(f"**Temperature Trend for {country_b}**")
            fig_b = px.line(df_b, x='last_updated_date', y='temperature_celsius', template='plotly_dark')
            fig_b.update_layout(xaxis_title='Date', yaxis_title='Temp (°C)', margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig_b.update_traces(line_color='#f87171') # Different color
            st.plotly_chart(fig_b, use_container_width=True, theme="streamlit")

# --- NEW: FORECASTING TAB FUNCTION ---
@st.cache_data
def run_forecast(data, periods):
    """Fits Prophet model and returns forecast."""
    m = Prophet()
    m.fit(data)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return m, forecast

def render_forecasting_tab(filtered_df, selected_country, selected_location):
    """Renders the content for the Forecasting tab."""
    st.header("🔮 Temperature Forecast")
    
    # Forecasting requires a single, specific location
    if selected_country == "All Countries" or selected_location == "All Locations":
        st.info("Please select a specific Country **and** Location from the sidebar to generate a forecast.")
        return

    if filtered_df.empty:
        st.warning("No data available for the selected location to create a forecast.")
        return
        
    st.markdown(f"Generating forecast for **{selected_location}, {selected_country}**.")
    st.markdown("This tool uses the Prophet forecasting model to predict future temperature trends. This is for illustrative purposes only.")

    # Add a slider to select forecast period
    periods_to_forecast = st.slider("Select forecast period (days):", 30, 365, 90)

    try:
        with st.spinner(f"Generating {periods_to_forecast}-day forecast... This may take a moment."):
            # Prepare data for Prophet
            # Prophet needs columns 'ds' (date) and 'y' (value)
            # We average the temperature per day for a cleaner forecast
            forecast_df = filtered_df.set_index('last_updated_date').resample('D')['temperature_celsius'].mean().reset_index()
            forecast_df = forecast_df.rename(columns={'last_updated_date': 'ds', 'temperature_celsius': 'y'})
            forecast_df = forecast_df.dropna() # Remove any days with no data

            if len(forecast_df) < 30:
                st.error("Not enough data points (< 30 days) for this location to generate a reliable forecast.")
                return

            # Run the forecast
            model, forecast = run_forecast(forecast_df, periods_to_forecast)

            st.subheader("Forecast Plot")
            # Use Prophet's built-in Plotly function
            fig_forecast = plot_plotly(model, forecast)
            # Customize for our dark theme
            fig_forecast.update_layout(
                template='plotly_dark', 
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Date",
                yaxis_title="Temperature (°C)"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.subheader("Forecast Components")
            st.markdown("Prophet breaks the forecast down into its underlying components:")
            # Use Prophet's component plot
            fig_components = model.plot_components(forecast)
            # This returns a matplotlib fig, so we use st.pyplot
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

    # --- Sidebar Filters ---
    st.sidebar.title("🌍 ClimateScope")
    st.sidebar.header("Dashboard Filters")
    st.sidebar.markdown("Use the filters below to customize the data shown on all tabs.")

    # Country Filter
    countries = sorted(df['country'].unique())
    countries.insert(0, "All Countries")
    selected_country = st.sidebar.selectbox(
        "Select a Country",
        countries,
        index=0,
    )

    # --- UPDATED: Dependent Location Filter ---
    locations = ["All Locations"]
    # If a specific country is chosen, find its locations
    if selected_country != "All Countries":
        # Get locations for the selected country
        country_df = df[df['country'] == selected_country]
        
        # Get unique locations, drop any nulls/NaNs, convert to string, and sort
        unique_locations = country_df['location_name'].dropna().unique()
        
        if len(unique_locations) > 0:
            # Convert all to string just in case, then sort
            sorted_locations = sorted([str(loc) for loc in unique_locations])
            locations.extend(sorted_locations)

    selected_location = st.sidebar.selectbox(
        "Select a Location",
        locations,
        index=0,
        help="Select a country first to populate this list. If no locations appear, that country has no specific location data."
    )
    # --- End of Updated Section ---

    # Date Range Filter
    min_date = df['last_updated_date'].min().date()
    max_date = df['last_updated_date'].max().date()
    selected_date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
    )

    # --- Data Filtering Logic (UPDATED) ---
    filtered_df = df.copy()
    map_zoom = 1
    filter_title = "" # For chart titles and messages

    # 1. Filter by Country
    if selected_country != "All Countries":
        filtered_df = df[df['country'] == selected_country]
        map_zoom = 4
        filter_title = selected_country
    else:
        filter_title = "All Countries"
        map_zoom = 1

    # 2. Filter by Location (NEW)
    if selected_location != "All Locations":
        # Filter the *already filtered* dataframe
        filtered_df = filtered_df[filtered_df['location_name'] == selected_location]
        filter_title += f" / {selected_location}" # Update the title
        map_zoom = 8 # Zoom in even more

    # 3. Filter by Date Range
    filter_message = f"Displaying data for: **{filter_title}**" # Updated message logic
    if len(selected_date_range) == 2:
        try:
            start_date = pd.to_datetime(selected_date_range[0])
            end_date = pd.to_datetime(selected_date_range[1])
            filtered_df = filtered_df[
                (filtered_df['last_updated_date'] >= start_date) &
                (filtered_df['last_updated_date'] <= end_date)
            ]
            filter_message += f" | Date Range: **{selected_date_range[0]}** to **{selected_date_range[1]}**"
        except Exception as e:
            st.error(f"Invalid date range: {e}")
            filtered_df = pd.DataFrame(columns=df.columns) # Empty df
            
    # --- Create Tabs (UPDATED) ---
    tab_list = [
        "🛰️ Global Overview",
        "📍 Location Deep-Dive",
        "🆚 Country Comparison", # NEW
        "📈 Trends & Correlations",
        "🌪️ Extreme Events & Wind",
        "🔮 Forecasting", # NEW
        "🗂️ Data Explorer",
        "ℹ️ About Project"
    ]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_list)

    # --- Render Tabs using functions (UPDATED) ---
    with tab1:
        render_global_overview_tab(filtered_df, global_avg_metrics, filter_message, selected_country, map_zoom)

    with tab2:
        render_location_deep_dive_tab(filtered_df, global_avg_metrics, filter_title, selected_country)
    
    with tab3:
        # Pass the original df and full country list for the selectors
        render_comparison_tab(df, countries) 
    
    with tab4:
        render_trends_tab(filtered_df, filter_message)

    with tab5:
        render_extremes_tab(filtered_df, filter_message)
        
    with tab6:
        # Pass the filtered_df and selections for the model
        render_forecasting_tab(filtered_df, selected_country, selected_location)

    with tab7:
        render_data_explorer_tab(filtered_df, filter_message)

    with tab8:
        render_about_tab()


else:
    st.error("Fatal Error: Data could not be loaded. The dashboard cannot be displayed.")
    st.warning("Please check that 'weather_cleaned.csv' is in the same directory as this script and is not empty or corrupted.")