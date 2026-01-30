import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib

# Page Configuration
st.set_page_config(page_title="Emergency Dept Dashboard", layout="wide")
st.title("Emergency Department Optimization & AI Prediction")

# --- 1. DATA LOADING FUNCTION ---
@st.cache_data
def load_data():
    try:
        # Load the CSV file using the correct separator
        df = pd.read_csv("data/raw/EventLog.csv", sep=";")
        # Data Cleaning
        df.columns = df.columns.str.strip()
        df['START'] = pd.to_datetime(df['START'], utc=True, errors='coerce')
        df['STOP'] = pd.to_datetime(df['STOP'], utc=True, errors='coerce')
        df = df.dropna(subset=['START', 'STOP'])
        df['Waiting_Time_Mins'] = (df['STOP'] - df['START']).dt.total_seconds() / 60
        df['Waiting_Time_Mins'] = df['Waiting_Time_Mins'].clip(lower=0)
        
        # Feature Engineering
        df['Arrival_Hour'] = df['START'].dt.hour
        df['Day_Name'] = df['START'].dt.strftime('%A')
        df['Year_Week'] = df['START'].dt.strftime('%Y - Week %U')
        
        return df, True # True means "Real Data"

    except FileNotFoundError:
        # Fallback: Simulated Data
        dates = pd.date_range(start="2023-01-01", periods=500, freq="h")
        df = pd.DataFrame({
            "START": dates,
            "STOP": dates + pd.to_timedelta(np.random.randint(10, 120, 500), unit='m')
        })
        df['Waiting_Time_Mins'] = (df['STOP'] - df['START']).dt.total_seconds() / 60
        df['Arrival_Hour'] = df['START'].dt.hour
        df['Day_Name'] = df['START'].dt.strftime('%A')
        df['Year_Week'] = df['START'].dt.strftime('%Y - Week %U')
        return df, False

# --- 2. AI MODEL LOADING FUNCTION ---
def load_model():
    # Attempt to load a trained model if it exists
    model_path = "models/xgb_model.json" 
    if os.path.exists(model_path):
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            return model, "XGBoost"
        except:
            return None, "Error"
    return None, "Missing"

# Load Resources
df, is_real_data = load_data()
model, model_status = load_model()

# --- SIDEBAR: CONTROL PANEL & PREDICTION ---
st.sidebar.header("Control Panel")

# A. Historical Filters
st.sidebar.subheader("1. Historical Analysis")
time_of_day = st.sidebar.selectbox("Filter Time of Day", ["All Day", "Day (06-18)", "Night (18-06)"])

if time_of_day == "Day (06-18)":
    df_filtered = df[(df['Arrival_Hour'] >= 6) & (df['Arrival_Hour'] < 18)]
elif time_of_day == "Night (18-06)":
    df_filtered = df[(df['Arrival_Hour'] >= 18) | (df['Arrival_Hour'] < 6)]
else:
    df_filtered = df.copy()

if is_real_data:
    unique_weeks = sorted(df_filtered['Year_Week'].unique())
    week_options = ["All Weeks"] + unique_weeks
    selected_week = st.sidebar.selectbox("Filter Weeks", week_options)
    if selected_week != "All Weeks":
        df_filtered = df_filtered[df_filtered['Year_Week'] == selected_week]
else:
    st.sidebar.warning("Using Simulated Data")

# B. AI Prediction Simulator
st.sidebar.markdown("---")
st.sidebar.subheader("2. Prediction Simulator")
st.sidebar.info("Predict waiting time for a new patient.")

input_day = st.sidebar.selectbox("Arrival Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
input_hour = st.sidebar.slider("Arrival Hour", 0, 23, 10)

# Prediction Logic
if st.sidebar.button("Run Prediction"):
    
    # 1. SIMPLE HEURISTIC FALLBACK (If model is missing, use historical average for that slot)
    avg_wait = df[
        (df['Arrival_Hour'] == input_hour) & 
        (df['Day_Name'] == input_day)
    ]['Waiting_Time_Mins'].mean()
    
    if pd.isna(avg_wait):
        avg_wait = 30.0 # Default fallback
        
    predicted_value = avg_wait 
    
    # Visual Output
    st.sidebar.success(f"Predicted Wait: {predicted_value:.0f} min")
    
    if predicted_value > 60:
        st.sidebar.error("High Congestion Expected")
    else:
        st.sidebar.write("Status: Normal")

# --- MAIN DASHBOARD ---

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", len(df_filtered))
col2.metric("Avg Waiting Time", f"{df_filtered['Waiting_Time_Mins'].mean():.1f} min")
col3.metric("AI Model Status", "Active" if model_status == "XGBoost" else "Demo Mode")

st.markdown("---")

# Visualizations
st.subheader("Operational Insights")

if not df_filtered.empty:
    # 1. Weekly Trends
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    if 'selected_week' in locals() and selected_week != "All Weeks":
        group_cols = ['Day_Name']
        title_text = "Daily Performance (Selected Week)"
    else:
        group_cols = ['Day_Name']
        title_text = "Global Average by Day of Week"

    df_daily = df_filtered.groupby(group_cols)['Waiting_Time_Mins'].mean().reindex(day_order).reset_index()
    
    fig1 = px.bar(
        df_daily, x='Day_Name', y='Waiting_Time_Mins',
        color='Waiting_Time_Mins', color_continuous_scale='Blues',
        text_auto='.0f',
        title=title_text
    )
    fig1.update_layout(yaxis_title="Minutes")
    st.plotly_chart(fig1)

    # 2. Hourly Trends
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### Hourly Bottlenecks")
        df_hourly = df_filtered.groupby('Arrival_Hour')['Waiting_Time_Mins'].mean().reset_index()
        fig2 = px.line(df_hourly, x='Arrival_Hour', y='Waiting_Time_Mins', markers=True)
        fig2.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Limit")
        st.plotly_chart(fig2)
        
    with col_right:
        st.markdown("#### Heatmap: Day vs Hour")
        # Pivot table for heatmap
        heatmap_data = df_filtered.pivot_table(
            index='Day_Name', columns='Arrival_Hour', 
            values='Waiting_Time_Mins', aggfunc='mean'
        ).reindex(day_order)
        
        fig3 = px.imshow(heatmap_data, text_auto=False, aspect="auto", color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig3)

else:
    st.warning("No data matches the current filters.")
