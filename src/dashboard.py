import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="Emergency Dept Dashboard", layout="wide")
st.title(" Emergency Department Optimization")

# Function to load data (or generate fake data if missing)
@st.cache_data
def load_data():
    try:
        # Attempt 1: Load real data
        df = pd.read_csv("data/raw/EventLog.csv")
        st.sidebar.success(" Real data loaded successfully!")
        return df
    except FileNotFoundError:
        # Attempt 2: Generate simulated data (Safety net)
        st.sidebar.warning("⚠️Real data not found. Using simulated data for demonstration.")
        
        # Create 100 fake patients
        dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
        df = pd.DataFrame({
            "Patient_ID": range(100),
            "Waiting_Time_Mins": np.random.randint(10, 120, 100), # Random mins between 10 and 120
            "Triage_Code": np.random.choice(["Red", "Yellow", "Green"], 100),
            "Arrival_Date": dates
        })
        return df

# Load the data
df = load_data()

# --- DASHBOARD LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Waiting Time Trends")
    # Determine which column to use for plotting
    y_col = "Waiting_Time_Mins" if "Waiting_Time_Mins" in df.columns else df.columns[1] 
    x_col = "Arrival_Date" if "Arrival_Date" in df.columns else df.columns[-1]
    
    st.line_chart(df.set_index(x_col)[y_col])

with col2:
    st.subheader(" Triage Code Distribution")
    # Identify the triage column (adaptable for real or fake data)
    triage_col = "Triage_Code" if "Triage_Code" in df.columns else "Triage"
    if triage_col in df.columns:
        st.bar_chart(df[triage_col].value_counts())
    else:
        st.write("Triage data not available.")

# Key Metrics
st.metric("Total Patients Processed", len(df))
st.write("---")
st.caption(" Dashboard running inside a Docker Container.")
