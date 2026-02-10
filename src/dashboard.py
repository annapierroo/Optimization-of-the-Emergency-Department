import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import holidays
import joblib

# --- AUTHENTICATION CONFIGURATION ---
def psw_check():
    USER = "admin"
    PASSWORD = "ED_Opt26"
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return

    st.title("Emergency Dept Dashboard - Login Required")
    user = st.text_input("Enter Username:")
    pwd = st.text_input("Enter Password:", type="password")
    
    if st.button("Login"):
        if user == USER and pwd == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    st.stop()

psw_check()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Emergency Dept Dashboard", layout="wide")
st.title("Emergency Department Optimization & AI Prediction")

# --- 1. DATA LOADING FUNCTION ---
@st.cache_data
def load_data():
    holiday = holidays.USA()
    try:
        df = pd.read_csv("data/raw/EventLog.csv", sep=";")
        df.columns = df.columns.str.strip()
        df['START'] = pd.to_datetime(df['START'], utc=True, errors='coerce')
        df['STOP'] = pd.to_datetime(df['STOP'], utc=True, errors='coerce')
        df = df.dropna(subset=['START', 'STOP'])
        
        df['Waiting_Time_Mins'] = (df['STOP'] - df['START']).dt.total_seconds() / 60
        df['Waiting_Time_Mins'] = df['Waiting_Time_Mins'].clip(lower=0)
        df['Arrival_Hour'] = df['START'].dt.hour
        df['Day_Name'] = df['START'].dt.strftime('%A')
        df['Year_Week'] = df['START'].dt.strftime('%Y - Week %U')
        is_real_data = True
    except FileNotFoundError:
        dates = pd.date_range(start="2023-01-01", periods=500, freq="h")
        df = pd.DataFrame({
            "START": dates,
            "STOP": dates + pd.to_timedelta(np.random.randint(10, 120, 500), unit='m')
        })
        is_real_data = False
        df['Waiting_Time_Mins'] = (df['STOP'] - df['START']).dt.total_seconds() / 60
        df['Arrival_Hour'] = df['START'].dt.hour
        df['Day_Name'] = df['START'].dt.strftime('%A')
        df['Year_Week'] = df['START'].dt.strftime('%Y - Week %U')

    def holiday_day(dt):
        if dt in holiday: return "Holiday"
        elif dt.weekday() >= 5: return "Weekend"
        else: return "Weekday"
    df['Day_Type'] = df['START'].apply(holiday_day)
    return df, is_real_data

# --- 2. LOAD MODELS ---

def load_waiting_time_model():
    """Load XGBoost Regressor for waiting time."""
    model_path = "models/xgb_model.json" 
    if os.path.exists(model_path):
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            return model, "XGBoost"
        except: return None, "Error"
    return None, "Missing"

@st.cache_resource
def load_next_activity_model():
    """Load XGBoost Classifier and separate Encoders."""
    try:
        import xgboost as xgb
        # Load Encoders
        enc_in = joblib.load("models/input_encoder.pkl")
        enc_out = joblib.load("models/output_encoder.pkl")
        
        # Load Model
        clf = xgb.XGBClassifier()
        clf.load_model("models/next_activity_xgb.json")
        return clf, enc_in, enc_out
    except Exception:
        return None, None, None

df, is_real_data = load_data()
model_wait, model_wait_status = load_waiting_time_model()
model_next, enc_in, enc_out = load_next_activity_model()

# --- SIDEBAR: CONTROL PANEL ---
st.sidebar.header("Control Panel")

st.sidebar.subheader("1. Historical Analysis")
time_of_day = st.sidebar.selectbox("Filter Time of Day", ["All Day", "Day (06-18)", "Night (18-06)"])
if time_of_day == "Day (06-18)": df_filtered = df[(df['Arrival_Hour'] >= 6) & (df['Arrival_Hour'] < 18)]
elif time_of_day == "Night (18-06)": df_filtered = df[(df['Arrival_Hour'] >= 18) | (df['Arrival_Hour'] < 6)]
else: df_filtered = df.copy()

if is_real_data:
    unique_weeks = sorted(df_filtered['Year_Week'].unique())
    week_options = ["All Weeks"] + unique_weeks
    selected_week = st.sidebar.selectbox("Filter Weeks", week_options)
    if selected_week != "All Weeks": df_filtered = df_filtered[df_filtered['Year_Week'] == selected_week]

selected_day_type = st.sidebar.selectbox("Type of day", ["All", "Weekday", "Weekend", "Holiday"])
if selected_day_type != "All": df_filtered = df_filtered[df_filtered['Day_Type'] == selected_day_type]

st.sidebar.markdown("---")

# B. AI Simulator: Waiting Time
st.sidebar.subheader("2. Prediction: Waiting Time")
st.sidebar.info("Predict waiting time for a new patient.")
input_day = st.sidebar.selectbox("Arrival Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
input_hour = st.sidebar.slider("Arrival Hour", 0, 23, 10)

if st.sidebar.button("Predict Wait"):
    avg_wait = df[(df['Arrival_Hour'] == input_hour) & (df['Day_Name'] == input_day)]['Waiting_Time_Mins'].mean()
    if pd.isna(avg_wait): avg_wait = 30.0 
    predicted_value = avg_wait 
    st.sidebar.success(f"Predicted Wait: {predicted_value:.0f} min")
    if predicted_value > 60: st.sidebar.error("High Congestion Expected")
    else: st.sidebar.write("Status: Normal")

st.sidebar.markdown("---")

# C. AI Simulator: Next Activity
st.sidebar.subheader("3. Prediction: Next Step")
st.sidebar.info("Predict the patient's next clinical activity.")

if model_next and enc_in and enc_out:
    # Dropdown uses Input Encoder (Current Activity)
    activity_list = sorted(enc_in.classes_)
    current_act = st.sidebar.selectbox("Current Activity", activity_list)
    
    if st.sidebar.button("Predict Next Step"):
        day_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
        day_num = day_map.get(input_day, 0)
        try:
            # Encode Input
            act_encoded = enc_in.transform([current_act])[0]
            X_new = pd.DataFrame([[act_encoded, input_hour, day_num]], 
                                columns=['Current_Activity_Encoded', 'Hour', 'Day_of_Week'])
            
            # Predict (Returns ID from Output Encoder)
            pred_idx = model_next.predict(X_new)[0]
            
            # Decode Output (Using Output Encoder)
            pred_label = enc_out.inverse_transform([pred_idx])[0]
            
            # Confidence
            probs = model_next.predict_proba(X_new)
            confidence = probs[0][pred_idx]
            
            st.sidebar.success(f"Next Step: **{pred_label}**")
            st.sidebar.caption(f"Confidence: {confidence:.0%}")
        except Exception as e:
            st.sidebar.error(f"Prediction Error: {e}")
else:
    st.sidebar.warning("Next Activity Model not ready. Run 'src/train_next_activity.py'.")

# --- MAIN DASHBOARD ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", len(df_filtered))
col2.metric("Avg Waiting Time", f"{df_filtered['Waiting_Time_Mins'].mean():.1f} min")
col3.metric("AI Models Status", "Active")

st.markdown("---")

# --- ALERT SYSTEM ---
CRITICAL_THRESHOLD = 60.0
if not df_filtered.empty:
    current_avg = df_filtered['Waiting_Time_Mins'].mean()
    hourly_check = df_filtered.groupby('Arrival_Hour')['Waiting_Time_Mins'].mean()
    bottleneck_hours = hourly_check[hourly_check > CRITICAL_THRESHOLD]
    
    if current_avg > CRITICAL_THRESHOLD:
        st.error(f"CRITICAL ALERT: Average waiting time is high ({current_avg:.1f} min). Immediate action required.")
    elif not bottleneck_hours.empty:
        max_bottleneck = bottleneck_hours.max()
        st.warning(f"BOTTLENECK DETECTED: Specific hours exceed {CRITICAL_THRESHOLD} mins (Max: {max_bottleneck:.1f} min). Please review the charts.")
    else:
        st.success("STATUS OPTIMAL: Waiting times are within safety limits.")

st.markdown("---")

# Visualizations
st.subheader("Operational Insights")
if not df_filtered.empty:
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if 'selected_week' in locals() and selected_week != "All Weeks":
        group_cols = ['Day_Name']; title_text = "Daily Performance (Selected Week)"
    else:
        group_cols = ['Day_Name']; title_text = "Global Average by Day of Week"

    df_daily = df_filtered.groupby(group_cols)['Waiting_Time_Mins'].mean().reindex(day_order).reset_index()
    fig1 = px.bar(df_daily, x='Day_Name', y='Waiting_Time_Mins', color='Waiting_Time_Mins', color_continuous_scale='Blues', text_auto='.0f', title=title_text)
    st.plotly_chart(fig1)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Hourly Bottlenecks")
        df_hourly = df_filtered.groupby('Arrival_Hour')['Waiting_Time_Mins'].mean().reset_index()
        fig2 = px.line(df_hourly, x='Arrival_Hour', y='Waiting_Time_Mins', markers=True)
        fig2.add_hline(y=CRITICAL_THRESHOLD, line_dash="dash", line_color="red", annotation_text="Limit (60m)")
        st.plotly_chart(fig2)
        
    with col_right:
        st.markdown("#### Heatmap: Day vs Hour")
        heatmap_data = df_filtered.pivot_table(index='Day_Name', columns='Arrival_Hour', values='Waiting_Time_Mins', aggfunc='mean').reindex(day_order)
        fig3 = px.imshow(heatmap_data, text_auto=False, aspect="auto", color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig3)
else:
    st.warning("No data matches the current filters.")