import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib


def preprocess_new_data(df, le_description, le_reason, le_time):
    data = df.copy()
    
    data['duration_hours'] = (data['STOP'] - data['START']).dt.total_seconds() / 3600
    
    data['has_reason'] = data['REASONDESCRIPTION'].notna().astype(int)
    data['REASONDESCRIPTION'] = data['REASONDESCRIPTION'].fillna('No Reason Specified')
    data['REASONCODE'] = data['REASONCODE'].fillna(0)
    
    Q1 = data['duration_hours'].quantile(0.25)
    Q3 = data['duration_hours'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    duration_99th = data['duration_hours'].quantile(0.99)
    data['is_outlier'] = ((data['duration_hours'] < lower_bound) | (data['duration_hours'] > upper_bound)).astype(int)
    data['duration_hours_capped'] = data['duration_hours'].clip(upper=duration_99th)
    
    data = data.sort_values(['PATIENT', 'START']).reset_index(drop=True)
    
    data['start_hour'] = data['START'].dt.hour
    data['start_day_of_week'] = data['START'].dt.dayofweek
    data['start_month'] = data['START'].dt.month
    data['start_year'] = data['START'].dt.year
    data['start_day'] = data['START'].dt.day
    
    data['season'] = data['start_month'].map({
        12: 1, 1: 1, 2: 1,
        3: 2, 4: 2, 5: 2,
        6: 3, 7: 3, 8: 3,
        9: 4, 10: 4, 11: 4
    })
    
    data['is_weekend'] = (data['start_day_of_week'] >= 5).astype(int)
    
    def get_time_of_day(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'
    
    data['time_of_day'] = data['start_hour'].apply(get_time_of_day)
    
    data['total_prior_encounters'] = 0
    data['avg_prior_duration'] = 0.0
    data['avg_prior_cost'] = 0.0
    data['days_since_last_encounter'] = 0.0
    data['encounters_last_30_days'] = 0
    data['encounters_last_90_days'] = 0
    
    for patient_id in data['PATIENT'].unique():
        patient_mask = data['PATIENT'] == patient_id
        patient_data = data[patient_mask].copy()
        
        for idx in patient_data.index:
            current_start = data.loc[idx, 'START']
            prior_encounters = patient_data[patient_data['START'] < current_start]
            
            if len(prior_encounters) > 0:
                data.loc[idx, 'total_prior_encounters'] = len(prior_encounters)
                data.loc[idx, 'avg_prior_duration'] = prior_encounters['duration_hours'].mean()
                data.loc[idx, 'avg_prior_cost'] = prior_encounters['BASE_COST'].mean()
                
                last_encounter_date = prior_encounters['START'].max()
                days_diff = (current_start - last_encounter_date).total_seconds() / (24 * 3600)
                data.loc[idx, 'days_since_last_encounter'] = days_diff
                
                encounters_30d = prior_encounters[prior_encounters['START'] >= (current_start - timedelta(days=30))]
                data.loc[idx, 'encounters_last_30_days'] = len(encounters_30d)
                
                encounters_90d = prior_encounters[prior_encounters['START'] >= (current_start - timedelta(days=90))]
                data.loc[idx, 'encounters_last_90_days'] = len(encounters_90d)
    
    data['description_encoded'] = le_description.transform(data['DESCRIPTION'])
    data['reason_encoded'] = le_reason.transform(data['REASONDESCRIPTION'])
    data['time_of_day_encoded'] = le_time.transform(data['time_of_day'])
    
    emergency_keywords = ['emergency', 'urgent', 'acute', 'trauma', 'critical']
    data['is_emergency'] = data['DESCRIPTION'].str.lower().str.contains('|'.join(emergency_keywords), na=False).astype(int)
    
    feature_columns = [
        'start_hour', 'start_day_of_week', 'start_month', 'start_year', 'season', 'is_weekend',
        'time_of_day_encoded',
        'total_prior_encounters', 'avg_prior_duration', 'avg_prior_cost',
        'days_since_last_encounter', 'encounters_last_30_days', 'encounters_last_90_days',
        'description_encoded', 'reason_encoded', 'CODE',
        'BASE_COST',
        'has_reason', 'is_emergency'
    ]
    
    X = data[feature_columns].copy()
    y = data['duration_hours_capped'].copy()
    
    return X, y


def retrain_random_forest(excel_path, X_old, y_old):
    encoders = joblib.load('saved/encoders.joblib')
    le_description = encoders['description']
    le_reason = encoders['reason']
    le_time = encoders['time_of_day']
    
    df_new = pd.read_excel(excel_path)
    X_new, y_new = preprocess_new_data(df_new, le_description, le_reason, le_time)
    
    X_combined = pd.concat([X_old, X_new], axis=0, ignore_index=True)
    y_combined = pd.concat([y_old, y_new], axis=0, ignore_index=True)
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_combined, y_combined)
    
    joblib.dump(rf_model, 'saved/best_los_model.joblib')
    
    return rf_model


def retrain_xgboost(excel_path, X_old, y_old):
    encoders = joblib.load('saved/encoders.joblib')
    le_description = encoders['description']
    le_reason = encoders['reason']
    le_time = encoders['time_of_day']
    
    df_new = pd.read_excel(excel_path)
    X_new, y_new = preprocess_new_data(df_new, le_description, le_reason, le_time)
    
    X_combined = pd.concat([X_old, X_new], axis=0, ignore_index=True)
    y_combined = pd.concat([y_old, y_new], axis=0, ignore_index=True)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_combined, y_combined)
    
    joblib.dump(xgb_model, 'saved/xgboost_model.joblib')
    
    return xgb_model