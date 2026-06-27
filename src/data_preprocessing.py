# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_engineer_features(filepath):
    """Loads sensor data and performs extensive feature engineering."""
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # 1. Rolling Metrics (Capturing wear-and-tear over time)
    window = 5
    df['Temp_Rolling_Mean'] = df['Temperature'].rolling(window=window, min_periods=1).mean()
    df['Vib_Rolling_Std'] = df['Vibration'].rolling(window=window, min_periods=1).std().fillna(0)
    df['Press_Rolling_Mean'] = df['Pressure'].rolling(window=window, min_periods=1).mean()
    
    # 2. Sequential/Lag Features
    df['Temp_Change'] = df['Temperature'].diff().fillna(0)
    
    # Define features and target
    feature_cols = [
        'Temperature', 'Vibration', 'Pressure', 'Operational_Hours',
        'Temp_Rolling_Mean', 'Vib_Rolling_Std', 'Press_Rolling_Mean', 'Temp_Change'
    ]
    
    X = df[feature_cols]
    y = df['Failure']
    
    return X, y

def prepare_datasets(X, y):
    """Splits and scales the datasets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns