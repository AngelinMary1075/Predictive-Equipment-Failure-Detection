import pandas as pd
import numpy as np
import os

np.random.seed(42)
n_rows = 1000

data = {
    'Timestamp': pd.date_range(start='2026-01-01', periods=n_rows, freq='H'),
    'Temperature': np.random.normal(loc=75, scale=10, size=n_rows),
    'Vibration': np.random.normal(loc=4.5, scale=1.2, size=n_rows),
    'Pressure': np.random.normal(loc=200, scale=30, size=n_rows),
    'Operational_Hours': np.arange(1, n_rows + 1)
}

df = pd.DataFrame(data)
df['Failure'] = ((df['Temperature'] > 92) | (df['Vibration'] > 6.8) | (df['Pressure'] > 250)).astype(int)

os.makedirs('data', exist_ok=True)
df.to_csv('data/sensor_data.csv', index=False)
print("Synthetic dataset created successfully inside the data/ folder!")
