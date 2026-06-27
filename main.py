# main.py
from src.data_preprocessing import load_and_engineer_features, prepare_datasets
from src.train import train_and_optimize
import pandas as pd

def main():
    print("🚀 Starting Predictive Equipment Failure Detection Pipeline...\n")
    
    data_path = 'data/sensor_data.csv'
    
    # Step 1: Feature Engineering
    print("[Step 1/3] Extracting and engineering features from sensor streams...")
    X, y = load_and_engineer_features(data_path)
    
    # Step 2: Prepare Datasets
    print("[Step 2/3] Splitting and scaling data partitions...")
    X_train, X_test, y_train, y_test, feature_names = prepare_datasets(X, y)
    
    # Step 3: Train and Benchmark Models
    print("[Step 3/3] Benchmarking machine learning architectures...")
    best_model = train_and_optimize(X_train, y_train, X_test, y_test)
    
    # Feature Importance Insight
    if hasattr(best_model, 'feature_importances_'):
        print("\n💡 Key Predictive Insights:")
        importances = pd.Series(best_model.feature_importances_, index=feature_names)
        print(importances.sort_values(ascending=False).to_string())
        
    print("\n✅ Pipeline executed successfully! Proactive maintenance insights generated.")

if __name__ == "__main__":
    main()