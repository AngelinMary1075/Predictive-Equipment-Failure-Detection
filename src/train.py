# src/train.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates the model and prints performance metrics."""
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"\n================ {model_name} Performance ================")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {auc:.4f}")
    
    cm = confusion_matrix(y_test, preds)
    print(f"Confusion Matrix:\n{cm}")
    # False Positives are what cause false maintenance alerts
    print(f"False Maintenance Alerts (False Positives): {cm[0][1]}") 
    return model

def train_and_optimize(X_train, y_train, X_test, y_test):
    """Trains and benchmarks Logistic Regression, Random Forest, and XGBoost."""
    
    # --- 1. Logistic Regression ---
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    evaluate_model(lr, X_test, y_test, "Logistic Regression")
    
    # --- 2. Random Forest (with Grid Search to optimize and reduce false alerts) ---
    print("\nOptimizing Random Forest...")
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='f1')
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    evaluate_model(best_rf, X_test, y_test, "Optimized Random Forest")
    
    # --- 3. XGBoost ---
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=1)
    xgb.fit(X_train, y_train)
    evaluate_model(xgb, X_test, y_test, "XGBoost")
    
    return best_rf  # Returning the optimized RF as an example winner