# Predictive Equipment Failure Detection

An intelligent, end-to-end predictive maintenance system that leverages industrial IoT sensor data streams to identify early signs of equipment failure. By building sequence-aware features and benchmarking multiple machine learning architectures, this pipeline optimizes predictive maintenance schedules, significantly minimizing industrial downtime and reducing false maintenance alerts.

---

## 🛠️ Tech Stack & Libraries

* **Core Programming:** Python
* **Data Manipulation & Processing:** Pandas, NumPy
* **Machine Learning Framework:** Scikit-Learn
* **Gradient Boosting:** XGBoost
* **Model Tuning:** Grid Search Cross-Validation (`GridSearchCV`)

---

## ⚙️ System Architecture & Workflow

The system processes continuous industrial streams through a modular, three-tiered data science pipeline:

### 1. Extensive Feature Engineering

Sensor streams naturally lack contextual historical awareness at a single point in time. To fix this, the pipeline constructs time-series features to capture wear-and-tear degradation:

* **Rolling Metrics:** Computes a 5-period moving average for temperature (`Temp_Rolling_Mean`) and pressure (`Press_Rolling_Mean`) to capture steady degradation trends.
* **Volatility Analysis:** Tracks rolling standard deviation of vibration (`Vib_Rolling_Std`) to identify unstable mechanical oscillations.
* **Lag Features:** Measures instantaneous thermal spikes (`Temp_Change`) using numerical differentiation.

### 2. Data Partitioning & Scaling

* Data is divided into an 80/20 train/test split.
* **Stratified Splitting** is implemented to maintain the exact ratio of failures across subsets, countering data imbalance.
* Features are standardized via `StandardScaler` to bring variance down to $1.0$ and a mean of $0.0$, satisfying the optimization prerequisites of linear estimators.

### 3. Model Benchmarking & Optimization

The system benchmarks three diverse model architectures to isolate the highest-performing predictor:

* **Logistic Regression:** Serves as the baseline linear classifier, equipped with balanced class weights.
* **Optimized Random Forest:** A non-linear ensemble tuned via Cross-Validation to actively balance Precision and Recall, directly targeting the elimination of **False Maintenance Alerts (False Positives)**.
* **XGBoost Classifier:** A gradient-boosted tree architecture optimized to capture complex, non-linear interactions among thermal and vibrational anomalies.

---

## 📂 Project Structure

```text
predictive_maintenance/
│
├── data/
│   └── sensor_data.csv        # Simulated IoT industrial sensor dataset
│
├── src/
│   ├── __init__.py            # Makes the directory a Python package
│   ├── data_preprocessing.py  # Cleans data, splits datasets, and engineers features
│   └── train.py               # Houses training loops, CV hyperparameter tuning, & evaluation
│
├── generate_data.py           # Synthetic dataset generator utility
├── main.py                    # Core orchestrator pipeline script
└── README.md                  # Comprehensive project documentation

```

---

## 🚀 How to Execute the Project

Follow these steps to run the pipeline locally on your machine.

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install pandas numpy scikit-learn xgboost

```

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Predictive-Equipment-Failure-Detection.git
cd Predictive-Equipment-Failure-Detection

```

### Step 2: Generate the Industrial Sensor Dataset

Run the data generator to create a realistic mock environment of 1,000 hourly industrial sensor readings containing temperatures (°C), vibrations (mm/s), and pressures (kPa):

```bash
python generate_data.py

```

### Step 3: Run the Complete Machine Learning Pipeline

Execute the master orchestration script to trigger feature engineering, split data, run model training cross-validations, and print out comparative performance reports:

```bash
python main.py

```

---

## 📊 Sample Output & Insights

When running `main.py`, the console outputs benchmarking classification reports alongside a **Feature Importance** breakdown to provide actionable industrial context:

```text
================ Optimized Random Forest Performance ================
              precision    recall  f1-score   support

           0       0.99      1.00      1.00       176
           1       1.00      0.96      0.98        24

    accuracy                           0.99       200

False Maintenance Alerts (False Positives): 0
ROC-AUC Score: 0.9995

💡 Key Predictive Insights:
Temperature            0.384291
Vib_Rolling_Std        0.245104
Vibration              0.198335
Temp_Rolling_Mean      0.103921
Pressure               0.041120
...

```

* **False Alert Reduction:** Through strict cross-validation on ensemble weights, the False Positive metric is constrained toward zero, preventing costly unnecessary inspections.
* **Root-Cause Insight:** Feature importances systematically confirm that raw `Temperature` and volatile spikes in vibration (`Vib_Rolling_Std`) act as the leading indicators of imminent equipment failure.
