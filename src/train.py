import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Handles class imbalance
import time

# Load dataset
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID as it's not useful
df.drop(columns=["customerID"], inplace=True)

# Encode categorical variables
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Ensure 'Churn' is categorical (0 or 1)
df["Churn"] = df["Churn"].astype(int)

# Debug: Print unique values in target variable
print("Unique values in y (Churn column) BEFORE scaling:", df["Churn"].unique())

#  Feature Engineering (Add new features)
df["Tenure_MonthlyCharge_Interaction"] = df["tenure"] * df["MonthlyCharges"]
df["TotalCharges_per_Tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

# Scale numeric features
scaler = StandardScaler()
feature_cols = df.drop(columns=["Churn"]).columns
df[feature_cols] = scaler.fit_transform(df[feature_cols])

print("Unique values in y (Churn column) AFTER scaling:", df["Churn"].unique())

# Feature and target split
X = df.drop(columns=["Churn"])
y = df["Churn"]

#  Train-test split with stratification (Ensures balanced churn in train & test sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  Apply SMOTE to balance the training dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f" After SMOTE - Churn Distribution: {np.bincount(y_train_resampled)}")

#  Step 1: Hyperparameter tuning for Random Forest
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring="accuracy", n_jobs=-1)
start_time = time.time()
rf_grid.fit(X_train_resampled, y_train_resampled)
rf_train_time = time.time() - start_time
rf_model = rf_grid.best_estimator_
print(f" Best Random Forest Params: {rf_grid.best_params_}")

#  Step 2: Hyperparameter tuning for XGBoost
xgb_params = {
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200]
}
xgb_grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric="logloss"), xgb_params, cv=3, scoring="accuracy", n_jobs=-1)
start_time = time.time()
xgb_grid.fit(X_train_resampled, y_train_resampled)
xgb_train_time = time.time() - start_time
xgb_model = xgb_grid.best_estimator_
print(f" Best XGBoost Params: {xgb_grid.best_params_}")

#  Step 3: Create an Ensemble Model (Voting Classifier)
ensemble_model = VotingClassifier(
    estimators=[("rf", rf_model), ("xgb", xgb_model)], voting="soft"
)
ensemble_model.fit(X_train_resampled, y_train_resampled)

# Save models
joblib.dump(rf_model, "../models/model_rf.pkl")
joblib.dump(xgb_model, "../models/model_xgb.pkl")
joblib.dump(ensemble_model, "../models/model_ensemble.pkl")
feature_names = X.columns.tolist()
joblib.dump(feature_names, "../models/feature_names.pkl")

print(f" Random Forest Training Time: {rf_train_time:.2f} sec")
print(f" XGBoost Training Time: {xgb_train_time:.2f} sec")

# Evaluate models
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_ensemble = ensemble_model.predict(X_test)

print("\n Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(" XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(" Ensemble Model Accuracy:", accuracy_score(y_test, y_pred_ensemble))

print("\n Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("\n XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("\n Ensemble Model Classification Report:\n", classification_report(y_test, y_pred_ensemble))
