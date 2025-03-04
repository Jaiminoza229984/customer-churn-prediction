import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

# Load trained models
rf_model = joblib.load("../models/model_rf.pkl")
xgb_model = joblib.load("../models/model_xgb.pkl")
ensemble_model = joblib.load("../models/model_ensemble.pkl")

# Load dataset for feature names
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop(columns=["customerID"], inplace=True)

# Load expected feature names
expected_features = joblib.load("../models/feature_names.pkl")

# Convert categorical columns to numeric for correlation analysis
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

#  Ensure Feature Engineering is applied
df["Tenure_MonthlyCharge_Interaction"] = df["tenure"] * df["MonthlyCharges"]
df["TotalCharges_per_Tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

# Ensure dataset columns match training features
missing_features = set(expected_features) - set(df.columns)
if missing_features:
    print(f" Warning: The following features are missing and will be added with default values: {missing_features}")
    for feature in missing_features:
        df[feature] = 0  # Default value for missing features

# Select only expected features
df = df[expected_features]

# Get feature importance
rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_
ensemble_importance = (rf_importance + xgb_importance) / 2  # Averaging feature importances

# Verify lengths before creating DataFrame
print(f" Number of features: {len(expected_features)}")
print(f" RF importance length: {len(rf_importance)}")
print(f" XGBoost importance length: {len(xgb_importance)}")
print(f" Ensemble importance length: {len(ensemble_importance)}")

# Create DataFrame for visualization
importance_df = pd.DataFrame({
    "Feature": expected_features,
    "RandomForest": rf_importance,
    "XGBoost": xgb_importance,
    "Ensemble": ensemble_importance
})

# Sort features by importance (average of all models)
importance_df["Avg_Importance"] = (importance_df["RandomForest"] + importance_df["XGBoost"] + importance_df["Ensemble"]) / 3
importance_df = importance_df.sort_values(by="Avg_Importance", ascending=False)

# Create results folder if it doesn't exist
if not os.path.exists("../results"):
    os.makedirs("../results")

# Feature Importance Comparison Barplot
plt.figure(figsize=(12, 6))
sns.barplot(x="XGBoost", y="Feature", data=importance_df, color="blue", label="XGBoost", alpha=0.7)
sns.barplot(x="RandomForest", y="Feature", data=importance_df, color="red", label="RandomForest", alpha=0.5)
sns.barplot(x="Ensemble", y="Feature", data=importance_df, color="green", label="Ensemble", alpha=0.3)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("../results/feature_importance_comparison.png")
plt.show()

# Heatmap for Feature Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("../results/feature_correlation_heatmap.png")
plt.show()

# Horizontal Bar Chart for Feature Importance
plt.figure(figsize=(10, 8))
sns.barplot(x="Avg_Importance", y="Feature", data=importance_df, palette="viridis")
plt.xlabel("Average Feature Importance Score")
plt.ylabel("Features")
plt.title("Overall Feature Importance (RF + XGB + Ensemble)")
plt.tight_layout()
plt.savefig("../results/feature_importance_horizontal.png")
plt.show()

print(" All comparison plots saved in the 'results/' folder!")
