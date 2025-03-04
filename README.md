# 🚀 Customer Churn Prediction with Machine Learning

### **📌 Project Overview**

This project predicts customer churn using **Machine Learning models** (**Random Forest, XGBoost, and an Ensemble Model**).\
It includes **real-time API deployment with FastAPI**, a **Streamlit UI for user interaction**, and **Docker for containerized execution**.

### **🔹 Key Features:**

- **Data Preprocessing & Feature Engineering**
- **SMOTE for Handling Class Imbalance**
- **Hyperparameter Tuning (GridSearchCV)**
- **FastAPI for Real-Time Predictions**
- **Streamlit UI for User Interaction**
- **Dockerized for Deployment & Portability**

---

## **📂 Project Structure**

```
📂 customer-churn-prediction
 ├── 📂 data                # Raw dataset
 │    ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
 ├── 📂 models              # Trained models
 │    ├── model_rf.pkl
 │    ├── model_xgb.pkl
 │    ├── model_ensemble.pkl
 │    ├── feature_names.pkl
 ├── 📂 results             # Model evaluation reports & plots
 │    ├── feature_importance_comparison.png
 │    ├── feature_correlation_heatmap.png
 │    ├── feature_importance_horizontal.png
 ├── 📂 src                 # Core project files
 │    ├── train.py          # Model training & tuning
 │    ├── compare.py        # Model comparison & visualization
 │    ├── api.py            # FastAPI for real-time inference
 │    ├── app.py            # Streamlit UI
 ├── 📜 requirements.txt    # Required Python libraries
 ├── 📜 Dockerfile          # Docker setup for deployment
 ├── 📜 README.md           # Project documentation
```

---

## **📊 Data & Model Training**

### **🔹 Dataset**

- **Source:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Contains **customer demographics, account details, and churn status**.
- **Target Variable:** `"Churn"` (`1` = Churned, `0` = Retained).

### **🔹 Machine Learning Models**

| Model              | Accuracy |
| ------------------ | -------- |
| **Random Forest**  | 76.7%    |
| **XGBoost**        | 76.5%    |
| **Ensemble Model** | 76.7%    |

- **SMOTE** applied to handle class imbalance.
- **GridSearchCV** for hyperparameter tuning.
- **Ensemble model** combines **Random Forest & XGBoost** for better performance.

---

## **📈 Results**

All model evaluation reports and visualizations are stored in the `results/` folder:
- **Feature Importance Comparison:** `results/feature_importance_comparison.png`
- **Feature Correlation Heatmap:** `results/feature_correlation_heatmap.png`
- **Overall Feature Importance (RF + XGB + Ensemble):** `results/feature_importance_horizontal.png`

These plots provide insights into **which features are most important** for predicting churn and **how features are correlated**.

---

## **⚡ Running the Project**

### **🔹 Install Dependencies**

```bash
pip install -r requirements.txt
```

### **🔹 Train & Save Models**

```bash
python src/train.py
```

### **🔹 Run FastAPI (Backend API)**

```bash
uvicorn src.api:app --reload
```

🔗 **Access API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### **🔹 Run Streamlit UI**

```bash
streamlit run src/app.py
```

🔗 **Access Streamlit UI:** [http://localhost:8501](http://localhost:8501)

---

## **🐳 Docker Containerization**

### **🔹 Build & Run in Docker**

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

🚀 Now FastAPI is running inside a **Docker container**.

---

## **🛠 Project Notes**

- **This project is NOT deployed online**. It is designed to be run locally using FastAPI and Streamlit.
- All results are stored in the `results/` folder for reference.
- The project provides an end-to-end machine learning pipeline including **data preprocessing, model training, evaluation, and visualization**.

---

## **📌 Future Improvements**

✔ Integrate a **database (PostgreSQL, MongoDB) to store predictions**\
✔ Deploy API using **Kubernetes (K8s) for scalability**\
✔ Implement **AutoML with Optuna for better hyperparameter tuning**\
✔ Add **Deep Learning models (LSTMs) for time-based churn prediction**

---


💡 **Author**: Jaimin Oza

 
