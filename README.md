# 📡 Telco Customer Churn Prediction

> **Architecture:** MVC Pattern (Model-View-Controller)  
> **Dataset:** Telco Customer Churn — Kaggle (Blastchar)  
> **Deployment:** Streamlit Web App + FastAPI REST API  
> **ML Framework:** Scikit-learn + XGBoost with 5 classification models

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Architecture](#project-architecture)
- [Folder Structure](#folder-structure)
- [Installation & Setup](#installation--setup)
- [Running the Project](#running-the-project)
  - [1. Train Models (MVC Pipeline)](#1-train-models-mvc-pipeline)
  - [2. Run Streamlit Web App](#2-run-streamlit-web-app)
  - [3. Run FastAPI Server](#3-run-fastapi-server)
  - [4. Explore Data (Notebook)](#4-explore-data-notebook)
- [Project Pipeline](#project-pipeline)
- [Model Results](#model-results)
- [API Endpoints](#api-endpoints)
- [Key Insights](#key-insights)
- [Business Recommendations](#business-recommendations)

---

## Project Overview

This project builds an **end-to-end machine learning pipeline** to predict whether a telecom customer will churn (cancel their subscription). It covers the full ML lifecycle:

1. Exploratory Data Analysis (EDA) with interactive Plotly charts
2. Data cleaning and preprocessing with a scikit-learn `Pipeline`
3. Training and comparing 5 classification models (with SMOTE balancing)
4. Model interpretation and feature importance
5. Deployment via a **Streamlit web app** and a **FastAPI REST API**

---

## 📂 Dataset Description

| Property | Value |
|----------|-------|
| Source | Kaggle — "Telco Customer Churn" by Blastchar |
| Rows | 7,043 customers |
| Columns | 21 (20 features + 1 target) |
| Target | `Churn` — Yes/No |
| Churn Rate | ~26.5% |

### Feature Groups

| Category | Features |
|----------|----------|
| **Demographics** | gender, SeniorCitizen, Partner, Dependents |
| **Account** | tenure, Contract, PaperlessBilling, PaymentMethod |
| **Phone Services** | PhoneService, MultipleLines |
| **Internet Services** | InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| **Billing** | MonthlyCharges, TotalCharges |

---

## ❓ Problem Statement

Telecom companies lose significant revenue when customers churn. The cost of acquiring a new customer is **5–7× higher** than retaining an existing one. By identifying at-risk customers **before they leave**, the business can:

- Offer targeted discounts or service upgrades
- Improve support for high-risk segments
- Reduce overall customer acquisition costs

**This model predicts:** Will a given customer churn? *(Binary classification)*

---

## Used Tools

| `Python 3.10+` | `Pandas` | `NumPy` | `Plotly` | `Scikit-learn` | `imbalanced-learn (SMOTE)` | `XGBoost` | `Joblib` | `Streamlit` | `FastAPI` | `Uvicorn` | `Jupyter` |

---

## 🔄 Project Pipeline

```
Raw CSV Data
    │
    ▼
Exploratory Data Analysis (Plotly)
    │  ├── Distribution plots
    │  ├── Churn rate by feature
    │  └── Correlation heatmap
    ▼
Data Cleaning
    │  ├── Drop customerID
    │  ├── Fix TotalCharges (string → float)
    │  ├── Encode target (Yes→1, No→0)
    │  └── Handle missing values
    ▼
Preprocessing Pipeline (sklearn)
    │  ├── Numerical: Impute → StandardScaler
    │  └── Categorical: Impute → OneHotEncoder
    ▼
SMOTE Balancing (Train Only)
    ▼
Model Training (5 models)
    │  ├── Logistic Regression
    │  ├── Decision Tree
    │  ├── Random Forest
    │  ├── Gradient Boosting
    │  └── XGBoost
    ▼
Evaluation (Recall, F1, ROC-AUC)
    ▼
Best Model Selection
    ▼
Feature Importance + Business Insights
    ▼
Save Pipeline (final_model.pkl)
    │
    ├── Streamlit App (app/app.py)
    └── FastAPI Service (api/main.py)
```



## ⚙️ How to Run the Notebook

### 1. Set up environment

```bash
# Clone or download the project
cd telco-churn-project

# Create virtual environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Jupyter notebook

The notebook will:
- Perform EDA with interactive Plotly charts
- Train and evaluate 5 models (with SMOTE on training split)
- Save the best model to `models/final_model.pkl`
- Save model comparison to `outputs/model_comparison.csv`


## 🌐 How to Run the Streamlit App

Make sure you have run the notebook first to generate `models/final_model.pkl`.

```bash
python -m streamlit run streamlit_app/app.py
```

Then open your browser to: **http://localhost:8501**

### App Features:
- Sidebar form to enter customer details
- Churn probability gauge chart
- Risk level classification (High / Medium / Low)
- Business action recommendations
- Dataset insight charts (churn by contract, internet, etc.)
- Model comparison table

---

## ⚡ FastAPI Usage

### Start the API

```bash
uvicorn api.main:app --reload
```

Visit the interactive Swagger docs: **http://127.0.0.1:8000/docs**

---

## 📖 API Endpoint Documentation

### `GET /`
**Health check** — confirms the API is running.

**Response:**
```json
{
  "message": "Telco Churn Prediction API is running"
}
```

---

### `POST /predict`
**Predict churn for a single customer.**

**Request body (JSON):**
```json
{
  "gender": "Female",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1026.0
}
```

**Response (churn):**
```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "churn_probability": 0.78,
  "risk_level": "High Risk"
}
```

**Response (no churn):**
```json
{
  "prediction": 0,
  "prediction_label": "No Churn",
  "churn_probability": 0.21,
  "risk_level": "Low Risk"
}
```

**Risk level logic:**
- `probability >= 0.70` → High Risk
- `0.40 <= probability < 0.70` → Medium Risk
- `probability < 0.40` → Low Risk

---

### `POST /predict/batch`
**Predict churn for multiple customers (max 100 per request).**

**Request body:** List of customer objects (same schema as `/predict`).

**Response:** List of prediction results with index.

---

### cURL Example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0
  }'
```

### Python `requests` Example

```python
import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0
}

response = requests.post(url, json=data)
print(response.json())
```

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.6693 | 0.4198 | **0.6613** | 0.5136 | **0.7376** |
| Random Forest | 0.7126 | 0.4610 | 0.5242 | 0.4906 | 0.7329 |
| XGBoost | 0.7495 | 0.5358 | 0.3817 | 0.4458 | 0.7329 |
| Gradient Boosting | 0.7438 | 0.5200 | 0.3844 | 0.4420 | 0.7295 |
| Decision Tree | 0.6828 | 0.4280 | 0.5995 | 0.4994 | 0.7114 |

All metrics above are generated using fixed `random_state=42` with SMOTE applied on the training split only.

**Best model (by Recall):** Logistic Regression — highest churn-capture rate (`Recall = 0.6613`).

**Why not accuracy?** With ~73% non-churners, predicting "No Churn" always gives 73% accuracy. We prioritize Recall (catching churners) and ROC-AUC (model discrimination).

---

## 💡 Key Insights

1. **Month-to-month customers** churn at ~43% vs ~3% for two-year contract holders
2. **Fiber optic customers** churn at ~42%, despite being on the premium service
3. **Electronic check** payment users churn more than auto-payment users
4. **New customers** (tenure < 12 months) are at highest risk
5. **Online security and tech support** subscribers are significantly less likely to churn
6. **Senior citizens** show slightly higher churn rates

---

## 📌 Business Recommendations

| Action | Target Segment | Expected Impact |
|--------|---------------|-----------------|
| Offer contract upgrade incentives | Month-to-month customers | ↓ Churn by ~30% |
| Bundle free tech support trial | Fiber optic users without support | ↓ Churn by ~15% |
| Auto-pay enrollment drive | Electronic check users | ↓ Churn by ~10% |
| New customer onboarding program | Tenure < 6 months | ↓ Early churn |
| Loyalty rewards | Tenure > 36 months | ↑ Retention |

---

## 🔮 Future Improvements

- **Hyperparameter tuning** — GridSearchCV / Optuna for better model performance
- **LightGBM / CatBoost** — Compare against current XGBoost baseline
- **SHAP values** — More robust model explanation beyond feature importance
- **Customer segmentation** — K-means clustering to identify distinct customer groups

---

## 📸 Screenshots

> *(Add screenshots of the Streamlit app and Swagger UI here)*

- `screenshots/app_main.png` — Main Streamlit app interface
- `screenshots/app_prediction_high_risk.png` — High churn risk result
- `screenshots/app_prediction_low_risk.png` — Low churn risk result
- `screenshots/api_swagger.png` — FastAPI Swagger UI
- `screenshots/eda_churn_by_contract.png` — EDA chart

---

## 👨‍💻 Authors

Ahmed Aboalesaad [LinkedIn](https://www.linkedin.com/in/ahmed-aboalesaad/)  
Hossam Elsherbiny [LinkedIn](https://www.linkedin.com/in/h-elsherbiny/)  
Marwan Ragab [LinkedIn](https://www.linkedin.com/in/marwan-ragab-fathy/)  
  
Dataset: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
