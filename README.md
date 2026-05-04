# 📡 Telco Customer Churn Prediction

> **Architecture:** MVC Pattern (Model-View-Controller)  
> **Dataset:** Telco Customer Churn — Kaggle (Blastchar)  
> **Deployment:** Streamlit Web App + FastAPI REST API  
> **ML Framework:** Scikit-learn with 4 classification models

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

## 🎯 Project Overview

This project builds an **end-to-end machine learning pipeline** to predict whether a telecom customer will churn (cancel their subscription). It covers the full ML lifecycle:

1. Exploratory Data Analysis (EDA) with interactive Plotly charts
2. Data cleaning and preprocessing with a scikit-learn `Pipeline`
3. Training and comparing 4 classification models
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

## 🛠️ Tools Used

| Tool | Purpose |
|------|---------|
| `Python 3.10+` | Core language |
| `Pandas` | Data loading, cleaning, manipulation |
| `NumPy` | Numerical operations |
| `Plotly` | Interactive EDA visualizations |
| `Scikit-learn` | Preprocessing pipeline, modeling, evaluation |
| `XGBoost` | Gradient boosting model |
| `Joblib` | Model serialization |
| `Streamlit` | Web app deployment |
| `FastAPI` | REST API deployment |
| `Uvicorn` | ASGI server for FastAPI |
| `Jupyter` | Notebook for analysis |

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
Model Training (4 models)
    │  ├── Logistic Regression
    │  ├── Decision Tree
    │  ├── Random Forest
    │  └── Gradient Boosting
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
- Train and evaluate 4 models
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
| Logistic Regression | ~0.80 | ~0.65 | ~0.55 | ~0.60 | **~0.84** |
| Gradient Boosting | ~0.80 | ~0.65 | ~0.54 | ~0.59 | ~0.83 |
| Random Forest | ~0.79 | ~0.63 | ~0.50 | ~0.56 | ~0.82 |
| Decision Tree | ~0.76 | ~0.55 | ~0.52 | ~0.53 | ~0.72 |

> *Metrics vary slightly based on random state and data split.*

**Best model:** Logistic Regression — highest ROC-AUC with good recall and interpretability.

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

1. **SMOTE or class weighting** — Address class imbalance for better recall on churners
2. **Hyperparameter tuning** — GridSearchCV / Optuna for better model performance
3. **XGBoost / LightGBM** — Potentially better tree-based models
4. **SHAP values** — More robust model explanation beyond feature importance
5. **Customer segmentation** — K-means clustering to identify distinct customer groups
6. **Time-series features** — If monthly data is available, track behavior trends
7. **A/B testing integration** — Connect model output to retention campaign system
8. **Model monitoring** — Track model drift over time in production

---

## 📸 Screenshots

> *(Add screenshots of the Streamlit app and Swagger UI here)*

- `screenshots/app_main.png` — Main Streamlit app interface
- `screenshots/app_prediction_high_risk.png` — High churn risk result
- `screenshots/app_prediction_low_risk.png` — Low churn risk result
- `screenshots/api_swagger.png` — FastAPI Swagger UI
- `screenshots/eda_churn_by_contract.png` — EDA chart

---

## 👨‍💻 Author

Built as a capstone project for the **Data Exploration and Preparation** course.  
Dataset: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
