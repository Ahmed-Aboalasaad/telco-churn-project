"""
Quick Start Guide - Telco Churn Project with MVC Architecture
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("""
╔════════════════════════════════════════════════════════════════╗
║    📡 TELCO CHURN PREDICTION - MVC ARCHITECTURE                ║
║    Quick Start Guide                                           ║
╚════════════════════════════════════════════════════════════════╝
""")

print("=" * 64)
print("STEP 1: Install Dependencies")
print("=" * 64)
print("""
Run in terminal:
  pip install -r requirements.txt

Or with conda:
  conda create -n churn-env python=3.10
  conda activate churn-env
  pip install -r requirements.txt
""")

print("=" * 64)
print("STEP 2: Train Models (MVC Pipeline)")
print("=" * 64)
print("""
Run in terminal:
  python mvc/run_pipeline.py

This will:
  ✓ Load and clean data
  ✓ Preprocess features
  ✓ Split into train/test
  ✓ Train 4 models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
  ✓ Evaluate and compare models
  ✓ Save best model: models/final_model.pkl
  ✓ Save preprocessor: models/preprocessor.pkl
  ✓ Save results: outputs/model_comparison.csv
  
Expected output:
  - Best Model: Gradient Boosting
  - ROC-AUC: 0.8705
  - Accuracy: 82.83%
""")

print("=" * 64)
print("STEP 3: Launch Streamlit Web App")
print("=" * 64)
print("""
Run in terminal:
  streamlit run streamlit_app/app.py

Opens in browser: http://localhost:8501

Features:
  🏠 Home - Dataset overview & churn statistics
  📊 EDA - Explore data patterns with interactive charts
  🔮 Predict - Make individual customer predictions
  📈 Performance - View model metrics & comparisons
""")

print("=" * 64)
print("STEP 4: (Optional) Run FastAPI Server")
print("=" * 64)
print("""
Run in terminal:
  uvicorn api.main:app --reload --port 8000

API Documentation: http://localhost:8000/docs
Redoc: http://localhost:8000/redoc

Endpoints:
  POST /predict - Make predictions
  GET /health - Server health check
""")

print("=" * 64)
print("STEP 5: (Optional) Explore Jupyter Notebook")
print("=" * 64)
print("""
Run in terminal:
  jupyter notebook notebooks/

Open: notebooks/02_visualization_eda.ipynb

Features:
  - Interactive Plotly visualizations
  - EDA and data exploration
  - Churn insights and patterns
  - Business recommendations
""")

print("=" * 64)
print("PROJECT STRUCTURE (MVC Architecture)")
print("=" * 64)
print("""
mvc/                           # MVC Application Layer
├── models/                    # MODEL - Data & ML logic
│   ├── preprocessing.py       # Data cleaning & feature engineering
│   └── trainer.py             # Model training & evaluation
├── controllers/               # CONTROLLER - Orchestration
│   └── pipeline.py            # Pipeline control & flow
├── views/                     # VIEW - Visualization
│   └── visualizer.py          # Plotly charts
└── run_pipeline.py            # Training script

streamlit_app/                 # Web Application
└── app.py                     # Interactive dashboard

notebooks/                     # Data Exploration
├── 01_eda_preprocessing_modeling.ipynb   # Original (all-in-one)
└── 02_visualization_eda.ipynb            # NEW (visualization only)

api/                           # REST API
└── main.py                    # FastAPI endpoints

data/                          # Dataset
└── WA_Fn-UseC_-Telco-Customer-Churn.csv

models/                        # Trained Models
├── final_model.pkl            # Gradient Boosting model
└── preprocessor.pkl           # Data preprocessor

outputs/                       # Results
└── model_comparison.csv       # Model metrics
""")

print("=" * 64)
print("USAGE EXAMPLES")
print("=" * 64)
print("""
1. Using MVC in Python scripts:

   from mvc.controllers.pipeline import ChurnPredictionController
   
   controller = ChurnPredictionController()
   controller.load_and_prepare_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
   controller.split_data()
   controller.train_models()
   controller.evaluate_models()
   predictions = controller.predict(new_data)

2. Using Visualizer:

   from mvc.views.visualizer import ChurnVisualizer
   
   viz = ChurnVisualizer(df)
   fig = viz.plot_categorical_churn_rate('Contract')
   fig.show()

3. Making predictions:

   import pandas as pd
   new_customer = pd.DataFrame({
       'tenure': [12],
       'MonthlyCharges': [65],
       'Contract': ['Month-to-month'],
       'InternetService': ['Fiber optic'],
       # ... other features
   })
   
   prediction = controller.predict(new_customer)
   probability = controller.predict_proba(new_customer)[0][1]
   
   if probability > 0.7:
       print("⚠️ High churn risk - Recommend retention offer")
""")

print("=" * 64)
print("TROUBLESHOOTING")
print("=" * 64)
print("""
Issue: Model files not found
  → Run: python mvc/run_pipeline.py first

Issue: "Module not found" errors
  → Check: PYTHONPATH includes project root
  → Or: cd to project root before running

Issue: Streamlit app won't start
  → Check: streamlit run streamlit_app/app.py
  → Check: models/final_model.pkl exists

Issue: Missing dependencies
  → Run: pip install -r requirements.txt
""")

print("=" * 64)
print("NEXT STEPS")
print("=" * 64)
print("""
1. ✓ Install dependencies: pip install -r requirements.txt
2. ✓ Train models: python mvc/run_pipeline.py
3. ✓ Run Streamlit: streamlit run streamlit_app/app.py
4. ✓ Explore notebook: jupyter notebook notebooks/02_visualization_eda.ipynb
5. ✓ Check API: uvicorn api.main:app --reload

Happy Data Science! 🚀
""")

# Check if models exist
print("\n" + "=" * 64)
print("SYSTEM CHECK")
print("=" * 64)

checks = {
    "Data file": PROJECT_ROOT / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv',
    "MVC preprocessing": PROJECT_ROOT / 'mvc' / 'models' / 'preprocessing.py',
    "MVC trainer": PROJECT_ROOT / 'mvc' / 'models' / 'trainer.py',
    "MVC controller": PROJECT_ROOT / 'mvc' / 'controllers' / 'pipeline.py',
    "MVC visualizer": PROJECT_ROOT / 'mvc' / 'views' / 'visualizer.py',
    "Streamlit app": PROJECT_ROOT / 'streamlit_app' / 'app.py',
    "Training script": PROJECT_ROOT / 'mvc' / 'run_pipeline.py',
    "Notebook (new)": PROJECT_ROOT / 'notebooks' / '02_visualization_eda.ipynb',
}

for name, path in checks.items():
    status = "✓" if path.exists() else "✗"
    print(f"{status} {name}: {path.name}")

print("\n" + "=" * 64)
