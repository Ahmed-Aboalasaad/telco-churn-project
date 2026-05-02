# Project Structure Visualization

```
TELCO CHURN PREDICTION PROJECT
═══════════════════════════════════════════════════════════════

┌───────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                │
│         (data/WA_Fn-UseC_-Telco-Customer-Churn.csv)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────────┐
│                  MVC ARCHITECTURE                             │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐  │
│ │              MODEL LAYER                                │  │
│ │  (mvc/models/)                                          │  │
│ │                                                         │  │
│ │  preprocessing.py                                       │  │
│ │  ├── DataProcessor class                               │  │
│ │  │   ├── load_data()                                   │  │
│ │  │   ├── prepare_features_and_target()                 │  │
│ │  │   ├── build_preprocessor()                          │  │
│ │  │   ├── fit_and_transform()                           │  │
│ │  │   └── save/load_preprocessor()                      │  │
│ │                                                         │  │
│ │  trainer.py                                             │  │
│ │  ├── ModelTrainer class                                │  │
│ │  │   ├── train_all_models()                            │  │
│ │  │   ├── evaluate_models()                             │  │
│ │  │   ├── get_best_model()                              │  │
│ │  │   └── save/load_model()                             │  │
│ │                                                         │  │
│ └─────────────────────────────────────────────────────────┘  │
│           │                                │                  │
│           │                                │                  │
│           ▼                                ▼                  │
│ ┌──────────────────────┐      ┌──────────────────────────┐   │
│ │  4 Trained Models    │      │  Preprocessor Pipeline   │   │
│ │  - Logistic Reg      │      │  - Scaling               │   │
│ │  - Decision Tree     │      │  - Encoding              │   │
│ │  - Random Forest     │      │  - Imputation            │   │
│ │  - Gradient Boost    │      │  - Feature Selection     │   │
│ └──────────────────────┘      └──────────────────────────┘   │
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐  │
│ │         CONTROLLER LAYER                                │  │
│ │  (mvc/controllers/)                                     │  │
│ │                                                         │  │
│ │  pipeline.py                                            │  │
│ │  ├── ChurnPredictionController class                    │  │
│ │  │   ├── load_and_prepare_data()                        │  │
│ │  │   ├── split_data()                                   │  │
│ │  │   ├── train_models()                                 │  │
│ │  │   ├── evaluate_models()                              │  │
│ │  │   ├── get_best_model_info()                          │  │
│ │  │   ├── predict()                                      │  │
│ │  │   ├── predict_proba()                                │  │
│ │  │   └── full_pipeline()  ◄─── ORCHESTRATES ALL         │  │
│ │                                                         │  │
│ └─────────────────────────────────────────────────────────┘  │
│                          │                                    │
│                          ▼                                    │
│ ┌─────────────────────────────────────────────────────────┐  │
│ │              VIEW LAYER                                 │  │
│ │  (mvc/views/)                                           │  │
│ │                                                         │  │
│ │  visualizer.py                                          │  │
│ │  ├── ChurnVisualizer class                              │  │
│ │  ├── EDA Visualizations                                 │  │
│ │  │   ├── plot_churn_distribution()                      │  │
│ │  │   ├── plot_numeric_distribution()                    │  │
│ │  │   ├── plot_categorical_churn_rate()                  │  │
│ │  │   ├── plot_boxplot_numeric()                         │  │
│ │  │   ├── plot_scatter()                                 │  │
│ │  │   └── plot_correlation_heatmap()                     │  │
│ │  │                                                       │  │
│ │  └── Model Visualizations                               │  │
│ │      ├── plot_confusion_matrix()                         │  │
│ │      ├── plot_model_comparison()                         │  │
│ │      ├── plot_feature_importance()                       │  │
│ │      └── plot_roc_curve()                                │  │
│ │                                                         │  │
│ └─────────────────────────────────────────────────────────┘  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ STREAMLIT  │  │   JUPYTER  │  │   FASTAPI  │
    │    APP     │  │  NOTEBOOK  │  │    API     │
    │            │  │            │  │            │
    │ • Home     │  │ • EDA      │  │ • /predict │
    │ • EDA      │  │ • Insights │  │ • /health  │
    │ • Predict  │  │ • Charts   │  │            │
    │ • Perfor.  │  │            │  │            │
    └────────────┘  └────────────┘  └────────────┘

═══════════════════════════════════════════════════════════════

FILE DEPENDENCIES
═════════════════

streamlit_app/app.py
├── imports: mvc.controllers.pipeline
├── imports: mvc.models.preprocessing
├── imports: mvc.views.visualizer
└── uses: models/final_model.pkl, models/preprocessor.pkl

notebooks/02_visualization_eda.ipynb
├── imports: mvc.models.preprocessing
├── imports: mvc.views.visualizer
└── uses: data/WA_Fn-UseC_-Telco-Customer-Churn.csv

api/main.py
├── imports: mvc.controllers.pipeline
└── uses: models/final_model.pkl, models/preprocessor.pkl

mvc/run_pipeline.py
├── imports: mvc.controllers.pipeline
└── outputs: models/final_model.pkl, models/preprocessor.pkl, outputs/model_comparison.csv

═══════════════════════════════════════════════════════════════

DATA FLOW
════════

Raw Data (CSV)
    │
    ▼
[DataProcessor.load_data()]
    │
    ▼
Clean Data (DataFrame)
    │
    ▼
[DataProcessor.prepare_features_and_target()]
    │
    ▼
Features (X) + Target (y)
    │
    ▼
[Train/Test Split]
    │
    ├────────────────────┬─────────────────────┬──────────────────────┐
    │                    │                     │                      │
    ▼                    ▼                     ▼                      ▼
[Preprocess]      [Train Models]       [Evaluate]              [Make Predictions]
    │                    │                     │                      │
    ▼                    ▼                     ▼                      ▼
Scaled X        4 Models Trained      Metrics & Scores        Churn Probabilities
    │                    │                     │                      │
    └────────────────────┴─────────────────────┴──────────────────────┘
                                │
                                ▼
                        [Best Model Selected]
                                │
                    ┌───────────┬───────────┐
                    │           │           │
                    ▼           ▼           ▼
                  Save      Display in   Use for
                 Model     Dashboard    Predictions

═══════════════════════════════════════════════════════════════

EXECUTION FLOWS
═══════════════

FLOW 1: Training (python mvc/run_pipeline.py)
────────────────────────────────────────────

controller.full_pipeline()
├── controller.load_and_prepare_data()
│   └── processor.load_data()
│       └── processor.prepare_features_and_target()
│
├── controller.split_data()
│   └── processor.fit_and_transform()
│
├── controller.train_models()
│   └── trainer.train_all_models()
│
├── controller.evaluate_models()
│   └── trainer.evaluate_models()
│
├── controller.get_best_model_info()
│   └── trainer.get_best_model()
│
└── controller.save_artifacts()
    ├── trainer.save_model()
    └── processor.save_preprocessor()


FLOW 2: Web App (streamlit run streamlit_app/app.py)
────────────────────────────────────────────────────

Load Pipeline
├── controller.load_and_prepare_data()
├── trainer.load_model()
└── processor.load_preprocessor()

Display Dashboard
├── Display EDA
│   └── viz.plot_*()  [Plotly Charts]
├── Make Predictions
│   └── controller.predict()
└── Show Performance
    └── trainer.get_results_dataframe()


FLOW 3: API (uvicorn api.main:app)
──────────────────────────────────

POST /predict
├── Validate Input
├── processor.transform()
├── trainer.predict()
└── Return JSON Response

═══════════════════════════════════════════════════════════════
```

---

## 🔄 Component Interaction Diagram

```
                    EXTERNAL REQUEST
                          │
                          ▼
                 ┌─────────────────┐
                 │   STREAMLIT     │
                 │   or API        │
                 │   or NOTEBOOK   │
                 └────────┬────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
        [PREDICT]              [VISUALIZE]
              │                       │
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │  CONTROLLER      │    │  CONTROLLER      │
    │  .predict()      │    │  (for data prep) │
    └────────┬─────────┘    └────────┬─────────┘
             │                       │
             ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │  DATA PROCESSOR  │    │  DATA PROCESSOR  │
    │  .transform()    │    │  .prepare()      │
    └────────┬─────────┘    └────────┬─────────┘
             │                       │
             ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │  MODEL           │    │  VISUALIZER      │
    │  .predict()      │    │  .plot_*()       │
    └────────┬─────────┘    └────────┬─────────┘
             │                       │
             ▼                       ▼
         PREDICTION               CHARTS
        (JSON/DataFrame)         (Plotly)
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                   RESPONSE TO USER

═══════════════════════════════════════════════════════════════
```
