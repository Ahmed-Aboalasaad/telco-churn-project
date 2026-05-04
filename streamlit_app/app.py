"""
Streamlit Web Application for Telco Churn Prediction
Interactive dashboard for customer churn prediction and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

# Add MVC modules to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'mvc'))

from controllers.pipeline import ChurnPredictionController
from views.visualizer import ChurnVisualizer
from models.preprocessing import DataProcessor

# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header {
        color: #1f77b4;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize paths
DATA_PATH = project_root / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODEL_PATH = project_root / 'models' / 'final_model.pkl'
PREPROCESSOR_PATH = project_root / 'models' / 'preprocessor.pkl'

@st.cache_resource
def load_pipeline():
    """Load and initialize the pipeline"""
    controller = ChurnPredictionController()
    try:
        # Load data
        df = controller.load_and_prepare_data(str(DATA_PATH))
        
        # Load trained model and preprocessor
        controller.model_trainer.load_model(str(MODEL_PATH))
        controller.data_processor.load_preprocessor(str(PREPROCESSOR_PATH))
        
        return controller, df
    except FileNotFoundError:
        st.error("❌ Model files not found. Please train the model first using the MVC pipeline.")
        return None, None

@st.cache_data
def compute_roc_curve_data():
    """Train baseline models and return ROC curve points for each model."""
    controller = ChurnPredictionController()
    controller.load_and_prepare_data(str(DATA_PATH))
    X_train, X_test, y_train, y_test = controller.split_data()
    controller.train_models()

    roc_data = []
    for model_name, model in controller.model_trainer.models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_data.append({
            'model': model_name,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(auc(fpr, tpr))
        })

    return roc_data

# ==================== Main App ====================

def main():
    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "📊 EDA & Insights", "🔮 Predict", "📈 Model Performance"]
    )
    
    # Load pipeline
    controller, df = load_pipeline()
    
    if controller is None or df is None:
        st.error("Failed to load the application. Please check the model files.")
        return
    
    # ==================== HOME PAGE ====================
    if page == "🏠 Home":
        st.markdown('<div class="header">📡 Telco Customer Churn Prediction</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        ### Welcome! 👋
        
        This application predicts whether a telecom customer will churn (cancel their subscription) 
        using machine learning models.
        
        **Features:**
        - 📊 **EDA & Insights** - Explore customer data and patterns
        - 🔮 **Predict** - Make predictions for individual or batch customers
        - 📈 **Model Performance** - View model metrics and comparisons
        
        **Dataset Overview:**
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📱 Total Customers", f"{len(df):,}")
        with col2:
            churn_count = df['Churn'].sum()
            st.metric("⚠️ Churned", f"{churn_count:,}")
        with col3:
            churn_rate = (churn_count / len(df) * 100)
            st.metric("📊 Churn Rate", f"{churn_rate:.1f}%")
        with col4:
            st.metric("🔧 Features", f"{df.shape[1] - 1}")
        
        st.markdown("---")
        
        st.subheader("🎯 Business Goal")
        st.write("""
        Identify customers at high risk of churn and recommend retention strategies 
        to reduce revenue loss and improve customer lifetime value.
        """)
        
        st.subheader("⚙️ How to Use")
        st.write("""
        1. **Explore Data** → Go to "EDA & Insights" to understand customer patterns
        2. **Make Predictions** → Go to "Predict" to enter customer data and get churn predictions
        3. **View Model Performance** → Check "Model Performance" to see how our models perform
        """)
    
    # ==================== EDA PAGE ====================
    elif page == "📊 EDA & Insights":
        st.title("📊 Exploratory Data Analysis")
        
        viz = ChurnVisualizer(df)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Distribution", "🎯 Churn Analysis", "📌 Correlations", "💡 Insights"
        ])
        
        with tab1:
            st.subheader("Numeric Features Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(viz.plot_numeric_distribution('tenure'), width='stretch')
            with col2:
                st.plotly_chart(viz.plot_numeric_distribution('MonthlyCharges'), width='stretch')
            
            st.plotly_chart(viz.plot_churn_distribution(), width='stretch')
        
        with tab2:
            st.subheader("Churn Rate by Key Features")
            
            feature_select = st.selectbox(
                "Select feature:",
                ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 
                 'TechSupport', 'gender', 'SeniorCitizen']
            )
            
            fig = viz.plot_categorical_churn_rate(feature_select)
            st.plotly_chart(fig, width='stretch')
        
        with tab3:
            st.subheader("Feature Correlations")
            st.plotly_chart(viz.plot_correlation_heatmap(), width='stretch')
        
        with tab4:
            st.subheader("🔍 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Churn Statistics:**
                - Total Churned: {df['Churn'].sum():,}
                - Churn Rate: {(df['Churn'].sum() / len(df) * 100):.2f}%
                - Avg Tenure (Churned): {df[df['Churn']==1]['tenure'].mean():.1f} months
                - Avg Tenure (Retained): {df[df['Churn']==0]['tenure'].mean():.1f} months
                """)
            
            with col2:
                st.warning(f"""
                **Highest Risk Segments:**
                - Month-to-month: {(df[df['Contract']=='Month-to-month']['Churn'].mean()*100):.1f}% churn
                - Fiber optic: {(df[df['InternetService']=='Fiber optic']['Churn'].mean()*100):.1f}% churn
                - E-check payment: {(df[df['PaymentMethod']=='Electronic check']['Churn'].mean()*100):.1f}% churn
                """)
    
    # ==================== PREDICT PAGE ====================
    elif page == "🔮 Predict":
        st.title("🔮 Churn Prediction")
        
        st.markdown("### Enter Customer Details")
        
        # Create columns for input
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        with col2:
            st.subheader("Account Details")
            tenure = st.slider("Tenure (months)", 0, 72, 24)
            contract = st.selectbox("Contract Type", 
                                   ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method",
                                         ["Electronic check", "Mailed check", 
                                          "Bank transfer (automatic)", 
                                          "Credit card (automatic)"])
        
        with col3:
            st.subheader("Services & Charges")
            internet_service = st.selectbox("Internet Service", 
                                           ["DSL", "Fiber optic", "No"])
            monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 70)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        
        st.divider()
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.subheader("Internet Services")
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        
        with col5:
            st.subheader("Entertainment")
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        # Make prediction button
        if st.button("🔮 Predict Churn", width='stretch'):
            try:
                # Create input dataframe with all required columns
                input_data = pd.DataFrame({
                    'customerID': ['SAMPLE001'],
                    'gender': [gender],
                    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'tenure': [tenure],
                    'PhoneService': [phone_service],
                    'MultipleLines': [multiple_lines],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [online_security],
                    'OnlineBackup': [online_backup],
                    'DeviceProtection': [device_protection],
                    'TechSupport': [tech_support],
                    'StreamingTV': [streaming_tv],
                    'StreamingMovies': [streaming_movies],
                    'Contract': [contract],
                    'PaperlessBilling': [paperless_billing],
                    'PaymentMethod': [payment_method],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [tenure * monthly_charges]
                })
                
                # Drop customerID as it's not needed for prediction
                input_data = input_data.drop(columns=['customerID'])
                
                # Make prediction
                prediction = controller.predict(input_data)[0]
                probability = controller.predict_proba(input_data)[0][1]
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 0:
                        st.success(f"✅ **WILL RETAIN** (No Churn)")
                        st.metric("Churn Probability", f"{probability*100:.1f}%")
                    else:
                        st.error(f"⚠️ **HIGH CHURN RISK** (Will Churn)")
                        st.metric("Churn Probability", f"{probability*100:.1f}%")
                
                with col2:
                    # Display recommendation
                    if probability > 0.7:
                        st.warning("""
                        **🎯 Recommendation:**
                        - Offer immediate retention offer
                        - Contact customer proactively
                        - Review service satisfaction
                        """)
                    elif probability > 0.4:
                        st.info("""
                        **🎯 Recommendation:**
                        - Monitor customer activity
                        - Send promotional offers
                        - Improve service quality
                        """)
                    else:
                        st.success("""
                        **🎯 Recommendation:**
                        - Continue regular support
                        - Maintain satisfaction levels
                        """)
            
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    # ==================== MODEL PERFORMANCE PAGE ====================
    elif page == "📈 Model Performance":
        st.title("📈 Model Performance")
        
        st.info("""
        This page shows the performance metrics of the trained ML models.
        The current production model is selected automatically based on the highest ROC-AUC score.
        """)
        
        # Display model comparison (from saved results if available)
        try:
            results_path = project_root / 'outputs' / 'model_comparison.csv'
            if results_path.exists():
                results_df = pd.read_csv(results_path)
                if not results_df.empty and 'ROC-AUC' in results_df.columns:
                    best_row = results_df.sort_values('ROC-AUC', ascending=False).iloc[0]
                    st.success(
                        f"🏆 Current best model: **{best_row['Model']}** "
                        f"(ROC-AUC: {best_row['ROC-AUC']:.4f})"
                    )
                
                st.subheader("📊 Model Comparison")
                st.dataframe(results_df, width='stretch')
                
                # Visualize comparison
                fig = px.bar(
                    results_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                    x='Model',
                    y='Score',
                    color='Metric',
                    barmode='group',
                    title='Model Performance Metrics',
                    height=400
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("Model comparison results not found. Run the training pipeline first.")
        
        except Exception as e:
            st.warning(f"Could not load model comparison: {str(e)}")

        st.subheader("📉 ROC Curves")
        try:
            roc_data = compute_roc_curve_data()
            roc_fig = go.Figure()

            for curve in roc_data:
                roc_fig.add_trace(go.Scatter(
                    x=curve['fpr'],
                    y=curve['tpr'],
                    mode='lines',
                    name=f"{curve['model']} (AUC={curve['auc']:.3f})",
                    line=dict(width=2)
                ))

            roc_fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray')
            ))

            roc_fig.update_layout(
                title='ROC Curves - All Models',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500,
                hovermode='closest'
            )
            st.plotly_chart(roc_fig, width='stretch')
        except Exception as e:
            st.warning(f"Could not generate ROC curves: {str(e)}")
        
        st.subheader("📌 Key Metrics Explained")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Accuracy** - Overall correctness of predictions
            
            **Precision** - Of predicted churners, how many actually churn
            
            **Recall** - Of actual churners, how many we identify
            """)
        
        with col2:
            st.markdown("""
            **F1-Score** - Balanced measure of precision and recall
            
            **ROC-AUC** - Model's ability to distinguish between classes
            """)

if __name__ == "__main__":
    main()
