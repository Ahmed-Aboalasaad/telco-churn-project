"""
Telco Customer Churn Predictor — Streamlit App
================================================
Course: Data Exploration and Preparation
Dataset: Telco Customer Churn (Kaggle — Blastchar)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import json

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "final_model.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
    df['Churn_num'] = (df['Churn'] == 'Yes').astype(int)
    return df

model = load_model()
df_ref = load_data()

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #1a73e8, #e91e63);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .subtitle { color: #666; font-size: 1rem; margin-bottom: 1.5rem; }
  .metric-card {
    background: #f8f9fa; border-radius: 12px;
    padding: 1rem 1.5rem; text-align: center;
    border: 1px solid #dee2e6;
  }
  .churn-high {
    background: linear-gradient(135deg, #ff6b6b, #ee5a24);
    color: white; border-radius: 16px; padding: 1.5rem 2rem;
    font-size: 1.6rem; font-weight: 700; text-align: center;
  }
  .churn-low {
    background: linear-gradient(135deg, #00b894, #00cec9);
    color: white; border-radius: 16px; padding: 1.5rem 2rem;
    font-size: 1.6rem; font-weight: 700; text-align: center;
  }
  .section-header {
    font-size: 1.1rem; font-weight: 600; color: #1a73e8;
    border-bottom: 2px solid #e8f0fe; padding-bottom: 0.4rem;
    margin-bottom: 1rem;
  }
  div[data-testid="stSidebar"] { background: #f0f4ff; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📡 Telco Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Data Exploration & Preparation Course Project · Predict whether a telecom customer will churn based on their profile and service details.</div>', unsafe_allow_html=True)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown('<div class="metric-card">🗃️<br><b>7,043</b><br><small>Total Customers</small></div>', unsafe_allow_html=True)
with col_m2:
    st.markdown('<div class="metric-card">📉<br><b>26.4%</b><br><small>Churn Rate</small></div>', unsafe_allow_html=True)
with col_m3:
    st.markdown('<div class="metric-card">🧠<br><b>Logistic Reg.</b><br><small>Best Model</small></div>', unsafe_allow_html=True)
with col_m4:
    st.markdown('<div class="metric-card">🎯<br><b>ROC-AUC 0.74</b><br><small>Model Score</small></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar Inputs ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Customer Profile")
    st.caption("Fill in the customer details below and click **Predict Churn** to get a result.")

    st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
    gender        = st.selectbox("Gender", ["Male", "Female"])
    senior        = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner       = st.selectbox("Partner", ["Yes", "No"])
    dependents    = st.selectbox("Dependents", ["Yes", "No"])

    st.markdown('<div class="section-header">📋 Account</div>', unsafe_allow_html=True)
    tenure        = st.slider("Tenure (months)", 0, 72, 12)
    contract      = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless     = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment       = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    st.markdown('<div class="section-header">📞 Phone Services</div>', unsafe_allow_html=True)
    phone_svc     = st.selectbox("Phone Service", ["Yes", "No"])
    multi_lines   = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    st.markdown('<div class="section-header">🌐 Internet Services</div>', unsafe_allow_html=True)
    internet      = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    svc_opts      = ["Yes", "No", "No internet service"]
    online_sec    = st.selectbox("Online Security",    svc_opts)
    online_bak    = st.selectbox("Online Backup",      svc_opts)
    device_prot   = st.selectbox("Device Protection",  svc_opts)
    tech_sup      = st.selectbox("Tech Support",       svc_opts)
    streaming_tv  = st.selectbox("Streaming TV",       svc_opts)
    streaming_mov = st.selectbox("Streaming Movies",   svc_opts)

    st.markdown('<div class="section-header">💳 Billing</div>', unsafe_allow_html=True)
    monthly       = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    auto_total    = tenure * monthly
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10000.0,
        value=float(round(auto_total, 2)), step=1.0,
        help="Auto-calculated as Tenure × Monthly Charges"
    )

    st.markdown("---")
    predict_btn   = st.button("🔮 Predict Churn", use_container_width=True, type="primary")

# ── Build input dataframe ───────────────────────────────────────────────────────
def build_input():
    return pd.DataFrame([{
        "gender":           gender,
        "SeniorCitizen":    senior,   # kept as Yes/No to match training
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_svc,
        "MultipleLines":    multi_lines,
        "InternetService":  internet,
        "OnlineSecurity":   online_sec,
        "OnlineBackup":     online_bak,
        "DeviceProtection": device_prot,
        "TechSupport":      tech_sup,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_mov,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment,
        "MonthlyCharges":   monthly,
        "TotalCharges":     total_charges,
    }])

# ── Main area ──────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    if predict_btn:
        customer_df = build_input()

        churn_prob  = model.predict_proba(customer_df)[0][1]
        churn_class = int(churn_prob >= 0.5)

        st.markdown("### 📊 Prediction Result")

        if churn_class == 1:
            st.markdown(f'<div class="churn-high">⚠️ High Churn Risk<br><span style="font-size:1rem;opacity:0.9">This customer is likely to leave</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="churn-low">✅ Low Churn Risk<br><span style="font-size:1rem;opacity:0.9">This customer is likely to stay</span></div>', unsafe_allow_html=True)

        st.markdown("")

        # Gauge chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            number={"suffix": "%", "font": {"size": 36}},
            delta={"reference": 26.4, "increasing": {"color": "#e74c3c"}, "decreasing": {"color": "#2ecc71"}},
            title={"text": "Churn Probability", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#e74c3c" if churn_class else "#2ecc71", "thickness": 0.3},
                "steps": [
                    {"range": [0, 40],  "color": "#d5f5e3"},
                    {"range": [40, 70], "color": "#fef9e7"},
                    {"range": [70, 100],"color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
            }
        ))
        gauge.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)

        # Risk level
        if churn_prob >= 0.70:
            risk, risk_color = "🔴 High Risk", "#e74c3c"
        elif churn_prob >= 0.40:
            risk, risk_color = "🟡 Medium Risk", "#f39c12"
        else:
            risk, risk_color = "🟢 Low Risk", "#2ecc71"

        st.markdown(f"**Risk Level:** <span style='color:{risk_color};font-weight:700'>{risk}</span> &nbsp;|&nbsp; **Probability:** `{churn_prob:.1%}`", unsafe_allow_html=True)

        # Explanation
        st.markdown("---")
        st.markdown("### 💡 What This Means")
        if churn_class == 1:
            st.error(
                "**This customer shows signs of potential churn.**\n\n"
                "Possible contributing factors based on the inputs:\n"
                f"- Contract type: **{contract}** — shorter contracts correlate with higher churn\n"
                f"- Payment method: **{payment}** — electronic check users churn more often\n"
                f"- Tenure: **{tenure} months** — newer customers are at greater risk\n\n"
                "**Recommended actions:**\n"
                "- 🎁 Offer a loyalty discount or contract upgrade\n"
                "- 🛡️ Provide free trial of Online Security or Tech Support\n"
                "- 📞 Schedule a proactive customer success call"
            )
        else:
            st.success(
                "**This customer appears stable and unlikely to churn.**\n\n"
                f"- Tenure: **{tenure} months** — long-tenure customers are loyal\n"
                f"- Contract: **{contract}** — longer contracts reduce churn risk\n\n"
                "**Recommended actions:**\n"
                "- 🌟 Maintain regular engagement and service quality\n"
                "- 📈 Offer upsell opportunities (premium services)\n"
                "- 💌 Enroll in a loyalty rewards program"
            )

    else:
        # Placeholder before prediction
        st.info("👈 **Fill in the customer details in the sidebar and click 'Predict Churn'** to see the prediction result and churn probability.")

        st.markdown("### 📌 How This Works")
        st.markdown("""
1. **Fill in** the customer's profile details in the left sidebar
2. **Click** the **Predict Churn** button
3. **View** the churn probability gauge, risk level, and interpretation
4. **Take action** based on the recommendations provided

The model was trained on **7,043 Telco customers** using their demographics,
account details, subscribed services, and billing information.
        """)

# ── Right column: Charts ───────────────────────────────────────────────────────
with right_col:
    st.markdown("### 📈 Dataset Insights")

    tab1, tab2, tab3 = st.tabs(["Churn Overview", "Customer Comparison", "Model Results"])

    with tab1:
        # Churn by Contract
        grp = df_ref.groupby('Contract')['Churn_num'].mean().reset_index()
        grp.columns = ['Contract', 'Churn Rate']
        grp['Churn Rate %'] = (grp['Churn Rate'] * 100).round(1)
        fig = px.bar(grp, x='Contract', y='Churn Rate %', text='Churn Rate %',
                     color='Churn Rate %', color_continuous_scale='RdYlGn_r',
                     title='Churn Rate by Contract Type', range_color=[0, 60])
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(height=280, margin=dict(t=40, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Churn by Internet Service
        grp2 = df_ref.groupby('InternetService')['Churn_num'].mean().reset_index()
        grp2.columns = ['InternetService', 'Churn Rate']
        grp2['Churn Rate %'] = (grp2['Churn Rate'] * 100).round(1)
        fig2 = px.bar(grp2, x='InternetService', y='Churn Rate %', text='Churn Rate %',
                      color='Churn Rate %', color_continuous_scale='RdYlGn_r',
                      title='Churn Rate by Internet Service', range_color=[0, 60])
        fig2.update_traces(texttemplate='%{text}%', textposition='outside')
        fig2.update_layout(height=280, margin=dict(t=40, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if predict_btn:
            # Compare customer vs dataset averages
            avg_monthly = df_ref['MonthlyCharges'].mean()
            avg_tenure  = df_ref['tenure'].mean()

            comparison = pd.DataFrame({
                'Metric': ['Monthly Charges ($)', 'Tenure (Months)'],
                'This Customer': [monthly, tenure],
                'Dataset Average': [avg_monthly, avg_tenure]
            })

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name='This Customer', x=comparison['Metric'],
                                     y=comparison['This Customer'],
                                     marker_color='#3498db', text=comparison['This Customer'].round(1),
                                     textposition='outside'))
            fig_cmp.add_trace(go.Bar(name='Dataset Average', x=comparison['Metric'],
                                     y=comparison['Dataset Average'],
                                     marker_color='#bdc3c7', text=comparison['Dataset Average'].round(1),
                                     textposition='outside'))
            fig_cmp.update_layout(barmode='group', title='Your Customer vs Dataset Average',
                                  height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Tenure bucket analysis
            df_ref['Tenure Group'] = pd.cut(df_ref['tenure'], bins=[0,12,24,48,72],
                                             labels=['0-12m', '13-24m', '25-48m', '49-72m'])
            grp_t = df_ref.groupby('Tenure Group', observed=True)['Churn_num'].mean().reset_index()
            grp_t.columns = ['Tenure Group', 'Churn Rate']
            grp_t['Churn Rate %'] = (grp_t['Churn Rate'] * 100).round(1)
            fig_t = px.line(grp_t, x='Tenure Group', y='Churn Rate %',
                            markers=True, title='Churn Rate by Tenure Group',
                            color_discrete_sequence=['#e74c3c'])
            fig_t.update_layout(height=280, margin=dict(t=40, b=20))
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.caption("Make a prediction first to see customer comparison charts.")

            # Scatter plot
            sample = df_ref.sample(500, random_state=42)
            fig_sc = px.scatter(sample, x='tenure', y='MonthlyCharges', color='Churn',
                                color_discrete_map={'No': '#2ecc71', 'Yes': '#e74c3c'},
                                opacity=0.5, title='Tenure vs Monthly Charges — Colored by Churn',
                                labels={'tenure': 'Tenure (Months)', 'MonthlyCharges': 'Monthly Charges ($)'})
            fig_sc.update_layout(height=400, margin=dict(t=40, b=20))
            st.plotly_chart(fig_sc, use_container_width=True)

    with tab3:
        # Model comparison from saved CSV
        model_cmp_path = os.path.join(BASE_DIR, "outputs", "model_comparison.csv")
        if os.path.exists(model_cmp_path):
            cmp_df = pd.read_csv(model_cmp_path)
            st.dataframe(cmp_df.style.format({
                'Accuracy': '{:.3f}', 'Precision': '{:.3f}',
                'Recall': '{:.3f}', 'F1': '{:.3f}',
                'Macro F1': '{:.3f}', 'ROC-AUC': '{:.3f}'
            }).highlight_max(subset=['Accuracy','Precision','Recall','F1','ROC-AUC'],
                             color='#d5f5e3'), use_container_width=True)

            # Bar chart
            fig_mc = go.Figure()
            for metric in ['Recall', 'F1', 'ROC-AUC']:
                fig_mc.add_trace(go.Bar(name=metric, x=cmp_df['Model'], y=cmp_df[metric], text=cmp_df[metric].round(3), textposition='outside'))
            fig_mc.update_layout(barmode='group', title='Key Metrics by Model',
                                  height=350, margin=dict(t=40, b=20),
                                  yaxis=dict(range=[0, 1.1]))
            st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.caption("Model comparison file not found. Run the notebook first.")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#999; font-size:0.85rem;'>
📡 Telco Customer Churn Predictor &nbsp;·&nbsp; Data Exploration & Preparation Course &nbsp;·&nbsp;
Built with Streamlit + Plotly + Scikit-learn
</div>
""", unsafe_allow_html=True)
