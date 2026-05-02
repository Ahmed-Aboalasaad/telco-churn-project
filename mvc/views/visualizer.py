"""
Visualization functions using Plotly
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChurnVisualizer:
    """Visualization class for exploratory data analysis"""
    
    def __init__(self, df):
        self.df = df
        self.numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_columns = [col for col in df.columns 
                                   if col not in self.numeric_columns + ['Churn']]
    
    # ==================== EDA Visualizations ====================
    
    def plot_churn_distribution(self):
        """Pie chart of churn distribution"""
        churn_counts = self.df['Churn'].value_counts()
        labels = ['No Churn', 'Churn']
        
        fig = go.Figure(data=[
            go.Pie(labels=labels, 
                   values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                   marker=dict(colors=['#2ecc71', '#e74c3c']))
        ])
        
        fig.update_layout(
            title="Customer Churn Distribution",
            font=dict(size=14),
            height=500
        )
        
        return fig
    
    def plot_numeric_distribution(self, column):
        """Histogram for numeric columns"""
        fig = px.histogram(
            self.df,
            x=column,
            nbins=30,
            color='Churn',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            title=f"Distribution of {column} by Churn Status",
            labels={'Churn': 'Churn Status'},
            height=500
        )
        
        fig.update_xaxes(title_text=column)
        fig.update_yaxes(title_text="Count")
        
        return fig
    
    def plot_categorical_churn_rate(self, column):
        """Bar chart of churn rate by categorical feature"""
        churn_rate = self.df.groupby(column)['Churn'].agg(['sum', 'count'])
        churn_rate['rate'] = (churn_rate['sum'] / churn_rate['count'] * 100).round(2)
        churn_rate = churn_rate.reset_index()
        churn_rate = churn_rate.sort_values('rate', ascending=False)
        
        fig = px.bar(
            churn_rate,
            x=column,
            y='rate',
            color='rate',
            color_continuous_scale='RdYlGn_r',
            title=f"Churn Rate by {column}",
            labels={'rate': 'Churn Rate (%)'},
            height=500
        )
        
        fig.update_yaxes(title_text="Churn Rate (%)")
        
        return fig
    
    def plot_boxplot_numeric(self, column):
        """Box plot for numeric columns by churn status"""
        fig = px.box(
            self.df,
            y=column,
            x='Churn',
            color='Churn',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            title=f"{column} Distribution by Churn Status",
            labels={'Churn': 'Churn Status'},
            height=500
        )
        
        fig.update_xaxes(title_text="Churn Status")
        fig.update_yaxes(title_text=column)
        
        return fig
    
    def plot_scatter(self, x_col, y_col):
        """Scatter plot for numeric features"""
        fig = px.scatter(
            self.df,
            x=x_col,
            y=y_col,
            color='Churn',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            title=f"{x_col} vs {y_col}",
            height=500
        )
        
        return fig
    
    def plot_correlation_heatmap(self):
        """Correlation heatmap for numeric features"""
        numeric_df = self.df[self.numeric_columns]
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Correlation Matrix - Numeric Features",
            height=500,
            width=600
        )
        
        return fig
    
    # ==================== Model Visualizations ====================
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['No Churn', 'Churn'],
            y=['No Churn', 'Churn'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix - {model_name}",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=500
        )
        
        return fig
    
    def plot_model_comparison(self, results_df):
        """Bar chart comparing model performance metrics"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['Model'],
                y=results_df[metric],
                text=np.round(results_df[metric], 4),
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Comparison - Performance Metrics",
            barmode='group',
            xaxis_title="Model",
            yaxis_title="Score",
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_feature_importance(self, model, feature_names, top_n=15):
        """Plot feature importance from tree-based models"""
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig = px.bar(
            importance_df,
            y='Feature',
            x='Importance',
            orientation='h',
            title=f"Top {top_n} Feature Importance",
            color='Importance',
            color_continuous_scale='Viridis',
            height=600
        )
        
        fig.update_yaxes(categoryorder='total ascending')
        
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC={roc_auc:.4f})',
            line=dict(width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title=f"ROC Curve - {model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=600,
            hovermode='closest'
        )
        
        return fig
