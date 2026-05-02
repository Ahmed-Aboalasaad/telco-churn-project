"""
Model Training and Evaluation for Telco Churn Prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import joblib
from pathlib import Path

class ModelTrainer:
    """Train and evaluate ML models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_all_models(self, X_train, y_train):
        """Train all models"""
        
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr
        
        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=6, min_samples_split=20, 
                                    random_state=42)
        dt.fit(X_train, y_train)
        self.models['Decision Tree'] = dt
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                        max_depth=4, random_state=42)
        gb.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
                'Confusion Matrix': confusion_matrix(y_test, y_pred),
                'Classification Report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            self.results[name] = metrics
        
        return self.results
    
    def get_best_model(self):
        """Get best model based on ROC-AUC"""
        if not self.results:
            raise ValueError("No models evaluated yet")
        
        best_score = -1
        best_name = None
        
        for name, metrics in self.results.items():
            if metrics['ROC-AUC'] > best_score:
                best_score = metrics['ROC-AUC']
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        return self.best_model_name, self.best_model
    
    def save_model(self, save_path):
        """Save best model"""
        if self.best_model is None:
            raise ValueError("No best model selected")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, save_path)
    
    def load_model(self, load_path):
        """Load saved model"""
        self.best_model = joblib.load(load_path)
        return self.best_model
    
    def get_results_dataframe(self):
        """Convert results to DataFrame for comparison"""
        results_list = []
        
        for model_name, metrics in self.results.items():
            results_list.append({
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1-Score': metrics['F1-Score'],
                'ROC-AUC': metrics['ROC-AUC']
            })
        
        return pd.DataFrame(results_list)
    
    def predict(self, X):
        """Make predictions using best model"""
        if self.best_model is None:
            raise ValueError("No model loaded")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.best_model is None:
            raise ValueError("No model loaded")
        
        return self.best_model.predict_proba(X)
