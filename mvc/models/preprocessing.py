"""
Data Preprocessing Pipeline for Telco Churn Dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

class DataProcessor:
    """Handle data cleaning and preprocessing"""
    
    def __init__(self):
        self.numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_columns = None
        self.preprocessor = None
    
    def load_data(self, data_path):
        """Load and clean data"""
        df = pd.read_csv(data_path)
        df = self._clean_data(df)
        return df
    
    def _clean_data(self, df):
        """Clean data - handle types, missing values"""
        # Remove spaces from column names
        df.columns = df.columns.str.strip()
        
        # Convert TotalCharges to numeric (handles spaces)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing values in TotalCharges
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Drop customerID as it's not needed
        if 'customerID' in df.columns:
            df.drop(columns=['customerID'], inplace=True)
        
        # Convert target to binary
        if 'Churn' in df.columns:
            df['Churn'] = (df['Churn'] == 'Yes').astype(int)
        
        return df
    
    def prepare_features_and_target(self, df):
        """Separate features and target"""
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        
        # Identify categorical columns (all except numeric)
        self.categorical_columns = [col for col in X.columns 
                                   if col not in self.numeric_columns]
        
        return X, y
    
    def build_preprocessor(self):
        """Build sklearn preprocessing pipeline"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_columns),
                ('cat', categorical_transformer, self.categorical_columns)
            ]
        )
        
        return self.preprocessor
    
    def fit_and_transform(self, X):
        """Fit preprocessor and transform data"""
        if self.preprocessor is None:
            self.build_preprocessor()
        
        X_processed = self.preprocessor.fit_transform(X)
        return X_processed
    
    def transform(self, X):
        """Transform data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet")
        
        X_processed = self.preprocessor.transform(X)
        return X_processed
    
    def save_preprocessor(self, save_path):
        """Save fitted preprocessor"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, save_path)
    
    def load_preprocessor(self, load_path):
        """Load preprocessor"""
        self.preprocessor = joblib.load(load_path)
