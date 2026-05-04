"""
Controller for managing the ML pipeline
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.preprocessing import DataProcessor
from models.trainer import ModelTrainer

class ChurnPredictionController:
    """Main controller for churn prediction pipeline"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.X = None
        self.y = None
    
    def load_and_prepare_data(self, data_path=None):
        """Load and prepare data for modeling"""
        if data_path:
            self.data_path = data_path
        
        if not self.data_path:
            raise ValueError("Data path not provided")
        
        # Load data
        self.df = self.data_processor.load_data(self.data_path)
        
        # Prepare features and target
        self.X, self.y = self.data_processor.prepare_features_and_target(self.df)
        
        print(f"✓ Data loaded: {self.df.shape[0]} rows, {self.X.shape[1]} features")
        return self.df
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        if self.X is None:
            raise ValueError("Data not loaded yet")
        
        # Preprocess data
        X_processed = self.data_processor.fit_and_transform(self.X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
        
        print(f"✓ Data split: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train all models"""
        if self.X_train is None:
            raise ValueError("Data not split yet")
        
        self.model_trainer.train_all_models(self.X_train, self.y_train)
        print("✓ All models trained")
        return self.model_trainer.models
    
    def evaluate_models(self):
        """Evaluate all models"""
        if self.X_test is None:
            raise ValueError("Data not split yet")
        
        results = self.model_trainer.evaluate_models(self.X_test, self.y_test)
        print("✓ Models evaluated")
        return results
    
    def get_best_model_info(self):
        """Get best model name and performance"""
        best_name, best_model = self.model_trainer.get_best_model()
        metrics = self.model_trainer.results[best_name]
        
        print(f"\n🏆 Best Model: {best_name}")
        print(f"   ROC-AUC: {metrics['ROC-AUC']:.4f}")
        print(f"   Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   Precision: {metrics['Precision']:.4f}")
        print(f"   Recall: {metrics['Recall']:.4f}")
        
        return best_name, metrics
    
    def save_artifacts(self, model_path, preprocessor_path):
        """Save model and preprocessor"""
        self.model_trainer.save_model(model_path)
        self.data_processor.save_preprocessor(preprocessor_path)
        
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Preprocessor saved: {preprocessor_path}")
    
    def full_pipeline(self, data_path, model_save_path, preprocessor_save_path):
        """Run complete pipeline: load -> split -> train -> evaluate -> save"""
        
        print("=" * 50)
        print("🫡 Starting Churn Prediction Pipeline")
        print("=" * 50)
        
        # Load and prepare
        self.load_and_prepare_data(data_path)
        
        # Split
        self.split_data()
        
        # Train
        self.train_models()
        
        # Evaluate
        self.evaluate_models()
        
        # Get best model
        self.get_best_model_info()
        
        # Save
        self.save_artifacts(model_save_path, preprocessor_save_path)
        
        # Get results
        results_df = self.model_trainer.get_results_dataframe()
        
        print("\n" + "=" * 50)
        print("📊 Model Comparison Results")
        print("=" * 50)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def predict(self, X_new):
        """Make predictions on new data"""
        X_processed = self.data_processor.transform(X_new)
        return self.model_trainer.predict(X_processed)
    
    def predict_proba(self, X_new):
        """Get prediction probabilities"""
        X_processed = self.data_processor.transform(X_new)
        return self.model_trainer.predict_proba(X_processed)
