"""
Training Script - Run the complete ML pipeline
Usage: python mvc/run_pipeline.py
"""

import sys
from pathlib import Path

# Add mvc to path
sys.path.insert(0, str(Path(__file__).parent))

from controllers.pipeline import ChurnPredictionController

def main():
    """Run the complete pipeline"""
    
    # Initialize paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    model_save_path = project_root / 'models' / 'final_model.pkl'
    preprocessor_save_path = project_root / 'models' / 'preprocessor.pkl'
    results_save_path = project_root / 'outputs' / 'model_comparison.csv'
    
    # Create output directory
    (project_root / 'models').mkdir(parents=True, exist_ok=True)
    (project_root / 'outputs').mkdir(parents=True, exist_ok=True)
    
    # Initialize controller
    controller = ChurnPredictionController()
    
    # Run pipeline
    results_df = controller.full_pipeline(
        data_path=str(data_path),
        model_save_path=str(model_save_path),
        preprocessor_save_path=str(preprocessor_save_path)
    )
    
    # Save results
    results_df.to_csv(results_save_path, index=False)
    print(f"\n✓ Results saved: {results_save_path}")
    
    print("\n" + "=" * 50)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print(f"\nNext steps:")
    print(f"1. Run Streamlit app: python -m streamlit run streamlit_app/app.py")
    print(f"2. Run FastAPI server: uvicorn api.main:app --reload")

if __name__ == "__main__":
    main()
