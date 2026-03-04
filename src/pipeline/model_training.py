"""
Pipeline script to orchestrate model training.
"""
import sys
import os
import logging

# 1. Add project root to Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import existing modules
from src.datas.loader import DataLoader
from src.model.train import Trainer

def main():
    # 5. Use logging (INFO level) for pipeline steps.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting Model Training Pipeline...")

    # 2. Use DataLoader to load dataset (Verify it exists / works first)
    dataset_path = "dataset/processed/burnout_processed.csv"
    loader = DataLoader(verbose=False)
    
    logger.info(f"Verifying dataset loading from {dataset_path} with DataLoader...")
    df = loader.load_csv(dataset_path)
    logger.info(f"Successfully loaded dataset with shape: {df.shape}")

    # 3. If feature engineering is required, apply it.
    # Note: Skipped as per user instructions (already done in notebook).

    # 4. Call Trainer class to orchestrate splitting, scaling, training, evaluation, saving
    logger.info("Initializing Trainer class...")
    trainer = Trainer(
        dataset_path=dataset_path,
        target_column="burnout_risk",
        artifacts_dir="artifacts/",
        scaler_save_dir="artifacts/scaling/",
        model_filename="xgboost_model.pkl",
        scaler_filename="scaler.pkl"
    )
    
    # Run all the steps orchestrated by the Trainer (splitting, scaling, weighting, training, evaluation, saving)
    trainer.run()

    # 6. Print final message: "Training pipeline completed successfully."
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
