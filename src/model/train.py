from __future__ import annotations

import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.datas.loader import DataLoader
from src.datas.scaling import DataScaler
from src.model.evaluation import ModelEvaluator
from src.model.xgboost_model import BurnoutXGBoostModel

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates the training pipeline for the Burnout Risk classification model.
    """

    def __init__(
        self,
        dataset_path: str = "dataset/processed/burnout_processed.csv",
        target_column: str = "burnout_label",
        artifacts_dir: str = "artifacts/",
        scaler_save_dir: str = "artifacts/scaling/",
        model_filename: str = "xgboost_model.pkl",
        scaler_filename: str = "scaler.pkl",
    ) -> None:
        """
        Initializes the Trainer with paths and configuration.

        Args:
            dataset_path: Path to the processed dataset CSV.
            target_column: The name of the target column to predict.
            artifacts_dir: Directory to save the trained model.
            scaler_save_dir: Directory to save the fitted scaler.
            model_filename: Name of the model pickle file.
            scaler_filename: Name of the scaler pickle file.
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.artifacts_dir = artifacts_dir
        self.scaler_save_dir = scaler_save_dir
        self.model_filename = model_filename
        self.scaler_filename = scaler_filename

        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.scaler_save_dir, exist_ok=True)

    def _compute_class_weights(self, y: pd.Series) -> np.ndarray:
        """
        Computes sample weights to handle class imbalance.

        Weight formula: total_samples / (num_classes * samples_in_class)

        Args:
            y: The target labels.

        Returns:
            An array of sample weights corresponding to each instance in y.

        Raises:
            ValueError: If target labels are empty.
        """
        if len(y) == 0:
            raise ValueError("Target labels cannot be empty.")

        class_counts = y.value_counts().to_dict()
        total_samples = len(y)
        num_classes = len(class_counts)

        weights_dict: dict[Any, float] = {
            cls: total_samples / (num_classes * count)
            for cls, count in class_counts.items()
        }

        sample_weights = y.map(weights_dict).to_numpy()
        return sample_weights

    def run(self) -> None:
        """
        Runs the full training pipeline using XGBoost and standard ML tools.

        Loads data, splits it, scales features, computes class weights, trains
        the model, evaluates it, and saves the resulting artifacts.
        """
        logger.info("Initializing data loader...")
        # Since loader has verbose=True by default, we'll keep it as default
        # or switch to False so our INFO logs take precedence. Let's use False to avoid duplicate prints.
        data_loader = DataLoader(verbose=False)

        logger.info(f"Loading dataset from {self.dataset_path} ...")
        df = data_loader.load_csv(self.dataset_path)
        logger.info("Dataset loaded.")

        logger.info(f"Splitting data with target column '{self.target_column}'...")
        X_train, X_val, y_train, y_val = data_loader.split_data(
            df, target_column=self.target_column
        )
        logger.info("Data split completed.")

        logger.info("Initializing and fitting data scaler...")
        scaler = DataScaler(method="standard", save_path=self.scaler_save_dir)
        scaler.fit_scaler(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        logger.info("Features scaled successfully.")

        logger.info("Computing sample weights for class imbalance...")
        sample_weight = self._compute_class_weights(y_train)
        logger.info("Class weights computed.")

        logger.info("Building XGBoost model...")
        model_builder = BurnoutXGBoostModel()
        model = model_builder.build()

        logger.info("Training started...")
        model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
        logger.info("Training completed.")

        logger.info("Evaluating model on validation set...")
        y_pred = model.predict(X_val_scaled)
        eval_metrics = ModelEvaluator.evaluate(y_true=y_val, y_pred=y_pred)

        report_save_path = os.path.join(self.artifacts_dir, "evaluation_report.txt")
        with open(report_save_path, "w") as f:
            f.write(f"Accuracy: {eval_metrics['accuracy']:.4f}\n\n")
            f.write(f"Classification Report:\n{eval_metrics['classification_report']}\n\n")
            f.write(f"Confusion Matrix:\n{eval_metrics['confusion_matrix']}\n")
        logger.info(f"Evaluation report saved to {report_save_path}")

        model_save_path = os.path.join(self.artifacts_dir, self.model_filename)
        joblib.dump(model, model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        scaler.save_scaler(filename=self.scaler_filename)
        # scaler.save_scaler() prints to stdout, our log adds specific INFO formatting
        scaler_full_path = os.path.join(self.scaler_save_dir, self.scaler_filename)
        logger.info(f"Scaler saved to {scaler_full_path}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
