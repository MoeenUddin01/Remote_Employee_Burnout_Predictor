from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator for classification models.
    """

    @staticmethod
    def evaluate(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, Any]:
        """
        Evaluates model predictions and logs accuracy, classification report, and confusion matrix.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            A dictionary containing the calculated metrics.

        Raises:
            ValueError: If y_true and y_pred have different lengths or are empty.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        if len(y_true) == 0:
            raise ValueError("y_true and y_pred cannot be empty.")

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": conf_matrix,
        }
