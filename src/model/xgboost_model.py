# src/model/xgboost.py

from __future__ import annotations
from xgboost import XGBClassifier


class BurnoutXGBoostModel:
    """
    XGBoost model builder for multi-class Burnout Risk classification.
    Classes:
        0 → Low
        1 → Medium
        2 → High
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def build(self) -> XGBClassifier:
        """
        Builds and returns configured XGBClassifier.
        """

        return XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,

            # 🔥 Multi-class configuration
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",

            use_label_encoder=False
        )
        
        
        
        
if __name__ == "__main__":
    print("Testing BurnoutXGBoostModel...\n")

    model_builder = BurnoutXGBoostModel()
    model = model_builder.build()

    print("Model built successfully ✅")
    print(model)