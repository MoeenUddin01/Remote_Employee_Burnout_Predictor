# Remote Employee Burnout Predictor

A machine learning classification project to predict the burnout risk of remote employees. 
The system trains an XGBoost model to classify employees into Low, Medium, or High risk categories.

## Recent Updates

1. **Training Pipeline Implementation**:
   - Developed the `Trainer` class in `src/model/train.py`.
   - Setup data loading, scaling, class weight computation for imbalanced nodes, and XGBoost training.
   - Saves final model to `artifacts/xgboost_model.pkl` and data scaler to `artifacts/scaling/scaler.pkl`.

2. **Evaluation Report Export**:
   - Implemented `ModelEvaluator` in `src/model/evaluation.py`.
   - The training script now automatically evaluates the model on the validation set.
   - Saves a compiled text report containing accuracy, the classification report, and the confusion matrix to **`artifacts/evaluation_report.txt`**.

3. **Feature Engineering & Inference Input Automation**:
   - The original dataset included mathematically derived columns that shouldn't be required from the end user.
   - Added `src/datas/fatigue_scorer.py`: Reverse-engineered linear regression formula to dynamically estimate `fatigue_score`.
   - Added `src/datas/feature_engineering.py`: Computes historical rolling averages (`work_hours_7d_avg`, `fatigue_3d_sum`, `meetings_7d_avg`) from baseline inputs.
   - **Required User Inputs** for inference: `day_type`, `work_hours`, `screen_time_hours`, `meetings_count`, `breaks_taken`, `after_hours_work`, `app_switches`, `sleep_hours`, `isolation_index`.
