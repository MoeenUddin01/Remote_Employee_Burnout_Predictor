# Remote Employee Burnout Predictor

A machine learning classification project to predict the burnout risk of remote employees.
The system trains an XGBoost model to classify employees into Low, Medium, or High risk categories.

## Recent Updates

1. **Training Pipeline Implementation**:
   - Developed the `Trainer` class in `src/model/train.py`.
   - Setup data loading, scaling, class weight computation for imbalanced nodes, and XGBoost training.
   - A unifying pipeline script `src/pipeline/model_training.py` orchestrates the entire flow.
   - Saves final model to `artifacts/xgboost_model.pkl` and data scaler to `artifacts/scaling/scaler.pkl`.

2. **Evaluation Report Export**:
   - Implemented `ModelEvaluator` in `src/model/evaluation.py`.
   - The training script automatically evaluates the model on a validation set.
   - Saves a compiled text report containing accuracy, the classification report, and the confusion matrix to **`artifacts/evaluation_report.txt`**.

3. **Feature Engineering & Inference Input Automation**:
   - The original dataset included mathematically derived columns that shouldn't be required from the end user.
   - Added `src/datas/fatigue_scorer.py`: Reverse-engineered linear regression formula to dynamically estimate `fatigue_score`.
   - Added `src/datas/feature_engineering.py`: Computes historical rolling averages (`work_hours_7d_avg`, `fatigue_3d_sum`, `meetings_7d_avg`) from baseline inputs.
   - **Required User Inputs** for inference: `day_type`, `work_hours`, `screen_time_hours`, `meetings_count`, `breaks_taken`, `after_hours_work`, `app_switches`, `sleep_hours`, `isolation_index`.
### 4. Interactive Web UI (FastAPI)
- Built a modern, dark-themed responsive web UI in `app/templates/index.html`.
- Implemented `app/app.py` serving the front-end directly via standard Jinja2 templating.
- The backend automatically executes the dynamically engineered math features and dynamically scales data so the user only has to input standard data.

## How to Run

### Installation
Ensure you have `uv` installed, or use standard `pip` with the virtual environment:
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install fastapi uvicorn pydantic pandas xgboost scikit-learn jinja2
```

### Model Training 
To run the full end-to-end model training pipeline:

```bash
uv run python src/pipeline/model_training.py
```

This script will:
1. Load the processed dataset (`dataset/processed/burnout_processed.csv`).
2. Split the data into training and validation sets.
3. Fit a `StandardScaler` and scale the features.
4. Calculate dynamic class weights to balance the dataset.
5. Train the `BurnoutXGBoostModel`.
6. Evaluate the model and output an accuracy report to `artifacts/evaluation_report.txt`.
7. Save the model to `artifacts/xgboost_model.pkl` and the scaler to `artifacts/scaling/scaler.pkl`.

### Running the Web UI & API
The project includes a fully functional FastAPI web interface. To start the application, run:

```bash
uv run uvicorn app.app:app --reload
```

Once running, navigate to:
- **Web UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Documentation (Swagger UX)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
