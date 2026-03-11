# Remote Employee Burnout Predictor

An end-to-end machine learning application that predicts the burnout risk of remote employees using **XGBoost**. The system classifies employees into **Low**, **Medium**, or **High** risk categories based on daily work and wellbeing metrics, and serves predictions through a modern React web UI backed by FastAPI.

---

## Features

- 🤖 **XGBoost classifier** trained on 14 dynamically engineered features
- ⚙️ **Auto-computed features** — fatigue score, effort-recovery ratio, 7-day rolling averages are all calculated server-side from minimal user input
- 🖥️ **React UI** — 2-step wizard form with animated risk ring gauge, probability bars, and contextual advice
- 📡 **FastAPI backend** with full OpenAPI docs at `/docs`
- 📊 **Evaluation report** auto-exported to `artifacts/evaluation_report.txt` after training

---

## Project Structure

```
├── app/
│   ├── app.py                  # FastAPI entry point + /predict endpoint
│   └── templates/index.html    # React 18 (CDN) single-page UI
├── src/
│   ├── datas/
│   │   ├── fatigue_scorer.py   # Reverse-engineered fatigue score formula
│   │   └── feature_engineering.py  # Rolling average feature computation
│   ├── model/
│   │   ├── train.py            # Trainer class (XGBoost training loop)
│   │   └── evaluation.py       # ModelEvaluator (accuracy, conf. matrix, report)
│   └── pipeline/
│       └── model_training.py   # End-to-end orchestration script
├── artifacts/
│   ├── xgboost_model.pkl       # Trained model (auto-saved after training)
│   ├── scaling/scaler.pkl      # StandardScaler (auto-saved after training)
│   └── evaluation_report.txt   # Accuracy + classification report
└── dataset/
    └── processed/burnout_processed.csv
```

---

## How to Run

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or with standard pip
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn pydantic pandas xgboost scikit-learn jinja2
```

### 2. Train the Model

Run the full end-to-end pipeline (data loading → scaling → training → evaluation → artifact save):

```bash
uv run python src/pipeline/model_training.py
```

This will:
1. Load `dataset/processed/burnout_processed.csv`
2. Split into train / validation sets
3. Fit a `StandardScaler` and scale features
4. Compute dynamic class weights to handle class imbalance
5. Train the `BurnoutXGBoostModel`
6. Save model → `artifacts/xgboost_model.pkl`
7. Save scaler → `artifacts/scaling/scaler.pkl`
8. Export evaluation report → `artifacts/evaluation_report.txt`

### 3. Start the Web App

```bash
uv run uvicorn app.app:app --reload
```

Then open:
- **Web UI** → [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Docs (Swagger)** → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API

### `POST /predict`

Accepts 10 user-provided inputs, computes all derived features server-side, and returns the risk prediction.

**Request body:**
```json
{
  "day_type": "Weekday",
  "work_hours": 9.0,
  "screen_time_hours": 7.0,
  "meetings_count": 5,
  "breaks_taken": 2,
  "after_hours_work": 2.0,
  "app_switches": 50,
  "sleep_hours": 6.0,
  "task_completion": 75.0,
  "isolation_index": 7
}
```

**Response:**
```json
{
  "burnout_risk_level": "High",
  "burnout_risk_class": 2,
  "fatigue_score": 8.52,
  "probabilities": {
    "Low": 0.03,
    "Medium": 0.09,
    "High": 0.88
  }
}
```

---

## Changelog

### v1.3 — React UI (Latest)
- Rewrote `app/templates/index.html` as a full **React 18 (CDN)** single-page app
- **2-step wizard form**: Step 1 (Work Metrics) → Step 2 (Wellbeing Metrics)
- Animated **SVG risk ring gauge** with adaptive glow per risk level
- **3-bar probability breakdown** with animated fills (Low / Medium / High)
- Contextual **advice banner** per risk level
- Fully responsive — single-column layout on mobile

### v1.2 — Evaluation Export
- `ModelEvaluator` in `src/model/evaluation.py` saves accuracy, classification report, and confusion matrix to `artifacts/evaluation_report.txt`

### v1.1 — Feature Engineering & Inference Automation
- `fatigue_scorer.py`: Reverse-engineered linear formula estimates `fatigue_score` dynamically
- `feature_engineering.py`: Derives rolling averages (`work_hours_7d_avg`, `fatigue_3d_sum`, `meetings_7d_avg`)
- Users only need to provide 10 plain inputs; all 14 model features are computed server-side

### v1.0 — Training Pipeline
- `Trainer` class in `src/model/train.py` with class-weight-balanced XGBoost training
- Full pipeline orchestration in `src/pipeline/model_training.py`
