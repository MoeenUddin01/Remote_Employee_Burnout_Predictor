from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import sys

# Add project root to path so we can import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datas.fatigue_scorer import calculate_fatigue_score
from src.datas.feature_engineering import calculate_derived_features

app = FastAPI(
    title="Burnout Prediction API",
    description="Predicts the risk of employee burnout using XGBoost.",
    version="1.0.0"
)

# Setup Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Load the model and scaler on startup
MODEL_PATH = os.path.join(project_root, "artifacts", "xgboost_model.pkl")
SCALER_PATH = os.path.join(project_root, "artifacts", "scaling", "scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    print(f"Error loading artifacts: {e}")
    # We don't exit here so the app can still boot and show the error in the endpoint if needed.

# ==========================================
# 1. Pydantic Models for Request Validation
# ==========================================
class BurnoutRequest(BaseModel):
    # Core user inputs
    day_type: str = Field(..., description="Either 'Weekday' or 'Weekend'")
    work_hours: float = Field(..., ge=0, description="Hours worked today")
    screen_time_hours: float = Field(..., ge=0, description="Hours spent looking at screens")
    meetings_count: int = Field(..., ge=0, description="Number of meetings today")
    breaks_taken: int = Field(..., ge=0, description="Number of breaks taken today")
    after_hours_work: float = Field(..., ge=0, description="Hours worked outside standard shifts")
    app_switches: int = Field(..., ge=0, description="Number of context switches / app switches")
    sleep_hours: float = Field(..., ge=0, description="Hours slept last night")
    task_completion: float = Field(..., ge=0, le=100, description="Percentage of tasks completed")
    isolation_index: int = Field(..., ge=1, le=10, description="Self-reported isolation score (1-10)")

class BurnoutResponse(BaseModel):
    burnout_risk_level: str
    burnout_risk_class: int
    fatigue_score: float
    probabilities: dict

# ==========================================
# 2. Main Endpoints
# ==========================================

@app.get("/")
def serve_ui(request: Request):
    """Serves the frontend user interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=BurnoutResponse)
def predict_burnout(request: BurnoutRequest):
    """
    Predicts burnout risk based on today's inputs. 
    It automatically calculates fatigue_score and other derived features in the backend.
    """
    if "model" not in globals() or "scaler" not in globals():
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded.")

    # 1. Calculate Fatigue Score dynamically
    fatigue_score = calculate_fatigue_score(
        work_hours=request.work_hours,
        screen_time_hours=request.screen_time_hours,
        meetings_count=request.meetings_count,
        breaks_taken=request.breaks_taken,
        after_hours_work=request.after_hours_work,
        app_switches=request.app_switches,
        sleep_hours=request.sleep_hours,
        isolation_index=request.isolation_index
    )

    # 2. Calculate Derived Features dynamically 
    # (Since this is a simple stateless API, we pass empty history for testing. 
    # In a real app, you'd fetch user history from a DB here)
    derived = calculate_derived_features(
        work_hours=request.work_hours,
        fatigue_score=fatigue_score,
        day_type=request.day_type,
        meetings_count=request.meetings_count,
        user_historical_data=None # Defaults to using today's data as the "average"
    )

    # Calculate remaining required mathematical features from the dataset
    effort_recovery_ratio = round((request.work_hours + request.after_hours_work) / max(request.sleep_hours, 1), 2)
    context_switch_load = round(request.app_switches / max(request.work_hours, 1), 2)

    # 3. Assemble the final feature dictionary in the exact order the model expects
    features_dict = {
        "day_type": 1 if request.day_type.lower() == "weekend" else 0, # Assuming standard encoding
        "work_hours": request.work_hours,
        "screen_time_hours": request.screen_time_hours,
        "meetings_count": request.meetings_count,
        "breaks_taken": request.breaks_taken,
        "after_hours_work": request.after_hours_work,
        "app_switches": request.app_switches,
        "sleep_hours": request.sleep_hours,
        "task_completion": request.task_completion,
        "isolation_index": request.isolation_index,
        "fatigue_score": fatigue_score,
        "work_hours_7d_avg": derived["work_hours_7d_avg"],
        "fatigue_3d_sum": derived["fatigue_3d_sum"],
        "is_weekend": int(derived["is_weekend"]),
        "meetings_7d_avg": derived["meetings_7d_avg"],
        "effort_recovery_ratio": effort_recovery_ratio,
        "context_switch_load": context_switch_load
    }

    # Convert to DataFrame for scaling
    df_features = pd.DataFrame([features_dict])

    # 4. Scale features
    # Note: Our DataScaler returns a pd.DataFrame with the same columns
    # We extract the underlying np array to feed to XGBoost to avoid feature name mismatch warnings
    try:
        scaled_features = scaler.transform(df_features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error scaling features: {str(e)}")

    # 5. Predict
    prediction_class = int(model.predict(scaled_features)[0])
    probabilities = model.predict_proba(scaled_features)[0]

    # Map class to readable label
    risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    risk_level = risk_labels.get(prediction_class, "Unknown")

    return BurnoutResponse(
        burnout_risk_level=risk_level,
        burnout_risk_class=prediction_class,
        fatigue_score=fatigue_score,
        probabilities={
            "Low": float(probabilities[0]),
            "Medium": float(probabilities[1]),
            "High": float(probabilities[2])
        }
    )

if __name__ == "__main__":
    import uvicorn
    # Make sure to run the server from the project root using:
    # uvicorn app.app:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
