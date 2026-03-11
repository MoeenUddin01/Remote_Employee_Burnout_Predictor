<!-- # Skill: FastAPI + React ML Prediction Service

## Purpose
Build a prediction API for the Remote Employee Burnout Predictor and connect it with a React frontend.

---

## Backend Technology
Use FastAPI for the backend API.

Rules:
- All endpoints must be written using FastAPI.
- Use Pydantic schemas for request validation.
- Load ML models from the artifacts directory.
- The model must be loaded once during startup.

Example structure:

app/
    main.py
    predict.py
    model_loader.py
    schemas.py

---

## Model Loading

The model and scaler must be loaded using a separate module.

model_loader.py should:
- Load model.pkl
- Load scaler.pkl
- Expose a function `get_model()`

---

## Prediction Endpoint

Create endpoint:

POST /predict

Input:
JSON with employee features.

Example:
{
  "age": 32,
  "work_hours": 9,
  "fatigue_score": 6.5,
  "isolation_index": 3.2
}

Output:
{
  "burnout_prediction": 1,
  "burnout_level": "High"
}

---

## Frontend Technology

Frontend must use React.

Rules:
- Create a form for employee data.
- Send request using fetch or axios.
- Display burnout prediction.

Example API call:

fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
})

---

## Integration Rules

- React must never access model files directly.
- All predictions must go through the FastAPI backend.
- Backend must return clean JSON responses. -->