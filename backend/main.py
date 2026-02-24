from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Annotated
import joblib
from contextlib import asynccontextmanager
import pandas as pd
from pathlib import Path
import shap

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "experiments" / "my_pipeline.joblib"

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading pipeline at startup...")
    try:
        pipeline = joblib.load(MODEL_PATH)
        app_state["pipeline"] = pipeline

        model = pipeline.named_steps["model"]
        app_state["explainer"] = shap.TreeExplainer(model)

        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        app_state["pipeline"] = None
        app_state["explainer"] = None
    yield
    print("Cleaning up resources...")

app = FastAPI(lifespan=lifespan)

# pydantic model for data validation
class UserInput(BaseModel):

    satisfaction_level: Annotated[float, Field(..., ge=0, le=1, description="Satisfaction levels of employee")]
    last_evaluation: Annotated[float, Field(..., ge=0, le=1, description="Evaluation levels of employee")]
    number_project: Annotated[int, Field(..., gt=0, description="Number of projects employee is assigned to")]
    average_montly_hours: Annotated[int, Field(..., gt=10, description="Monthly average working hours of employee")]
    time_spend_company: Annotated[int, Field(..., gt=0, description="Number of years employee worked")]
    Work_accident: Annotated[Literal[0,1], Field(...,description="Did any work accidents take place")]
    promotion_last_5years: Annotated[Literal[0,1], Field(..., description="Did the employee got promoted in last 5 years")]
    Departments: Annotated[Literal['sales','accounting','hr','technical','support','management','IT','product_mng','marketing','RandD'], Field(..., description="Department of employee")]
    salary: Annotated[Literal['high','medium','low'], Field(..., description="Salary of employee")]


@app.post('/predict')
def predict_attrition(data: UserInput):

    pipeline = app_state.get("pipeline")

    input_df = pd.DataFrame([{
        'satisfaction_level': data.satisfaction_level,
        'last_evaluation': data.last_evaluation,
        'number_project': data.number_project,
        'average_montly_hours': data.average_montly_hours,
        'time_spend_company': data.time_spend_company,
        'Work_accident': data.Work_accident,
        'promotion_last_5years': data.promotion_last_5years,
        'Departments ': data.Departments,
        'salary': data.salary,
    }])

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not available")
    pred = pipeline.predict(input_df)[0]
    print(pred)
    prediction_probability = pipeline.predict_proba(input_df)[0]
    print(prediction_probability)
    explainer = app_state.get("explainer")

    # Transform input
    processed_input = pipeline.named_steps["preprocess"].transform(input_df)

    shap_values = explainer(processed_input)

    # For binary classification → class 1
    shap_class1 = shap_values.values[0, :, 1]
    # print(type(shap_class1))
    # print(shap_class1.shape)

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

    feature_impacts = list(zip(feature_names, shap_class1))

    # Sort by absolute contribution
    top_features = sorted(
        feature_impacts,
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]
    top_factors = []
    for feature, value in top_features:
        clean_name = feature.split("__")[-1]
        effect = "increases_leave_risk" if value > 0 else "decreases_leave_risk"
        top_factors.append({
            "feature": clean_name,
            "impact": effect,
            "contribution_strength": round(float(value), 3)
        })

    prediction = "Leave" if pred == 1 else "Stay"
    if prediction_probability[1]<0.4:
        risk_level = 'Low'
    elif prediction_probability[1]<0.7:
        risk_level = 'Medium'
    else:
        risk_level = 'High'  
    return JSONResponse(status_code=200, content={'prediction': prediction,
    'attrition_probability': round(prediction_probability[0],2),
    'leave_probability': round(prediction_probability[1],2),
    'risk_level': risk_level,
    'top_factors': top_factors})