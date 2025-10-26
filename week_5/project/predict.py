import pickle
import uvicorn

from fastapi import FastAPI
from typing import Dict,Any
from pydantic import BaseModel,Field

class Client(BaseModel):
    lead_source: str = Field(..., description="Source of the lead, e.g., 'organic_search'")
    number_of_courses_viewed: int = Field(..., ge=0, description="Number of courses the client has viewed")
    annual_income: float = Field(..., ge=0, description="Annual income of the client in USD")

app = FastAPI(title="subscription-prediction")

@app.get("/")
def home():
    return "homepage"

@app.get("/ping")
def ping():
    return "PONG"


def predict_single(client):
    with open('pipeline_v1.bin', 'rb') as f_in:
        pipeline = pickle.load(f_in)
    
    result = pipeline.predict_proba(client.dict())[0,1]
    return float(result)


@app.post("/predict")
def predict(client: Client):
    sub_prob = predict_single(client)
    
    return {
        "subscription_probability": sub_prob,
        "subscription": bool(sub_prob >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)