from src.predict import LogiRisk_Predictor
from app.schemas import ShipmentInput
from fastapi import FastAPI

app = FastAPI(title='LogiRisk-ML API')
logirisk_predictor = LogiRisk_Predictor()

@app.post('/predict')
async def predict_risk(data: ShipmentInput):
    raw_data = [data.model_dump()]
    risk_score = logirisk_predictor.predict(raw_data)

    return {
        'risk_score': round(risk_score, 4),
        'late_risk': risk_score > 0.5,
        "status": "success"
    }