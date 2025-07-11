from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.model import Features, predict_sales

app = FastAPI(title="Sales Forecasting API")

# 1) Add this CORS block **before** you define any routes:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:63342"],  # your front-end origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome..."}

@app.post("/predict")
def predict(features: Features):
    try:
        return {"predicted_sales": predict_sales(features)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
