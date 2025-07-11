import os
import joblib
from pydantic import BaseModel, Field

# Define request schema matching only the trained features
class Features(BaseModel):
    TV: float = Field(..., example=150.0)
    Radio: float = Field(..., example=30.0)
    Newspaper: float = Field(..., example=20.0)

# Load trained model
this_dir = os.path.dirname(__file__)
model_path = os.path.join(this_dir, '..', 'models', 'sales_model.joblib')
model = joblib.load(model_path)

# Prediction function
def predict_sales(features: Features) -> float:
    data = [[features.TV, features.Radio, features.Newspaper]]
    return float(model.predict(data)[0])