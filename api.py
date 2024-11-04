"""
{
  "gender": 2,
  "cholesterol": 1,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1,
  "age_years": 50,
  "bmi": 21.9666,
  "bp_category": 4,
  "mean_arterial_pressure": 90,
  "pulse_pressure": 30
}
"""

import joblib 
import json
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

model = joblib.load('Cardio_predict.pkl')

class model_input(BaseModel):
    gender: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int
    age_years: int
    bmi: float
    bp_category: int
    mean_arterial_pressure: float
    pulse_pressure: int
    

@app.post('/prediction')
def prediction(input_format: model_input):
    
    input_data = input_format.json()
    input_directory= json.loads(input_data)
    
    gender = input_directory['gender']
    cholesterol = input_directory['cholesterol']
    gluc = input_directory['gluc']
    smoke = input_directory['smoke']
    alco = input_directory['alco']
    active = input_directory['active']
    age_years = input_directory['age_years']
    bmi = input_directory['bmi']
    bp_category = input_directory['bp_category']
    mean_arterial_pressure = input_directory['mean_arterial_pressure']
    pulse_pressure = input_directory['pulse_pressure']
    
    input_list = [gender,cholesterol, gluc, smoke, alco, active, age_years, bmi, bp_category, mean_arterial_pressure, pulse_pressure]
    prediction = model.predict([input_list])
    
    if prediction[0]==0:
        return "False"
    else:
        return "True"
