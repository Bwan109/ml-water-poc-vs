import pickle
import numpy as np
from fastapi import FastAPI
from typing import List
import json

from pydantic import BaseModel
#aluminium - dangerous if greater than 2.8
#arsenic - dangerous if greater than 0.01
# cadmium - dangerous if greater than 0.005
# chloramine - dangerous if greater than 4
# chromium - dangerous if greater than 0.1
class request_body(BaseModel):
    aluminium : float
    arsenic : float
    cadmium : float
    chloramine : float
    chromium : float

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to API"}

@app.post("/predict")
def predict(data: request_body):
    test_data = [[
            data.aluminium,
            data.arsenic,
            data.cadmium,
            data.chloramine,
            data.chromium
    ]]
    with open("./app/modelknn.pkl", "rb") as f:
        model = pickle.load(f)

#test_data = [[2.1,	0.001,	0.001,	3, 0.01]]
    #test_data = np.array(test_data).reshape(1, -1)

    prediction = model.predict(test_data)
    if (prediction == 0):
        return {"prediction": "Not Safe"}

    else:
        return {"prediction": "Safe"}



