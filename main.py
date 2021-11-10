# Put the code for your API here.
from src.train_data import split_and_train
from src.prepare_data import process_data
from src import model
import pandas as pd
import numpy as np
import uvicorn
import yaml
import os

from fastapi import FastAPI
from typing import Union, List, Optional
from pydantic import BaseModel, Field

import json
from fastapi.testclient import TestClient

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc config core.hardlink_lock true")
    if os.system("dvc pull -r s3remote") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

def test_(data_in: dict, expected_output: str):
    with TestClient(app) as client:
        r = client.post("/predict", data=json.dumps(data_in), 
                        headers={"Content-Type": "application/json"})
        assert r.json()["predictions"] == expected_output, "Unexpected output of the model"

class DataIn(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example = "State-gov")
    fnlgt: int = Field(..., example = 77516)
    education: str = Field(..., example = "Bachelors")
    education_num: int = Field(..., alias = "education-num", example = 13)
    marital_status: str = Field(..., alias = "marital-status", example = "Never-married")
    occupation: str = Field(..., example = "Adm-clerical")
    relationship: str = Field(..., example = "Not-in-family")
    race: str = Field(..., example = "White")
    sex: str = Field(..., example = "Male")
    capital_gain: int = Field(..., alias = "capital-gain", example = 2174)
    capital_loss: int = Field(..., alias = "capital-loss", example = 0)
    hours_per_week: int = Field(..., alias = "hours-per-week", example = 40)
    native_country: str = Field(..., alias = "native-country", example = "United-States")
    salary: Optional[str]


# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def dummy_get():
    return "Hola"

@app.post("/predict")
async def get_preds(input_data: DataIn):

    df_input = pd.DataFrame.from_dict([input_data.dict(by_alias=True)])

    parameters = yaml.safe_load(open('params.yaml'))
    cat_features = parameters['data']['categorical_features']

    onehot_enc = pd.read_pickle("data/encoders/onehotenc.pkl")
    
    X_in, _, _, _ = process_data(
                                df_input,
                                categorical_features=cat_features,
                                label="salary",
                                training=False,
                                encoder=onehot_enc,
                                lb=None
                                )

    model_rfc = pd.read_pickle("model/model/rfc.pkl")

    return {"predictions": f"{model.inference(model_rfc, X_in)[0]}"}

if __name__ == '__main__':

    test_data_neg = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    test_data_pos = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 193524,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
    }

    test_(test_data_neg, "0")
    test_(test_data_pos, "1")