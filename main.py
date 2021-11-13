import os
import uvicorn
from src.train_data import split_and_train
from src.model import performance_data_slice, compute_model_metrics, inference
from src import fastAPI_main
import pandas as pd

def extract_metrics_slice():

    test_data = pd.read_pickle("data/prepared/test.pkl")
    encoder = pd.read_pickle("data/encoders/onehotenc.pkl")
    model_f = pd.read_pickle("model/model/rfc.pkl")

    X = test_data[:,:-1]
    y = test_data[:,-1]

    performance_data_slice(encoder, model_f, X, y, 'education')
    return None

def get_global_metrics():

    test_data = pd.read_pickle("data/prepared/test.pkl")
    model_f = pd.read_pickle("model/model/rfc.pkl")

    X = test_data[:,:-1]
    y = test_data[:,-1]

    y_preds = inference(model_f, X)

    return compute_model_metrics(y, y_preds)

if __name__ == "__main__":
    uvicorn.run("src.fastAPI_main:app")
    #split_and_train()
    #extract_metrics_slice()
    #print(get_global_metrics())