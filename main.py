# Put the code for your API here.
from src.train_data import split_and_train
from src import model
import pandas as pd
import numpy as np
#split_and_train()

#df = pd.read_csv("data/census.csv")
#print(df.columns.values)
xy_test = pd.read_pickle("data/prepared/test.pkl")
model_ = pd.read_pickle("model/model/rfc.pkl")
encoder_ = pd.read_pickle("data/encoders/onehotenc.pkl")

X_test = xy_test[:,:-1]
y_test = xy_test[:,-1]

print(model.performance_data_slice(encoder_, model_, X_test,y_test, 'education'))