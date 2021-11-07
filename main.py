# Put the code for your API here.
from src.train_data import split_and_train
import pandas as pd
#split_and_train()

df = pd.read_csv("data/census.csv")
print(df.columns.values)
