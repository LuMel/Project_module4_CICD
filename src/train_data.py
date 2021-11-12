# Script to train machine learning model.
from sklearn import model_selection
if __name__ == "src.train_data":
    from src.prepare_data import process_data, remove_spaces
    from src import model
elif __name__ == "Project_module4_CICD.src.train_data":
    from ..src.prepare_data import process_data, remove_spaces
    from ..src import model
else:
    print(__name__)
    raise ModuleNotFoundError
#from src.prepare_data import process_data, remove_spaces
#from src import model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import yaml
import pickle
import os

def split_and_train():
    # Add code to load in the data.
    parameters = yaml.safe_load(open('params.yaml'))

    input_path = parameters['data']['input_path']
    data_raw = pd.read_csv(input_path + "census.csv")

    # pre cleaning
    data = remove_spaces(data_raw, parameters['data']['categorical_features'])

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, 
                                    test_size=float(parameters['data']['train_test_split']))

    cat_features = list(parameters['data']['categorical_features'])
    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    Xy_train = np.concatenate([X_train, y_train.reshape(y_train.shape[0],1)], axis=1)
    Xy_test = np.concatenate([X_test, y_test.reshape(y_test.shape[0],1)], axis=1)

    os.makedirs("data/prepared", exist_ok=True)
    # save processed train/test data
    with open("data/prepared/train.pkl", 'wb') as f:
            pickle.dump(Xy_train, f)

    with open("data/prepared/test.pkl", 'wb') as f:
            pickle.dump(Xy_test, f)

    depth_ = parameters['model']['max_depth']
    imbalance_ = parameters['model']['imbalance']
    # Train and save a model.
    fitted_model = model.train_model(X_train, y_train, max_depth = depth_, imbalance=imbalance_)

    os.makedirs("model/model", exist_ok=True)
    # save model to disk
    with open('model/model/rfc.pkl', 'wb') as file:
        pickle.dump(fitted_model, file)

    return None

if __name__ == '__main__':
    split_and_train()