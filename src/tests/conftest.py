import pytest
import pandas as pd

@pytest.fixture(scope='session')
def data():
    '''
    Fixture to get the raw data for the tests
    '''
    # load raw data
    df = pd.read_csv("data/census.csv")
    return df

@pytest.fixture(scope='session')
def data_train_test():
    '''
    Fixture to get the training & test data for the tests
    '''
    # load raw data
    df_train = pd.read_pickle("data/prepared/train.pkl")
    df_test = pd.read_pickle("data/prepared/test.pkl")
    return df_train, df_test