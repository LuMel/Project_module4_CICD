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

@pytest.fixture(scope='session')
def data_negative_API():
    '''
    Fixture of an example with negative outcome
    '''
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

    return test_data_neg


@pytest.fixture(scope='session')
def data_positive_API():
    '''
    Fixture of an example with positive outcome
    '''
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
    return test_data_pos
