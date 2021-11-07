import pytest
import pandas as pd

@pytest.fixture(scope='session')
def data():
    '''
    Fixture to get the data for the tests
    '''
    # load raw data
    df = pd.read_csv("data/census.csv")
    return df