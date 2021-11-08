import pandas as pd
import numpy as np
from ..prepare_data import remove_spaces
from typing import Tuple

def test_column_names_raw(data: pd.DataFrame):
    """
    Tests whether the column names and column order are as expected
    (Note: in many columns, this implies having a space at the beginning)
    """
    expected_columns = ["age", " workclass", " fnlgt", " education", 
                       " education-num", " marital-status", " occupation",
                        " relationship", " race", " sex", " capital-gain",
                        " capital-loss", " hours-per-week", " native-country", 
                        " salary"]

    assert list(expected_columns) == list(data.columns.values), "Columns are not as expeted"


def test_age_range(data: pd.DataFrame):
    """
    Tests whether the age column takes values between the expected range.
    """
    assert data['age'].dropna().between(0, 100).all() , "Age is not in the exptected range"


def test_split_sizes(data_train_test: Tuple[np.ndarray, np.ndarray]):
    """
    Tests whether the training set has appropiate dimensions wrt the test set
    """

    train, test = data_train_test

    assert train.shape[0] > 2*(test.shape[0]), "train and test sets don't have expected proportions"

def test_remove_spaces_col_names(data: pd.DataFrame):
    """
    Tests whether the column names and column order are as expected
    after applying function remove_spaces
    """
    expected_columns = ["age", "workclass", "fnlgt", "education", 
                       "education-num", "marital-status", "occupation",
                        "relationship", "race", "sex", "capital-gain",
                        "capital-loss", "hours-per-week", "native-country", 
                        "salary"]

    data_rm = remove_spaces(data)
    assert list(expected_columns) == list(data_rm.columns.values), "Columns are not as expeted"

def test_split_type(data_train_test: Tuple[np.ndarray, np.ndarray]):
    """
    Check if the first two outputs of the function prepare_data are ndarrays
    """
    train, test = data_train_test

    assert isinstance(train, np.ndarray), "training set does not have the expected type"
    assert isinstance(test, np.ndarray), "test set does not have the expected type"

