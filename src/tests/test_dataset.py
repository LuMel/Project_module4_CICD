import pandas as pd

def test_column_names(data):
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


def test_age_range(data):
    """
    Tests whether the age column takes values between the expected range.
    """
    assert data['age'].dropna().between(0, 100).all() , "Age is not in the exptected range"