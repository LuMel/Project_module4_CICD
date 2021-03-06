import numpy as np
import pandas as pd
import os
import pickle
import yaml
from typing import List, Optional, Tuple
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def remove_spaces(raw_data: pd.DataFrame, categorical_feats: List[str]) -> pd.DataFrame:
    """
    Removes spaces from columns & values in raw data
    Inputs
    ------
    raw_data : pd.DataFrame
                DataFrame containing the raw data
    Output
    ------
    pd.DataFrame identical to raw_data but with the spaces in the column's names & values removed
    """    
    raw_data.columns = [col.strip() for col in raw_data]
    raw_data = raw_data.apply(lambda x: x.str.strip() if x.name in categorical_feats else x)
    return raw_data

def process_data(
    X: pd.DataFrame, categorical_features: List[str] = [], label: Optional[str] =None, 
    training: bool = True, encoder: Optional[OneHotEncoder] = None, lb: Optional[LabelBinarizer] = None
) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder, LabelBinarizer]:
    """ Process the data used in the machine learning pipeline.
    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.
    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.
    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()

        # save encoder
        os.makedirs("data/encoders", exist_ok=True)
        with open("data/encoders/onehotenc.pkl", 'wb') as f:
            pickle.dump(encoder, f)
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass
    
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

