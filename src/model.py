from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
import yaml



# Optional: implement hyperparameter tuning.
def train_model(
    X_train: np.ndarray, y_train: np.ndarray, 
                max_depth: int, imbalance: Optional[str]
) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    max_depth: int 
        Maximum depth of Random Forest
    imbalance: Optional[str]
        whether we apply class wheight (imbalanced datasets)
    Returns
    -------
    model
        Trained machine learning model.
    """
    if imbalance:
        rfc = RandomForestClassifier(max_depth=max_depth, class_weight='balanced').fit(X_train, y_train)
    else:
        rfc = RandomForestClassifier(max_depth=max_depth).fit(X_train, y_train)
    return rfc


def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def performance_data_slice(onehotenc: OneHotEncoder, 
                           fitted_model: RandomForestClassifier, 
                           X: np.ndarray, 
                           y: np.ndarray, 
                           col: str, 
                           value: Optional[str] = None,
                           save: bool = True) -> Union[pd.DataFrame, Tuple[float, float, float]]:

    """ Computes metrics for given data slice
    Inputs
    ------
    onehotenc : sklearn.preprocessing.OneHotEncoder
    fitted_model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    y : np.ndarray
        Data used for evaluation
    col : str
        name of the categorical column used for performance-slicing
    value : str
        Value of the categorical value to do the slice. 
        If None it will evaluate the perfomance of every value of col 
    Returns
    -------
    Evaluation of the performance on the input slices
    """
    parameters = yaml.safe_load(open('params.yaml'))
    cat_features = parameters['data']['categorical_features']
    num_features = parameters['data']['numerical_features']
 
    try:
        idx_col = np.where(np.array(cat_features) == col)[0][0]
    except IndexError as err:
        print('{0} not in categorical columns'.format(col))
        raise err
    
    cols_look = [col for col in onehotenc.get_feature_names_out() if col[3:] in onehotenc.__dict__['categories_'][idx_col]]
    cols_idx = [len(num_features) + i for i, e in enumerate(onehotenc.get_feature_names_out()) if e in cols_look]

    if value is None:
        performance = list()
        for idx in cols_idx:
            X_test_f = X[X[:, idx] == 1.].copy()
            y_test_f = y[X[:, idx] == 1.].copy()
            performance.append(compute_model_metrics(y_test_f, inference(fitted_model, X_test_f)))
        df_performance = pd.DataFrame(performance, columns = ['precision', 'recall', 'fbeta'], index = cols_look) 
        df_performance.index.name = col
        if save:
            df_performance.to_csv("data/prepared/slice_output.txt", sep=' ', mode='a')
        return df_performance

    value_look = [col for col in onehotenc.get_feature_names_out() if col[3:] == value]
    value_idx = [len(num_features) + i for i, e in enumerate(onehotenc.get_feature_names_out()) if e in value_look]

    assert len(list(set(cols_idx) & set(value_idx))) > 0,  "value does not belong to column"
    assert len(value_idx) == 1, 'value was either not found or found multiple times'
    X_test_f = X[X[:, value_idx[0]] == 1.].copy()
    y_test_f = y[X[:, value_idx[0]] == 1.].copy()
    
    return compute_model_metrics(y_test_f, inference(fitted_model, X_test_f))



