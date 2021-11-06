from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Optional, Tuple



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