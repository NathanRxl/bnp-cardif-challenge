import numpy as np


def logloss(y_pred_proba, y_true):
    epsilon = 1e-15
    pred_proba = y_pred_proba.copy()
    pred_proba = np.maximum(epsilon, pred_proba)
    pred_proba = np.minimum(1 - epsilon, pred_proba)

    logloss = (
        - sum(
            y_true * np.log(pred_proba) +
            (1 - y_true) * np.log(1 - pred_proba)
        )
        / len(y_true)
    )
    return logloss
