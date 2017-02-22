import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb


class LogReg:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("logreg", LogisticRegression(solver='lbfgs',
                                          class_weight='balanced'))
        ])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]


class RandomForest:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("randomforest", RandomForestClassifier(n_estimators=150,
                                                    n_jobs=-1))
        ])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]


class XGBoost:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("xgb", xgb.XGBClassifier(
                learning_rate=0.1,
                n_estimators=100,
                objective='binary:logistic',
                nthread=-1,
                subsample=0.7,
                colsample_bytree=0.5
            ))
        ])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]


class Stacking:
    def __init__(self):
        logreg_model = LogReg()
        random_forest_model = RandomForest()
        xgb_model = XGBoost()

        self.stacked_models = {
            "LogReg": logreg_model,
            "RandomForest": random_forest_model,
            "XGBoost": xgb_model
        }

    def fit(self, X_train, y_train):
        for model_name in self.stacked_models:
            self.stacked_models[model_name].fit(X_train, y_train)

    def predict(self, X_test):
        return np.argmax(self.predict_proba(X_test))

    def predict_proba(self, X_test):
        y_pred_proba = np.zeros(shape=(X_test.shape[0], ))
        nb_of_stacked_models = len(self.stacked_models)

        for model_name in self.stacked_models:
            y_pred_proba += (
                (1 / nb_of_stacked_models) *
                self.stacked_models[model_name].predict_proba(X_test)
            )

        return y_pred_proba
