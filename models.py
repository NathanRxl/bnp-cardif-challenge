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
