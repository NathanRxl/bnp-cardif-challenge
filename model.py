from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class CardifChallengeModel:
    def __init__(self, **kwargs):
        self.model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("logreg", LogisticRegression(**kwargs)),
            ]
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def predict_log_proba(self, X_test):
        return self.model.predict_log_proba(X_test)
