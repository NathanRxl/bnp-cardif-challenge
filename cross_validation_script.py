from time import time
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import metrics
import models

initial_time = time()

print("Cross-validation script", end="\n\n")

# Load train data
train_path = "data/preprocessed_train.csv"
train_df = pd.read_csv(train_path, index_col='ID')
# Split train data in X_train, y_train
X_train = train_df.drop('target', axis=1)
y_train = train_df['target'].values

# Cross validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
cv_scores = []
cv_splits = kf.split(X_train.index.tolist())

# Define the model
model = models.LogReg()

for n_fold, (train_fold_idx, test_fold_idx) in enumerate(cv_splits):

    print(
        "Start working on fold number", n_fold + 1, "... ",
        end="",
        flush=True
    )

    X_fold_train = X_train.iloc[train_fold_idx].as_matrix()
    y_fold_train = y_train[train_fold_idx]

    X_fold_test = X_train.iloc[test_fold_idx].as_matrix()
    y_fold_test = y_train[test_fold_idx]

    model.fit(X_fold_train, y_fold_train)
    y_pred_proba = model.predict_proba(X_fold_test)

    fold_score = metrics.logloss(y_pred_proba, y_fold_test)

    print(round(fold_score, 5))
    cv_scores.append(fold_score)

cv_score_mean = round(np.mean(cv_scores), 5)
cv_score_std = round(np.std(cv_scores), 5)
cv_score_min = round(np.min(cv_scores), 5)
cv_score_max = round(np.max(cv_scores), 5)

print("\nMean score :", cv_score_mean)
print("Standard deviation :", cv_score_std)
print("Min score :", cv_score_min)
print("Max score :", cv_score_max)


print("\nCross-validation script completed in %0.2f seconds"
      % (time() - initial_time))
