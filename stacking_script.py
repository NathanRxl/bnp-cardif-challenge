from time import time
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold

import models

initial_time = time()

print("Stacking script", end="\n\n")

data_folder = "data/"

# Load train data
train_path = data_folder + "preprocessed_train.csv"
train_df = pd.read_csv(train_path, index_col='ID')
# Split train data in X_train, y_train
X_train = train_df.drop('target', axis=1)
y_train = train_df['target'].values

# Load test data
test_path = data_folder + "preprocessed_test.csv"
test_df = pd.read_csv(test_path, index_col='ID')

X_test = test_df.values

# KFold Stacking
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
stacking_splits = kf.split(X_train.index.tolist())

# Define the models to stack
models_to_stack = {
    "LogReg": models.LogReg(),
    "RandomForest": models.RandomForest(),
    "XGBoost": models.XGBoost(),
}

X_train_stacked = pd.DataFrame({"target": train_df["target"]})
X_test_stacked = pd.DataFrame(index=test_df.index)

for model_name, model in models_to_stack.items():
    X_test_stacked[model_name] = np.zeros((test_df.shape[0],))

for n_fold, (train_fold_idx, test_fold_idx) in enumerate(stacking_splits):

    print(
        "Start stacking fold number", n_fold + 1, "... ",
        end="",
        flush=True
    )

    X_fold_train = X_train.iloc[train_fold_idx]
    y_fold_train = y_train[train_fold_idx]

    X_fold_test = X_train.iloc[test_fold_idx]
    y_fold_test = y_train[test_fold_idx]

    for model_name, model in models_to_stack.items():
        model.fit(X_fold_train, y_fold_train)
        y_fold_pred_proba = model.predict_proba(X_fold_test)
        stack_fold_idx = X_train_stacked.iloc[test_fold_idx].index
        X_train_stacked.loc[stack_fold_idx, model_name] = y_fold_pred_proba

        y_test_pred_proba = model.predict_proba(X_test)
        X_test_stacked[model_name] += (1 / n_splits) * y_test_pred_proba

    print("OK")

X_train_stacked.to_csv(data_folder + "train_stacked.csv")
X_test_stacked.to_csv(data_folder + "test_stacked.csv")

print("\nStacking script completed in %0.2f seconds"
      % (time() - initial_time))
