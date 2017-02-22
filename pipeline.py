import pandas as pd
from time import time

import metrics
import models
from tools import Submissioner


initial_time = time()

print("Pipeline script", end="\n\n")

# Initiate Submissioner
submissioner = Submissioner()

# Load train data
print("\tLoad train data ... ", end="", flush=True)
train_path = "data/preprocessed_train.csv"
train_df = pd.read_csv(train_path, index_col='ID')
# Split train data in X_train, y_train
X_train = train_df.drop('target', axis=1).values
y_train = train_df['target'].values
print("OK")

# Initiate the predictor
print("\tInitiate the predictor and fit the train data ... ", end="", flush=True)
cardif_model = models.LogReg()

# Fit the predictor to the train data
cardif_model.fit(X_train, y_train)
print("OK", end="\n\n")


# Compute the training score
print("\tTraining score: ", end="", flush=True)
y_pred_train = cardif_model.predict_proba(X_train)
train_score = metrics.logloss(y_pred_train, y_train)
print(round(train_score, 5), end="\n\n")


# Load test data
print("\tLoad test data ... ", end="", flush=True)
test_path = "data/preprocessed_test.csv"
test_df = pd.read_csv(test_path, index_col='ID')
X_test = test_df.values
print("OK")

# Predict the probabilites of the target of the test data
print("\tMake predictions on test data ... ", end="", flush=True)
y_pred_proba = cardif_model.predict_proba(X_test)

# Save the predictions
# submissioner.save(y_pred_proba)
print("OK", end="\n\n")


# Create submission file
print("\tCreate submission file ... ", end="", flush=True)
submissioner.create_submission(y_pred_proba)
print("OK")

print("\nPipeline completed in %0.2f seconds" % (time() - initial_time))
