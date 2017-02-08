import pandas as pd
from time import time

from model import CardifChallengeModel
from tools import Submissioner

initial_time = time()

print("Pipeline script", end="\n\n")

# Initiate Submissioner
submissioner = Submissioner()

# Load train data
print("\tLoad train data ... ", end="", flush=True)
train_path_low = "data/preprocessed_train_low.csv"
train_df_low = pd.read_csv(train_path_low, index_col='ID')
# Split train data in X_train, y_train
X_train_low = train_df_low.drop('target', axis=1).values
y_train_low = train_df_low['target'].values

train_path_high = "data/preprocessed_train_high.csv"
train_df_high = pd.read_csv(train_path_high, index_col='ID')
# Split train data in X_train, y_train
X_train_high = train_df_high.drop('target', axis=1).values
y_train_high = train_df_high['target'].values
print("OK")

# Initiate the predictor
print("\tInitiate the predictor and fit the train data ... ", end="", flush=True)
cardif_low_model = CardifChallengeModel()
cardif_high_model = CardifChallengeModel()

# Fit the predictor to the train data
cardif_low_model.fit(X_train_low, y_train_low)
cardif_high_model.fit(X_train_high, y_train_high)
print("OK", end="\n\n")

# Load test data
print("\tLoad test data ... ", end="", flush=True)
test_path_low = "data/preprocessed_test_low.csv"
test_df_low = pd.read_csv(test_path_low, index_col='ID')
X_test_low = test_df_low.values

test_path_high = "data/preprocessed_test_high.csv"
test_df_high = pd.read_csv(test_path_high, index_col='ID')
X_test_high = test_df_high.values
print("OK")

# Predict the probabilites of the target of the test data
print("\tMake predictions on test data ... ", end="", flush=True)
y_pred_proba_low = cardif_low_model.predict_proba(X_test_low)
y_pred_proba_high = cardif_high_model.predict_proba(X_test_high)
print("OK")

# TODO: This should definitely be the role of the Submissioner to aggregate these predictions
# Merging predictions on the clusters
print("\tMerging predictions ... ", end="", flush=True)
test_df_low['prediction'] = y_pred_proba_low[:, 0]
test_df_high['prediction'] = y_pred_proba_high[:, 0]
test_df = pd.concat([test_df_low, test_df_high], axis=0)
test_df = test_df.sort_index()

# Save the predictions
index_test = test_df.index
y_pred_proba = test_df['prediction'].values
print("OK", end="\n\n")

# Create submission file
print("\tCreate submission file ... ", end="", flush=True)
submissioner.create_submission(index_test, y_pred_proba)
print("OK")

print("\nPipeline completed in %0.2f seconds" % (time() - initial_time))
