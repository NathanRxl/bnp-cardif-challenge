import pandas as pd
from time import time

from preprocessing import Preprocessor

initial_time = time()

print("Preprocessing script", end="\n\n")

preprocessor = Preprocessor()

# Preprocess train
print("Preprocessing train csv ... ")
train_path = "data/train.csv"
train_df = pd.read_csv("data/train.csv")
preprocessed_train_df = preprocessor.preprocess_data("train", train_df)
print("\t Creating data/preprocessed_train.csv ... ", end="", flush=True)
preprocessed_train_df.to_csv("data/preprocessed_train.csv", index=False)
print("OK", flush=True)

# Preprocess test
print("Preprocessing test csv ... ")
test_path = "data/test.csv"
test_df = pd.read_csv("data/test.csv")
preprocessed_test_df = preprocessor.preprocess_data("test", test_df)
print("\t Creating data/preprocessed_test.csv ... ", end="", flush=True)
preprocessed_test_df.to_csv("data/preprocessed_test.csv", index=False)
print("OK", flush=True)

print("\nPreprocessing completed in %0.2f seconds" % (time() - initial_time))
