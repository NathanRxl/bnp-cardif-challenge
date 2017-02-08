import pandas as pd
from time import time
from tools import Clusterizer
from preprocessing import Preprocessor

initial_time = time()

print("Preprocessing script", end="\n\n")

preprocessor = Preprocessor()


# Get the data
print("\tGetting train data ... ", end="", flush=True)
train_path = "data/train.csv"
train_df = pd.read_csv("data/train.csv")
print("OK")

print("\tGetting test data ... ", end="", flush=True)
test_path = "data/test.csv"
test_df = pd.read_csv("data/test.csv")
print("OK", end="\n\n")


# TODO: The clustering should appear after the preprocessing, shouldn't it ?
# TODO: Possibly into the Preprocessor itself ?
# Cluster the data
print("\tClustering train ... ", end="", flush=True)
clusterizer = Clusterizer.Clusterizer()
train_df_low, train_df_high = clusterizer.clusterize(train_df)
print("OK")

print("\tClustering test ... ", end="", flush=True)
test_df_low, test_df_high = clusterizer.clusterize(test_df)
print("OK", end="\n\n")


# Preprocess train
print("\tPreprocessing training sparse data cluster ...")

preprocessed_train_low_df = preprocessor.preprocess_data("train", train_df_low)
print("\t\tCreating sparse data cluster 'data/preprocessed_train_low.csv' ... ", end="", flush=True)
preprocessed_train_low_df.to_csv("data/preprocessed_train_low.csv", index=False)
print("OK", end="\n\n")

print("\tPreprocessing training dense data cluster ...")

preprocessed_train_high_df = preprocessor.preprocess_data("train", train_df_high)
print("\t\tCreating dense data cluster 'data/preprocessed_train_high.csv' ... ", end="", flush=True)
preprocessed_train_high_df.to_csv("data/preprocessed_train_high.csv", index=False)
print("OK", end="\n\n", flush=True)


# Preprocess test
print("\tPreprocessing test sparse data cluster ...")

preprocessed_test_low_df = preprocessor.preprocess_data("test", test_df_low)
print("\t\tCreating sparse data cluster 'data/preprocessed_test_low.csv' ... ", end="", flush=True)
preprocessed_test_low_df.to_csv("data/preprocessed_test_low.csv", index=False)
print("OK", flush=True)

print("\tPreprocessing test dense data cluster ...")

preprocessed_test_low_df = preprocessor.preprocess_data("test", test_df_high)
print("\t\tCreating dense data cluster 'data/preprocessed_test_high.csv' ... ", end="", flush=True)
preprocessed_test_low_df.to_csv("data/preprocessed_test_high.csv", index=False)
print("OK", flush=True)


print("\nPreprocessing completed in %0.2f seconds" % (time() - initial_time))
