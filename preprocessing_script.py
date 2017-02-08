import pandas as pd
from time import time
from tools import Clusterizer
from preprocessing import Preprocessor

initial_time = time()

print("Preprocessing script", end="\n\n")

preprocessor = Preprocessor()
# Get Data
print("Getting train data...")
train_path = "data/train.csv"
train_df = pd.read_csv("data/train.csv")
print("OK\n\nGetting test data...")
test_path = "data/test.csv"
test_df = pd.read_csv("data/test.csv")
print("OK")

# Clustering
print ("Clustering train...")
clusterizer = Clusterizer.Clusterizer()
train_df_low, train_df_high = clusterizer.clusterize(train_df)
print ("OK\n\nClustering test...")
test_df_low, test_df_high = clusterizer.clusterize(test_df)
print ("OK")
# Preprocess train
print("Preprocessing trainning cluster 1... ")

preprocessed_train_low_df = preprocessor.preprocess_data("train", train_df_low)
print("\t Creating data/preprocessed_train_low.csv ... ", end="", flush=True)
preprocessed_train_low_df.to_csv("data/preprocessed_train_low.csv", index=False)
print("OK", flush=True)

print("Preprocessing trainning cluster 2... ")

preprocessed_train_high_df = preprocessor.preprocess_data("train", train_df_high)
print("\t Creating data/preprocessed_train_high.csv ... ", end="", flush=True)
preprocessed_train_high_df.to_csv("data/preprocessed_train_high.csv", index=False)
print("OK", flush=True)


# Preprocess test
print("Preprocessing testing cluster 1... ")

preprocessed_test_low_df = preprocessor.preprocess_data("test", test_df_low)
print("\t Creating data/preprocessed_test_low.csv ... ", end="", flush=True)
preprocessed_test_low_df.to_csv("data/preprocessed_test_low.csv", index=False)
print("OK", flush=True)

print("Preprocessing testing cluster 2... ")

preprocessed_test_low_df = preprocessor.preprocess_data("test", test_df_high)
print("\t Creating data/preprocessed_test_high.csv ... ", end="", flush=True)
preprocessed_test_low_df.to_csv("data/preprocessed_test_high.csv", index=False)
print("OK", flush=True)
print("\nPreprocessing completed in %0.2f seconds" % (time() - initial_time))
