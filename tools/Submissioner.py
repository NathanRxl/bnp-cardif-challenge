import pandas as pd
import numpy as np
import time
import os


class Submissioner:

    def __init__(self):
        pass

    @staticmethod
    def create_submission(index_test, y_predict, submission_folder="submissions"):
        predicted_prob = pd.DataFrame(y_predict, columns=["PredictedProb"])
        predicted_index = pd.DataFrame(index_test, columns=["ID"])
        submission_df = pd.concat([predicted_index, predicted_prob], axis=1)

        # create folder to store submission file if the folder doesn't exist yet
        if not os.path.exists(submission_folder):
            os.makedirs(submission_folder)

        submission_df.to_csv("%s/submission_%s.txt" % (submission_folder, time.strftime("%Y%m%d_%H%M%S")), index=False)
