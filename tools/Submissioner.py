import pandas as pd
import numpy as np
import time
import os


class Submissioner:

    def __init__(self):
        pass

    @staticmethod
    def save(index_test, y_predict, folder="submissions"):
        sub = pd.DataFrame(y_predict, columns=["PredictedProb"])
        ID = pd.DataFrame(index_test, columns=["ID"])
        sub1 = pd.concat([ID, sub], axis=1)
        sub1.to_csv("output/submission.txt", index=False)

    def create_submission(self, ref_submission_path="input/sample_submission.csv", folder="submissions"):
        # retrieve submission file reference
        ref_submission = pd.read_csv(ref_submission_path)

        # set predictions in dataframe
        ref_submission.set_index(["ID"], inplace=True)
        ref_submission.loc[self.multi_index] = np.array(self.PredictedProb).reshape((len(self.PredictedProb), 1))
        submission = ref_submission.reset_index()

        # convert dataframe to string
        submission_content = submission.to_csv(index=False)

        # create folder to store submission file if the folder doesn't exist yet
        if not os.path.exists(folder):
            os.makedirs(folder)

        # create file name according to current date
        file_name = "%s/submission_%s.txt" % (folder, time.strftime("%Y%m%d_%H%M%S"))

        # write into submission file
        with open(file_name, 'w') as file:
            file.write(submission_content)
