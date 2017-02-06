import pandas as pd
import time


class Submissioner:

    def __init__(self):
        pass

    def create_submission(self, y_predict, folder="submissions"):

        ID = pd.read_csv("submissions/sample_submission.csv", usecols=["ID"])
        predicted_prob = pd.DataFrame(y_predict[:, 1], columns=["PredictedProb"])

        submission_df = pd.concat([ID, predicted_prob], axis=1)

        # Write the submission
        submission_filename = "%s/submission_%s.txt" % (folder, time.strftime("%Y%m%d_%H%M%S"))
        submission_df.to_csv(submission_filename, index=False)
