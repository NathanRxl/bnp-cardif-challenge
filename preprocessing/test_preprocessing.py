import unittest

import numpy as np
import pandas as pd

from Preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = Preprocessor()

    def assertDataFrameEquals(self, preprocessed_df, expected_preprocessed_df):
        self.assertTrue(
            preprocessed_df.equals(expected_preprocessed_df),
            msg=(
                "\n\nPreprocessed DataFrame \n {} \n\nExpected preprocessed DataFrame \n {}"
                .format(preprocessed_df, expected_preprocessed_df)
            )
        )

    def test_preprocessing_script_non_c_feature(self):
        preprocessing_test_df = pd.DataFrame(data={
            'non_c_feature_1': pd.Series([0.5, 10, np.nan]),
            'non_c_feature_2': pd.Series([0, 0.5, 18.5]),
        })
        preprocessed_test_df = self.preprocessor.preprocess_data("train", preprocessing_test_df, verbose=False)

        expected_preprocessed_test_df = pd.DataFrame(data={
            'non_c_feature_1': pd.Series([0.5, 10, -1]),
            'non_c_feature_2': pd.Series([0, 0.5, 18.5]),
        })
        self.assertDataFrameEquals(preprocessed_test_df, expected_preprocessed_test_df)

    def test_preprocessing_script_c_feature(self):
        preprocessing_test_df = pd.DataFrame(data={
            'c_feature_1': pd.Series(["A", np.nan, "B", "B", "A"]),
            'c_feature_2': pd.Series([np.nan, "D", "C", "C", "C"]),
            'ID': pd.Series([0, 1, 2, 3, 4]),
            'target': pd.Series([1, 0, 1, 0, 1]),
        })
        preprocessed_test_df = self.preprocessor.preprocess_data("train", preprocessing_test_df, verbose=False)

        expected_preprocessed_test_df = pd.DataFrame(data={
            'c_feature_1': pd.Series([1, 0, 0.5, 0.5, 1]),
            'c_feature_2': pd.Series([1, 0, 2/3, 2/3, 2/3]),
            'ID': pd.Series([0, 1, 2, 3, 4]),
            'target': pd.Series([1, 0, 1, 0, 1]),
        })
        self.assertDataFrameEquals(preprocessed_test_df, expected_preprocessed_test_df)

    def test_preprocessing_script(self):
        preprocessing_test_df = pd.DataFrame(data={
            'non_c_feature': pd.Series([0.5, 10, np.nan, np.nan]),
            'c_feature': pd.Series(["A", np.nan, "B", "A"]),
            'ID': pd.Series([0, 1, 2, 3]),
            'target': pd.Series([1, 1, 0, 0]),
        })
        preprocessed_test_df = self.preprocessor.preprocess_data("train", preprocessing_test_df, verbose=False)

        expected_preprocessed_test_df = pd.DataFrame(data={
            'non_c_feature': pd.Series([0.5, 10, -1, -1]),
            'c_feature': pd.Series([0.5, 1, 0, 0.5]),
            'ID': pd.Series([0, 1, 2, 3]),
            'target': pd.Series([1, 1, 0, 0]),
        })
        self.assertDataFrameEquals(preprocessed_test_df, expected_preprocessed_test_df)

if __name__ == "__main__":
    unittest.main()
