import numpy as np


class Preprocessor:

    def __init__(self):
        self.categorical_likelihood_mapping = dict()

    @staticmethod
    def preprocess_numerical_data(train_or_test, df_to_preprocess):

        df_transformed = df_to_preprocess.copy()

        # Non categorical features preprocessing
        non_c_features = df_transformed.dtypes[df_transformed.dtypes != 'object'].index.tolist()
        df_transformed[non_c_features] = df_transformed[non_c_features].fillna(value=-1, inplace=False)

        return df_transformed

    @staticmethod
    def build_categorical_likelihood_mapping(df_to_transform, c_feature):
        df_group_by_category_label = (
            df_to_transform[['target', 'ID', c_feature]].groupby([c_feature, 'target']).count()['ID']
        )
        # Compute the likelihood mapping
        categorical_likelihood_mapping = dict()
        c_feature_median = list()
        c_feature_unique_categories = df_to_transform[c_feature].unique()
        for category in c_feature_unique_categories:
            try:
                category_occurences = df_group_by_category_label[category].sum()
                category_label_one_frequency = float(df_group_by_category_label[category][1]) / category_occurences
                categorical_likelihood_mapping[category] = category_label_one_frequency
                c_feature_median.append(category_label_one_frequency)
            except KeyError:
                categorical_likelihood_mapping[category] = 0.0
                c_feature_median.append(0.0)

        # Fill possible new categories (not present in the categorical likelihood) by the median
        # Typically, in test, 40 new users appear in v22 : we fill them by the median likelihood of all other users
        categorical_likelihood_mapping["median"] = np.median(c_feature_median)

        return categorical_likelihood_mapping

    def preprocess_categorical_data(self, train_or_test, df_to_preprocess):

        df_transformed = df_to_preprocess.copy()

        # Categorical features preprocessing
        c_features = df_transformed.dtypes[df_transformed.dtypes == 'object'].index.tolist()
        # Fill NaN values with a new category for these features
        df_transformed[c_features] = df_transformed[c_features].fillna(value='NaN', inplace=False)

        # Compute the likelihood mapping between category and likelihood of target for each feature
        # Do the preprocessing for each feature on-the-fly
        for c_feature in c_features:
            if train_or_test == 'train':
                # Compute the likelihood mapping
                self.categorical_likelihood_mapping[c_feature] = (
                    self.build_categorical_likelihood_mapping(df_transformed, c_feature)
                )

            # Preprocessing on-the-fly
            df_transformed[c_feature] = df_transformed[c_feature].map(self.categorical_likelihood_mapping[c_feature])
            df_transformed[c_feature] = df_transformed[c_feature].fillna(
                value=self.categorical_likelihood_mapping[c_feature]["median"], inplace=False
            )

        return df_transformed

    def preprocess_data(self, train_or_test, df_to_preprocess, verbose=True):

        if verbose:
            print("\t Preprocessing numerical data ... ", end="", flush=True)

        df_numerical_transformed = self.preprocess_numerical_data(train_or_test, df_to_preprocess)
        if verbose:
            print("OK", flush=True)
            print("\t Preprocessing categorical data ... ", end="", flush=True)
        df_numerical_categorical_transformed = self.preprocess_categorical_data(train_or_test, df_numerical_transformed)
        if verbose:
            print("OK", flush=True)

        return df_numerical_categorical_transformed
