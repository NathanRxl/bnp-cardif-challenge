# -*- coding: utf-8 -*-

import numpy as np


class Clusterizer:
    def __init__(self, nan_label='n_nan', threshold=60):
        self._nan_label = nan_label
        self._threshold = threshold

    def clusterize(self, data_frame, save=False, path_low='', path_high=''):
        df = self._add_nan_count(data_frame)
        df_low_nan = df[df[self._nan_label] < self._threshold]
        df_high_nan = df[df[self._nan_label] >= self._threshold]
        if save:
            df_low_nan.to_csv(path_low, sep=',')
            df_high_nan.to_csv(path_high, sep=',')
        return df_low_nan, df_high_nan

    def _add_nan_count(self, data_frame):
        n_nan = []
        df = data_frame.copy()
        df_null = df.isnull()
        for i in range(df.shape[0]):
            n_nan.append(np.sum(df_null.loc[i].values))
        df[self._nan_label] = n_nan
        return df