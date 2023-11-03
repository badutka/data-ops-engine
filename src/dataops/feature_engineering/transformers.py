from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

from dataops.stats.analyzers import CorrelationAnalyzer
from dataops import settings

import logging


class CategoricalFeatDropTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold):
        self.columns_to_drop = None
        self.correlation_threshold = correlation_threshold
        self.correlation_analyzer = CorrelationAnalyzer()
        self.logger = logging.getLogger(settings.common.logger_name)

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = np.delete(X, self.columns_to_drop, axis=1)
        elif isinstance(X, pd.DataFrame):
            X = X.drop(self.columns_to_drop, axis=1)
        return X

    def fit(self, X, y=None):
        if y is not None:
            self.correlation_analyzer.calc_point_biserial_corr(X, y)
            correlations = self.correlation_analyzer.point_biserial
            self.columns_to_drop = [feature for feature, corr in correlations.items() if np.abs(corr) < self.correlation_threshold]
            self.logger.info(f"Number of total categorical features: {X.shape[1]}, features to drop: {len(self.columns_to_drop)}, remaining features: {X.shape[1] - len(self.columns_to_drop)}")
        return self

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.columns_to_drop
        remaining_columns = [input_features[i] for i in range(len(input_features)) if i not in self.columns_to_drop]
        return remaining_columns
