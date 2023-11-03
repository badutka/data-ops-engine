import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from dataops.utils.utils import set_sns_font, set_plot_size
import dataops.messages as messages
from dataops import settings
from dataops.visualization.plotters import CorrelationsPlotter
import scipy.stats as stats


class CorrelationAnalyzer:
    def __init__(self):
        self.correlation_plotter = CorrelationsPlotter()

        self.df_corr = None
        self.assoc_cramers = None
        self.assoc_chi2 = None
        self.point_biserial = dict()

    def calc_point_biserial_corr(self, cat_feat_data: pd.DataFrame, target: pd.Series):
        if isinstance(cat_feat_data, pd.DataFrame):
            # If cat_feat_data is a DataFrame, extract the feature names
            features = list(cat_feat_data.columns)
            if target.name in features:
                features.remove(target.name)
        elif isinstance(cat_feat_data, np.ndarray):
            # If cat_feat_data is a NumPy array, create a list of column indices
            features = list(range(cat_feat_data.shape[1]))
        else:
            raise ValueError("Input cat_feat_data must be a Pandas DataFrame or NumPy array.")

        for feature in features:
            if isinstance(cat_feat_data, pd.DataFrame):
                # If cat_feat_data is a DataFrame, use column names to access data
                feature_data = cat_feat_data[feature]
            elif isinstance(cat_feat_data, np.ndarray):
                # If cat_feat_data is a NumPy array, use column index to access data
                feature_data = cat_feat_data[:, feature]

            correlation_coefficient, _ = stats.pointbiserialr(feature_data, target)
            self.point_biserial[feature] = correlation_coefficient

    def calc_correlation_numerical(self, df, method='pearson'):
        self.df_corr = df.corr(method=method, numeric_only=True)

    def get_chi2(self, contingency_table):
        """
        Perform the chi-square test of independence
        """
        chi2, p, _, _ = chi2_contingency(contingency_table)  # , correction=False
        return chi2, p

    def get_cramers_v(self, contingency_table, chi2):
        return np.sqrt((chi2 / contingency_table.to_numpy().sum()) / (min(contingency_table.shape) - 1))  # np.sqrt(phi2 / min(r - 1, k - 1))

    def get_association(self, df):
        if any(df.nunique() > 10):
            logging.getLogger('dataops-logger').warning(messages.COR_EX_001_MSG)

        columns = df.columns

        self.assoc_chi2 = pd.DataFrame(index=columns, columns=columns)
        self.assoc_cramers = pd.DataFrame(index=columns, columns=columns)

        for col1 in columns:
            for col2 in columns:
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, _ = self.get_chi2(contingency_table)
                self.assoc_chi2.loc[col1, col2] = self.get_chi2(contingency_table)[0]
                self.assoc_cramers.loc[col1, col2] = self.get_cramers_v(contingency_table, chi2)

    def plot_point_biserial(self):
        self.correlation_plotter.plot_point_biserial(self.point_biserial)

    def plot_correlations(self):
        self.correlation_plotter.plot_correlation_numerical(self.df_corr)

    def plot_associations(self, assoc_type):
        if assoc_type == 'cramers-v':
            self.correlation_plotter.plot_association(association_df=self.assoc_cramers)
        elif assoc_type == 'chi2':
            self.correlation_plotter.plot_association(association_df=self.assoc_chi2)
        else:
            pass
