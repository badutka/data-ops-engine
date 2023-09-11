import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from dataops.utils.utils import set_sns_font
import dataops.messages as messages
from dataops import settings
from dataops.visualization.plotters import CorrelationsPlotter


class CorrelationAnalyzer:
    def __init__(self):
        self.correlation_plotter = CorrelationsPlotter()

        self.df_corr = None
        self.assoc_cat = dict()

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

    def get_association(self, df, method='chi2'):
        if any(df.nunique() > 10):
            logging.getLogger('dataops-logger').warning(messages.COR_EX_001_MSG)

        columns = df.columns

        # Calculate Associations
        if 'chi2' in method:
            self.assoc_cat['chi2'] = pd.DataFrame(index=columns, columns=columns)
        if 'cramers-v' in method:
            self.assoc_cat['cramers-v'] = pd.DataFrame(index=columns, columns=columns)

        for col1 in columns:
            for col2 in columns:
                # Create a contingency table between the two categorical columns
                contingency_table = pd.crosstab(df[col1], df[col2])

                if 'chi2' in method:
                    self.assoc_cat['chi2'].loc[col1, col2] = self.get_chi2(contingency_table)[0]

                if 'cramers-v' in method:
                    chi2, _ = self.get_chi2(contingency_table)
                    self.assoc_cat['cramers-v'].loc[col1, col2] = self.get_cramers_v(contingency_table, chi2)

    def plot_correlations(self):
        self.correlation_plotter.plot_correlation_numerical(self.df_corr)

    def plot_associations(self, assoc_type):
        self.correlation_plotter.plot_association(association_df=self.assoc_cat[assoc_type])
