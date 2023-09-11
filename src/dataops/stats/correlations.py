import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from dataops.utils.utils import set_sns_font
import dataops.messages as messages
from dataops import settings


def get_correlation_numerical(df, method='pearson'):
    return df.corr(method=method, numeric_only=True)


@set_sns_font(0.7)
def plot_correlation_numerical(df_corr, method='pearson', annot=True, mask='triu-ones', fmt='.0%', cmap='crest'):
    if mask == 'triu-ones':
        mask = np.triu(np.ones(df_corr.shape[1]), k=1)

    sns.heatmap(df_corr, annot=annot, mask=mask, fmt=fmt, cmap=cmap)
    plt.title(f'{method.capitalize()} correlation heatmap between numerical variables')
    plt.show()


def get_chi2(contingency_table):
    # Perform the chi-square test of independence
    chi2, p, _, _ = chi2_contingency(contingency_table)  # , correction=False
    return chi2, p


def get_cramers_v(contingency_table, chi2):
    return np.sqrt((chi2 / contingency_table.to_numpy().sum()) / (min(contingency_table.shape) - 1))  # np.sqrt(phi2 / min(r - 1, k - 1))


def get_association(df, method='chi2'):
    if any(df.nunique() > 10):
        logging.getLogger('dataops-logger').warning(messages.COR_EX_001_MSG)

    columns = df.columns

    # Calculate Associations
    association = {}
    if 'chi2' in method:
        association['chi2'] = pd.DataFrame(index=columns, columns=columns)
    if 'cramers-v' in method:
        association['cramers-v'] = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:
        for col2 in columns:
            # Create a contingency table between the two categorical columns
            contingency_table = pd.crosstab(df[col1], df[col2])

            if 'chi2' in method:
                association['chi2'].loc[col1, col2] = get_chi2(contingency_table)[0]

            if 'cramers-v' in method:
                chi2, _ = get_chi2(contingency_table)
                association['cramers-v'].loc[col1, col2] = get_cramers_v(contingency_table, chi2)

    return association


@set_sns_font(settings.multiclass.assoc_plot_font)
def plot_association(association_df, annot=True, fmt='.2f', cmap='crest'):
    plt.figure(figsize=(settings.multiclass.assoc_plot_width, settings.multiclass.assoc_plot_height))
    sns.heatmap(association_df.astype(float), annot=annot, fmt=fmt, cmap=cmap)
    plt.show()
