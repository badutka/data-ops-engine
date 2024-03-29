import numpy as np
import pandas as pd
from typing import Dict, Union, AnyStr

import seaborn as sns
import matplotlib.pyplot as plt

from dataops import settings
from dataops.utils.utils import set_sns_font, set_plot_size


class MetricPlotter():
    def plot_confusion_matrices(self, confusion_matrices, nrows, ncols):
        """
        https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
        """
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 4 * nrows))

        for i, (name, cm) in enumerate(confusion_matrices.items()):
            if nrows > 1 and ncols > 1:
                ax = axs[np.unravel_index(i, (nrows, ncols))]
            else:
                if nrows == 1 and ncols == 1:
                    axs = np.array([axs])
                ax = axs[i]
            sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, fmt='.2%')
            ax.set_title(name)

        plt.tight_layout()
        plt.show()

    @set_plot_size(plot_size=(10, 10))
    def plot_metrics_heatmap(self, metrics_df):
        sns.heatmap(metrics_df, annot=True, fmt='.2%', cmap='crest')
        plt.title('Metrics for every model')
        plt.show()


class CorrelationsPlotter():
    def plot_point_biserial(self, correlation_coefficients: Dict[str, float], plot_type='heatmap'):
        plt.figure(figsize=(10, 6))

        if plot_type == 'heatmap':
            sns.heatmap(pd.DataFrame.from_dict(correlation_coefficients, orient='index', columns=['Correlation Coefficient']), annot=True, fmt='.2%', cmap='crest')
        if plot_type == 'bar':
            sns.barplot(x=list(correlation_coefficients.values()), y=list(correlation_coefficients.keys()), orient='h')

        plt.xlabel("Point-Biserial Correlation Coefficient")
        plt.ylabel("Binary Categorical Features")
        plt.title("Correlations between Features and Target")
        plt.show()

    @set_sns_font(0.7)
    def plot_correlation_numerical(self, df_corr, method='pearson', annot=True, mask='triu-ones', fmt='.0%', cmap='crest'):
        if mask == 'triu-ones':
            mask = np.triu(np.ones(df_corr.shape[1]), k=1)

        sns.heatmap(df_corr, annot=annot, mask=mask, fmt=fmt, cmap=cmap)
        plt.title(f'{method.capitalize()} correlation heatmap between numerical variables')
        plt.show()

    @set_sns_font(settings.multiclass.assoc_plot_font)
    def plot_association(self, association_df, annot=True, fmt='.2f', cmap='crest'):
        plt.figure(figsize=(settings.multiclass.assoc_plot_width, settings.multiclass.assoc_plot_height))
        sns.heatmap(association_df.astype(float), annot=annot, fmt=fmt, cmap=cmap)
        plt.show()
