from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict, Tuple, AnyStr
import numpy as np


def get_metrics_bc(models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.DataFrame, Dict[AnyStr, ConfusionMatrixDisplay]]:
    metrics_df = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1_score", "roc_auc_score"], index=list(models.keys()), dtype=float)
    confusion_matrices = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics_df.at[name, "accuracy"] = models[name].score(X_test, y_test)
        metrics_df.at[name, "precision"] = precision_score(y_test, y_pred)
        metrics_df.at[name, "recall"] = recall_score(y_test, y_pred)
        metrics_df.at[name, "f1_score"] = f1_score(y_test, y_pred)
        metrics_df.at[name, "roc_auc_score"] = roc_auc_score(y_test, y_pred)

        # confusion_matrices[name] = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=[False, True])
        # ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=[False, True])
        confusion_matrices[name] = confusion_matrix(y_test, y_pred, normalize='all')

    return metrics_df, confusion_matrices


def plot_confusion_matrices(confusion_matrices, nrows, ncols):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 4 * nrows))

    # https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        if nrows > 1 and ncols > 1:
            ax = axs[np.unravel_index(i, (nrows, ncols))]
        else:
            ax = axs[i]
        sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, fmt='.2%')
        ax.set_title(name)

    plt.tight_layout()
    plt.show()


def plot_metrics_heatmap(metrics_df):
    sns.heatmap(metrics_df, annot=True, fmt='.2%', cmap='crest')
    plt.title('Metrics for every model')
    plt.show()
