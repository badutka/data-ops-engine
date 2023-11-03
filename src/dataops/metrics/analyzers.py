import pandas as pd
import numpy as np

from typing import Dict, Tuple, AnyStr, Union, List
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

from dataops.visualization.plotters import MetricPlotter
from dataops.utils.utils import set_plot_size


class MetricAnalyzer:
    def __init__(self):
        self.metric_plotter = MetricPlotter()
        self.confusion_matrices = dict()
        self.gen_metrics_df: Union[Tuple[pd.DataFrame, Dict[AnyStr, ConfusionMatrixDisplay]], None] = None

    def calculate_metrics(self, models: Dict, X_test: pd.DataFrame, y_true: pd.Series, average: str = 'binary', metric_list=None, score=True) -> None:
        if metric_list is None:  # if metric_list is None then prepare some generic default metrics
            metric_list = ['accuracy', 'precision', 'f1_score', 'recall']

        if average != 'binary' and "roc_auc_score" in metric_list:  # if average is not set to binary (binary classifier), then remove roc_auc_score metric from list, if it contains it
            metric_list.remove("roc_auc_score")

        self.gen_metrics_df = pd.DataFrame(columns=metric_list, index=list(models.keys()), dtype=float)  # initialize metric df (model names as indices, metric names as columns)

        for name, model in models.items():
            y_pred = model.predict(X_test)

            if score is True:  # if score is set to True, then also use models' score method
                self.gen_metrics_df.at[name, "score"] = models[name].score(X_test, y_true)

            for metric in metric_list:  # loop over metrics and call appropriate functions
                self.gen_metrics_df.at[name, metric] = self.calculate_metric(metric=metric, y_true=y_true, y_pred=y_pred, average=average)

            self.confusion_matrices[name] = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='all')

    def calculate_metric(self, metric, y_true, y_pred, average):
        match metric:
            case 'accuracy':
                return metrics.accuracy_score(y_true, y_pred)
            case 'f1_score':
                return metrics.f1_score(y_true, y_pred, average=average)
            case 'precision':
                return metrics.precision_score(y_true, y_pred, average=average, zero_division=np.nan)
            case 'recall':
                return metrics.recall_score(y_true, y_pred, average=average)
            case 'balanced_accuracy':
                return metrics.balanced_accuracy_score(y_true, y_pred)
            case other:
                raise ValueError(f"Metric '{metric}' is not supported.")

    def plot_metrics(self, nrows=1, ncols=2):
        self.metric_plotter.plot_confusion_matrices(self.confusion_matrices, nrows, ncols)
        self.metric_plotter.plot_metrics_heatmap(self.gen_metrics_df)
