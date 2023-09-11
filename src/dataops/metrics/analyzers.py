import pandas as pd
import numpy as np

from typing import Dict, Tuple, AnyStr, Union
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dataops.visualization.plotters import MetricPlotter


class MetricAnalyzer:
    def __init__(self):
        self.metric_plotter = MetricPlotter()

        self.metrics = ["accuracy", "precision", "recall", "f1_score"]
        self.confusion_matrices = dict()
        self.gen_metrics_df: Union[Tuple[pd.DataFrame, Dict[AnyStr, ConfusionMatrixDisplay]], None] = None

        self._conf_matrix_nrows = 1
        self._conf_matrix_ncols = 2

    def calculate_metrics(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series, average: str = 'binary') -> None:

        if average != 'binary' and "roc_auc_score" in self.metrics:
            self.metrics.remove("roc_auc_score")

        self.gen_metrics_df = pd.DataFrame(columns=self.metrics, index=list(models.keys()), dtype=float)

        for name, model in models.items():
            y_pred = model.predict(X_test)
            self.gen_metrics_df.at[name, "accuracy"] = models[name].score(X_test, y_test)
            self.gen_metrics_df.at[name, "precision"] = precision_score(y_test, y_pred, average=average, zero_division=np.nan)
            self.gen_metrics_df.at[name, "recall"] = recall_score(y_test, y_pred, average=average)
            self.gen_metrics_df.at[name, "f1_score"] = f1_score(y_test, y_pred, average=average)

            if average == 'binary' and "roc_auc_score" in self.metrics:
                self.gen_metrics_df.at[name, "roc_auc_score"] = roc_auc_score(y_test, y_pred, average=average)

            # confusion_matrices[name] = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=[False, True])
            # ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=[False, True])
            self.confusion_matrices[name] = confusion_matrix(y_test, y_pred, normalize='all')

    def plot_metrics(self):
        self.metric_plotter.plot_confusion_matrices(self.confusion_matrices, self._conf_matrix_nrows, self._conf_matrix_ncols)
        self.metric_plotter.plot_metrics_heatmap(self.gen_metrics_df)