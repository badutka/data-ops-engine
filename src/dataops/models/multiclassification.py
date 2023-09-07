import pandas as pd
import numpy as np
import logging

from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import SVR

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from dataops import settings
from dataops import Exceptions
from dataops.utils.utils import get_df_details, drop_columns, setup_param_grid, read_yaml, is_numeric, is_below_nunique_limit
from dataops.preprocessing import correlations  # import get_correlation_numerical, plot_correlation_numerical, get_association, plot_association
from dataops.preprocessing import feature_engineering
from dataops.utils.timing import time, strftime
from dataops.metrics.generic_metrics import get_metrics_bc, plot_confusion_matrices, plot_metrics_heatmap


class MultiClfModel:
    def __init__(self):
        self.max_unique_for_cat = settings.multiclass.max_nunique_for_column
        self.corr_heatmap = settings.multiclass.corr_heatmap
        self.assoc_heatmap = settings.multiclass.assoc_heatmap
        self.metric_average = settings.multiclass.metric_average
        self.parameters_file_name = settings.common.parameters_file_name

        self.models_params = read_yaml(self.parameters_file_name)

    def setup_pipelines(self, df):
        pp_ohe = ColumnTransformer(
            transformers=[
                ('encoder', OneHotEncoder(drop='first'), df.columns[~is_numeric(df)])
            ],
            remainder='passthrough'  # Handle the remaining columns as they are (numeric features)
        )

        # rfe = RFECV(SVR(kernel="linear"), step=1, cv=settings.multiclass.rfecv)
        rfe = RFE(SVR(kernel="linear"), step=1)

        prep_pipeline = Pipeline(steps=[('pp', pp_ohe), ('drt', rfe)])

        model_pipelines = {}

        classifiers = {
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
        }

        for name, clf in classifiers.items():
            model_pipelines[name] = Pipeline(steps=[(name, clf)])

        return prep_pipeline, model_pipelines

    def get_train_test_X_y(self, df, target, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = feature_engineering.get_train_test_X_y(df=df, target=target, test_size=test_size, random_state=random_state)  # split data
        return X_train, X_test, y_train, y_test

    def fit_prep_pipeline(self, prep_pipeline, X_train, y_train):
        prep_pipeline.fit(X_train, y_train)

    def preprocess_via_pipeline(self, prep_pipeline, X_train, X_test):
        X_train_prep = prep_pipeline.transform(X_train)
        X_test_prep = prep_pipeline.transform(X_test)
        return X_train_prep, X_test_prep

    def fit_model_pipelines(self, X_train, y_train, model_pipelines):
        fit_models = dict()

        for name, model_pipeline in model_pipelines.items():

            start_time = time.perf_counter()

            logger = logging.getLogger('dataops-logger')
            logger.debug(f"Preparing the fitting process for {name} pipeline... .")

            # pipeline_to_html(pipeline=pipeline, display='diagram', write='a')

            param_grid = setup_param_grid(self.models_params, name)

            if not param_grid:
                logger.warning(Exceptions.ModelInputsNotProvidedWarning(f"Parameter grid for {name} not provided. Running CV with libs' default parameters."))

            model = RandomizedSearchCV(model_pipeline, param_distributions=param_grid, cv=self.models_params[name]['CV'], scoring='accuracy')

            logger.debug(f"Fitting data using {name}... .")

            fit_models[name] = model.fit(X_train, y_train).best_estimator_  # grid_rf.best_score_ | grid_rf.best_params_

            logger.debug(f"Fitting for {name} completed successfully.")
            # logger.debug(f"Model parameters: \n{json.dumps(models[name][name].get_params(), indent=2)}")
            logger.debug(f"Model parameters: {fit_models[name][name].get_params()}")
            logger.debug(f"Time elapsed: {strftime(time.perf_counter() - start_time)}")

        return fit_models

    def measure_model_perf(self, fit_models, X_test, y_test):
        metrics_df, confusion_matrices = get_metrics_bc(fit_models, X_test, y_test, average=self.metric_average)
        plot_confusion_matrices(confusion_matrices, 1, 2)
        plot_metrics_heatmap(metrics_df)

    def show_data_stats(self, df):
        if self.max_unique_for_cat:
            df = df.loc[:, lambda x: (is_numeric(x) | (~is_numeric(x) & is_below_nunique_limit(x)))]

        if self.corr_heatmap:
            df_corr = correlations.get_correlation_numerical(df=df, method=self.corr_heatmap)
            correlations.plot_correlation_numerical(df_corr=df_corr)

        if self.assoc_heatmap:
            df_assoc = feature_engineering.encode_data(df=df, encoder='ohe', enc_columns=df.select_dtypes(include='object').columns.tolist())  # exclude=[np.number]
            assoc_cat = correlations.get_association(df_assoc, method=self.assoc_heatmap)
            correlations.plot_association(association_df=assoc_cat[self.assoc_heatmap])
