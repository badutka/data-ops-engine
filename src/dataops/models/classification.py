import pandas as pd
import logging
from typing import Union

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from dataops import settings
from dataops import Exceptions
from dataops import messages
from dataops.data_manager import dataframe
from dataops.utils.utils import setup_param_grid, read_yaml
from dataops.utils.timing import timefunc
from dataops.feature_engineering import feature_engineering
from dataops.metrics.analyzers import MetricAnalyzer
from dataops.stats.analyzers import CorrelationAnalyzer


class MultiClfModel():
    def __init__(self):
        self.parameters_file_name = settings.common.parameters_file_name
        self.logger_name = settings.common.logger_name

        self.max_unique_for_cat = settings.multiclass.max_nunique_for_column
        self.corr_heatmap = settings.multiclass.corr_heatmap
        self.assoc_type_heatmap = settings.multiclass.assoc_heatmap
        self.metric_average = settings.multiclass.metric_average
        self.scoring = settings.multiclass.grid_search_scoring
        self.__data_file_name = settings.multiclass.file_name
        self.__data_file_delimiter = settings.multiclass.file_delimiter

        self.logger = logging.getLogger(self.logger_name)
        self.df = pd.read_csv(self.__data_file_name, delimiter=self.__data_file_delimiter)  # read df from memory
        self.models_params = read_yaml(self.parameters_file_name)

        self.X_train: Union[pd.DataFrame, None] = None
        self.X_test: Union[pd.DataFrame, None] = None
        self.y_train: Union[pd.Series, None] = None
        self.y_test: Union[pd.Series, None] = None
        self.X_train_feat: Union[pd.DataFrame, None] = None
        self.X_test_feat: Union[pd.DataFrame, None] = None
        self.feature_eng_pipeline: Union[Pipeline, None] = None
        self.model_pipelines: dict = dict()
        self.fit_models: dict = dict()

        self.correlation_analyzer = CorrelationAnalyzer()
        self.performance_analyzer = MetricAnalyzer()

    def setup_pipelines(self):
        pp_ohe = ColumnTransformer(
            transformers=[
                ('encoder', OneHotEncoder(drop='first'), self.df.columns[~dataframe.is_numeric(self.df)])
            ],
            remainder='passthrough'  # Handle the remaining columns as they are (numeric features)
        )

        # rfe = RFECV(SVR(kernel="linear"), step=1, cv=settings.multiclass.rfecv)
        rfe = RFE(SVR(kernel="linear"), step=1)

        self.feature_eng_pipeline = Pipeline(steps=[('pp', pp_ohe), ('drt', rfe)])

        classifiers = {
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
        }

        for name, clf in classifiers.items():
            self.model_pipelines[name] = Pipeline(steps=[(name, clf)])

    def get_train_test_X_y(self, target, test_size=settings.multiclass.test_size, random_state=settings.multiclass.random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = feature_engineering.get_train_test_X_y(df=self.df, target=target, test_size=test_size, random_state=random_state)  # split data

    def run_preprocessing_pipeline(self):
        pass

    def run_feature_eng_pipeline(self):
        self.feature_eng_pipeline.fit(self.X_train, self.y_train)
        self.X_train_feat = self.feature_eng_pipeline.transform(self.X_train)
        self.X_test_feat = self.feature_eng_pipeline.transform(self.X_test)

    def fit_model_pipelines(self):
        if self.X_train_feat is not None and self.X_test_feat is not None:
            X_train = self.X_train_feat
        else:
            X_train = self.X_train

        for name, model_pipeline in self.model_pipelines.items():
            self.__fit_model_pipeline(X_train, self.y_train, name, model_pipeline)

        return self.fit_models

    def measure_model_perf(self, save_path=None):
        if self.X_train_feat is not None and self.X_test_feat is not None:
            X_test = self.X_test_feat
        else:
            X_test = self.X_test

        self.performance_analyzer.calculate_metrics(self.fit_models, X_test, self.y_test, average=self.metric_average)
        self.performance_analyzer.plot_metrics()

    def show_data_stats(self):
        if self.max_unique_for_cat:
            self.df = self.df.loc[:, lambda x: (dataframe.is_numeric(x) | (~dataframe.is_numeric(x) & dataframe.is_below_nunique_limit(x)))]

        if self.corr_heatmap:
            self.correlation_analyzer.calc_correlation_numerical(df=self.df, method=self.corr_heatmap)
            self.correlation_analyzer.plot_correlations()

        if self.assoc_type_heatmap:
            df_assoc = feature_engineering.encode_data(df=self.df, encoder='ohe', enc_columns=self.df.select_dtypes(include='object').columns.tolist())  # exclude=[np.number]
            self.correlation_analyzer.get_association(df_assoc, method=self.assoc_type_heatmap)
            self.correlation_analyzer.plot_associations(assoc_type=self.assoc_type_heatmap)

    @timefunc(logger=logging.getLogger(settings.common.logger_name))
    def __fit_model_pipeline(self, X_train, y_train, name, model_pipeline):
        self.logger.debug(messages.MULTI_CLF_FIT_MSG_001.format(name))

        # pipeline_to_html(pipeline=pipeline, display='diagram', write='a')

        param_grid = setup_param_grid(self.models_params, name)

        if not param_grid:
            self.logger.warning(Exceptions.ModelInputsNotProvidedWarning(messages.MULTI_CLF_FIT_MSG_002.format(name)))

        model = RandomizedSearchCV(model_pipeline, param_distributions=param_grid, cv=self.models_params[name]['CV'], scoring=self.scoring)

        self.logger.debug(messages.MULTI_CLF_FIT_MSG_003.format(name))

        self.fit_models[name] = model.fit(X_train, y_train).best_estimator_  # grid_rf.best_score_ | grid_rf.best_params_

        self.logger.debug(messages.MULTI_CLF_FIT_MSG_004.format(name))
        # logger.debug(f"Model parameters: \n{json.dumps(models[name][name].get_params(), indent=2)}")
        self.logger.debug(messages.MULTI_CLF_FIT_MSG_005.format(self.fit_models[name][name].get_params()))
