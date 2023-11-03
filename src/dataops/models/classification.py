import pandas as pd
import logging
from typing import Union

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV

from dataops import settings
from dataops import Exceptions
from dataops import messages
from dataops.data_manager import dataframe
from dataops.utils.utils import setup_param_grid, read_yaml, pipeline_to_html
from dataops.utils.timing import timefunc
from dataops.feature_engineering import feature_engineering
from dataops.feature_engineering import transformers
from dataops.metrics.analyzers import MetricAnalyzer
from dataops.stats.analyzers import CorrelationAnalyzer
from dataops.graphs.histogram_plots.feature_histograms import feature_desc_hist_array
import matplotlib.pyplot as plt


class MultiClfModel():
    def __init__(self):
        self.parameters_file_name = settings.common.parameters_file_name
        self.logger_name = settings.common.logger_name

        self.target = settings.multiclass.target
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
        self.num_cols, self.cat_cols = dataframe.get_num_and_cat_feats(df=self.df, target=self.target)

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
        num_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', MinMaxScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first')),
            ('col_drop', transformers.CategoricalFeatDropTransformer(correlation_threshold=0.1))
        ])

        col_trans = ColumnTransformer(transformers=[
            ('num_pipeline', num_pipeline, self.num_cols),
            ('cat_pipeline', cat_pipeline, self.cat_cols)
        ],
            remainder='drop',
            n_jobs=-1)

        # pp_ohe = ColumnTransformer(
        #     transformers=[
        #         ('encoder', OneHotEncoder(drop='first'), self.df.columns[~dataframe.is_numeric(self.df)]),  # todo: drop target
        #         # ('columnDropper', transformers.ColumnDropperTransformer([]), self.df.columns.drop(self.target)),
        #     ],
        #     remainder='passthrough'  # Handle the remaining columns as they are (numeric features)
        # )
        #
        # pp_drop = ColumnTransformer(
        #     transformers=[
        #         ('columnDropper', transformers.ColumnDropperTransformer(columns_to_drop=[3], correlation_threshold=0.1), dataframe.get_features_index(self.df.columns, self.target)),
        #         # 'encoder__higher_yes', 'remainder__age', 'remainder__Medu'
        #     ],
        #     remainder='passthrough'
        # )

        # rfe = RFECV(SVR(kernel="linear"), step=1, cv=settings.multiclass.rfecv)
        rfe = RFE(SVR(kernel="linear"), step=1)

        # self.feature_eng_pipeline = Pipeline(steps=[('col_trans', col_trans), ('rfe', rfe)])
        self.feature_eng_pipeline = Pipeline(steps=[('col_trans', col_trans)])

        classifiers = {
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
        }

        for name, clf in classifiers.items():
            # self.model_pipelines[name] = Pipeline(steps=[
            #     ('col_trans', col_trans),
            #     (name, clf)
            # ])

            self.model_pipelines[name] = Pipeline(steps=[(name, clf)])

    def get_train_test_X_y(self, test_size=settings.multiclass.test_size, random_state=settings.multiclass.random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = feature_engineering.get_train_test_X_y(df=self.df, target=self.target, test_size=test_size, random_state=random_state)  # split data

    def run_preprocessing_pipeline(self):
        pass

    def run_feature_eng_pipeline(self, impute_target=False):
        if impute_target:
            y_imputer = SimpleImputer(strategy='most_frequent')
            self.y_train = y_imputer.fit_transform(self.y_train.values.reshape(-1, 1)).ravel()
            self.y_test = y_imputer.transform(self.y_test.values.reshape(-1, 1)).ravel()

        self.feature_eng_pipeline.fit(self.X_train, self.y_train)
        self.X_train_feat = self.feature_eng_pipeline.transform(self.X_train)
        self.X_test_feat = self.feature_eng_pipeline.transform(self.X_test)

        # f1 = self.feature_eng_pipeline['col_trans'].transformers_[1][1]['col_drop'].get_feature_names_out()
        # print(f1)  # if 32 features pre-ohe and 41 features post-ohe, assuming 18 are to drop, 23 remain.

    def fit_model_pipelines(self):
        if self.X_train_feat is not None and self.X_test_feat is not None:
            X_train = self.X_train_feat
        else:
            X_train = self.X_train
        for name, model_pipeline in self.model_pipelines.items():
            self.__fit_model_pipeline(X_train, self.y_train, name, model_pipeline)

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

        df_assoc = feature_engineering.encode_data(df=self.df, encoder='ohe', enc_columns=dataframe.get_cat_columns(self.df))

        if self.assoc_type_heatmap:
            self.correlation_analyzer.get_association(df_assoc)
            self.correlation_analyzer.plot_associations(assoc_type=self.assoc_type_heatmap)

        if True:
            self.correlation_analyzer.calc_point_biserial_corr(df_assoc, self.df[self.target])
            self.correlation_analyzer.plot_point_biserial()

    @timefunc(logger=logging.getLogger(settings.common.logger_name))
    def __fit_model_pipeline(self, X_train, y_train, name, model_pipeline):
        self.logger.debug(messages.MULTI_CLF_FIT_MSG_001.format(name))

        pipeline_to_html(pipeline=model_pipeline, display='diagram', write='a')

        param_grid = setup_param_grid(self.models_params, name)

        if not param_grid:
            self.logger.warning(Exceptions.ModelInputsNotProvidedWarning(messages.MULTI_CLF_FIT_MSG_002.format(name)))

        model = RandomizedSearchCV(model_pipeline, param_distributions=param_grid, cv=self.models_params[name]['CV'], scoring=self.scoring, verbose=True)

        self.logger.debug(messages.MULTI_CLF_FIT_MSG_003.format(name))

        self.fit_models[name] = model.fit(X_train, y_train).best_estimator_  # grid_rf.best_score_ | grid_rf.best_params_
        self.logger.debug(messages.MULTI_CLF_FIT_MSG_004.format(name))
        # logger.debug(f"Model parameters: \n{json.dumps(models[name][name].get_params(), indent=2)}")
        self.logger.debug(messages.MULTI_CLF_FIT_MSG_005.format(self.fit_models[name][name].get_params()))
