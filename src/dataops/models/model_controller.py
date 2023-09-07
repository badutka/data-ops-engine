import pandas as pd
from dataops.models.multiclassification import MultiClfModel
from dataops import settings


def run_multi_clfn():
    """
    https://scikit-learn.org/stable/modules/multiclass.html
    """

    df = pd.read_csv(settings.multiclass.file_name, delimiter=settings.multiclass.file_delimiter)  # read df from memory

    multi_clfn_model = MultiClfModel()  # create multi-classification model instance

    multi_clfn_model.show_data_stats(df)  # show stats info for dataframe

    prep_pipeline, model_pipelines = multi_clfn_model.setup_pipelines(df)  # setup preprocessing and models pipelines

    X_train, X_test, y_train, y_test = multi_clfn_model.get_train_test_X_y(df, settings.multiclass.target)  # split data

    multi_clfn_model.fit_prep_pipeline(prep_pipeline, X_train, y_train)  # fit preprocessing pipeline on TRAINING data

    X_train_prep, X_test_prep = multi_clfn_model.preprocess_via_pipeline(prep_pipeline, X_train, X_test)  # use prep pipeline to transform X_train and X_test

    fit_models = multi_clfn_model.fit_model_pipelines(X_train_prep, y_train, model_pipelines)  # fit model pipelines on TRAINING data

    multi_clfn_model.measure_model_perf(fit_models, X_test_prep, y_test)  # plot metrics of model performence
