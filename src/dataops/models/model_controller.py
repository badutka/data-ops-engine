import pandas as pd
from dataops.models.classification import MultiClfModel
from dataops import settings


def run_multi_clfn():
    """
    https://scikit-learn.org/stable/modules/multiclass.html
    """

    # df = pd.read_csv(settings.multiclass.file_name, delimiter=settings.multiclass.file_delimiter)  # read df from memory

    multi_clfn_model = MultiClfModel()  # create multi-classification model instance

    # multi_clfn_model.show_data_stats()  # show stats info for dataframe

    multi_clfn_model.setup_pipelines()  # setup preprocessing and models pipelines

    multi_clfn_model.get_train_test_X_y()  # split data

    multi_clfn_model.run_feature_eng_pipeline()  # fit preprocessing pipeline on TRAINING data

    multi_clfn_model.fit_model_pipelines()  # fit model pipelines on TRAINING data

    multi_clfn_model.measure_model_perf()  # plot metrics of model performance
