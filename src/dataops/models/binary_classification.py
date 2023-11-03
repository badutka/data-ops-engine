import pandas as pd
from logging import getLogger

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline

from dataops.utils.timing import time, strftime
from dataops.custom_types.custom_types import ParamsDict, ColumnTransformer
# from dataops.stats.analyzers import get_correlation_numerical, plot_correlation_numerical, get_association, plot_association
from dataops.feature_engineering import feature_engineering
#from dataops.metrics.generic_metrics import get_metrics_bc, plot_confusion_matrices, plot_metrics_heatmap
from dataops.utils.utils import setup_param_grid, pipeline_to_html, read_yaml#, drop_columns, get_df_details
from dataops.Exceptions import ModelInputsNotProvidedWarning


def binary_classification():
    # Display all columns in pandas DF
    pd.set_option('display.max_columns', None)

    df, target, columns, enc_columns = data_preparation()

    statistical_analysis(df, columns)

    df = drop_columns(df=df, columns=columns, keep='True')

    get_df_details(df)

    # Create the pipeline with ColumnTransformer for encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(drop='first'), enc_columns)
        ],
        remainder='passthrough'  # Handle the remaining columns as they are (numeric features)
    )

    # Define the classifiers to use
    classifiers = {
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'XGBClassifier': XGBClassifier()
    }

    # Train test split
    X_train, X_test, y_train, y_test = feature_engineering.get_train_test_X_y(df, target, test_size=0.2, random_state=42)

    # Read models' parameters
    models_params = read_yaml("parameters.yaml")

    # Fit data to pipelines
    models = fit_pipelines(classifiers, preprocessor, models_params, X_train, y_train)

    metrics_df, confusion_matrices = get_metrics_bc(models, X_test, y_test)

    plot_confusion_matrices(confusion_matrices, 1, 5)

    plot_metrics_heatmap(metrics_df)

    # sklearn_bc.run_classification_pipeline(df=df, target=target, classifiers=classifiers, preprocessor=preprocessor, enc_columns=enc_columns, resampler=None)


def data_preparation():
    # Read Dataset CSV file
    df = pd.read_csv(r"C:\Users\tooba\Desktop\Customer-Churn-Records.csv")

    # Special case
    df["Products"] = df["NumOfProducts"].apply(lambda x: "monoproduct" if x < 2 else "multiproduct")
    df = df.drop("NumOfProducts", axis=1)

    # Set Target variable
    target = "Exited"

    # Set columns to be dropped (or kept)
    # columns = ["RowNumber", "CustomerId", "Surname"]
    columns = ["Age", "Balance", "Exited", "Geography", "Gender", "IsActiveMember", "Products"]

    # Set columns to be encoded among all kept columns
    enc_columns = ['Geography', 'Gender', "Products"]

    return df, target, columns, enc_columns


def statistical_analysis(df, columns):
    # Get correlation between numerical variables
    df_corr = get_correlation_numerical(df=df)

    # Plot the numerical correlations matrix via heatmap
    plot_correlation_numerical(df_corr=df_corr)

    # Prep data to calculate associations between categorical features
    df_assoc = drop_columns(df=df, columns=['Surname'])  # drop unwanted columns# todo: test drop=None
    df_assoc = feature_engineering.encode_data(df=df_assoc, encoder='ohe', enc_columns=df_assoc.select_dtypes(include='object').columns.tolist())  # exclude=[np.number]

    # Calculate and plot associations between categorical features
    assoc_cat = get_association(df_assoc, method=['chi2', 'cramers-v'])
    plot_association(association_df=assoc_cat['chi2'])
    plot_association(association_df=assoc_cat['cramers-v'], fmt='.2%')

    df = drop_columns(df=df, columns=columns, keep='True')

    get_df_details(df)


def fit_pipelines(classifiers, preprocessor: ColumnTransformer, models_params: ParamsDict, X_train: pd.DataFrame, y_train: pd.Series):
    models = dict()

    for name, clf in classifiers.items():

        start_time = time.perf_counter()

        logger = getLogger('dataops-logger')
        logger.debug(f"Preparing a Pipeline for {name}... .")

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            (name, clf)
        ])

        pipeline_to_html(pipeline=pipeline, display='diagram', write='a')

        param_grid = setup_param_grid(models_params, name)

        if not param_grid:
            logger.warning(ModelInputsNotProvidedWarning(f"Parameter grid for {name} not provided. Running CV with libs' default parameters."))

        model = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv=models_params[name]['CV'], scoring='accuracy')

        logger.debug(f"Fitting data using {name}... .")

        models[name] = model.fit(X_train, y_train).best_estimator_  # grid_rf.best_score_ | grid_rf.best_params_

        logger.debug(f"Fitting for {name} completed successfully.")
        # logger.debug(f"Model parameters: \n{json.dumps(models[name][name].get_params(), indent=2)}")
        logger.debug(f"Model parameters: {models[name][name].get_params()}")
        logger.debug(f"Time elapsed: {strftime(time.perf_counter() - start_time)}")

    return models
