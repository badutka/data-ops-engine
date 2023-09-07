import yaml
import os
import logging
import coloredlogs
from dataclasses import dataclass
from pydantic import BaseModel
import typing

# lib version
__version__ = "0.0.1"


def load_settings():
    """
    Read root:settings.yaml file into a dictionary
    https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py

    Returns:
        dict: settings loaded from yaml file
    """
    current_file_path = os.path.abspath(__file__)
    file_path = os.path.join(os.path.dirname(current_file_path), "settings.yaml")

    with open(file_path, "r") as file:
        settings = yaml.safe_load(file)

    return settings


class MulticlassSettings(BaseModel):
    """
    Pydantic class to store and validate settings for multi-classification models
    """
    file_name: str
    file_delimiter: typing.Union[str, None]
    target: str
    max_nunique_for_column: typing.Union[int, None]

    rfecv: typing.Union[int, None]
    metric_average: str

    corr_heatmap: typing.Union[str, None]
    assoc_heatmap: typing.Union[str, None]

    assoc_plot_font: float
    assoc_plot_width: float
    assoc_plot_height: float


class CommonSettings(BaseModel):
    """
    Pydantic class to store and validate common settings
    """
    parameters_file_name: str


class Settings(BaseModel):
    """
        Pydantic class to gather all lower level settings into a single object
    """
    common: CommonSettings
    multiclass: MulticlassSettings


settings_data = load_settings()
settings = Settings(**settings_data)


def setup_logger(logger_name="dataops-logger", file_name="logs.log", loger_level='DEBUG'):
    """
    Create a logger object with two handlers - console and file.

    Args:
        logger_name (str): name of the logger object used to retrieve throughout the library (only one instance per name is created)
        file_name (str): name of the file messages are logged into
        loger_level (str): level at which logger starts to acknowledge messages

    Returns (None): None

    """
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(file_name)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    coloredlogs.install(
        level=loger_level,
        logger=logger,
        level_styles={
            'debug': {'color': 'blue'},
            'info': {'color': 'green'},
            'warning': {'color': 'yellow'},
            'error': {'color': 'red'},
            'critical': {'color': 'red', 'bold': True}
        })


setup_logger()
