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
    # todo: https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py
    current_file_path = os.path.abspath(__file__)
    file_path = os.path.join(os.path.dirname(current_file_path), "settings.yaml")
    with open(file_path, "r") as file:
        settings = yaml.safe_load(file)
    return settings


class MulticlassSettings(BaseModel):
    file_name: str
    file_delimiter: typing.Union[str, None]
    max_nunique_for_column: typing.Union[int, None]


class Settings(BaseModel):
    multiclass: MulticlassSettings


def setup_logger():
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger("dataops-logger")

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler("logs.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    coloredlogs.install(
        level='DEBUG',
        logger=logger,
        level_styles={
            'debug': {'color': 'blue'},
            'info': {'color': 'green'},
            'warning': {'color': 'yellow'},
            'error': {'color': 'red'},
            'critical': {'color': 'red', 'bold': True}
        })


setup_logger()
settings_data = load_settings()
settings = Settings(**settings_data)
