import yaml
import os
import logging
import coloredlogs


class SettingsSingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


# Function to load the settings
def load_settings():
    # todo: https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py
    current_file_path = os.path.abspath(__file__)
    file_path = os.path.join(os.path.dirname(current_file_path), "settings.yaml")
    with open(file_path, "r") as file:
        settings = yaml.safe_load(file)
    return settings


class Settings(metaclass=SettingsSingletonMeta):
    __settings: dict = load_settings()

    @staticmethod
    def get(name):
        return Settings.__settings[name]

    @staticmethod
    def set(name, value):
        raise NotImplementedError("Settings values for configuration is not yet permitted.")


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

    # By default the install() function installs a handler on the root logger,
    # this means that log messages from your code and log messages from the
    # libraries that you use will all show up on the terminal.
    # coloredlogs.install(level='DEBUG')

    # If you don't want to see log messages from libraries, you can pass a
    # specific logger object to the install() function. In this case only log
    # messages originating from that logger will show up on the terminal.
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

    return logger


setup_logger()
settings = Settings()
