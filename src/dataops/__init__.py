import yaml
import os
import logging
import coloredlogs

# Global variable to store the settings
SETTINGS = None

# Local variable to store the settings file name
SETTINGS_FILE_NAME = "settings.yaml"


# Function to load the settings
def load_settings():
    # todo: https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py
    global SETTINGS
    current_file_path = os.path.abspath(__file__)
    file_path = os.path.join(os.path.dirname(current_file_path), SETTINGS_FILE_NAME)
    with open(file_path, "r") as file:
        SETTINGS = yaml.safe_load(file)



# Function to get the settings
def get_settings():
    if SETTINGS is None:
        load_settings()
    return SETTINGS


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


get_settings()
setup_logger()
