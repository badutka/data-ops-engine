import yaml
import os

# Global variable to store the settings
SETTINGS = None

# Local variable to store the settings file name
SETTINGS_FILE_NAME = "settings.yaml"


# Function to load the settings
def load_settings():
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


get_settings()
