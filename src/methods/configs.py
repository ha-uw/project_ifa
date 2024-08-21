"""
configs.py

Module for loading and populating configuration data from YAML files.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ConfigLoader:
    """
    A class for loading and populating configuration data.

    Methods:
    - load_config(config_path): Loads the configuration data from the specified file path.

    Attributes:
    - config_path: The file path of the configuration file.
    """

    def load_config(self, config_path):
        if not Path(config_path).is_file():
            raise FileNotFoundError(
                f"The configuration file {config_path} does not exist."
            )
        self.config_path = config_path
        with open(self.config_path) as file:
            self._data = yaml.safe_load(file)
        self._populate_attrs(self._data)

    def _populate_attrs(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                nested_cl = ConfigLoader()
                nested_cl._populate_attrs(value)
                setattr(self, key, nested_cl)
            else:
                setattr(self, key, value)
