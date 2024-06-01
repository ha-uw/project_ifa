import os
import yaml
from dataclasses import dataclass


@dataclass
class ConfigLoader:
    def load_config(self, config_path):
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"The configuration file {config_path} does not exist."
            )
        self.config_path = config_path
        with open(self.config_path, "r") as file:
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
