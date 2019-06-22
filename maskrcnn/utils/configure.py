import yaml
import argparse

class Config(argparse.Namespace):
    """Loads config file. Modified to allow for attribute updating.
    """

    def update(self, path):
        """Update with a new config file. Existing parameters will be overwritten.

        Args:
            path (str): path to the new config file
        """

        with open(path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        for key, value in cfg_dict.items():
            setattr(self, key, value)