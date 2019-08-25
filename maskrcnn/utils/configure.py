import yaml
import argparse


class Config(argparse.Namespace):
    """Loads config file. Modified to allow for attribute updating.
    """

    def update(self, paths):
        """Update with new config files. Existing parameters will be overwritten.

        Later files take priority.

        Args:
            path (list of str): paths to the new config files
        """
        for path in paths:
            with open(path, 'r') as f:
                cfg_dict = yaml.safe_load(f)

            for key, value in cfg_dict.items():
                setattr(self, key, value)
