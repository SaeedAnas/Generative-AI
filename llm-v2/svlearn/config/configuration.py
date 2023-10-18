import logging as log
import os
from pathlib import Path

import ruamel.yaml as yaml
from pykwalify.core import Core
from ruamel.yaml import CommentedMap

from svlearn.common import SVError, file_exists

BOOTCAMP_ROOT_DIR = os.getenv("BOOTCAMP_ROOT_DIR")
BOOTCAMP_CONFIG_YAML = 'bootcamp-config.yaml'
BOOTCAMP_CONFIG_SCHEMA = 'schema-bootcamp-config.yaml'


class ConfigurationMixin:
    def load_config(self, config_file: Path = None) -> CommentedMap:
        """
        Loads the configuration from a YAML file
        :rtype: an instance of CommentedMap, a map-like object that preserves the order of keys
        :param config_file: path to the YAML configuration file
        :return: configuration object
    """
        default_config_file_dir = BOOTCAMP_ROOT_DIR
        if default_config_file_dir is None:
            default_config_file_dir = str(Path.cwd())

        print(default_config_file_dir)
        if config_file is None:
            log.warning(f'No configuration file specified. Trying the default location')
            # We are going to try the default location
            config_file = f'{default_config_file_dir}/{BOOTCAMP_CONFIG_YAML}'
            log.warning(f'Loading configuration from {config_file} if it exists')

        if not file_exists(config_file):
            errorMsg = f'Configuration file not found: {config_file}'
            log.error(errorMsg)
            raise SVError(errorMsg)
        else:
            log.info(f'Configuration file found: {config_file}')

        # check for schema validation file
        schema_file = f'{default_config_file_dir}/{BOOTCAMP_CONFIG_SCHEMA}'
        if not file_exists(schema_file):
            log.error(f'Configuration schema validation file not found: {schema_file}')
            raise SVError(f'Configuration schema validation file not found: {schema_file}')

        log.info(f'Loading configuration from {config_file}')
        loader = yaml.YAML()

        try:
            with open(config_file, 'r') as config_file:
                config = loader.load(config_file)
                log.info(f'Configuration loaded from {config_file}')

        except FileNotFoundError:
            log.error(f'Configuration file not found: {config_file}')
            raise SVError(f'Configuration file not found: {config_file}')
        except Exception as e:
            log.error(f'Error loading configuration from {config_file}: {e}')
            raise SVError(f'Error loading configuration from {config_file}: {e}')

        # Now, validate the configuration
        # Use pykwalify to validate the YAML against a schema

        #
        # Keep it disabled till we have create a valid schema
        #
        if False:
            try:
                core = Core(source_data=config, schema_files=[schema_file])
                core.validate(raise_exception=True)
                log.info(f'Configuration validated against {schema_file}')
            except Exception as e:
                log.error(f'Error validating configuration against {schema_file}: {e}')
                raise SVError(f'Error validating configuration against {schema_file}: {e}')

        return config


if __name__ == '__main__':
    # before running this, make sure the cwd() is set to the project root.
    mixin = ConfigurationMixin()
    config = mixin.load_config()
    print(config['database'])
    variant_ = config["database"]["variant"]
    print(f'---{variant_}---')
