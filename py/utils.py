import json
import logging
import logging.config
import os
from pathlib import Path

import pkg_resources
import yaml


def load_config(config_file):
    with open(config_file) as config_file:
        return json.load(config_file)


def get_module_version(module):
    return pkg_resources.get_distribution(module).version


def setup_logging(module, level=logging.DEBUG):
    logging_conf_file = os.path.join(Path(__file__).parent.parent.absolute(), "conf", "logging.yml")
    with open(logging_conf_file, 'r') as f_conf:
        dict_conf = yaml.load(f_conf, Loader=yaml.FullLoader)
    logging.config.dictConfig(dict_conf)
    logger = logging.getLogger(module)
    logger.setLevel(level)
    return logger


log = setup_logging(__name__)
log.debug("test debug logging")
