import json
import os
from loguru import logger

class Config:
    static_class = False
    table = {}

    def __init__(self):
        pass

    def load_from_json(self, config_path=False):
        if not config_path:
            if os.path.isfile('config.json'):
                config_path = 'config.json'
            else:
                config_path = './conf/config.json'
        with open(config_path) as config_file:
            self.table = json.load(config_file)

    def get_value(self, key, default_value=""):
        if self.table.get(key):
            return self.table[key]
        return default_value

    @staticmethod
    def get(key='', default_value=""):
        #up_key=
        if os.getenv(key.upper()):
            logger.info(f'key: {key.upper()}, getenv: {os.getenv(key.upper())}')
            return os.getenv(key.upper())
        if not Config.static_class:
            Config.static_class = Config()
            Config.static_class.load_from_json()
        return Config.static_class.get_value(key, default_value)
