
from collections import OrderedDict
import datetime
from os.path import dirname, abspath, join, expanduser, isfile, exists
from os import environ, makedirs
import pytz
import re
from socket import gethostname

def get_root_path():
    return dirname(abspath(__file__))


def get_data_path():
    return join(get_root_path(), 'data')


def get_configs_path():
    return join(get_root_path(), 'runConfigs')


def get_models_path():
    return join(get_root_path(), 'models')

def get_saved_path():
    return join(get_root_path(), 'saved') # cached torch.datasets, splitted according to a config