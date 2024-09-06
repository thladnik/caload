from __future__ import annotations

from caload.analysis import open_analysis

__author__ = 'Tim Hladnik'
__copyright__ = 'Copyright 2024'
__license__ = 'GPLv3'
__version__ = '0.1.0'
__maintainer__ = 'Tim Hladnik'
__email__ = 'tim.github@hladnik.de'
__status__ = 'Development'


__all__ = ['sqltables', 'entities', 'analysis', 'filter', 'open_analysis', 'utils','s2p_autorun',
           'less', 'lessequal', 'equal', 'greaterequal', 'greater', 'is_true', 'is_false']

default_bulk_format = 'hdf5'
default_max_blob_size = 2 ** 20  # 2^20 ~ 1MB


def less(name, value):
    return name, '<', value


def lessequal(name, value):
    return name, '<=', value


def equal(name, value):
    return name, '==', value


def greaterequal(name, value):
    return name, '>=', value


def greater(name, value):
    return name, '>', value


def is_true(name):
    return name, '==', True


def is_false(name):
    return name, '==', False


def has(name):
    return name, 'has', None


def has_not(name):
    return name, 'hasnot', None


if __name__ == '__main__':
    pass
