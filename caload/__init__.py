from __future__ import annotations

from caload.analysis import open_analysis

__all__ = ['sqltables', 'entities', 'digest', 'analysis', 'open_analysis', 'utils','s2p_autorun',
           'less', 'lessequal', 'equal', 'greaterequal', 'greater', 'is_true', 'is_false']


def less(name, value):
    return name, 'l', value


def lessequal(name, value):
    return name, 'le', value


def equal(name, value):
    return name, 'e', value


def greater(name, value):
    return name, 'g', value


def greaterequal(name, value):
    return name, 'ge', value


def is_true(name):
    return name, 'e', True


def is_false(name):
    return name, 'e', False


if __name__ == '__main__':
    pass
