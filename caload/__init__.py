import os


from caload import base
from caload import sqltables as sql

__all__ = ['base', 'sql', 'digest', 'open_analysis', 'less', 'lessequal', 'equal', 'greaterequal', 'greater']


def open_analysis(analysis_path: str, mode=base.Mode.analyse) -> base.Analysis:

    meta_path = f'{analysis_path}/metadata.db'
    if not os.path.exists(meta_path):
        raise ValueError(f'Path {meta_path} not found')

    summary = base.Analysis(analysis_path, mode=mode)

    return summary


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


if __name__ == '__main__':

    # path = '../../../../TEMP_testdata/default_orientation/analysis01'
    # analysis = open_analysis(path)

    analysis_path = 'C:/Users/thladnik/TEMP'
    analysis_filename = 'analysis01'
    analysis = open_analysis(os.path.join(analysis_path, analysis_filename))

    # Filter examples
    analysis.rois(greater('s2p/radius', 1.))

    print('Fin')