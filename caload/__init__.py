from __future__ import annotations

from caload.analysis import open_analysis
from caload.schemata import ca_imaging_s2p_vxpy

__author__ = 'Tim Hladnik'
__copyright__ = 'Copyright 2024'
__license__ = 'GPLv3'
__version__ = '0.1.0'
__maintainer__ = 'Tim Hladnik'
__email__ = 'tim.github@hladnik.de'
__status__ = 'Development'

__all__ = ['sqltables', 'entities', 'analysis', 'filter', 'open_analysis', 'utils', 's2p_autorun', 'ca_imaging_s2p_vxpy']

default_bulk_format = 'hdf5'
default_max_blob_size = 2 ** 20  # 2^20 ~ 1MB
