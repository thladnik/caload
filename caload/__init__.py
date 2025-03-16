from __future__ import annotations

import argparse
import os

from caload.analysis import open_analysis
from caload.schemata import ca_imaging_s2p_vxpy

__author__ = 'Tim Hladnik'
__copyright__ = 'Copyright 2024'
__license__ = 'GPLv3'
__version__ = '0.1.0'
__maintainer__ = 'Tim Hladnik'
__email__ = 'tim.github@hladnik.de'
__status__ = 'Development'

__all__ = ['sqltables', 'entities', 'analysis', 'filter', 'files', 'open_analysis', 's2p_autorun', 'ca_imaging_s2p_vxpy']

default_bulk_format = 'hdf5'
default_max_blob_size = 2 ** 20  # 2^20 ~ 1MB


def run():

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, choices=['create', 'delete', 'update'],
                        help='Command to run')
    parser.add_argument('-p', '--path', type=str, default=os.getcwd(),
                        help=f'Path to the analysis folder on which to run command on (defaults to "{os.getcwd()}")')
    parser.add_argument('-s', '--schema', type=str, default=None,
                        help='Import path to entity schema which should be used for analysis creation')
    parser.add_argument('--dbhost', type=str, default=None,
                        help='Database host address')
    parser.add_argument('--dbuser', type=str, default=None,
                        help='Database username')
    parser.add_argument('--dbname', type=str, default=None,
                        help='Database schema name for given analysis')
    parser.add_argument('--dbpassword', type=str, default=None,
                        help='Database password for dbuser')

    # Parse
    args = parser.parse_args()

    # Run appropriate function
    if args.command == 'create':

        if args.schema is None:
            raise AttributeError('No schema provided')

        analysis.create_analysis(analysis_path=args.path, entity_schema=args.schema,
                                 dbhost=args.dbhost, dbname=args.dbname,
                                 dbuser=args.dbuser, dbpassword=args.dbpassword)

    elif args.command == 'delete':
        analysis.delete_analysis(analysis_path=args.path)

    elif args.command == 'update':
        analysis.update_analysis(analysis_path=args.path, )
        print(f'Delete analysis on analysis path {args.path}')
