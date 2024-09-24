from __future__ import annotations

import logging
import os
import pprint
import sys
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import yaml
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Query, Session

import caload
from caload.entities import *
from caload.filter import *
from caload.sqltables import *

__all__ = ['Analysis', 'Mode', 'open_analysis']


class Mode(Enum):
    create = 1
    analyse = 2


log = logging.getLogger(__name__)


def create_analysis(analysis_path: str, data_root: str, digest_fun: Callable, schema: List[Type[Entity]],
                    dbhost: str = None, dbname: str = None, dbuser: str = None, dbpassword: str = None,
                    bulk_format: str = None, compression: str = 'gzip', compression_opts: Any = None,
                    shuffle_filter: bool = True, max_blob_size: int = None,
                    **kwargs) -> Analysis:

    analysis_path = Path(analysis_path).as_posix()

    if os.path.exists(analysis_path):
        raise FileExistsError(f'Analysis path {analysis_path} already exists')

    # Get connection parameters

    if dbhost is None:
        dbname = input(f'MySQL host name [default: "localhost"]: ')
        if dbname == '':
            dbname = 'localhost'

    if dbname is None:
        default_dbname = analysis_path.split('/')[-1]
        dbname = input(f'New schema name on host "{dbhost}" [default: "{default_dbname}"]: ')
        if dbname == '':
            dbname = default_dbname

    if dbuser is None:
        dbuser = input(f'User name for schema "{dbname}" [default: caload_user]: ')
        if dbuser == '':
            dbuser = 'caload_user'

    if dbpassword is None:
        import getpass
        dbpassword = getpass.getpass(f'Password for user {dbuser}: ')

    # Set bulk format type
    if bulk_format is None:
        bulk_format = caload.default_bulk_format

    # Set max blob size
    if max_blob_size is None:
        max_blob_size = caload.default_max_blob_size

    print(f'Create new analysis at {analysis_path}')

    # Create schema
    print(f'> Create database {dbname}')
    engine = create_engine(f'mysql+pymysql://{dbuser}:{dbpassword}@{dbhost}')
    with engine.connect() as connection:
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS {dbname}'))
    engine.dispose()

    # Create engine for dbname
    print('> Select database')
    engine = create_engine(f'mysql+pymysql://{dbuser}:{dbpassword}@{dbhost}/{dbname}')

    # Create tables
    print('> Create tables')
    SQLBase.metadata.create_all(engine)

    # Create hierarchy
    print('>> Set up type hierarchy')
    hierarchy = parse_hierarchy(schema)
    print('---')
    pprint.pprint(hierarchy)
    print('---')

    with Session(engine) as session:
        def _create_entity_type(_hierarchy: Dict[str,  ...], parent_row: Union[EntityTypeTable, None]):
            for name, children in _hierarchy.items():
                row = EntityTypeTable(name=name, parent=parent_row)
                session.add(row)
                _create_entity_type(children, row)

        _create_entity_type(hierarchy, None)
        session.commit()

    # # Create procedures
    # with engine.connect() as connection:
    #     connection.execute(text(
    #         """
    #         CREATE PROCEDURE insert_attribute_name(IN attribute_name VARCHAR(255))
    #         BEGIN
    #             IF NOT EXISTS (SELECT 1 FROM attributes WHERE name = attribute_name) THEN
    #                 INSERT INTO attributes (name) VALUES (attribute_name);
    #             END IF;
    #         END;
    #         """))

    print('> Create analysis folder')
    # Create analysis data folder
    os.mkdir(analysis_path)

    # Set config data
    config = {'data_root': data_root,
              'dbhost': dbhost,
              'dbname': dbname,
              'dbuser': dbuser,
              'dbpassword': dbpassword,
              'bulk_format': bulk_format,
              'max_blob_size': max_blob_size,
              'compression': compression,
              'compression_opts': compression_opts,
              'shuffle_filter': shuffle_filter}

    # Create config file
    with open(os.path.join(analysis_path, 'caload.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    # Create analysis
    print('> Open analysis folder')
    analysis = Analysis(analysis_path, mode=Mode.create, **kwargs)

    # Start digesting recordings
    print('> Start data digest')
    digest_fun(analysis)

    return analysis


def parse_hierarchy(entity_types: List[Type]) -> Dict[str, Dict[str, ...]]:

    # Parse into flat dictionary of child:parent relations between entity types
    flat = {}
    for _type in entity_types:
        parent = getattr(_type, 'parent_type', None)
        if parent is not None:
            parent = parent.__name__
        flat[_type.__name__] = parent

    # Creates nested representation of hierarchy
    def build_nested_dict(flat):
        def nest_key(key):
            nested = {}
            for k, v in flat.items():
                if v == key:
                    nested[k] = nest_key(k)
            return nested

        nested_dict = {}
        for key, value in flat.items():
            if value is None:
                nested_dict[key] = nest_key(key)

        return nested_dict

    # print(flat)
    # pprint.pprint(build_nested_dict(flat))

    return build_nested_dict(flat)


def update_analysis(analysis_path: str, digest_fun: Callable, **kwargs):
    print(f'Open existing analysis at {analysis_path}')

    analysis = open_analysis(analysis_path, mode=Mode.create, **kwargs)

    # Start digesting recordings
    digest_fun(analysis)


def delete_analysis(analysis_path: str):

    # Convert
    analysis_path = Path(analysis_path).as_posix()

    if not os.path.exists(analysis_path):
        raise ValueError(f'Analysis path {analysis_path} does not exist')

    # Load config
    config_path = os.path.join(analysis_path, 'caload.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'No configuration file found at {config_path}. Not a valid analysis path')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print('-----\nCAUTION - DANGER ZONE\n-----')
    print(f'Are you sure you want to delete the analysis {analysis_path}?')
    print('This means that all data for this analysis will be lost!')
    print('If you are sure, type the databse schema name ("dbname" in caload.yaml) to verify')

    print('-----\nCAUTION - DANGER ZONE\n-----')
    verification_str = input('Schema name: ')

    if verification_str != config['dbname']:
        print('Abort.')

    print(f'Deleting all data for analysis {analysis_path}')

    print('> Remove directories and files')

    # Use pathlib recursive unlinker by mitch from https://stackoverflow.com/a/49782093
    def rmdir(directory, counter):
        directory = Path(directory)
        for item in directory.iterdir():
            if item.is_dir():
                counter = rmdir(item, counter)
            else:
                item.unlink()

        if counter % 10 == 0:
            # print(' ' * 500, end='\n')
            sys.stdout.write(f'\rRemove {directory.as_posix()}')
        counter += 1
        directory.rmdir()

        return counter

    # Delete tree
    rmdir(analysis_path, 0)
    print('')
    print('> Drop schema')
    engine = create_engine(f'mysql+pymysql://{config["dbuser"]}:{config["dbpassword"]}@{config["dbhost"]}')
    with engine.connect() as connection:
        connection.execute(text(f'DROP SCHEMA IF EXISTS {config["dbname"]}'))
        connection.commit()

    print(f'Successfully deleted analysis {analysis_path}')


def open_analysis(analysis_path: str, **kwargs) -> Analysis:

    print(f'Open analysis {analysis_path}')
    summary = Analysis(analysis_path, **kwargs)

    return summary


class Analysis:
    mode: Mode
    analysis_path: str
    sql_engine: Engine
    session: Session
    write_timeout = 3.  # s
    lazy_init: bool
    echo: bool

    # Entity type name -> entity type PK
    _entity_type_pk_map: Dict[str, int]
    # Child entity type -> parent entity type
    _entity_hierachy_map: Dict[str, Union[str, None]]

    def __init__(self, path: str, mode: Mode = Mode.analyse, lazy_init: bool = False, echo: bool = False):

        # Set mode
        self.mode = mode
        self.is_create_mode = mode is Mode.create
        self._entity_type_pk_map = {}
        self._entity_hierachy_map = {}

        # Set path as posix
        self._analysis_path = Path(path).as_posix()

        # Load config
        config_path = os.path.join(self.analysis_path, 'caload.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'No configuration file found at {config_path}. Not a valid analysis path')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set optionals
        self.lazy_init = lazy_init
        self.echo = echo

        # Open SQL session by default
        if not lazy_init:
            self.open_session()

    def _load_entity_hierarchy(self):
        entity_type_rows = self.session.query(EntityTypeTable).order_by(EntityTypeTable.pk).all()

        # Create name to pk map
        self._entity_type_pk_map = {str(row.name): int(row.pk) for row in entity_type_rows}

        # Create hierarchy map
        self._entity_hierachy_map = {}
        for row in entity_type_rows:
            parent = None
            if row.parent is not None:
                parent = str(row.parent.name)

            self._entity_hierachy_map[str(row.name)] = parent

    def __repr__(self):
        return f"Analysis('{self.analysis_path}')"

    def add_entity(self, entity_type: Type[Entity], entity_id: str, parent_entity: Entity = None):

        self.open_session()

        if not isinstance(entity_type, str):
            entity_type_name = entity_type.__name__
        else:
            entity_type_name = entity_type

        # Check if type is known
        if entity_type_name not in self._entity_type_pk_map:
            raise ValueError(f'Entity type {entity_type_name} does not exist')

        # Check if parent has correct entity type
        if parent_entity is not None:
            if parent_entity.row.entity_type.name != self._entity_hierachy_map[entity_type_name]:
                raise ValueError(f'Entity type {entity_type_name} has not parent type {parent_entity.row.entity_type.name}')

            parent_entity_row = parent_entity.row
        else:
            parent_entity_row = None

        # Add row
        row = EntityTable(parent=parent_entity_row, entity_type_pk=self._entity_type_pk_map[entity_type_name], id=entity_id)
        self.session.add(row)
        self.session.commit()

        # Add entity
        entity = Entity(row=row, analysis=self)
        entity.create_file()

        return entity

    @property
    def analysis_path(self):
        return self._analysis_path

    @property
    def bulk_format(self):
        return self.config['bulk_format']

    @property
    def max_blob_size(self) -> int:
        return self.config['max_blob_size']

    @property
    def compression(self) -> str:
        return self.config['compression']

    @property
    def compression_opts(self) -> Any:
        return self.config['compression_opts']

    @property
    def shuffle_filter(self) -> Any:
        return self.config['shuffle_filter']

    def open_session(self, pool_size: int = 20):

        # Only open if necessary
        if hasattr(self, 'session'):
            return

        # Create engine
        connstr = f'{self.config["dbuser"]}:{self.config["dbpassword"]}@{self.config["dbhost"]}/{self.config["dbname"]}'
        self.sql_engine = create_engine(f'mysql+pymysql://{connstr}', echo=self.echo, pool_size=pool_size)

        # Create a session
        self.session = Session(self.sql_engine)

        # Load entity types
        if len(self._entity_type_pk_map) == 0:
            self._load_entity_hierarchy()

    def close_session(self):

        # TODO: make it so connection errors lead to a call of close_session
        self.session.close()
        self.sql_engine.dispose()

        # Session attribute *needs* to be deleted, to prevent serialization error
        #  during multiprocess' pickling, because it contains weakrefs
        del self.session
        del self.sql_engine

    def get(self, entity_type: Union[str, Type[Entity]], *filter_expressions, entity_query: Query = None, **equalities) -> EntityCollection:

        self.open_session()

        # Get str
        if isinstance(entity_type, Entity):
            entity_type_name = entity_type.__name__
        else:
            entity_type_name = entity_type

        # Add equality filters to filter expressions
        for k, v in equalities.items():
            filter_expressions = filter_expressions + (f'{k} == {v}',)

        # Concat expression
        expr = ' AND '.join(filter_expressions)

        query = get_entity_query_by_attributes(entity_type_name, self.session, expr, entity_query=entity_query)

        return EntityCollection(entity_type_name, analysis=self, query=query)

    def get_temp_path(self, path: str):
        temp_path = os.path.join(self.analysis_path, 'temp', path)
        if not os.path.exists(temp_path):
            # Avoid error if concurrent process already created it in meantime
            os.makedirs(temp_path, exist_ok=True)

        return temp_path

