from __future__ import annotations

import os
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

import h5py
import yaml
from sqlalchemy import Engine, create_engine, or_, and_
from sqlalchemy.orm import Query, Session, aliased

from caload import utils
from caload.entities import Animal, Recording, Roi, Phase, \
    EntityCollection, AnimalCollection, RecordingCollection, RoiCollection, PhaseCollection
from caload.sqltables import AnimalValueTable, AttributeTable, PhaseValueTable, RecordingValueTable, RoiValueTable, \
    SQLBase, \
    AnimalTable, RecordingTable, \
    RoiTable, \
    PhaseTable

__all__ = ['Analysis', 'Mode', 'open_analysis']


class Mode(Enum):
    create = 1
    analyse = 2


class Analysis:
    _default_bulk_format = 'hdf5'
    _default_max_blob_size = 2 ** 20  # 2^20 ~ 1MB
    mode: Mode
    analysis_path: str
    sql_engine: Engine
    session: Session
    write_timeout = 3.  # s
    lazy_init: bool
    echo: bool
    entities: Dict[tuple, Union[Animal, Recording, Roi, Phase]]

    def __init__(self, path: str, mode: Mode = Mode.analyse, bulk_format: str = None,
                 lazy_init: bool = False, echo: bool = False, max_blob_size: int = None,
                 compression: str = 'gzip', compression_opts: Any = None, shuffle_filter: bool = True):

        # Set mode
        self.mode = mode
        self.is_create_mode = mode is Mode.create

        # Set path as posix
        self._analysis_path = Path(path).as_posix()

        # Set up entity dictionary
        self.entities = {}

        # Generate config
        self.config = {'bulk_format': bulk_format,
                       'max_blob_size': max_blob_size,
                       'compression': compression,
                       'compression_opts': compression_opts,
                       'shuffle_filter': shuffle_filter}

        if self.is_create_mode:

            # Set bulk format type
            if bulk_format is None:
                self.config['bulk_format'] = self._default_bulk_format

            # Set max blob size
            if max_blob_size is None:
                self.config['max_blob_size'] = self._default_max_blob_size

            with open(f'{self.analysis_path}/configuration.yaml', 'w') as f:
                yaml.safe_dump(self.config, f)

        else:
            if bulk_format is not None or max_blob_size is not None:
                Warning('bulk_format and max_blob_size may only be set during creation of new analysis')

            with open(f'{self.analysis_path}/configuration.yaml', 'r') as f:
                self.config.update(yaml.safe_load(f))

        # Set optionals
        self.lazy_init = lazy_init
        self.echo = echo

        # Open SQL session by default
        if not lazy_init:
            self.open_session()
        # else:
        #     if self.is_create_mode:
        #         raise Exception('Cannot lazily initialize in create mode')
        #
        # # Preload all entities for fast lookup of existing entities during creation
        # if self.is_create_mode:
        #     for row in self.session.query(AnimalTable).all():
        #         Animal(row=row, analysis=self)
        #
        #     for row in self.session.query(RecordingTable).all():
        #         Recording(row=row, analysis=self)
        #
        #     for row in self.session.query(RoiTable).all():
        #         Roi(row=row, analysis=self)
        #
        #     for row in self.session.query(PhaseTable).all():
        #         Phase(row=row, analysis=self)

    def __repr__(self):
        return f"Analysis('{self.analysis_path}')"

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

    def open_session(self):

        # Create engine
        engine = create_engine(f'sqlite:///{self.analysis_path}/metadata.db', echo=self.echo)
        SQLBase.metadata.create_all(engine)

        # Create a session
        self.session = Session(engine)

    def close_session(self):
        self.session.close()

        # Session attribute *needs* to be deleted, to prevent serialization error
        #  during multiprocess' pickling, because it contains weakrefs
        del self.session

    def add_animal(self, animal_id: str) -> Animal:
        return Animal.create(analysis=self, animal_id=animal_id)

    def animals(self, *attr_filters, **kwargs) -> AnimalCollection:
        query = _get_entity_query_by_attributes(self, AnimalTable, AnimalValueTable, *attr_filters, **kwargs)

        return AnimalCollection(analysis=self, query=query)

    def get_animal_by_id(self, animal_id: str = None) -> Animal:

        # Build query
        query = _get_entity_query_by_ids(self, AnimalTable, animal_id=animal_id)

        # Return data
        if query.count() == 0:
            raise Exception('No animals for given identifiers')
        if query.count() == 1:
            return Animal(analysis=self, row=query.first())
        raise Exception('This should not happen')

    def recordings(self, *attr_filters, **kwargs) -> RecordingCollection:
        query = _get_entity_query_by_attributes(self, RecordingTable, RecordingValueTable, *attr_filters, **kwargs)

        return RecordingCollection(analysis=self, query=query)

    def get_recordings_by_id(self, rec_id: str = None, rec_date: Union[str, date] = None, animal_id: str = None) \
            -> Union[Recording, RecordingCollection]:

        # Build query
        query = _get_entity_query_by_ids(self, RecordingTable, rec_id=rec_id,
                                         rec_date=rec_date, animal_id=animal_id)

        # Return data
        if query.count() == 0:
            raise Exception('No recordings for given identifiers')
        if query.count() == 1:
            return Recording(analysis=self, row=query.first())
        return RecordingCollection(analysis=self, query=query)

    def rois(self, *attr_filters, **kwargs) -> RoiCollection:
        query = _get_entity_query_by_attributes(self, RoiTable, RoiValueTable, *attr_filters, **kwargs)

        return RoiCollection(analysis=self, query=query)

    def get_rois_by_id(self,
                       roi_id: int = None, rec_id: str = None,
                       rec_date: Union[str, date] = None, animal_id: str = None) \
            -> Union[Roi, RoiCollection]:

        # Build query
        query = _get_entity_query_by_ids(self, RoiTable,
                                         roi_id=roi_id, rec_id=rec_id,
                                         rec_date=rec_date, animal_id=animal_id)

        # Return data
        if query.count() == 0:
            raise Exception('No roi for given identifiers')
        if query.count() == 1:
            return Roi(analysis=self, row=query.first())
        return RoiCollection(analysis=self, query=query)

    def phases(self, *attr_filters, **kwargs) -> PhaseCollection:
        query = _get_entity_query_by_attributes(self, PhaseTable, PhaseValueTable, *attr_filters, **kwargs)

        return PhaseCollection(analysis=self, query=query)

    def get_phases_by_id(self,
                         phase_id: int = None, rec_id: str = None,
                         rec_date: Union[str, date] = None, animal_id: str = None) \
            -> Union[Phase, PhaseCollection]:

        # Build query
        query = _get_entity_query_by_ids(self, PhaseTable,
                                         phase_id=phase_id, rec_id=rec_id,
                                         rec_date=rec_date, animal_id=animal_id)

        # Return data
        if query.count() == 0:
            raise Exception('No phases for given identifiers')
        if query.count() == 1:
            return Phase(analysis=self, row=query.first())
        return PhaseCollection(analysis=self, query=query)

    def export(self, path: str):
        # TODO: add some data export and stuff
        with h5py.File(path, 'w') as f:
            for animal in self.animals():
                animal.export_to(f)

    def get_temp_path(self, path: str):
        temp_path = os.path.join(self.analysis_path, 'temp', path)
        if not os.path.exists(temp_path):
            # Avoid error if concurrent process already created it in meantime
            os.makedirs(temp_path, exist_ok=True)

        return temp_path


def open_analysis(analysis_path: str, mode=Mode.analyse, **kwargs) -> Analysis:
    meta_path = f'{analysis_path}/metadata.db'
    if not os.path.exists(meta_path):
        raise ValueError(f'Path {meta_path} not found')

    summary = Analysis(analysis_path, mode=mode, **kwargs)

    return summary


def _get_entity_query_by_ids(analysis: Analysis,
                             base_table: Type[AnimalTable, RecordingTable, RoiTable, PhaseTable],
                             animal_id: str = None,
                             rec_date: Union[str, date, datetime] = None,
                             rec_id: str = None,
                             roi_id: int = None,
                             phase_id: int = None) -> Query:
    # Convert date
    rec_date = utils.parse_date(rec_date)

    # Create query
    query = analysis.session.query(base_table)

    # Join parents and filter by ID
    if base_table in (RoiTable, PhaseTable):
        if rec_date is not None or rec_id is not None:
            query = query.join(RecordingTable)

        if rec_date is not None:
            query = query.filter(RecordingTable.date == rec_date)
        if rec_id is not None:
            query = query.filter(RecordingTable.id == rec_id)

    if base_table in (RecordingTable, RoiTable, PhaseTable) and animal_id is not None:
        query = query.join(AnimalTable).filter(AnimalTable.id == animal_id)

    # Filter bottom entities
    if base_table == RoiTable and roi_id is not None:
        query = query.filter(RoiTable.id == roi_id)
    if base_table == PhaseTable and phase_id is not None:
        query = query.filter(PhaseTable.id == phase_id)

    return query


def _get_entity_query_by_attributes(analysis: Analysis,
                                    base_table: Type[AnimalTable, RecordingTable, RoiTable, PhaseTable],
                                    attr_value_table: Type[AnimalValueTable, RecordingValueTable, RoiValueTable, PhaseValueTable],
                                    *attribute_filters: List[Tuple[str, str, Any]],
                                    entity_query: Query = None):
    # Create query
    _query = analysis.session.query(base_table)

    if entity_query is not None:
        _subquery = entity_query.subquery().primary_key
        _query = _query.filter(base_table.pk.in_(_subquery))

    for filt in attribute_filters:
        if isinstance(filt, tuple):
            name, comp, value = filt
        elif isinstance(filt, str):
            name, comp, value = filt.split(' ')

            cast_value = {'true': True, 'false': False}.get(value.lower(), None)
            if cast_value is None:
                for _type in (int, float):
                    try:
                        cast_value = _type(value)
                    except:
                        pass
                    else:
                        break

            if cast_value is not None:
                value = cast_value
            # If no valid conversion, assume string is correct one

        else:
            raise Exception(f'Invalid filter argument {filt}')

        # Create alias
        _alias = aliased(attr_value_table)

        if comp in ('has', 'hasnot'):
            # Build subquery to filter attribute name
            subquery = analysis.session.query(AttributeTable).filter(AttributeTable.name == name).subquery()
            # Build WHERE clause based on attribute name subquery
            if comp == 'has':
                _query = _query.filter(subquery.c.pk == _alias.attribute_pk)
            else:
                _query = _query.filter(subquery.c.pk != _alias.attribute_pk)

            # Skip rest
            continue

        # Determine value type
        if isinstance(value, (bool, int, float, str, date, datetime)):
            _aliased_value_field = getattr(_alias, f'value_{str(type(value).__name__)}')
        else:
            raise TypeError(f'Invalid type "{type(value)}" to filter for in attributes')

        # Apply value filter
        if comp == '<':
            w1 = _aliased_value_field < value
        elif comp == '<=':
            w1 = _aliased_value_field <= value
        elif comp == '==':
            w1 = _aliased_value_field == value
        elif comp == '>=':
            w1 = _aliased_value_field >= value
        elif comp == '>':
            w1 = _aliased_value_field > value
        else:
            raise ValueError('Invalid filter format')

        # Build subquery to filter attribute name
        subquery = analysis.session.query(AttributeTable).filter(AttributeTable.name == name).subquery()
        # Build WHERE clause based on attribute name subquery
        w = and_(w1, subquery.c.pk == _alias.attribute_pk)

        # Add to main query
        _query = _query.join(_alias).filter(w)

    return _query
