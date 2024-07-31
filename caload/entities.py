from __future__ import annotations

import os
import pickle
import pprint

import sys
import time
from abc import abstractmethod

from datetime import datetime, date, timedelta
from typing import Any, Callable, Dict, List, TYPE_CHECKING, Tuple, Type, Union

import h5py
import numpy as np
import pandas as pd
from sqlalchemy.orm import Query, aliased, joinedload
from tqdm import tqdm

import caload
from caload.sqltables import *
from caload import utils

if TYPE_CHECKING:
    from caload.analysis import Analysis

__all__ = ['EntityCollection', 'Animal', 'Recording', 'Roi', 'Phase']


class Entity:
    _analysis: Analysis
    _collection: EntityCollection
    _row: Union[AnimalTable, RecordingTable, RoiTable, PhaseTable]
    _attribute_name_pk_map: Dict[str, int] = {}

    attr_value_table: Union[Type[AnimalValueTable, RecordingValueTable, RoiValueTable, PhaseValueTable]]
    # Keep references to parent table instances,
    # to avoid cold references during multiprocessing,
    # caused by lazy loading
    parents: List[EntityTable]

    # def __new__(cls, *args, **kwargs):
    #
    #     row = kwargs.get('row')
    #     entities = getattr(kwargs.get('analysis'), 'entities')
    #
    #     # Check if instance of entity already exists in this analysis
    #     if cls.unique_identifier(row) in entities:
    #         return entities[cls.unique_identifier(row)]
    #
    #     # If entity does not exist yet, create new one
    #     new_entity = super(Entity, cls).__new__(cls)
    #     entities[new_entity.unique_identifier(row)] = new_entity
    #
    #     return new_entity

    def __init__(self,
                 row,
                 analysis,
                 collection=None):
        self._analysis = analysis
        self._row = row
        self._collection = collection

        self.parents = []

    def __contains__(self, item):
        value_query = (self.analysis.session.query(self.attr_value_table)
                       .filter(self.attr_value_table.attribute.name == item)
                       .filter(self.attr_value_table.entity_pk == self.row.pk))
        return value_query.count() > 0

    def __getitem__(self, item: str):

        value_query = (self.analysis.session.query(self.attr_value_table)
                       .join(AttributeTable)
                       .filter(self.attr_value_table.entity_pk == self.row.pk)
                       .filter(AttributeTable.name == item))

        if value_query.count() == 0:
            raise KeyError(f'Attribute {item} not found for entity {self}')

        # Fetch first (only row)
        value_row = value_query.first()

        column_str = value_row.column_str
        value = value_row.value

        # Anything that isn't a referenced path gets returned immediately
        if column_str != 'value_path':
            return value

        # Load and return referenced dataset
        file_type, *file_info = value.split(':')

        if file_type == 'hdf5':
            file_path, key = file_info
            with h5py.File(os.path.join(self.analysis.analysis_path, file_path), 'r') as f:
                return f[key][:]

        elif file_type == 'pkl':
            file_path, = file_info
            with open(os.path.join(self.analysis.analysis_path, file_path), 'rb') as f2:
                return pickle.load(f2)

        raise KeyError(f'No attribute with name {item} in {self}')

    def __setitem__(self, key: str, value: Any):

        # Find corresponding builtin python scalar type for numpy scalars and shape () arrays
        if isinstance(value, np.generic) or (isinstance(value, np.ndarray) and np.squeeze(value).shape == ()):
            # Convert to the corresponding Python built-in type using the item() method
            value = value.item()

        # Get attribute name entry
        if key not in self._attribute_name_pk_map:
            query_name = self.analysis.session.query(AttributeTable).filter(AttributeTable.name == key)
            if query_name.count() > 0:
                attribute_row = query_name.first()
            else:
                try:
                    attribute_row = AttributeTable(name=key)
                    self.analysis.session.add(attribute_row)
                    self.analysis.session.commit()
                except Exception as _exc:
                    # If insert fails, assume it was already added by concurrent process and re-query
                    self.analysis.session.rollback()
                    if query_name.count() == 0:
                        print(f'Failed to add non-existent attribute name {key}, Traceback:')
                        raise _exc
                    attribute_row = query_name.first()

            # Add PK to map
            self._attribute_name_pk_map[key] = attribute_row.pk

        attribute_row_pk = self._attribute_name_pk_map[key]

        value_row = None
        # Query attribute row if not in create mode
        if not self.analysis.is_create_mode:
            # Build query
            value_query = (self.analysis.session.query(self.attr_value_table)
                           .join(AttributeTable)
                           .filter(self.attr_value_table.entity_pk == self.row.pk)
                           .filter(AttributeTable.name == key))

            # Evaluate
            if value_query.count() == 1:
                value_row = value_query.one()
            elif value_query.count() > 1:
                raise ValueError('Wait a minute...')

        # Create row if it doesn't exist yet
        value_row_is_new = False
        if value_row is None:
            value_row = self.attr_value_table(entity_pk=self.row.pk, attribute_pk=attribute_row_pk,
                                              is_persistent=self.analysis.is_create_mode)
            self.analysis.session.add(value_row)
            value_row_is_new = True

        # Set scalars
        if type(value) in (str, float, int, bool, date, datetime):

            # Set value type
            value_type_map = {str: 'str', float: 'float', int: 'int',
                              bool: 'bool', date: 'date', datetime: 'datetime'}
            column_str = f'value_{value_type_map.get(type(value))}'

        # NOTE: there is no universal way to get the byte number of objects
        # Builtin object have __sizeof__(), but this only returns the overhead for numpy.ndarrays
        # For numpy arrays it's numpy.ndarray.nbytes
        # Set small objects
        elif (not isinstance(value, np.ndarray) and value.__sizeof__() < self.analysis.max_blob_size) \
                or (isinstance(value, np.ndarray) and value.nbytes < self.analysis.max_blob_size):

            # Set value type
            column_str = 'value_blob'

            # Create new blob row
            if value_row_is_new:
                blob_row = AttributeBlobTable()
                self.analysis.session.add(blob_row)
                value_row.value_blob = blob_row

        # Set large objects
        else:

            # Set value type
            column_str = 'value_path'

            # Write any non-scalar data that is too large according to specified bulk storage format
            value = getattr(self, f'_write_{self.analysis.bulk_format}')(key, value, value_row.value)

        # Reset old value in case it was set to different type before
        if type(value_row.column_str) is str and value_row.column_str != column_str and value_row.value is not None:
            value_row.value = None

        # Set row type and value
        value_row.column_str = column_str
        value_row.value = value

        # Commit changes (right away if not in create mode)
        if not self.analysis.is_create_mode:
            self.analysis.session.commit()

    @classmethod
    @abstractmethod
    def unique_identifier(cls, row: Union[AnimalTable, RecordingTable, RoiTable, PhaseTable]) -> tuple:
        pass

    def _write_hdf5(self, key: str, value: Any, data_path) -> str:

        # If not data_path is provided, generate it
        if data_path is None:
            if isinstance(value, np.ndarray):
                data_path = f'hdf5:{self.path}/data.hdf5:{key}'
            else:
                data_path = f'pkl:{self.path}/{key.replace("/", "_")}'

        # Decode data path
        file_type, *file_info = data_path.split(':')
        if file_type == 'hdf5':
            path, key = file_info
        else:
            path, = file_info

        # Write data to file
        if file_type == 'hdf5':
            pending = True
            start = time.perf_counter()
            while pending:
                try:
                    with h5py.File(os.path.join(self.analysis.analysis_path, path), 'a') as f:
                        if key not in f:
                            f.create_dataset(key, data=value)
                        else:
                            if value.shape != f[key].shape:
                                del f[key]
                                f.create_dataset(key, data=value)
                            else:
                                f[key][:] = value

                except Exception as _exc:
                    if (time.perf_counter() - start) > self.analysis.write_timeout:
                        import traceback
                        raise TimeoutError(f'Failed to write attribute {key} to {key} '
                                           f'// Traceback: {traceback.format_exc()}')
                    else:
                        time.sleep(10 ** -6)
                else:
                    pending = False

        # TODO: alternative to pickle dumps? Writing arbitrary raw binary data to HDF5 seems difficult
        # Dump all other types as binary strings
        else:
            with open(os.path.join(self.analysis.analysis_path, path), 'wb') as f:
                pickle.dump(value, f)

        return data_path

    def _write_asdf(self, key: str, value: Any, row: AttributeValueTable = None):
        pass

    @property
    def scalar_attributes(self):
        # Get all
        query = (self.analysis.session.query(self.attr_value_table)
                 .filter(self.attr_value_table.entity_pk == self.row.pk)
                 .filter(self.attr_value_table.column_str.not_in(['value_path', 'value_blob'])))
        return {value_row.attribute.name: value_row.value for value_row in query.all()}

    @property
    def attributes(self):
        # Get all
        query = (self.analysis.session.query(self.attr_value_table)
                 .filter(self.attr_value_table.entity_pk == self.row.pk))

        # Update
        attributes = {}
        for value_row in query.all():
            if value_row.column_str == 'value_path':
                attributes[value_row.attribute.name] = self[value_row.attribute.name]
            else:
                attributes[value_row.attribute.name] = value_row.value

        return attributes

    def update(self, data: Dict[str, Any]):
        """Implement update method for usage like in dict.update"""
        for key, value in data.items():
            self[key] = value

    @property
    @abstractmethod
    def parent(self):
        pass

    @property
    @abstractmethod
    def path(self) -> str:
        pass

    @property
    def row(self):
        return self._row

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    def _create_file(self):
        entity_path = f'{self.analysis.analysis_path}/{self.path}'

        # Create directoty of necessary
        if not os.path.exists(entity_path):
            os.makedirs(entity_path)

        # Create data file
        path = f'{entity_path}/data.hdf5'
        with h5py.File(path, 'w') as _:
            pass


class Animal(Entity):
    attr_value_table = AnimalValueTable

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)

    def __repr__(self):
        return f"Animal(id='{self.id}')"

    @classmethod
    def unique_identifier(cls, row: AnimalTable) -> tuple:
        return Animal.__name__, row.id

    @staticmethod
    def create(animal_id: str, analysis: Analysis):
        # Add row
        row = AnimalTable(id=animal_id)
        analysis.session.add(row)
        analysis.session.commit()

        # Add entity
        entity = Animal(row=row, analysis=analysis)
        entity._create_file()
        return entity

    @property
    def parent(self):
        return None

    @property
    def path(self) -> str:
        return f'animals/{self.id}'

    @property
    def id(self) -> str:
        return self._row.id

    def add_recording(self, *args, **kwargs) -> Recording:
        return Recording.create(*args, animal=self, analysis=self.analysis, **kwargs)

    def recordings(self, *args, **kwargs) -> RecordingCollection:
        return Recording.filter(self.analysis, animal_id=self.id, *args, **kwargs)

    def rois(self, *args, **kwargs) -> RoiCollection:
        return Roi.filter(self.analysis, animal_id=self.id, *args, **kwargs)

    def phases(self, *args, **kwargs) -> PhaseCollection:
        return Phase.filter(self.analysis, animal_id=self.id, *args, **kwargs)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               **key_filters) -> AnimalCollection:
        query = _filter(analysis,
                        AnimalTable,
                        [],
                        AnimalValueTable,
                        *attr_filters,
                        *[caload.equal(k, v) for k, v in key_filters.items()])

        # TODO: Maybe add separate method to filter exclusively by indexed entity columns ('key_filters') for faster filtering?

        return AnimalCollection(analysis=analysis, query=query)


class Recording(Entity):
    attr_value_table = RecordingValueTable

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)
        self.parents.append(self.row.parent)

    def __repr__(self):
        return f"Recording(id={self.id}, " \
               f"rec_date='{self.rec_date}', " \
               f"animal_id='{self.animal_id}')"

    @classmethod
    def unique_identifier(cls, row: RecordingTable) -> tuple:
        return Recording.__name__, row.parent.id, row.date, row.id

    @staticmethod
    def create(animal: Animal, rec_date: date, rec_id: str, analysis: Analysis):
        # Add row
        row = RecordingTable(parent=animal.row, date=utils.parse_date(rec_date), id=rec_id)
        analysis.session.add(row)
        analysis.session.commit()

        # Add entity
        entity = Recording(row=row, analysis=analysis)
        entity._create_file()

        return entity

    def add_roi(self, *args, **kwargs) -> Roi:
        return Roi.create(*args, recording=self, analysis=self.analysis, **kwargs)

    def add_phase(self, *args, **kwargs) -> Phase:
        return Phase.create(*args, recording=self, analysis=self.analysis, **kwargs)

    def rois(self, *args, **kwargs) -> RoiCollection:
        return Roi.filter(self.analysis, animal_id=self.animal_id, rec_id=self.id, rec_date=self.rec_date, *args,
                          **kwargs)

    def phases(self, *args, **kwargs) -> PhaseCollection:
        return Phase.filter(self.analysis, animal_id=self.animal_id, rec_id=self.id, rec_date=self.rec_date, *args,
                            **kwargs)

    @property
    def parent(self):
        return self.animal

    @property
    def path(self) -> str:
        return f'{self.parent.path}/recordings/{self.rec_date}_{self.id}'

    @property
    def id(self) -> str:
        return self._row.id

    @property
    def rec_date(self) -> date:
        return self._row.date

    @property
    def animal_id(self) -> str:
        return self._row.parent.id

    @property
    def animal(self) -> Animal:
        return Animal(analysis=self.analysis, row=self._row.parent)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               **key_filters) -> RecordingCollection:
        query = _filter(analysis,
                        RecordingTable,
                        [AnimalTable],
                        RecordingValueTable,
                        *attr_filters,
                        *[caload.equal(k, v) for k, v in key_filters.items()])

        return RecordingCollection(analysis=analysis, query=query)


class Phase(Entity):
    attr_value_table = PhaseValueTable

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)

        self.parents.append(self.row.parent)
        self.parents.append(self.row.parent.parent)

        self.analysis.session.flush()

    def __repr__(self):
        return f"Phase(id={self.id}, " \
               f"rec_date='{self.rec_date}', " \
               f"rec_id='{self.rec_id}', " \
               f"animal_id='{self.animal_id}')"

    @classmethod
    def unique_identifier(cls, row: PhaseTable) -> tuple:
        return Phase.__name__, row.parent.parent.id, row.parent.id, row.parent.date, row.id

    @staticmethod
    def create(recording: Recording, phase_id: int, analysis: Analysis):
        # Add row
        row = PhaseTable(parent=recording.row, id=phase_id)
        analysis.session.add(row)
        # Immediately commit if not in create mode
        if not analysis.is_create_mode:
            analysis.session.commit()

        # Add entity
        entity = Phase(row=row, analysis=analysis)
        entity._create_file()

        return entity

    @property
    def parent(self):
        return self.recording

    @property
    def path(self) -> str:
        return f'{self.parent.path}/phases/{self.id}'

    @property
    def id(self) -> int:
        return self._row.id

    @property
    def animal_id(self) -> str:
        return self.parent.parent.id

    @property
    def rec_date(self) -> date:
        return self.recording.rec_date

    @property
    def rec_id(self) -> str:
        return self.recording.id

    @property
    def animal(self) -> Animal:
        return Animal(analysis=self.analysis, row=self.row.parent.parent)

    @property
    def recording(self) -> Recording:
        return Recording(analysis=self.analysis, row=self.row.parent)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               **key_filters) -> PhaseCollection:
        query = _filter(analysis,
                        PhaseTable,
                        [RecordingTable, AnimalTable],
                        PhaseValueTable,
                        *attr_filters,
                        *[caload.equal(k, v) for k, v in key_filters.items()])

        return PhaseCollection(analysis=analysis, query=query)

    def export_to(self, f: h5py.File):

        with h5py.File(f'{self.analysis.analysis_path}/{self.path}/data.hdf5', 'r') as f2:
            f2.copy(source='/', dest=f, name=self.path)

        try:
            f[self.path].attrs.update(self.scalar_attributes)
        except:
            for k, v in self.scalar_attributes.items():
                try:
                    f[self.path].attrs[k] = v
                except:
                    print(f'Failed to export scalar attribute {k} in {self} (type: {type(v)})')


class Roi(Entity):
    attr_value_table = RoiValueTable

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)

        self.parents.append(self.row.parent)
        self.parents.append(self.row.parent.parent)

        self.analysis.session.flush()

    def __repr__(self):
        return f"Roi(id={self.id}, " \
               f"rec_date='{self.rec_date}', " \
               f"rec_id='{self.rec_id}', " \
               f"animal_id='{self.animal_id}')"

    @classmethod
    def unique_identifier(cls, row: RoiTable) -> tuple:
        return Roi.__name__, row.parent.parent.id, row.parent.id, row.parent.date, row.id

    @staticmethod
    def create(recording: Recording, roi_id: int, analysis: Analysis):
        # Add row
        row = RoiTable(parent=recording.row, id=roi_id)
        analysis.session.add(row)
        # Immediately commit if not in create mode
        if not analysis.is_create_mode:
            analysis.session.commit()

        # Add entity
        entity = Roi(row=row, analysis=analysis)
        entity._create_file()

        return entity

    @property
    def path(self) -> str:
        return f'{self.parent.path}/rois/{self.id}'

    @property
    def parent(self):
        return self.recording

    @property
    def id(self) -> int:
        return self._row.id

    @property
    def animal_id(self) -> str:
        return self.animal.id

    @property
    def rec_date(self) -> date:
        return self.recording.rec_date

    @property
    def rec_id(self) -> str:
        return self.recording.id

    @property
    def animal(self) -> Animal:
        return Animal(analysis=self.analysis, row=self.row.parent.parent)

    @property
    def recording(self) -> Recording:
        return Recording(analysis=self.analysis, row=self.row.parent)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               **key_filters) -> RoiCollection:
        query = _filter(analysis,
                        RoiTable,
                        [RecordingTable, AnimalTable],
                        RoiValueTable,
                        *attr_filters,
                        *[caload.equal(k, v) for k, v in key_filters.items()])

        return RoiCollection(analysis=analysis, query=query)

    def export_to(self, f: h5py.File):

        with h5py.File(f'{self.analysis.analysis_path}/{self.path}/data.hdf5', 'r') as f2:
            f2.copy(source='/', dest=f, name=self.path)

        try:
            f[self.path].attrs.update(self.scalar_attributes)
        except:
            for k, v in self.scalar_attributes.items():
                try:
                    f[self.path].attrs[k] = v
                except:
                    print(f'Failed to export scalar attribute {k} in {self} (type: {type(v)})')


def _filter(analysis: Analysis,
            base_table: Type[AnimalTable, RecordingTable, RoiTable, PhaseTable],
            joined_tables: List[Type[AnimalTable, RecordingTable, RoiTable, PhaseTable]],
            attr_value_table: Type[AnimalValueTable, RecordingValueTable, RoiValueTable, PhaseValueTable],
            *attribute_filters: List[Tuple[str, str, Any]],
            **kwargs):
    from sqlalchemy import or_, and_
    from sqlalchemy.orm import aliased

    _query = analysis.session.query(base_table)

    for filt in attribute_filters:
        if isinstance(filt, tuple):
            name, comp, value = filt

            # Create alias
            _alias = aliased(attr_value_table)

            if comp == 'has':
                # Build subquery to filter attribute name
                subquery = analysis.session.query(AttributeTable).filter(AttributeTable.name == name).subquery()
                # Build WHERE clause based on attribute name subquery
                _query = _query.filter(subquery.c.pk == _alias.attribute_pk)
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


def _filter_old(analysis: Analysis,
                base_table: Type[AnimalTable, RecordingTable, RoiTable, PhaseTable],
                joined_tables: List[Type[AnimalTable, RecordingTable, RoiTable, PhaseTable]],
                attr_value_table: Type[AnimalValueTable, RecordingValueTable, RoiValueTable, PhaseValueTable],
                *attribute_filters: Tuple[(str, str, Any)],
                animal_id: str = None,
                rec_date: Union[str, date, datetime] = None,
                rec_id: str = None,
                roi_id: int = None,
                phase_id: int = None) -> Query:
    # Possible solution for future:
    """
    Apply multiple attribute filters:
    SELECT * FROM rois
        LEFT JOIN roi_attributes as attr ON rois.pk = attr.entity_pk
        GROUP BY rois.pk
        HAVING  SUM(CASE WHEN attr.name == 's2p/npix' AND attr.value_int < 50 THEN 1 ELSE 0 END) > 0 AND
                SUM(CASE WHEN attr.name == 's2p/radius' AND attr.value_float > 5.0 THEN 1 ELSE 0 END) > 0
    """

    # Convert date
    rec_date = utils.parse_date(rec_date)

    # Join parent tables
    query = analysis.session.query(base_table)
    for t in joined_tables:
        query = query.join(t)

    # Filter
    if animal_id is not None:
        query = query.filter(AnimalTable.id == animal_id)
    if rec_date is not None:
        query = query.filter(RecordingTable.date == rec_date)
    if rec_id is not None:
        query = query.filter(RecordingTable.id == rec_id)
    if roi_id is not None:
        query = query.filter(RoiTable.id == roi_id)
    if phase_id is not None:
        query = query.filter(PhaseTable.id == phase_id)

    # Apply attribute filters
    _attr_filters = []
    if len(attribute_filters) > 0:

        for name, comp, value in attribute_filters:

            if query.join(attr_value_table).filter(attr_value_table.attribute.name == name).count() == 0:
                raise KeyError(f'Unkown filter for {attr_value_table} with name {name}')

            attr_value_alias = aliased(attr_value_table)
            query = query.join(attr_value_alias)

            # Filter attribute name
            query = query.filter(attr_value_alias.attribute.name == name)

            # Only check if entity has an attribute of "name"
            if comp == 'contains':
                continue

            attr_value_field = None
            # Booleans need to be checked first, because they isinstance(True/False, int) evaluates to True
            if isinstance(value, bool):
                attr_value_field = attr_value_alias.value_bool
            elif isinstance(value, int):
                attr_value_field = attr_value_alias.value_int
            elif isinstance(value, float):
                attr_value_field = attr_value_alias.value_float
            elif isinstance(value, str):
                attr_value_field = attr_value_alias.value_str

            if attr_value_field is None:
                raise TypeError(f'Invalid type "{type(value)}" to filter for in attributes')

            if comp == 'l':
                query = query.filter(attr_value_field < value)
            elif comp == 'le':
                query = query.filter(attr_value_field <= value)
            elif comp == 'e':
                query = query.filter(attr_value_field == value)
            elif comp == 'ge':
                query = query.filter(attr_value_field >= value)
            elif comp == 'g':
                query = query.filter(attr_value_field > value)
            else:
                raise ValueError('Invalid filter format')

    return query


if __name__ == '__main__':
    pass


class EntityCollection:
    analysis: Analysis
    _entity_type: Type[Entity]
    _query: Query
    _iteration_count: int = -1
    _batch_offset: int = 0
    _batch_size: int = 100
    _batch_results: List[EntityTable]

    def __init__(self, analysis: Analysis, query: Query):
        self.analysis = analysis
        self._query = query
        self._query_custom_orderby = False

    def __len__(self):
        return self.query.count()

    def __iter__(self):
        return self

    def __next__(self):

        # Increment count
        self._iteration_count += 1

        # Fetch the next batch of results
        if self._iteration_count == 0 or (self._iteration_count == self._batch_offset + self._batch_size):
            self._batch_offset = self._iteration_count
            self._batch_results = self.query.offset(self._batch_offset).limit(self._batch_size).all()

        # No more results: reset iteration counter and offset and stop iteration
        if len(self._batch_results) == 0 or self._iteration_count >= self._batch_offset + len(self._batch_results):
            self._iteration_count = -1
            self._batch_offset = 0

            raise StopIteration

        # Return single result
        return self._get_entity(self._batch_results[self._iteration_count % self._batch_size])

    def __getitem__(self, item) -> Union[Entity, List[Entity], pd.DataFrame]:

        # Return single entity
        if isinstance(item, (int, np.integer)):
            if item < 0:
                item = len(self) + item
            return self._get_entity(self.query.offset(item).limit(1)[0])

        # Return slice
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step == 0:
                raise KeyError('Invalid step size 0')

            # Get data
            result = [self._get_entity(row) for row in self.query.offset(start).limit(stop - start)]

            # TODO: there should be a way to directly query the n-th row using 'ROW_NUMBER() % n'
            #  but it's not clear how is would work in SQLAlchemy ORM; figure out later
            # result = [self._get_entity(row) for row in self.query.offset(start).limit(stop-start)][::abs(step)]

            # Return in order
            return result[::step]

        # Return multiple attributes for all entities in collection
        if isinstance(item, (str, list, tuple)):
            if isinstance(item, str):
                item = [item]

            df = self._dataframe_of(attribute_names=item, include_bulk=True, include_blobs=True)

            # For single column, return pd.Series
            if len(df.columns) == 1:
                return df.iloc[:, 0]

            return df

        raise KeyError(f'Invalid key {item}')

    @property
    def query(self):
        """Property which should be used *exclusively* to access the Query object.
        This is important, because there is no default order to SELECTs (unless specified).
        This means that repeated iterations over the EntityCollection instance
        may return differently ordered results.
        """
        if not self._query_custom_orderby:
            self._query = self._query.order_by(None).order_by('pk')

        return self._query

    def _column(self, attribute: Union[str, int], include_blobs: bool = False):

        # If attribute is string, fetch corresponding primary key
        if isinstance(attribute, str):
            pk_query = self.analysis.session.query(AttributeTable.pk).filter(AttributeTable.name == attribute)

            if pk_query.count() == 0:
                raise KeyError(f'No attribute with name {attribute} found for {self}')
            attribute_pk = pk_query.first().pk
        else:
            attribute_pk = attribute

        # Subquery on entity primary keys
        entity_query = self.query.subquery().primary_key

        # Query attribute values
        value_query = (self.analysis.session.query(self._entity_type.attr_value_table)
                       .filter(self._entity_type.attr_value_table.entity_pk.in_(entity_query))
                       .filter(self._entity_type.attr_value_table.attribute_pk == attribute_pk))

        if include_blobs:
            value_query = value_query.options(joinedload(self._entity_type.attr_value_table.value_blob))

        return value_query.all()

    def _dataframe_of(self, attribute_names: List[str] = None, attribute_pks: List[int] = None,
                      include_blobs: bool = False, include_bulk: bool = False) -> pd.DataFrame:

        # Add to excluded list
        excluded = []
        if not include_blobs:
            excluded.append('value_blob')
        if not include_bulk:
            excluded.append('value_path')

        # Prepare args
        if attribute_pks is None:
            attribute_pks = [None] * len(attribute_names)
        elif attribute_names is None:
            attribute_names = [None] * len(attribute_pks)

        if attribute_pks is None or attribute_names is None:
            raise ValueError('No attribute list specified')

        # Iterate through specified attributes
        cols = []
        for attr_pk, attr_name in zip(attribute_pks, attribute_names):

            # Fetch rows
            rows = self._column(attr_name if attr_pk is None else attr_pk, include_blobs)

            # If no attribute name was give, fetch it
            if attr_name is None:
                self.analysis.session.query(AttributeTable.name).filter(AttributeTable.pk == attr_pk)

            # Get indices and data
            idcs = [v.entity_pk for v in rows if v.column_str not in excluded]
            data = [v.value for v in rows if v.column_str not in excluded]

            # Create series
            series = pd.Series(index=idcs, data=data, name=attr_name)

            # Include if not empty
            if series.count() > 0:
                cols.append(series)

        return pd.concat(cols, axis=1)

    @property
    def dataframe(self):

        # Fetch all entity related, unique attributes
        unique_value_rows = self.analysis.session.query(RoiValueTable).group_by(RoiValueTable.attribute_pk).all()

        # Return dataframe of all attributes
        return self._dataframe_of(attribute_names=[row.attribute.name for row in unique_value_rows],
                                  attribute_pks=[row.attribute.pk for row in unique_value_rows],
                                  include_blobs=False, include_bulk=False)

    @property
    def extended_dataframe(self):
        # Fetch all entity related, unique attributes
        unique_value_rows = self.analysis.session.query(RoiValueTable).group_by(RoiValueTable.attribute_pk).all()

        # Return dataframe of all attributes
        return self._dataframe_of(attribute_names=[row.attribute.name for row in unique_value_rows],
                                  attribute_pks=[row.attribute.pk for row in unique_value_rows],
                                  include_blobs=True, include_bulk=False)

    def _get_entity(self, row: EntityTable) -> Entity:
        return self._entity_type(row=row, analysis=self.analysis)

    def sortby(self, name: str, order: str = 'ASC'):
        # TODO: implement sorting
        pass

    def map(self, fun: Callable, **kwargs) -> Any:
        print(f'Run function {fun.__name__} on {self} with args '
              f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {len(self)} entities')

        for entity in tqdm(self):
            fun(entity, **kwargs)

    def map_async(self, fun: Callable, chunk_size: int = None, worker_num: int = None, **kwargs) -> Any:
        print(f'Run function {fun.__name__} on {self} with args '
              f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {len(self)} entities')

        # Prepare pool and entities
        import multiprocessing as mp
        if worker_num is None:
            worker_num = mp.cpu_count() - 1
            if len(self) < worker_num:
                worker_num = len(self)
        print(f'Start pool with {worker_num} workers')

        kwargs = tuple([(k, v) for k, v in kwargs.items()])
        if chunk_size is None:
            worker_args = [(fun, e, kwargs) for e in self]
            chunk_size = 1
        else:
            chunk_num = int(np.ceil(len(self) / chunk_size))
            worker_args = [(fun, self[i * chunk_size:(i + 1) * chunk_size], kwargs) for i in range(chunk_num)]
            print(f'Entity chunksize {chunk_size}')

        # Close session first
        self.analysis.close_session()

        # Map entities to process pool
        execution_times = []
        start_time = time.time()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        print(f'Start processing at {formatted_time}')
        with mp.Pool(processes=worker_num) as pool:
            iterator = pool.imap_unordered(self.worker_wrapper, worker_args)
            for iter_num in range(1, len(self) + 1):

                # Iterate while looking out for exceptions
                try:
                    exec_time = next(iterator)
                except StopIteration:
                    pass
                except Exception as _exc:
                    raise _exc

                # Calcualate timing info
                execution_times.append(exec_time)
                mean_exec_time = np.mean(execution_times) if len(execution_times) > 0 else 0
                time_per_entity = mean_exec_time / (worker_num * chunk_size)
                time_elapsed = time.time() - start_time
                time_rest = time_per_entity * (len(self) - iter_num * chunk_size)

                # Print timing info
                sys.stdout.write('\r'
                                 f'[{iter_num * chunk_size}/{len(self)}] '
                                 f'{time_per_entity:.2f}s/iter '
                                 f'- {timedelta(seconds=int(time_elapsed))}'
                                 f'/{timedelta(seconds=int(time_elapsed + time_rest))} '
                                 f'-> {timedelta(seconds=int(time_rest))} remaining ')

                # # Truncate
                # if len(execution_times) > 1000:
                #     execution_times = execution_times[len(execution_times) - 1000:]
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(f'\nFinish processing at {formatted_time}')

        # Re-open session
        self.analysis.open_session()

    @staticmethod
    def worker_wrapper(args):

        start_time = time.perf_counter()
        # Unpack args
        fun: Callable = args[0]
        entity: Union[Entity, EntityCollection, List[Entity]] = args[1]
        kwargs = {k: v for k, v in args[2]}

        # Re-open session in worker
        if isinstance(entity, list):
            entity[0].analysis.open_session()
            close_session = entity[0].analysis.close_session
        else:
            entity.analysis.open_session()
            close_session = entity.analysis.close_session

        # Run function on entity
        res = fun(entity, **kwargs)

        # Close session again
        close_session()

        elapsed_time = time.perf_counter() - start_time

        return elapsed_time


class AnimalCollection(EntityCollection):
    _entity_type = Animal

    def _get_entity(self, row: AnimalTable) -> Animal:
        return super()._get_entity(row)


class RecordingCollection(EntityCollection):
    _entity_type = Recording

    def _get_entity(self, row: RecordingTable) -> Recording:
        return super()._get_entity(row)


class RoiCollection(EntityCollection):
    _entity_type = Roi

    def _get_entity(self, row: RoiTable) -> Roi:
        return super()._get_entity(row)


class PhaseCollection(EntityCollection):
    _entity_type = Phase

    def _get_entity(self, row: PhaseTable) -> Phase:
        return super()._get_entity(row)
