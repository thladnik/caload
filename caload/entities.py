from __future__ import annotations

import os
import pickle

import sys
import time
from abc import abstractmethod

from datetime import datetime, date, timedelta
from typing import Any, Callable, Dict, List, TYPE_CHECKING, Tuple, Type, Union

import h5py
import numpy as np
import pandas as pd
from sqlalchemy.orm import Query, aliased
from tqdm import tqdm

from caload.sqltables import EntityTable, AnimalTable, RecordingTable, RoiTable, PhaseTable, \
    AttributeTable, AnimalAttributeTable, RecordingAttributeTable, RoiAttributeTable, PhaseAttributeTable
from caload import utils

if TYPE_CHECKING:
    from caload.analysis import Analysis

__all__ = ['EntityCollection', 'Animal', 'Recording', 'Roi', 'Phase']


class Entity:
    _analysis: Analysis
    _collection: EntityCollection
    _row:  Union[AnimalTable, RecordingTable, RoiTable, PhaseTable]

    attr_table: Union[Type[AnimalAttributeTable, RecordingAttributeTable, RoiAttributeTable, PhaseAttributeTable]]
    # Keep references to parent table instances,
    # to avoid cold references during multiprocessing,
    # caused by lazy loading
    parents: List[EntityTable]

    def __init__(self,
                 row,
                 analysis,
                 collection = None):
        self._analysis = analysis
        self._row = row
        self._collection = collection

        self.parents = []

    def __getitem__(self, item: str):

        query = (self.analysis.session.query(self.attr_table)
                 .filter(self.attr_table.name == item)
                 .filter(self.attr_table.entity_pk == self.row.pk))

        if query.count() == 0:
            raise KeyError(f'Attribute {item} not found for entity {self}')

        # Fetch first (only row)
        attr_row = query.first()

        column_str = attr_row.column_str
        value = attr_row.value

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

        row = None
        # Query attribute row if not in create mode
        if not self.analysis.is_create_mode:
            # Build query
            query = (self.analysis.session.query(self.attr_table)
                     .filter(self.attr_table.entity_pk == self.row.pk)
                     .filter(self.attr_table.name == key))

            # Evaluate
            if query.count() == 1:
                row = query.one()
            elif query.count() > 1:
                raise ValueError('Wait a minute...')

        # Create row if it doesn't exist yet
        if row is None:
            row = self.attr_table(entity_pk=self.row.pk, name=key, is_persistent=self.analysis.is_create_mode)
            self.analysis.session.add(row)

        # Set scalars
        if type(value) in (str, float, int, bool, date, datetime):

            # Set value type
            value_type_map = {str: 'str', float: 'float', int: 'int',
                              bool: 'bool', date: 'date', datetime: 'datetime'}
            column_str = f'value_{value_type_map.get(type(value))}'

        # Set small objects
        elif value.__sizeof__() < self.analysis.max_blob_size:

            # Set value type
            column_str = 'value_blob'

        # Set large objects
        else:

            # Set value type
            column_str = 'value_path'

            # Write any non-scalar data that is too large according to specified bulk storage format
            value = getattr(self, f'_write_{self.analysis.bulk_format}')(key, value, row.value)

        # Reset old value in case it was set to different type before
        if type(row.column_str) is str and row.column_str != column_str and row.value is not None:
            row.value = None

        # Set row type and value
        row.column_str = column_str
        row.value = value

        # Commit changes (right away if not in create mode)
        if not self.analysis.is_create_mode:
            self.analysis.session.commit()

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

    def _write_asdf(self, key: str, value: Any, row: AttributeTable = None):
        pass

    @property
    def scalar_attributes(self):
        # Get all
        query = (self.analysis.session.query(self.attr_table)
                 .filter(self.attr_table.entity_pk == self.row.pk)
                 .filter(self.attr_table.column_str != 'value_path'))

        # Update
        return {row.name:  row.value for row in query.all()}

    @property
    def attributes(self):
        # Get all
        query = (self.analysis.session.query(self.attr_table).filter(self.attr_table.entity_pk == self.row.pk))

        # Update
        attributes = {}
        for row in query.all():
            if row.column_str == 'value_path':
                attributes[row.name] = self[row.name]
            else:
                attributes[row.name] = row.value

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
    attr_table = AnimalAttributeTable

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)

    def __repr__(self):
        return f"Animal(id='{self.id}')"

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

    def recordings(self, *args, **kwargs) -> EntityCollection:
        return Recording.filter(self.analysis, animal_id=self.id, *args, **kwargs)

    def rois(self, *args, **kwargs) -> EntityCollection:
        return Roi.filter(self.analysis, animal_id=self.id, *args, **kwargs)

    def phases(self, *args, **kwargs) -> EntityCollection:
        return Phase.filter(self.analysis, animal_id=self.id, *args, **kwargs)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               animal_id: str = None) -> EntityCollection:
        query = _filter(analysis,
                        AnimalTable,
                        [],
                        AnimalAttributeTable,
                        *attr_filters,
                        animal_id=animal_id)

        return EntityCollection(analysis=analysis, entity_type=cls, query=query)


class Recording(Entity):
    attr_table = RecordingAttributeTable

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)
        self.parents.append(self.row.parent)

    def __repr__(self):
        return f"Recording(id={self.id}, " \
               f"rec_date='{self.rec_date}', " \
               f"animal_id='{self.animal_id}')"

    @staticmethod
    def create(animal: Animal, rec_date: date, rec_id: str, analysis: Analysis):
        # Add row
        row = RecordingTable(parent_pk=animal.row.pk, date=utils.parse_date(rec_date), id=rec_id)
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

    def rois(self, *args, **kwargs) -> EntityCollection:
        return Roi.filter(self.analysis, animal_id=self.animal_id, rec_id=self.id, rec_date=self.rec_date, *args,
                          **kwargs)

    def phases(self, *args, **kwargs) -> EntityCollection:
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
               animal_id: str = None,
               rec_date: Union[str, date, datetime] = None,
               rec_id: str = None) -> EntityCollection:
        query = _filter(analysis,
                        RecordingTable,
                        [AnimalTable],
                        RecordingAttributeTable,
                        *attr_filters,
                        animal_id=animal_id,
                        rec_date=rec_date,
                        rec_id=rec_id)

        return EntityCollection(analysis=analysis, entity_type=cls, query=query)


class Phase(Entity):
    attr_table = PhaseAttributeTable

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

    @staticmethod
    def create(recording: Recording, phase_id: int, analysis: Analysis):
        # Add row
        row = PhaseTable(parent_pk=recording.row.pk, id=phase_id)
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
               animal_id: str = None,
               rec_date: Union[str, date, datetime] = None,
               rec_id: str = None,
               phase_id: int = None, ) -> EntityCollection:
        query = _filter(analysis,
                        PhaseTable,
                        [RecordingTable, AnimalTable],
                        PhaseAttributeTable,
                        *attr_filters,
                        animal_id=animal_id,
                        rec_date=rec_date,
                        rec_id=rec_id,
                        phase_id=phase_id)

        return EntityCollection(analysis=analysis, entity_type=cls, query=query)

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
    attr_table = RoiAttributeTable

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

    @staticmethod
    def create(recording: Recording, roi_id: int, analysis: Analysis):
        # Add row
        row = RoiTable(parent_pk=recording.row.pk, id=roi_id)
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
               animal_id: str = None,
               rec_date: Union[str, date, datetime] = None,
               rec_id: str = None,
               roi_id: int = None, ) -> EntityCollection:
        query = _filter(analysis,
                        RoiTable,
                        [RecordingTable, AnimalTable],
                        RoiAttributeTable,
                        *attr_filters,
                        animal_id=animal_id,
                        rec_date=rec_date,
                        rec_id=rec_id,
                        roi_id=roi_id)

        return EntityCollection(analysis=analysis, entity_type=cls, query=query)

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
            attribute_table: Type[AnimalAttributeTable, RecordingAttributeTable, RoiAttributeTable, PhaseAttributeTable],
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

            alias = aliased(attribute_table)
            query = query.join(alias)

            attr_value_field = None
            # Booleans need to be checked first, because they isinstance(True/False, int) evaluates to True
            if isinstance(value, bool):
                attr_value_field = alias.value_bool
            elif isinstance(value, int):
                attr_value_field = alias.value_int
            elif isinstance(value, float):
                attr_value_field = alias.value_float
            elif isinstance(value, str):
                attr_value_field = alias.value_str

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

            query = query.filter(alias.name == name)

    return query


if __name__ == '__main__':
    pass


class EntityCollection:
    analysis: Analysis
    _entity_type: Type[Entity]
    _query: Query

    def __init__(self, analysis: Analysis, entity_type: Type[Entity], query: Query):
        self.analysis = analysis
        self._entity_type = entity_type
        self._query = query
        self._query_custom_orderby = False

    def __len__(self):
        return self.query.count()

    def __iter__(self) -> Entity:
        for row in self.query.all():
            yield self._get_entity(row)

    def __getitem__(self, item) -> Union[Entity, List[Entity]]:
        if isinstance(item, slice):
            indices = item.indices(len(self))
            if indices[2] != 1:
                raise KeyError(f'Invalid key {item} with indices {indices}')
            # Return slice
            return [self._get_entity(row) for row in self.query.offset(indices[0]).limit(indices[1])]
        if isinstance(item, (int, np.integer)):
            return self._get_entity(self.query.offset(item).limit(1)[0])
        print(type(item))
        raise KeyError(f'Invalid key {item}')

    @property
    def query(self):
        """Property which should be used exclusively to access the Query object.
        This is important, because there is no default order to SELECTs (unless specified).
        This means that repeated iterations over the EntityCollection instance
        may return differently ordered results.
        """
        if not self._query_custom_orderby:
            self._query = self._query.order_by(None).order_by('pk')

        return self._query

    @property
    def dataframe(self):
        return pd.DataFrame([entity.scalar_attributes for entity in self])

    @property
    def extended_dataframe(self):
        df = self.dataframe
        ext_df = pd.DataFrame([entity.parent.extended_dataframe for entity in self])
        return pd.DataFrame([entity.df for entity in self])

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
            for iter_num in range(1, len(self)+1):

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
