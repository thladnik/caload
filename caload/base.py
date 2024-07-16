from __future__ import annotations

import copy
import os.path
import pickle
import sys
import time
from abc import abstractmethod
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple, Type, Union

import h5py
import numpy as np
import pandas as pd
from pandas._typing import IndexLabel, Scalar
from sqlalchemy import create_engine, Engine, false, or_, and_, true
from sqlalchemy.orm import Session, Query, aliased
from tqdm import tqdm

from caload import sqltables as sql
from caload import utils

__all__ = ['Analysis', 'EntityCollection', 'Entity',
           'Animal', 'Recording', 'Roi', 'Phase', 'Mode']


class Mode(Enum):
    create = 1
    analyse = 2


class Analysis:
    mode: Mode
    analysis_path: str
    sql_engine: Engine
    session: Session
    index: h5py.File
    entities: Dict[Union[sql.Animal, sql.Recording, sql.Roi, sql.Phase], Entity] = {}
    write_timeout = 3.  # s

    def __init__(self, path: str, mode: Mode = Mode.analyse):
        self.mode = mode
        self.analysis_path = Path(path).as_posix()

        self.open_session()

    def __repr__(self):
        return f"Analysis('{self.analysis_path}')"

    def open_session(self):

        # Create engine
        engine = create_engine(f'sqlite:///{self.analysis_path}/metadata.db', echo=False)
        sql.SQLBase.metadata.create_all(engine)

        # Create a session
        self.session = Session(engine)

    def close_session(self):
        self.session.close()

        # Session attribute *needs* to be deleted, to prevent serialization error
        #  during multiprocess' pickling, because it contains weakrefs
        del self.session

    def add_animal(self, animal_id: str) -> Animal:
        return Animal.create(analysis=self, animal_id=animal_id)

    def animals(self, *args, **kwargs) -> EntityCollection:
        return Animal.filter(self, *args, **kwargs)

    def recordings(self, *args, **kwargs) -> EntityCollection:
        return Recording.filter(self, *args, **kwargs)

    def rois(self, *args, **kwargs) -> EntityCollection:
        return Roi.filter(self, *args, **kwargs)

    def phases(self, *args, **kwargs) -> EntityCollection:
        return Phase.filter(self, *args, **kwargs)

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


class EntityCollection:
    analysis: Analysis
    _entity_type: Type[Entity]
    _query: Query

    def __init__(self, analysis: Analysis, entity_type: Type[Entity], query: Query):
        self.analysis = analysis
        self._entity_type = entity_type
        self._query = query

    def __len__(self):
        return self._query.count()

    def __iter__(self):
        for row in self._query.all():
            yield self._get_entity(row)

    def __getitem__(self, item) -> Union[Entity, List[Entity]]:
        if isinstance(item, slice):
            indices = item.indices(len(self))
            if indices[2] != 1:
                raise KeyError(f'Invalid key {item} with indices {indices}')
            # Return slice
            return [self._get_entity(row) for row in self._query.offset(indices[0]).limit(indices[1])]
        if isinstance(item, int):
            return self._get_entity(self._query.offset(item).limit(1)[0])
        raise KeyError(f'Invalid key {item}')

    @property
    def dataframe(self):
        return pd.DataFrame([entity.scalar_attributes for entity in self])

    def _get_entity(self, row: Union[sql.Animal, sql.Recording, sql.Roi, sql.Phase]) -> Entity:
        return self._entity_type(row=row, analysis=self.analysis)

    def sortby(self, name: str, order: str = 'ASC'):
        # TODO: implement sorting
        # self._query = self._query.order_by()
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

                # Truncate
                if len(execution_times) > 1000:
                    execution_times = execution_times[len(execution_times) - 1000:]
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


# class ReferencedDataFrame(pd.DataFrame):
#
#     def _set_item(self, key, value) -> None:
#         pass
#
#     def _set_value(self, index: IndexLabel, col, value: Scalar, takeable: bool = False) -> None:
#         row = self[index]


class Entity:
    _analysis: Analysis
    attribute_table = Union[Type[sql.Animal], Type[sql.Recording], Type[sql.Roi]]
    _collection: EntityCollection

    # TODO: find better way to handle scalar attribute caching
    #  current approach does not work for parallelized enviornments
    #  if multiple processes modify the same entity (should this even happen?)

    # TODO: add attribute buffering for blobs

    def __init__(self, row: Union[sql.Animal, sql.Recording, sql.Roi, sql.Phase], analysis: Analysis, collection: EntityCollection = None):
        self._analysis = analysis
        self._row: Union[sql.Animal, sql.Recording, sql.Roi] = row
        self._scalar_attributes: Dict[str, Any] = None
        self._collection = collection
        # self._parents: List[Entity] = []

    def __getitem__(self, item: str):

        query = (self.analysis.session.query(self.attribute_table)
                 .filter(self.attribute_table.name == item)
                 .filter(self.attribute_table.entity_pk == self.row.pk))

        if query.count() == 0:
            raise KeyError(f'Attribute {item} not found for entity {self}')

        # Fetch first (only row)
        attr_row = query.first()

        value_column = attr_row.value_column
        value = attr_row.value

        if value_column != 'value_path':
            return value

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

        # Set scalars
        if type(value) in (str, float, int, bool, date, datetime):

            # Query attribute row if not in create mode
            row = None
            if not self.analysis.mode == Mode.create:
                # Build query
                query = (self.analysis.session.query(self.attribute_table)
                         .filter(self.attribute_table.entity_pk == self.row.pk)
                         .filter(self.attribute_table.name == key))

                # Evaluate
                if query.count() == 1:
                    row = query.one()
                elif query.count() > 1:
                    raise ValueError('Wait a minute...')

            # Create new row
            if row is None:
                value_type_map = {str: 'str', float: 'float', int: 'int',
                                  bool: 'bool', date: 'date', datetime: 'datetime'}
                value_type_str = value_type_map.get(type(value))
                row = self.attribute_table(entity_pk=self.row.pk, name=key, value_column=f'value_{value_type_str}')
                self.analysis.session.add(row)

            # Set value
            row.value = value

        # Set non-scalar data
        else:
            if isinstance(value, np.ndarray):
                hdf5_path = f'{self.path}/data.hdf5'
                data_path = f'hdf5:{hdf5_path}:{key}'
                pending = True
                start = time.perf_counter()
                while pending:
                    try:
                        with h5py.File(os.path.join(self.analysis.analysis_path, hdf5_path), 'a') as f:
                            # Save numpy arrays as datasets
                            if isinstance(value, np.ndarray):
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
                            time.sleep(10**-6)
                    else:
                        pending = False

            # TODO: alternative to pickle dumps? Writing arbitrary raw binary data to HDF5 seems difficult
            # Dump all other types as binary strings
            else:
                pkl_path = f'{self.path}/{key.replace("/", "_")}'
                data_path = f'pkl:{pkl_path}'

                with open(os.path.join(self.analysis.analysis_path, pkl_path), 'wb') as f:
                    pickle.dump(value, f)

            # Add row for attribute to SQL
            row = self.attribute_table(entity_pk=self.row.pk, name=key, value_column='value_path')
            row.value = data_path
            self.analysis.session.add(row)

        # Commit changes (right away if not in create mode)
        if self.analysis.mode != Mode.create:
            self.analysis.session.commit()

    @property
    def scalar_attributes(self):
        # Get all
        query = (self.analysis.session.query(self.attribute_table)
                 .filter(self.attribute_table.entity_pk == self.row.pk)
                 .filter(self.attribute_table.value_column != 'value_path'))

        # Update
        return {row.name:  row.value for row in query.all()}

    @property
    def attributes(self):
        # Get all
        query = (self.analysis.session.query(self.attribute_table)
                 .filter(self.attribute_table.entity_pk == self.row.pk))

        # Update
        attributes = {}
        for row in query.all():
            if row.value_column == 'value_path':
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
    def path(self) -> str:
        pass

    @property
    def row(self) -> sql.Animal:
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

    # def export(self, path: str):
    #     with h5py.File(path, 'a') as f:
    #         self.export_to(f.require_group(self.path))
    #
    #         for entity in self._parents:
    #             entity.export_to(f.require_group(entity.path))
    #
    @abstractmethod
    def export_to(self, grp: h5py.Group):
        pass


class Animal(Entity):
    attribute_table = sql.AnimalAttribute

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)

    def __repr__(self):
        return f"Animal(id='{self.id}')"

    @staticmethod
    def create(animal_id: str, analysis: Analysis):
        # Add row
        row = sql.Animal(id=animal_id)
        analysis.session.add(row)
        analysis.session.commit()

        # Add entity
        entity = Animal(row=row, analysis=analysis)
        entity._create_file()
        return entity

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
                        sql.Animal,
                        [],
                        sql.AnimalAttribute,
                        *attr_filters,
                        animal_id=animal_id)

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

        for rec in self.recordings():
            rec.export_to(f)


class Recording(Entity):
    attribute_table = sql.RecordingAttribute

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)
        self._animal: sql.Animal = self.row.animal

        # Add parents
        # self._parents.append(self.animal)

    def __repr__(self):
        return f"Recording(id={self.id}, " \
               f"rec_date='{self.rec_date}', " \
               f"animal_id='{self.animal_id}')"

    @staticmethod
    def create(animal: Animal, rec_date: date, rec_id: str, analysis: Analysis):
        # Add row
        row = sql.Recording(animal_pk=animal.row.pk, date=utils.parse_date(rec_date), id=rec_id)
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
    def path(self) -> str:
        return f'{self.animal.path}/recordings/{self.rec_date}_{self.id}'

    @property
    def row(self) -> sql.Recording:
        return self._row

    @property
    def id(self) -> str:
        return self._row.id

    @property
    def rec_date(self) -> date:
        return self._row.date

    @property
    def animal_id(self) -> str:
        return self._animal.id

    @property
    def animal(self) -> Animal:
        return Animal(analysis=self.analysis, row=self._animal)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               animal_id: str = None,
               rec_date: Union[str, date, datetime] = None,
               rec_id: str = None) -> EntityCollection:
        query = _filter(analysis,
                        sql.Recording,
                        [sql.Animal],
                        sql.RecordingAttribute,
                        *attr_filters,
                        animal_id=animal_id,
                        rec_date=rec_date,
                        rec_id=rec_id)

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

        for phase in self.phases():
            phase.export_to(f)

        for roi in self.rois():
            roi.export_to(f)


class Phase(Entity):
    attribute_table = sql.PhaseAttribute

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)
        self.analysis.session.flush()
        self._recording: sql.Recording = self.row.recording
        self._animal: sql.Animal = self._recording.animal

        # Add parents
        # self._parents.append(self.animal)
        # self._parents.append(self.recording)

    def __repr__(self):
        return f"Phase(id={self.id}, " \
               f"rec_date='{self.rec_date}', " \
               f"rec_id='{self.rec_id}', " \
               f"animal_id='{self.animal_id}')"

    @staticmethod
    def create(recording: Recording, phase_id: int, analysis: Analysis):
        # Add row
        row = sql.Phase(recording_pk=recording.row.pk, id=phase_id)
        analysis.session.add(row)
        # Immediately commit if not in create mode
        if analysis.mode != Mode.create:
            analysis.session.commit()

        # Add entity
        entity = Phase(row=row, analysis=analysis)
        entity._create_file()

        return entity

    @property
    def path(self) -> str:
        return f'{self.recording.path}/phases/{self.id}'

    @property
    def id(self) -> int:
        return self._row.id

    @property
    def animal_id(self) -> str:
        return self._animal.id

    @property
    def rec_date(self) -> date:
        return self._recording.date

    @property
    def rec_id(self) -> str:
        return self._recording.id

    @property
    def row(self) -> sql.Roi:
        return self._row

    @property
    def animal(self) -> Animal:
        return Animal(analysis=self.analysis, row=self._animal)

    @property
    def recording(self) -> Recording:
        return Recording(analysis=self.analysis, row=self._recording)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               animal_id: str = None,
               rec_date: Union[str, date, datetime] = None,
               rec_id: str = None,
               phase_id: int = None, ) -> EntityCollection:
        query = _filter(analysis,
                        sql.Phase,
                        [sql.Recording, sql.Animal],
                        sql.PhaseAttribute,
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
    attribute_table = sql.RoiAttribute

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)
        self.analysis.session.flush()
        self._recording: sql.Recording = self.row.recording
        self._animal: sql.Animal = self._recording.animal

        # Add parents
        # self._parents.append(self.animal)
        # self._parents.append(self.recording)

    def __repr__(self):
        return f"Roi(id={self.id}, " \
               f"rec_date='{self.rec_date}', " \
               f"rec_id='{self.rec_id}', " \
               f"animal_id='{self.animal_id}')"

    @staticmethod
    def create(recording: Recording, roi_id: int, analysis: Analysis):
        # Add row
        row = sql.Roi(recording_pk=recording.row.pk, id=roi_id)
        analysis.session.add(row)
        # Immediately commit if not in create mode
        if analysis.mode != Mode.create:
            analysis.session.commit()

        # Add entity
        entity = Roi(row=row, analysis=analysis)
        entity._create_file()

        return entity

    @property
    def path(self) -> str:
        return f'{self.recording.path}/rois/{self.id}'

    @property
    def id(self) -> int:
        return self._row.id

    @property
    def animal_id(self) -> str:
        return self._animal.id

    @property
    def rec_date(self) -> date:
        return self._recording.date

    @property
    def rec_id(self) -> str:
        return self._recording.id

    @property
    def row(self) -> sql.Roi:
        return self._row

    @property
    def animal(self) -> Animal:
        return Animal(analysis=self.analysis, row=self._animal)

    @property
    def recording(self) -> Recording:
        return Recording(analysis=self.analysis, row=self._recording)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               animal_id: str = None,
               rec_date: Union[str, date, datetime] = None,
               rec_id: str = None,
               roi_id: int = None, ) -> EntityCollection:
        query = _filter(analysis,
                        sql.Roi,
                        [sql.Recording, sql.Animal],
                        sql.RoiAttribute,
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
            base_table: Type[sql.Animal, sql.Recording, sql.Roi, sql.Phase],
            joined_tables: List[Type[sql.Animal, sql.Recording, sql.Roi, sql.Phase]],
            attribute_table: Type[sql.AnimalAttribute, sql.RecordingAttribute, sql.RoiAttribute, sql.PhaseAttribute],
            *attribute_filters: Tuple[(str, str, Any)],
            animal_id: str = None,
            rec_date: Union[str, date, datetime] = None,
            rec_id: str = None,
            roi_id: int = None,
            phase_id: int = None) -> Query:

    # TODO: currently filtering only works on a single attribute at a time
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
        query = query.filter(sql.Animal.id == animal_id)
    if rec_date is not None:
        query = query.filter(sql.Recording.date == rec_date)
    if rec_id is not None:
        query = query.filter(sql.Recording.id == rec_id)
    if roi_id is not None:
        query = query.filter(sql.Roi.id == roi_id)
    if phase_id is not None:
        query = query.filter(sql.Phase.id == phase_id)

    # Apply attribute filters
    _attr_filters = []
    if len(attribute_filters) > 0:

        for name, comp, value in attribute_filters:

            alias = aliased(attribute_table)
            query = query.join(alias)

            attr_value_field = None
            if isinstance(value, float):
                attr_value_field = alias.value_float
            elif isinstance(value, int):
                attr_value_field = alias.value_int
            elif isinstance(value, str):
                attr_value_field = alias.value_str
            elif isinstance(value, bool):
                attr_value_field = alias.value_bool

            if attr_value_field is None:
                raise TypeError(f'Invalid type "{type(value)}" to filter for in attributes')

            if comp == 'l':
                query = query.filter(attr_value_field < value)
            elif comp == 'le':
                query = query.filter(attr_value_field <= value)
            elif comp == 'e':
                # if isinstance(value, bool):
                #     print(value)
                #     if value:
                #         query = query.filter(attr_value_field.is_(true()))
                #     else:
                #         query = query.filter(attr_value_field.is_(false()))
                # else:
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
