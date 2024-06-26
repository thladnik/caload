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
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session, Query
from tqdm import tqdm

from caload import sqltables as sql
from caload import utils

__all__ = ['Analysis', 'EntityCollection', 'Entity', 'Animal', 'Recording', 'Roi', 'Phase', 'Mode']


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
        if hasattr(self, '_sql_session'):
            return

        # Create engine
        engine = create_engine(f'sqlite:///{self.analysis_path}/metadata.db', echo=False)
        sql.SQLBase.metadata.create_all(engine)

        # Create a session
        self.session = Session(engine)

    def close_session(self):
        self.session.close()

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

    # def export(self, path: str, entities: List[Entity]):
    #     # TODO: add some data export and stuff
    #     for entity in entities:
    #         entity.export(path)


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

    def __iter__(self) -> Iterator[Entity]:
        for row in self._query.all():
            yield self._get_entity(row)

    def __getitem__(self, item) -> List[Entity]:
        if isinstance(item, slice):
            indices = item.indices(len(self))
            if indices[2] != 1:
                raise KeyError(f'Invalid key {item} with indices {indices}')
            # Return slice
            return [self._get_entity(row) for row in self._query.offset(indices[0]).limit(indices[1])]
        raise KeyError(f'Invalid key {item}')

    def _get_entity(self, row: Union[sql.Animal, sql.Recording, sql.Roi, sql.Phase]) -> Entity:
        return self._entity_type(row=row, analysis=self.analysis)

    def sortby(self, name: str, order: str = 'ASC'):
        # TODO: implement sorting
        # self._query = self._query.order_by()
        pass

    def map(self, fun: Callable, **kwargs) -> Any:
        for entity in tqdm(self):
            fun(entity, **kwargs)

    def map_async(self, fun: Callable, chunk_size: int = None, worker_num: int = None, **kwargs) -> Any:

        # Prepare pool and entities
        import multiprocessing as mp
        if worker_num is None:
            worker_num = mp.cpu_count() - 1
        print(f'Start pool with {worker_num} workers')
        pool = mp.Pool(processes=worker_num)

        entities = self[:]

        kwargs = tuple([(k, v) for k, v in kwargs.items()])
        if chunk_size is None:
            worker_args = [(fun, e, kwargs) for e in entities]
            chunk_size = 1
        else:
            chunk_num = int(np.ceil(len(entities) / chunk_size))
            worker_args = [(fun, entities[i * chunk_size:(i + 1) * chunk_size], kwargs) for i in range(chunk_num)]
            print(f'Entity chunksize {chunk_size}')

        # Close session first
        self.analysis.close_session()

        # Map entities to process pool
        execution_times = []
        start_time = time.perf_counter()
        iter_num = 0
        for exec_time in pool.imap_unordered(self.worker_wrapper, worker_args):
            iter_num += 1

            # Calcualate timing info
            execution_times.append(exec_time)
            mean_exec_time = np.mean(execution_times)
            time_per_entity = mean_exec_time / (worker_num * chunk_size)
            time_elapsed = time.perf_counter() - start_time
            time_rest = time_per_entity * (len(entities) - iter_num * chunk_size)

            # Print timing info
            sys.stdout.write('\r'
                             f'[{iter_num * chunk_size}/{len(entities)}] '
                             f'{time_per_entity:.2f}s/iter '
                             f'- {timedelta(seconds=int(time_elapsed))}'
                             f'/{timedelta(seconds=int(time_elapsed + time_rest))} '
                             f'-> {timedelta(seconds=int(time_rest))} remaining ')

            # Truncate
            if len(execution_times) > 100:
                execution_times = execution_times[len(execution_times) - 100:]

        # Re-open session
        self.analysis.open_session()


    @staticmethod
    def worker_wrapper(args):

        start_time = time.perf_counter()
        # Unpack args
        fun: Callable = args[0]
        entity: Union[Entity, List[Entity]] = args[1]
        kwargs = {k: v for k, v in args[2]}

        # Re-open session in worker
        if isinstance(entity, list):
            entity[0].analysis.open_session()
        else:
            entity.analysis.open_session()

        # Run function on entity/entities
        fun(entity, **kwargs)

        # Close session again
        if isinstance(entity, list):
            entity[0].analysis.close_session()
        else:
            entity.analysis.close_session()

        return time.perf_counter() - start_time


class Entity:
    _analysis: Analysis
    attribute_table = Union[Type[sql.Animal], Type[sql.Recording], Type[sql.Roi]]
    _collection: EntityCollection

    def __init__(self, row: Union[sql.Animal, sql.Recording, sql.Roi, sql.Phase], analysis: Analysis, collection: EntityCollection = None):
        self._analysis = analysis
        self._row: Union[sql.Animal, sql.Recording, sql.Roi] = row
        self._scalar_attributes: Dict[str, Any] = None
        self._collection = collection
        # self._parents: List[Entity] = []

    def __getitem__(self, item):

        # If scalar values aren't loaded yet, do it
        if self._scalar_attributes is None:
            self._update_scalar_attributes()

        # If scalar value exists, return it
        if item in self._scalar_attributes:
            return self._scalar_attributes[item]

        # Otherwise, try to get it from file

        # Is there an HDF5 dataset?
        entity_path = f'{self.analysis.analysis_path}/{self.path}'
        with h5py.File(f'{entity_path}/data.hdf5', 'r') as f:
            if item in f:
                return f[item][:]

        # Is there a binary blob?
        if item in os.listdir(entity_path):
            with open(os.path.join(entity_path, item), 'rb') as f2:
                return pickle.load(f2)

        raise KeyError(f'No attribute with name {item} in {self}')

    def __setitem__(self, key: str, value: Any):

        # Find corresponding builtin python scalar type for numpy scalars and shape () arrays
        if isinstance(value, np.generic) or (isinstance(value, np.ndarray) and np.squeeze(value).shape == ()):
            # Convert to the corresponding Python built-in type using the item() method
            value = value.item()

        # Set scalars
        if type(value) in (str, float, int, bool, date, datetime):

            # If scalar values aren't loaded yet, do it
            if self._scalar_attributes is None:
                self._update_scalar_attributes()

            # Set local cache
            self._scalar_attributes[key] = value

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

            # Commit changes (right away if not in create mode)
            if self.analysis.mode != Mode.create:
                self.analysis.session.commit()

        # Set arrays
        else:
            # TODO: alternative to pickle dumps? Writing arbitrary raw binary data to HDF5 seems difficult
            entity_path = f'{self.analysis.analysis_path}/{self.path}'
            pending = True
            success = False
            last_exception = None
            start = time.perf_counter()
            while pending:
                try:
                    with h5py.File(f'{entity_path}/data.hdf5', 'a') as f:
                        # Save numpy arrays as datasets
                        if isinstance(value, (list, tuple, np.ndarray)):
                            if key not in f:
                                f.create_dataset(key, data=value)
                            else:
                                f[key][:] = value

                        # Dump all other types as binary strings
                        else:
                            with open(os.path.join(entity_path, key), 'wb') as f:
                                pickle.dump(value, f)
                except Exception as _exc:
                    import traceback
                    last_exception = traceback.format_exc()
                    if (time.perf_counter() - start) > self.analysis.write_timeout:
                        pending = False
                else:
                    pending = False
                    success = True

            if not success:
                raise TimeoutError(f'Failed to write attribute {key} to {key} // Traceback: {last_exception}')

    def _update_scalar_attributes(self):

        # Initialize
        if self._scalar_attributes is None:
            self._scalar_attributes = {}

        # Get all
        query = (self.analysis.session.query(self.attribute_table)
                 .filter(self.attribute_table.entity_pk == self.row.pk))

        # Update
        for row in query.all():
            self._scalar_attributes[row.name] = row.value

    @property
    def scalar_attributes(self):
        if self._scalar_attributes is None:
            self._update_scalar_attributes()
        return self._scalar_attributes

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
    # # @abstractmethod
    # def export_to(self, grp: h5py.Group):
    #     pass


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
    if len(attribute_filters) > 0:

        query = query.join(attribute_table)

        for name, comp, value in attribute_filters:
            attr_value_field = None
            if isinstance(value, float):
                attr_value_field = attribute_table.value_float
            elif isinstance(value, int):
                attr_value_field = attribute_table.value_int
            elif isinstance(value, str):
                attr_value_field = attribute_table.value_str
            elif isinstance(value, bool):
                attr_value_field = attribute_table.value_bool

            if attr_value_field is None:
                raise TypeError(f'Invalid type "{type(value)}" to filter for in attributes')

            if valid := comp == 'l':
                query = query.filter(attr_value_field < value)
            elif valid := comp == 'le':
                query = query.filter(attr_value_field <= value)
            elif valid := comp == 'e':
                query = query.filter(attr_value_field == value)
            elif valid := comp == 'ge':
                query = query.filter(attr_value_field >= value)
            elif valid := comp == 'g':
                query = query.filter(attr_value_field > value)

            # Filter for attribute name if any value filter was valid
            if valid:
                query = query.filter(attribute_table.name == name)

    return query


if __name__ == '__main__':
    pass
