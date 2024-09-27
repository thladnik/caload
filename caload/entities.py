from __future__ import annotations

import math
import os
import pickle
import pprint

import sys
import time
from abc import abstractmethod

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, TYPE_CHECKING, Tuple, Type, TypeVar, Union


import cloudpickle
import h5py
import numpy as np
import pandas as pd
from sqlalchemy import case, func, text

from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Query, aliased, joinedload
from tqdm import tqdm

import caload
from caload.sqltables import *
from caload import files, utils

if TYPE_CHECKING:
    from caload.analysis import Analysis

__all__ = ['Entity', 'EntityCollection']


class Entity:
    _type: Type[Entity]
    _analysis: Analysis
    _row: EntityTable
    entity_type_name: str
    _attribute_name_pk_map: Dict[str, int] = {}

    # Keep references to parent table instances,
    # to avoid cold references during multiprocessing,
    # caused by lazy loading
    _parent_row: EntityTable

    def __init__(self, row: EntityTable, analysis: Analysis):
        self._analysis = analysis
        self._row = row
        self.entity_type_name = row.entity_type.name
        self._parent_row = row.parent

    def __repr__(self):
        return f"{self.__class__.__name__}(id='{self.row.id}', parent={self.parent})"

    def __contains__(self, item):
        # subquery = self.analysis.session.query(AttributeTable.pk).filter(AttributeTable.name == item).subquery()
        value_query = (self.analysis.session.query(AttributeTable)
                       .filter(AttributeTable.name == item, AttributeTable.entity_pk == self.row.pk))
        return value_query.count() > 0

    def __getitem__(self, item: str):

        value_query = (self.analysis.session.query(AttributeTable)
                       .filter(AttributeTable.name == item, AttributeTable.entity_pk == self.row.pk))

        if value_query.count() == 0:
            raise KeyError(f'Attribute {item} not found for entity {self}')

        # Fetch first (only row)
        value_row = value_query.first()

        column_str = value_row.data_type
        value = value_row.value

        # Anything that isn't a referenced path gets returned immediately
        if column_str != 'path':
            return value

        # Read from filepath
        return files.read(self.analysis, value)

    def __setitem__(self, key: str, value: Any):

        # Find corresponding builtin python scalar type for numpy scalars
        if isinstance(value, np.generic):
            value = value.item()

        attribute_row = None
        # Query attribute row if not in create mode
        if not self.analysis.is_create_mode:
            # Build query
            attribute_row = (self.analysis.session.query(AttributeTable)
                           .filter(AttributeTable.name == key, AttributeTable.entity_pk == self.row.pk))

            # Evaluate
            if attribute_row.count() == 1:
                attribute_row = attribute_row.one()
            elif attribute_row.count() > 1:
                raise ValueError('Wait a minute...')

        # Create row if it doesn't exist yet
        if attribute_row is None:
            attribute_row = AttributeTable(entity=self.row, name=key, is_persistent=self.analysis.is_create_mode)
            self.analysis.session.add(attribute_row)

        # Set scalars
        if type(value) in (str, float, int, bool, date, datetime):

            # Set value type
            value_type_map = {str: 'str', float: 'float', int: 'int',
                              bool: 'bool', date: 'date', datetime: 'datetime'}
            value_type = value_type_map.get(type(value))

            # Some SQL dialects don't support inf float values
            if value_type == 'float' and math.isinf(value):
                value_type = 'blob'

            # Set column string
            data_type_str = value_type

        # NOTE: there is no universal way to get the byte number of objects
        # Builtin object have __sizeof__(), but this only returns the overhead for some numpy.ndarrays
        # For numpy arrays it's numpy.ndarray.nbytes
        # Set small objects
        elif (not isinstance(value, np.ndarray) and value.__sizeof__() < self.analysis.max_blob_size) \
                or (isinstance(value, np.ndarray) and value.nbytes < self.analysis.max_blob_size):

            # Set value type
            data_type_str = 'blob'

        # Set large objects
        else:

            # Set value type
            data_type_str = 'path'

            # Write any non-scalar data that is too large according to specified bulk storage format
            data_path = attribute_row.value

            # If not data_path is set yet, generate it
            if data_path is None:
                if isinstance(value, np.ndarray):
                    data_path = f'hdf5:{self.path}/data.hdf5:{key}'
                else:
                    data_path = f'pkl:{self.path}/{key.replace("/", "_")}'

            # Write to file
            files.write(self.analysis, key, value, data_path)

            # Set value to data_path to write to database
            value = data_path

        # Reset old value in case it was set to different type before
        if type(attribute_row.data_type) is str and attribute_row.data_type != data_type_str and attribute_row.value is not None:
            attribute_row.value = None

        # Set row type and value
        attribute_row.data_type = data_type_str
        attribute_row.value = value

        # Commit changes (right away if not in create mode)
        if not self.analysis.is_create_mode:
            self.analysis.session.commit()

    @property
    def pk(self):
        return self.row.pk

    @property
    def id(self):
        return self.row.id

    def add_child_entity(self, entity_type: Union[str, Type[Entity]], entity_id: str):
        return self.analysis.add_entity(entity_type, entity_id, parent_entity=self)

    @property
    def scalar_attributes(self):
        # Get all
        query = (self.analysis.session.query(AttributeTable)
                 .filter(AttributeTable.entity_pk == self.row.pk)
                 .filter(AttributeTable.data_type.not_in(['path', 'blob'])))
        return {value_row.name: value_row.value for value_row in query.all()}

    @property
    def attributes(self):
        # Get all
        query = (self.analysis.session.query(AttributeTable)
                 .filter(AttributeTable.entity_pk == self.row.pk))

        # Update
        attributes = {}
        for value_row in query.all():
            if value_row.data_type == 'path':
                attributes[value_row.name] = self[value_row.name]
            else:
                attributes[value_row.name] = value_row.value

        return attributes

    def update(self, data: Dict[str, Any]):
        """Implement update method for usage like in dict.update"""
        for key, value in data.items():
            self[key] = value

    @property
    def parent(self):
        if self._parent_row is None:
            return None
        return Entity(row=self._parent_row, analysis=self.analysis)

    @property
    def path(self) -> str:
        if self.parent is not None:
            return Path(os.path.join(self.parent.path, self.entity_type_name.lower(), self.id)).as_posix()
        return Path(os.path.join('entities', self.entity_type_name.lower(), self.id)).as_posix()

    @property
    def row(self):
        return self._row

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    def create_file(self):
        entity_abs_path = os.path.join(self.analysis.analysis_path, self.path)

        # Create directoty of necessary
        if not os.path.exists(entity_abs_path):
            os.makedirs(entity_abs_path)

        # Create data file
        path = os.path.join(entity_abs_path, 'data.hdf5')
        if not os.path.exists(path):
            with h5py.File(path, 'w') as _:
                pass


E = TypeVar('E')


class EntityCollection:
    analysis: Analysis
    _query: Query
    _entity_count: int = -1
    _iteration_count: int = -1
    _batch_offset: int = 0
    _batch_size: int = 100
    _batch_results: List[EntityTable]
    _entity_type: Union[Type[Entity], Type[E]]
    _entity_type_name: str

    def __init__(self, entity_type: Union[str, Type[E]], analysis: Analysis, query: Query):

        if isinstance(entity_type, str):
            self._entity_type_name = entity_type
            self._entity_type = Entity
        else:
            if type(entity_type) is not type or not issubclass(entity_type, Entity):
                raise TypeError('Entity type has to be either str or a subclass of Entity')

            self._entity_type = entity_type
            self._entity_type_name = self._entity_type.__name__

            # if self.entity_type_name in globals() and issubclass(globals()[self.entity_type_name], Entity):
            #     self.entity_type = globals()[self.entity_type_name]
            # else:
            #     self.entity_type = Entity

        self.analysis = analysis
        self._query = query
        self._query_custom_orderby = False

    def __repr__(self):
        return f'{self._entity_type_name}Collection({len(self)})'

    def __len__(self):
        if self._entity_count < 0:
            self._entity_count = self.query.count()
        return self._entity_count

    def __iter__(self):
        return self

    def __next__(self) -> Union[Entity, E]:

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
        return self._entity_type(row=self._batch_results[self._iteration_count % self._batch_size], analysis=self.analysis)

    def __getitem__(self, item) -> Union[Entity, E, List[Entity], List[E], pd.DataFrame]:

        # Return single entity
        if isinstance(item, (int, np.integer)):
            if item < 0:
                item = len(self) + item
            return self._entity_type(row=self.query.offset(item).limit(1)[0], analysis=self.analysis)

        # Return slice
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step == 0:
                raise KeyError('Invalid step size 0')

            # Get data
            result = [self._entity_type(row=row, analysis=self.analysis) for row in self.query.offset(start).limit(stop - start)]

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

    def __setitem__(self, key, df):
        """Set attributes on individual entities in collection
        """
        if key != slice(None, None, None):
            raise KeyError(f'Invalid key {key}')
        if not isinstance(df, pd.DataFrame):
            if not isinstance(df, pd.Series):
                raise ValueError(f'Invalid value of type {type(df)}. Needs to be pandas.Series or pandas.DataFrame')
            df = pd.DataFrame(df)

        for attr_name in df.columns:

            # Create df for insert
            df_insert = pd.DataFrame(df[attr_name])

            # Determine data type
            dtype = str(df_insert[attr_name].dtype).lower()
            if 'int' in dtype:
                dtype_str = 'int'
            elif 'float' in dtype:
                dtype_str = 'float'
            elif 'object' in dtype:
                dtype_str = 'str'
            else:
                raise TypeError('')

            # Rename value column
            data_type_str = dtype_str
            df_insert.rename(columns={attr_name: data_type_str}, inplace=True)

            # Add PK set
            df_insert['entity_pk'] = df_insert.index
            df_insert['name'] = attr_name
            df_insert['data_type'] = data_type_str
            df_insert['is_persistent'] = self.analysis.is_create_mode

            insert_attr_data = df_insert.to_dict('records')

            insert_stmt = mysql_insert(AttributeTable).values(insert_attr_data)

            update_attr_data = {data_type_str: getattr(insert_stmt.inserted, data_type_str),
                                'is_persistent': insert_stmt.inserted.is_persistent,
                                'data_type': data_type_str}

            upsert_stmt = insert_stmt.on_duplicate_key_update(update_attr_data)
            self.analysis.session.execute(upsert_stmt)
            self.analysis.session.commit()

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

    def _column(self, attribute_name: str, include_blobs: bool = False):

        # Subquery on entity primary keys
        entity_query = self.query.subquery().primary_key

        # Query attribute values
        value_query = (self.analysis.session.query(AttributeTable)
                       .filter(AttributeTable.name == attribute_name)
                       .filter(AttributeTable.entity_pk.in_(entity_query)))

        return value_query.all()

    def _dataframe_of(self, attribute_names: List[str] = None,
                      include_blobs: bool = False, include_bulk: bool = False) -> pd.DataFrame:

        # Add to excluded list
        excluded = []
        if not include_blobs:
            excluded.append('blob')
        if not include_bulk:
            excluded.append('path')

        # Query distinct attribute names and their corresponding types
        attribute_query = self.analysis.session.query(AttributeTable.name, AttributeTable.data_type).join(
            EntityTable).join(EntityTypeTable).filter(EntityTypeTable.name == self._entity_type_name,
                                                      AttributeTable.data_type.notin_(excluded))
        # Filter for user-defined attributes
        if attribute_names is not None:
            attribute_query = attribute_query.filter(AttributeTable.name.in_(attribute_names))

        # Get names and types
        selected_attribute_names, attribute_types = zip(*attribute_query.distinct().all())

        if attribute_names is not None and len(attribute_names) != len(selected_attribute_names):
            raise Warning(f'Number selected attributes does not match number requested. '
                          f'{len(attribute_types)} != {len(selected_attribute_names)}. '
                          f'Some requested attributes may be blob or path and not included')

        # Build case for each attribute so query returns correct type field
        cases = []
        for attr_name in selected_attribute_names:
            cases.append(
                func.max(case(
                    (AttributeTable.name == attr_name,
                     case((AttributeTable.data_type == 'int', AttributeTable.value_int),
                          (AttributeTable.data_type == 'str', AttributeTable.value_str),
                          (AttributeTable.data_type == 'float', AttributeTable.value_float),
                          (AttributeTable.data_type == 'date', AttributeTable.value_date),
                          (AttributeTable.data_type == 'datetime', AttributeTable.value_datetime),
                          (AttributeTable.data_type == 'blob', AttributeTable.value_blob),
                          (AttributeTable.data_type == 'path', AttributeTable.value_path),
                          else_=None)
                     ),
                    else_=None)).label(name=attr_name)
            )

        # Create query
        query = (
            self.analysis.session.query(
                EntityTable.pk,
                EntityTable.id,
                *cases  # Dynamically added case expressions
            )
            .join(AttributeTable, EntityTable.pk == AttributeTable.entity_pk)
            # .filter(EntityTable.pk.in_([13, 15, 4201]))
            .filter(EntityTable.pk.in_(self.query.subquery().primary_key))
            .group_by(EntityTable.pk, EntityTable.id)
        )

        # Add PK and ID to columns to match query result
        columns = ('pk', 'entity_id') + selected_attribute_names

        # Create DataFrame from query result
        df = pd.DataFrame(columns=columns, data=query.all())

        # Convert all types correctly (default result will likely contain bytestring values
        for attr_name, attr_type in zip(selected_attribute_names, attribute_types):
            if attr_type == 'int':
                df[attr_name] = df[attr_name].astype(int)
            elif attr_type == 'float':
                df[attr_name] = df[attr_name].astype(float)
            elif attr_type == 'str':
                df[attr_name] = df[attr_name].astype(str)
            elif attr_type == 'date':
                df[attr_name] = pd.to_datetime(df[attr_name].apply(lambda s: s.decode()), format='%Y-%m-%d')
            elif attr_type == 'datetime':
                df[attr_name] = pd.to_datetime(df[attr_name].apply(lambda s: s.decode()), format='%Y-%m-%d %H:%M:%S')
            elif attr_type == 'blob':
                df[attr_name] = df[attr_name].apply(lambda s: pickle.loads(s) if s is not None else None)
            elif attr_type == 'path':
                df[attr_name] = df[attr_name].apply(lambda s: files.read(self.analysis, s) if s is not None else None)

        # Set row index to PKs
        df.set_index('pk', drop=True, inplace=True)

        # Return final DataFrame
        return df

    def where(self, *filter_expressions, entity_query: Query = None, **equalities) -> EntityCollection:
        # query = caload.filter.get_entity_query_by_attributes(self._entity_type_name, self.analysis.session,
        #                                                      ' AND '.join(expr), entity_query=self.query)

        return self.analysis.get(self._entity_type, *filter_expressions, entity_query=entity_query, **equalities)

        # return EntityCollection(self._entity_type_name, analysis=self.analysis, query=query)

    @property
    def scalar_attributes(self):
        """Get a pandas.DataFrame of all entities in this collection (rows) and their attributes
        """
        return self._dataframe_of(attribute_names=None, include_blobs=False, include_bulk=False)

    @property
    def attributes(self):
        """Get a pandas.DataFrame of all entities in this collection (rows) and their attributes, including binary data
        """
        return self._dataframe_of(attribute_names=None, include_blobs=True, include_bulk=False)

    # def sortby(self, name: str, order: str = 'ASC'):
    #     # TODO: implement sorting
    #     pass

    def map(self, fun: Callable, **kwargs) -> Any:
        """Sequentially apply a function to each Entity of the collection (kwargs are passed onto the function)
        """

        print(f'Run function {fun.__name__} on {self} with args '
              f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {len(self)} entities')

        for entity in tqdm(self):
            fun(entity, **kwargs)

    def map_async(self, fun: Callable, chunk_size: int = None, worker_num: int = None, **kwargs) -> Any:
        """Concurrently apply a function to each Entity of the collection (kwargs are passed onto the function)

        worker_num: int number of subprocess workers to spawn for parallel execution
        chunk_size: int (optional) size of chunks for batched execution of function to decrease overhead
            (note that for batch execution the first argument
            of fun is going to be of type List[Entity] instead of Entity)
        """

        print(f'Run function {fun.__name__} on {self} with args '
              f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {len(self)} entities')

        # Prepare pool and entities
        import multiprocessing as mp
        if worker_num is None:
            worker_num = mp.cpu_count() - 4
            if len(self) < worker_num:
                worker_num = len(self)
        print(f'Start pool with {worker_num} workers')

        print(f'Prepare entities')
        t = time.perf_counter()
        kwargs = tuple([(k, v) for k, v in kwargs.items()])
        if chunk_size is None:
            worker_args = [(fun, e, kwargs) for e in self]
            chunk_size = 1
        else:
            chunk_num = int(np.ceil(len(self) / chunk_size))
            worker_args = [(fun, self[i * chunk_size:(i + 1) * chunk_size], kwargs) for i in range(chunk_num)]
            print(f'Entity chunksize {chunk_size}')
        print(f'> Preparation finished in {time.perf_counter() - t:.2f}s')

        # Close session first
        self.analysis.close_session()

        # Map entities to process pool
        execution_times = []
        start_time = time.time()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        print(f'Start processing at {formatted_time}')
        with (mp.Pool(processes=worker_num) as pool,
              tqdm(total=len(worker_args), desc='Processing', unit='iter', smoothing=0.) as pbar):
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

                pbar.update(chunk_size)
                pbar.set_postfix({
                    'time_per_iter': f'{time_per_entity:.2f}s',
                    'elapsed': str(timedelta(seconds=int(time_elapsed))),
                    'eta': str(timedelta(seconds=int(time_rest))),
                })

                # # Print timing info
                # sys.stdout.write('\r'
                #                  f'[{iter_num * chunk_size}/{len(self)}] '
                #                  f'{time_per_entity:.2f}s/iter '
                #                  f'- {timedelta(seconds=int(time_elapsed))}'
                #                  f'/{timedelta(seconds=int(time_elapsed + time_rest))} '
                #                  f'-> {timedelta(seconds=int(time_rest))} remaining ')

        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(f'\nFinish processing at {formatted_time}')

        # Re-open session
        self.analysis.open_session()

    @staticmethod
    def worker_wrapper(args):
        """Subprocess wrapper function for concurrent execution, which handles the MySQL session
        and provides feedback on execution time to parent process
        """

        start_time = time.perf_counter()
        # Unpack args
        fun: Callable = args[0]
        entity: Union[Entity, EntityCollection, List[Entity]] = args[1]
        kwargs = {k: v for k, v in args[2]}

        # Re-open session in worker
        if isinstance(entity, list):
            entity[0].analysis.open_session(pool_size=1, echo=False)
            close_session = entity[0].analysis.close_session
        else:
            entity.analysis.open_session(pool_size=1, echo=False)
            close_session = entity.analysis.close_session

        # Run function on entity
        res = fun(entity, **kwargs)

        # Close session again
        close_session()

        elapsed_time = time.perf_counter() - start_time

        return elapsed_time

    def save_to(self, filepath: Union[str, os.PathLike]):
        pass

    # def create_task(self, fun: Callable, **kwargs):
    #
    #     print(f'Create new task for {fun.__name__} with args '
    #           f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {len(self)} entities')
    #
    #     funstr = cloudpickle.dumps(fun)
    #     argstr = cloudpickle.dumps(kwargs)
    #
    #     task_row = TaskTable(target_fun=funstr, target_args=argstr)
    #     self.analysis.session.add(task_row)
    #     self.analysis.session.commit()
    #
    #     for entity in self:
    #
    #         if isinstance(entity, Animal):
    #             _id = {'animal_pk': entity.row.pk}
    #         elif isinstance(entity, Recording):
    #             _id = {'recording_pk': entity.row.pk}
    #         elif isinstance(entity, Roi):
    #             _id = {'roi_pk': entity.row.pk}
    #         elif isinstance(entity, Phase):
    #             _id = {'phase_pk': entity.row.pk}
    #         else:
    #             raise ValueError(f'What is a {entity}?')
    #
    #         # Add to task
    #         self.analysis.session.add(TaskedEntityTable(task_pk=task_row.pk, **_id))
    #
    #     self.analysis.session.commit()
