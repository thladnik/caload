from __future__ import annotations
from datetime import datetime, date, timedelta
import math
import os
from pathlib import Path
import pickle
import pprint
import shutil
import time
from typing import Any, Callable, Dict, List, TYPE_CHECKING, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
import sqlalchemy.exc
from sqlalchemy import case, func
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Query
from tqdm import tqdm

import caload
import caload.filter
import caload.files
from caload.handling import retry_on_operational_failure
from caload.sqltables import *

if TYPE_CHECKING:
    from caload.analysis import Analysis

__all__ = ['AnalysisEntity', 'Entity', 'EntityCollection']


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
    _synced_dataframe: SyncedDataFrame = None

    def __init__(self, entity_type: Union[str, Type[E]], analysis: Analysis, query: Query):

        # If entity type is only provided as string, set tyoe to Entity base class
        if isinstance(entity_type, str):
            self._entity_type_name = entity_type
            self._entity_type = Entity
        else:
            if type(entity_type) is not type or not issubclass(entity_type, Entity):
                raise TypeError('Entity type has to be either str or subtype of Entity')

            self._entity_type = entity_type
            self._entity_type_name = self._entity_type.__name__

        self.analysis = analysis
        self._query = query
        self._query_custom_orderby = None
        self._cache = pd.DataFrame()
        self._pending_changes: Dict[str, List[int]] = {}

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self)})'

    @retry_on_operational_failure
    def __len__(self):
        if self._entity_count < 0:
            self._entity_count = self.query.count()
        return self._entity_count

    def __iter__(self):
        self._iteration_count = -1
        self._batch_offset = 0
        return self

    @retry_on_operational_failure
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
        return self._entity_type(row=self._batch_results[self._iteration_count % self._batch_size],
                                 analysis=self.analysis)

    @retry_on_operational_failure
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
            result = [self._entity_type(row=row, analysis=self.analysis) for row in
                      self.query.offset(start).limit(stop - start)]

            # TODO: there should be a way to directly query the n-th row using 'ROW_NUMBER() % n'
            #  but it's not clear how is would work in SQLAlchemy ORM; figure out later
            # result = [self._get_entity(row) for row in self.query.offset(start).limit(stop-start)][::abs(step)]

            # Return in order
            return result[::step]

        # Return multiple attributes for all entities in collection
        if isinstance(item, (str, list, tuple)):
            if isinstance(item, str):
                item = [item]

            df = self.dataframe_of(attribute_names=item)

            # For single column, return pd.Series
            if len(df.columns) == 1:
                return df.iloc[:, 0]

            return df

        raise KeyError(f'Invalid key {item}')

    @retry_on_operational_failure
    def __setitem__(self, key, df):
        """Set attributes on individual entities in collection
        """
        if key != slice(None, None, None):
            raise KeyError(f'Invalid key {key}')
        if not isinstance(df, pd.DataFrame):
            if not isinstance(df, pd.Series):
                raise ValueError(f'Invalid value of type {type(df)}. Needs to be pandas.Series or pandas.DataFrame')
            df = pd.DataFrame(df)

        self.update(df)

    def __delitem__(self, key):

        query = (self.analysis.session.query(AttributeTable)
                 .filter(AttributeTable.name == key)
                 .filter(AttributeTable.entity_pk.in_(self.query.subquery().primary_key)))
        query.delete()
        self.analysis.session.commit()

    def restart_session(self, *args, **kwargs):
        self.analysis.restart_session(*args, **kwargs)

    @retry_on_operational_failure
    def update(self, df: pd.DataFrame):  # , overwrite: bool = False):

        _dtypes = ['str', 'float', 'int', 'date', 'datetime', 'bool', 'blob', 'path']

        # Do an upsert for each attribute to be updated
        for attr_name in df.columns:

            # Create df for insert
            df_insert = pd.DataFrame(df[attr_name])

            # Determine data type
            dtype = str(df_insert[attr_name].dtype).lower()
            if 'int' in dtype:
                data_type_str = 'int'
            elif 'float' in dtype:
                data_type_str = 'float'
            elif 'bool' in dtype:
                data_type_str = 'bool'
            elif 'object' in dtype:
                _dt = type(df_insert[attr_name].head(1).values[0])
                if _dt is str:
                    data_type_str = 'str'
                elif _dt is date:
                    data_type_str = 'date'
                elif _dt is datetime:
                    data_type_str = 'datetime'
                else:
                    data_type_str = 'blob'
            else:
                data_type_str = 'blob'

            # Rename value column
            df_insert.rename(columns={attr_name: f'value_{data_type_str}'}, inplace=True)

            # Add PK set
            df_insert['entity_pk'] = df_insert.index
            df_insert['name'] = attr_name
            df_insert['data_type'] = data_type_str
            df_insert['is_persistent'] = self.analysis.is_create_mode

            # Perform upsert
            insert_attr_data = df_insert.to_dict('records')
            insert_stmt = mysql_insert(AttributeTable).values(insert_attr_data)
            update_attr_data = {f'value_{data_type_str}': getattr(insert_stmt.inserted, f'value_{data_type_str}'),
                                # On update, reset all other value fields to None:
                                **{f'value_{dt}': None for dt in list(set(_dtypes) - {data_type_str})},
                                'is_persistent': insert_stmt.inserted.is_persistent,
                                'data_type': data_type_str}
            upsert_stmt = insert_stmt.on_duplicate_key_update(update_attr_data)
            self.analysis.session.execute(upsert_stmt)
            self.analysis.session.commit()

            # Update cache
            self._cache[df.columns] = df

    @property
    def query(self):
        """Property which should be used *exclusively* to access the Query object.
        This is important, because there is no default order to SELECTs (unless specified).
        This means that repeated iterations over the EntityCollection instance
        may return differently ordered results.
        """
        if self._query_custom_orderby is None:
            self._query = self._query.order_by(None).order_by('pk')

        return self._query

    @property
    def dataframe(self):
        if self._synced_dataframe is None:
            self._synced_dataframe = SyncedDataFrame(entity_collection=self)
        return self._synced_dataframe

    @property
    @retry_on_operational_failure
    def attribute_rows(self) -> List[Tuple[str, str]]:
        attr_query = (self.analysis.session.query(AttributeTable.name, AttributeTable.data_type)
                      .join(EntityTable)
                      .filter(EntityTable.pk.in_(self.query.subquery().primary_key))
                      .join(EntityTypeTable)
                      .filter(EntityTypeTable.name == self._entity_type_name)
                      .distinct())
        return [(row.name, row.data_type) for row in attr_query.all()]

    def info(self):

        _info = {
            'Entity type: ': self._entity_type,
            'Entity count': len(self),
            'Attributes': {name: data_type for name, data_type in self.attribute_rows}
        }

        pprint.pprint(_info, sort_dicts=False, width=120)

    def dataframe_of(self, attribute_names: List[str] = None, reload_cached: bool = False) -> pd.DataFrame:

        # If all attributes are in cache, return cached result
        if not reload_cached and len(set(attribute_names) & set(self._cache.columns)) == len(attribute_names):
            return self._cache[attribute_names].copy()

        # Check which attributes to load
        if not reload_cached:
            _attributes_cached = list(set(attribute_names) & set(self._cache.columns.tolist()))
            _attributes_to_fetch = list(set(attribute_names) - set(self._cache.columns.tolist()))
        else:
            _attributes_cached = []
            _attributes_to_fetch = attribute_names

        if self.analysis.debug:
            print('Cached attributes:', _attributes_cached)
            print('Attributes to fetch: ', _attributes_to_fetch)

        # Load attributes from database
        self._load_attributes(_attributes_to_fetch)

        # Return final DataFrame
        return self._cache[attribute_names].copy()

    @retry_on_operational_failure
    def _load_attributes(self, attribute_names: List[str]):

        # Get names and types
        all_attribute_names, _attribute_types = zip(*self.attribute_rows)

        selected_attribute_names = list(set(all_attribute_names) & set(attribute_names))
        attribute_types = {name: data_type for name, data_type in zip(all_attribute_names, _attribute_types)}

        if len(set(selected_attribute_names)) != len(selected_attribute_names):
            raise Warning(f'Attribute list contains duplicates. '
                          f'This most likely means that some attribute names in collection have different data types')

        if len(selected_attribute_names) < len(attribute_names):
            raise KeyError(f'Failed to load attributes: {list(set(attribute_names) - set(selected_attribute_names))}')

        # Build cases which return correct value field based on attr_name's data_type
        cases = []
        for attr_name in selected_attribute_names:

            # Get data type
            data_type = attribute_types[attr_name]

            if self.analysis.debug:
                print(f'> Fetch {attr_name} of type {data_type} from DB')

            # Use the appropriate column for the data_type
            cases.append(
                func.max(case(
                    (AttributeTable.name == attr_name, getattr(AttributeTable, f'value_{data_type}')),
                    else_=None)).label(attr_name)
            )

        # Construct query
        query = (
            self.analysis.session.query(
                EntityTable.pk,
                # EntityTable.id,
                *cases
            )
            .join(AttributeTable, EntityTable.pk == AttributeTable.entity_pk)
            .filter(EntityTable.pk.in_(self.query.subquery().primary_key))
            .group_by(EntityTable.pk, EntityTable.id)
        )

        # Add PK and ID to columns to match query result
        columns = ['pk', *selected_attribute_names]

        # Create DataFrame from query result
        df_new = pd.DataFrame(columns=columns, data=query.all())

        # Convert all types correctly (default result will likely contain bytestring values
        for attr_name in selected_attribute_names:

            # Get type
            attr_type = attribute_types[attr_name]

            # Cast to type
            if attr_type == 'int':
                df_new[attr_name] = df_new[attr_name].astype(int)
            elif attr_type == 'float':
                df_new[attr_name] = df_new[attr_name].astype(float)
            elif attr_type == 'str':
                df_new[attr_name] = df_new[attr_name].astype(str)
            elif attr_type == 'bool':
                df_new[attr_name] = df_new[attr_name].astype(bool)
            elif attr_type == 'date':
                df_new[attr_name] = pd.to_datetime(df_new[attr_name].apply(lambda s: s.decode()), format='%Y-%m-%d')
            elif attr_type == 'datetime':
                df_new[attr_name] = pd.to_datetime(df_new[attr_name].apply(lambda s: s.decode()), format='%Y-%m-%d %H:%M:%S')
            # Load blobs
            elif attr_type == 'blob':
                df_new[attr_name] = df_new[attr_name].apply(lambda s: pickle.loads(s) if s is not None else None)
            # Load data from path
            elif attr_type == 'path':
                df_new[attr_name] = df_new[attr_name].apply(lambda s: caload.files.read(self.analysis, s) if s is not None else None)

        # Set row index to primary key
        df_new.set_index('pk', drop=True, inplace=True)

        # Update cache
        self._cache[df_new.columns] = df_new

    def where(self, *filter_expressions, **equalities) -> EntityCollection:
        return self.analysis.get(self._entity_type, *filter_expressions, entity_query=self.query, **equalities)

    @property
    def attributes(self):
        """Get a SyncedDataFrame of all entities in this collection (rows) with their scalar attributes preloaded
        """

        # Preload scalar attributes
        self.dataframe.load_columns([n for n, dtype in self.attribute_rows if dtype not in ['path', 'blob']])

        # Return whole synced dataframe
        return self.dataframe

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
              f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {len(self)} {self._entity_type_name} entities')

        if len(self) == 0:
            print('No entities to operate on in collection')
            return

        # Prepare pool and entities
        import multiprocessing as mp
        if worker_num is None:
            worker_num = mp.cpu_count() - 4
            if len(self) < worker_num:
                worker_num = len(self)
        print(f'Start pool with {worker_num} workers')

        # Package entities together with their mapped function and arguments
        #  and make the entity table instances transient
        print(f'Prepare entities')
        t = time.perf_counter()
        kwargs = tuple([(k, v) for k, v in kwargs.items()])
        if chunk_size is None:
            worker_args = []
            chunk_size = 1
            for entity in self:
                entity.unload_row()
                worker_args.append((fun, entity, kwargs))
        else:
            chunk_num = int(np.ceil(len(self) / chunk_size))
            worker_args = []
            for i in range(chunk_num):
                entities = []
                for entity in self[i * chunk_size:(i + 1) * chunk_size]:
                    entity.unload_row()
                    entities.append((fun, entity, kwargs))
                worker_args.append(entities)

            print(f'Entity chunk_size {chunk_size}')

        print(f'> Preparation finished in {time.perf_counter() - t:.2f}s')

        # Close session first
        self.analysis.close_session()

        # Map entities to process pool
        execution_times = []
        start_time = time.time()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        print(f'Start processing at {formatted_time}')
        with (mp.Pool(processes=worker_num) as pool, tqdm(total=len(worker_args), desc='Processing', unit='entities',
                                                          smoothing=0.) as pbar):
            iterator = pool.imap_unordered(self.worker_wrapper, worker_args)
            for iter_num in range(1, len(self) + 1):

                # Next iteration
                try:
                    exec_time = next(iterator)

                # Catch
                except StopIteration:
                    pass

                # Re-raise any exception raised by worker wrapper
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
        entity: Union[Entity, List[Entity]] = args[1]
        kwargs = {k: v for k, v in args[2]}

        # Re-open session in worker and merge instances and run function
        if isinstance(entity, list):
            # Open
            entity[0].analysis.open_session(pool_size=1, echo=False)
            for e in entity:
                # Run
                EntityCollection.worker_call(fun, e, **kwargs)
            # Close
            entity[0].analysis.close_session()

        else:
            # Open
            entity.analysis.open_session(pool_size=1, echo=False)
            # Run
            EntityCollection.worker_call(fun, entity, **kwargs)
            # Close
            entity.analysis.close_session()

        elapsed_time = time.perf_counter() - start_time

        return elapsed_time

    @staticmethod
    def worker_call(fun, entity, **kwargs):

        retry_counter = 0

        while True:
            # Iterate while looking out for exceptions
            try:
                _ = fun(entity, **kwargs)

            # Catch SQL operational errors
            except sqlalchemy.exc.OperationalError as _exc:

                retry_counter += 1
                if retry_counter > 3:
                    print('Connection lost repeatedly')
                    raise _exc
                print(f'WARNING: lost connection. Retry no {retry_counter}')

                # Try reconnect
                entity.restart_session(pool_size=1)
                # entity.analysis.close_session()
                # entity.analysis.open_session(pool_size=1)

            # Raise other relevant exceptions
            except Exception as _exc:
                raise _exc
            else:
                break

    # def save_to(self, filepath: Union[str, os.PathLike]):
    #     pass

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


class Entity:
    collection_type: Type[EntityCollection] = EntityCollection
    _analysis: Analysis
    _row_pk: Union[int, None] = None
    _row: Union[EntityTable, None] = None

    # Keep references to parent table instances,
    # to avoid cold references during multiprocessing,
    # caused by lazy loading
    _parent: Union[Entity, None] = None

    def __init__(self, row: EntityTable, analysis: Analysis):
        self._analysis = analysis
        self._row = row

        self.load()

    def __repr__(self):
        return f"{self.row.entity_type.name}(id='{self.row.id}', parent={self.parent})"

    @retry_on_operational_failure
    def __contains__(self, item):
        # subquery = self.analysis.session.query(AttributeTable.pk).filter(AttributeTable.name == item).subquery()
        value_query = (self.analysis.session.query(AttributeTable)
                       .filter(AttributeTable.name == item, AttributeTable.entity_pk == self.row.pk))
        return value_query.count() > 0

    @retry_on_operational_failure
    def __delitem__(self, key):

        query = (self.analysis.session.query(AttributeTable)
                 .filter(AttributeTable.name == key)
                 .filter(AttributeTable.entity_pk == self.row.pk))
        query.delete()
        self.analysis.session.commit()

    @retry_on_operational_failure
    def __getitem__(self, item: str):

        value_query = (self.analysis.session.query(AttributeTable)
                       .filter(AttributeTable.name == item)
                       .filter(AttributeTable.entity_pk == self.row.pk))

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
        return caload.files.read(self.analysis, value)

    @retry_on_operational_failure
    def __setitem__(self, key: str, value: Any):

        attribute_row = None
        pre_data_type_str = ''

        # Get corresponding builtin python scalar type for numpy scalars
        if isinstance(value, np.generic):
            value = value.item()

        # Query attribute row if not in create mode
        if not self.analysis.is_create_mode:
            # Build query
            attribute_query = (self.analysis.session.query(AttributeTable)
                               .filter(AttributeTable.name == key)
                               .filter(AttributeTable.entity_pk == self.row.pk))

            # Evaluate
            if attribute_query.count() == 1:
                attribute_row = attribute_query.one()
                pre_data_type_str = attribute_row.data_type

            elif attribute_query.count() > 1:
                raise ValueError('Wait a minute...')

        # Create row if it doesn't exist yet
        if attribute_row is None:
            attribute_row = AttributeTable(entity=self.row, name=key, is_persistent=self.analysis.is_create_mode)
            self.analysis.session.add(attribute_row)

        # Determine data type of new value

        # Scalars
        if type(value) in (str, float, int, bool, date, datetime):

            # Set value type
            value_type_map = {str: 'str', float: 'float', int: 'int',
                              bool: 'bool', date: 'date', datetime: 'datetime'}
            value_type = value_type_map.get(type(value))

            # Some SQL dialects don't support inf float values
            if value_type == 'float' and math.isinf(value):
                value_type = 'blob'

            # Set column string
            new_data_type_str = value_type

        # Small objects
        # NOTE: there is no universal way to get the byte number of objects
        # Builtin object have __sizeof__(), but this only returns the overhead for some numpy.ndarrays
        # For numpy arrays it's numpy.ndarray.nbytes
        elif (not isinstance(value, np.ndarray) and value.__sizeof__() < self.analysis.max_blob_size) \
                or (isinstance(value, np.ndarray) and value.nbytes < self.analysis.max_blob_size):

            new_data_type_str = 'blob'

        # Large objects or object of unkown type
        else:
            new_data_type_str = 'path'

        # Handle deletion of old values
        if new_data_type_str != pre_data_type_str:

            # Delete old files
            if pre_data_type_str == 'path':
                caload.files.delete(attribute_row.value)

            # Set old value to None
            attribute_row.value = None

        # Handle path types
        if new_data_type_str == 'path':

            # Get previous path (if available)
            data_path = attribute_row.value

            # If no data_path is set yet, generate it
            if data_path is None:
                if isinstance(value, np.ndarray):
                    data_path = f'hdf5:{self.path}/data.hdf5:{key}'
                else:
                    data_path = f'pkl:{self.path}/{key.replace("/", "_")}'

            # Write to file
            caload.files.write(self.analysis, key, value, data_path)

            # Set value to data_path to write to database
            value = data_path

        # Set row type and value
        attribute_row.data_type = new_data_type_str
        attribute_row.value = value

        # Commit changes (right away if not in create mode)
        if not self.analysis.is_create_mode:
            self.analysis.session.commit()

    def __matmul__(self, other):
        return LinkEntity(linker=self, linkee=other, analysis=self.analysis)

    def restart_session(self, *args, **kwargs):
        self.analysis.restart_session(*args, **kwargs)

    @property
    def pk(self):
        return self.row.pk

    @property
    def id(self):
        return self.row.id

    def unload_row(self):

        # Save primary key of row
        self._row_pk = self.row.pk

        # Propagate
        if self.parent is not None:
            self.parent.unload_row()

        # Set row to None
        self._row = None

    def load(self):
        """To be implemented in Entity subclass. Is called after Entity.__init__
        """

    def add_child_entity(self, entity_type: Union[str, Type[Entity]], entity_id: Union[str, List[str]]):
        return self.analysis.add_entity(entity_type, entity_id, parent_entity=self)

    @property
    @retry_on_operational_failure
    def scalar_attributes(self):
        # Get all
        query = (self.analysis.session.query(AttributeTable)
                 .filter(AttributeTable.entity_pk == self.row.pk)
                 .filter(AttributeTable.data_type.not_in(['path', 'blob'])))
        return {value_row.name: value_row.value for value_row in query.all()}

    @property
    @retry_on_operational_failure
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
                try:
                    attributes[value_row.name] = value_row.value
                except Exception as e:
                    print(f'Failed to load attribute with name {value_row.name}')
                    raise e

        return attributes

    def update(self, data: Dict[str, Any]):
        """Implement update method for usage like in dict.update"""
        for key, value in data.items():
            self[key] = value

    @property
    def parent(self) -> Union[Entity, None]:
        if self._parent is None and self.row.parent is not None:
            self._parent = Entity(row=self.row.parent, analysis=self.analysis)
        return self._parent

    @property
    def path(self) -> str:
        return Path(os.path.join('entities', self.row.entity_type.name.lower(), f'{self.row.pk}_{self.id}')).as_posix()

    @property
    @retry_on_operational_failure
    def row(self):
        if self._row is None:
            if self._row_pk is None:
                raise Exception('Missing primary key to retrieve table row')
            self._row = self.analysis.session.query(EntityTable).filter(EntityTable.pk == self._row_pk).first()
        return self._row

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    def dump_file(self, src_path: str, name: str) -> str:
        """Copy arbitrary files to a dump subfolder for entity"""

        # Get original source filename
        fn = Path(src_path).as_posix().split('/')[-1]

        # Set destination path
        dest_path = os.path.join(self.path, 'dump', fn)

        # Copy file
        shutil.copy(src_path, dest_path)

        # Save relative path
        self[f'__dump_path_{name}'] = dest_path

    def get_rel_dump_path(self, name: str):
        return self[f'__dump_path_{name}']

    def get_abs_dump_path(self, name: str):
        return os.path.join(self.analysis.analysis_path, name)


class AnalysisEntity(Entity):
    pass


class LinkEntity(Entity):
    _link_row: LinkTable = None
    linker: Entity = None
    linkee: Entity = None

    def __init__(self,
                 analysis: Analysis,
                 linker: Entity = None, linkee: Entity = None,
                 link_row: LinkTable = None):

        # If link
        row = None
        if link_row is None:
            assert linker is not None and linkee is not None, ('If link_row is not provided, '
                                                               'linker and linkee are required')

            if not analysis.is_create_mode:
                link_row = (self.analysis.session.query(LinkTable)
                            .filter(LinkTable.linker == linker.pk)
                            .filter(LinkTable.linkee == linkee.pk).first())

            # Create new link row and corresponding link entity
            if link_row is None:
                row = EntityTable(entity_type=analysis.entity_type_row_map['Link'])
                link_row = LinkTable(linker=self.linker.row, linkee=self.linkee.row, entity=row)

        assert link_row is not None, 'No link_row provided'

        if row is None:
            row = link_row.entity

        if linker is None or linkee is None:
            self.linker = link_row.linker
            self.linkee = link_row.linkee

        Entity.__init__(self, row=row, analysis=analysis)


E = TypeVar('E')


class SyncedDataFrame(pd.DataFrame):

    def __init__(self, entity_collection: EntityCollection, *args, **kwargs):

        # Set stuff
        object.__setattr__(self, '_entity_collection', entity_collection)
        object.__setattr__(self, '_pending_columns', [])
        object.__setattr__(self, '_loaded_columns', [])

        # Get attribute data shape
        row_indices = [row.pk for row in entity_collection.query.all()]
        attr_rows = entity_collection.attribute_rows

        # Set up empty dataframe with placeholder values
        _data = {name: [f'<{data_type}>'] * len(row_indices) for name, data_type in attr_rows}

        # Call DataFrame init to set up DataFrame structure, but without the data yet
        super().__init__(*args, data=_data, index=row_indices, **kwargs)

    @property
    def entity_collection(self) -> EntityCollection:
        return object.__getattribute__(self, '_entity_collection')

    @property
    def pending_columns(self) -> List[str]:
        return object.__getattribute__(self, '_pending_columns')

    @property
    def loaded_columns(self) -> List[str]:
        return object.__getattribute__(self, '_loaded_columns')

    def load_columns(self, column_names: List[str]):

        # Compile list of attributes to fetch (attributes which are not loaded and weren't newly added)
        attributes_to_fetch = list(set(column_names) - set(self.loaded_columns) - set(self.pending_columns))

        # Load missing columns
        if len(attributes_to_fetch) > 0:
            df = self.entity_collection.dataframe_of(attribute_names=attributes_to_fetch)

            # Set newly loaded items
            self[df.columns] = df

            # Add to loaded columns
            self.loaded_columns.extend(attributes_to_fetch)

            # The implicit call to __setitem__ above causes newly loaded columns
            #  to be set as updated_columns, so we remove them here
            _new_updated_columns = list(set(self.pending_columns) - set(attributes_to_fetch))
            object.__setattr__(self, '_pending_columns', _new_updated_columns)

    def __getitem__(self, item):

        # Make sure all required columns are loaded from database
        if isinstance(item, (list, str)):
            self.load_columns(item if isinstance(item, list) else [item])

        # Pass original call to parent
        return pd.DataFrame.__getitem__(self, item)

    def __setitem__(self, key, value):

        _key = key
        if isinstance(_key, str):
            _key = [_key]

        if isinstance(_key, list):
            # Eliminate duplicates
            _new = list(set(_key) - set(self.pending_columns))

            # Add newly added ones to list
            self.pending_columns.extend(_new)

        # Pass original call to parent
        pd.DataFrame.__setitem__(self, key, value)

    def commit(self):
        """Save all pending changes to the database
        """

        # Call entity collection's update method with updated columns
        self.entity_collection.update(self[self.pending_columns])

        # Add changed columns to list of loaded one
        self.loaded_columns.extend(self.pending_columns)
        # Clear all pending columns since the changes are in database now
        self.pending_columns.clear()
