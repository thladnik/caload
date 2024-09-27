from __future__ import annotations

import logging
import os
import pprint
import sys
from datetime import date
from enum import Enum
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, TypeVar, Union

import yaml
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Query, Session

import caload
from caload.entities import *
from caload.filter import *
from caload.sqltables import *

__all__ = ['Analysis', 'Mode', 'open_analysis']


E = TypeVar('E')


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

        # Create type hierarchy
        def _create_entity_type(_hierarchy: Dict[str,  ...], parent_row: EntityTypeTable):
            for name, children in _hierarchy.items():
                row = EntityTypeTable(name=name, parent=parent_row)
                session.add(row)
                _create_entity_type(children, row)

        # Add Analysis type
        analysis_type_row = EntityTypeTable(name='Analysis', parent=None)
        session.add(analysis_type_row)

        # Add custom types
        _create_entity_type(hierarchy, analysis_type_row)
        session.commit()

        # Add root entity
        analysis_row = EntityTable(entity_type=analysis_type_row, id='analysis_00')
        session.add(analysis_row)
        session.commit()


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

    # Root analysis entity row
    _row: EntityTable
    # Entity type name -> entity type PK
    _entity_type_pk_map: Dict[str, int]
    # Child entity type -> parent entity type
    _entity_type_hierachy_map: Dict[str, Union[str, None]]
    # # Entity type name -> entity type row
    _entity_type_row_map: Dict[str, EntityTypeTable]

    def __init__(self, path: str, mode: Mode = Mode.analyse, lazy_init: bool = False,
                 echo: bool = False, select_analysis: str = None):

        # Set mode
        self.mode = mode
        self.is_create_mode = mode is Mode.create
        self._entity_type_pk_map = {}
        self._entity_type_hierachy_map = {}
        self._entity_type_row_map = {}

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
        # if not lazy_init:
        self.open_session()

        # Select analysis entity row
        self.select_analysis(select_analysis)

    def _load_entity_hierarchy(self):
        entity_type_rows = self.session.query(EntityTypeTable).order_by(EntityTypeTable.pk).all()

        # Create name to pk map
        self._entity_type_row_map = {str(row.name): row for row in entity_type_rows}
        self._entity_type_pk_map = {str(row.name): int(row.pk) for row in entity_type_rows}

        # Create hierarchy map
        self._entity_type_hierachy_map = {}
        for row in entity_type_rows:
            parent = None
            if row.parent is not None:
                parent = str(row.parent.name)

            self._entity_type_hierachy_map[str(row.name)] = parent

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
            if parent_entity.row._entity_type.name != self._entity_type_hierachy_map[entity_type_name]:
                raise ValueError(f'Entity type {entity_type_name} has not parent type {parent_entity.row._entity_type.name}')

            parent_entity_row = parent_entity.row
        else:
            # If no parent entity was provided, this should be top level
            if self._entity_type_hierachy_map[entity_type_name] != 'Analysis':
                raise Exception('No parent entity provided, but entity type is not top level')
            parent_entity_row = self._row

        # Add row
        # row = EntityTable(parent=parent_entity_row, entity_type_pk=self._entity_type_pk_map[entity_type_name], id=entity_id)
        row = EntityTable(parent=parent_entity_row, entity_type=self._entity_type_row_map[entity_type_name], id=entity_id)
        self.session.add(row)

        # Commit
        if not self.is_create_mode:
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

    def select_analysis(self, analysis_name: str):
        query = self.session.query(EntityTable)
        query = query.join(EntityTypeTable).filter(EntityTypeTable.name == 'Analysis')

        if analysis_name is not None:
            query = query.filter(EntityTable.id == analysis_name)

        self._row = query.order_by(EntityTable.pk).first()

    def open_session(self, pool_size: int = 20):

        # Only open if necessary
        if hasattr(self, 'session'):
            return

        # Create engine
        connstr = f'{self.config["dbuser"]}:{self.config["dbpassword"]}@{self.config["dbhost"]}/{self.config["dbname"]}'
        self.sql_engine = create_engine(f'mysql+pymysql://{connstr}', echo=echo, pool_size=pool_size)

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

    def get(self, entity_type: Union[str, Type[Entity], Type[E]], *filter_expressions: str,
            entity_query: Query = None, **equalities: Dict[str, Any]) -> EntityCollection:

        self.open_session()

        # Get entity name
        if isinstance(entity_type, str):
            entity_type_name = entity_type
        else:
            if type(entity_type) is not type or not issubclass(entity_type, Entity):
                raise TypeError('Entity type has to be either str or a subclass of Entity')

            entity_type_name = entity_type.__name__

        # Add equality filters to filter expressions
        for k, v in equalities.items():

            # Add quotes to string
            if isinstance(v, str):
                v = f'"{v}"'

            filter_expressions = filter_expressions + (f'{k} == {v}',)

        # Concat expression
        expr = ' AND '.join(filter_expressions)

        # Get filter query
        query = get_entity_query_by_attributes(entity_type_name, self.session, expr, entity_query=entity_query)

        # Return collection for resulting query
        return EntityCollection(entity_type, analysis=self, query=query)

    def get_temp_path(self, path: str):
        temp_path = os.path.join(self.analysis_path, 'temp', path)
        if not os.path.exists(temp_path):
            # Avoid error if concurrent process already created it in meantime
            os.makedirs(temp_path, exist_ok=True)

        return temp_path

    def start_slurm_task_processor(self, partition='bigmem', node_num=1, cpu_num=64, batch_size=500):

        # Create random job id
        job_id = time.strftime('%Y-%m-%d-%H-%S-%M')

        # Set shell script path
        slurm_job_path = self.get_temp_path(f'slurm_jobs/{job_id}')
        run_filepath = os.path.join(slurm_job_path, 'slurm_task_processor_run.sh')

        # Create shell script
        venv_path = os.environ['VIRTUAL_ENV']

#        job_script = \
# f"""#!/bin/sh
#
# source {venv_path}/bin/activate
# python -c "from caload.analysis import Analysis; {self}.process_tasks(batch_size={batch_size})"
# deactivate
# """

#         job_script = \
# f"""#!/bin/bash
# #SBATCH --partition={partition}
# #SBATCH --job-name=f'task_processor_job_{job_id}'      # Job name
# #SBATCH --nodes={node_num}                       # Number of nodes (X)
# #SBATCH --ntasks-per-node=1             # Number of tasks per node
# #SBATCH --time=24:00:00                 # Time limit (hh:mm:ss)
# #SBATCH --output=my_python_job_%j.out   # Standard output and error log
#
# srun --mpi=pmi2 {run_filepath}
# """

        run_script = \
            f"""#!/bin/bash
source {venv_path}/bin/activate
python -c "from caload.analysis import Analysis; {self}.process_tasks(batch_size={batch_size})"
deactivate
"""

        print(f'Create slurm run script {run_filepath}:')
        print('-' * 16)
        pprint.pprint(run_script[0])
        print('-' * 16)

        # Create shell script file for running python processing
        with open(run_filepath, 'w') as f:
            f.writelines(run_script)

        command = ['sbatch',
                   f'--partition={partition}',
                   f'--nodes={node_num}',
                   '--ntasks-per-node=1',
                   f'--cpus-per-task={cpu_num}',
                   # '-n', str(core_num),
                   f'--job-name=task_processor_{job_id}',
                   f'-o {os.path.join(slurm_job_path, "slurm-%j.out")}',
                   run_filepath]

        # command = ['sbatch',
        #            '-N', str(node_num),
        #            '-T'
        #            '-n', str(core_num),
        #            '-p', partition,
        #            '-J', f'TaskProcessorJob{job_id}',
        #            job_filepath]

        # command = ['sbatch', str(job_filepath)]
        print(f'Run command {command}')

        # Run slurm batch
        subprocess.run(command)

    def process_tasks(self, batch_size: int = 500):

        query_task = self.session.query(TaskTable).filter(TaskTable.status != 1).order_by('pk')

        while query_task.count() > 0:

            # Fetch first unfinished task
            task_row = query_task.first()

            # Continue while there are tasked entities left
            while True:

                # Query to fetch and lock first <batch_size> pending rows
                tasked_entity_query = (self.session.query(TaskedEntityTable).with_for_update()
                                       .filter(TaskedEntityTable.task_pk == task_row.pk)
                                       .filter(TaskedEntityTable.status == 0).limit(batch_size))

                # If no rows are left, break loop
                if tasked_entity_query.count() == 0:
                    break

                # Get a batch of rows
                tasked_entity_rows = tasked_entity_query.all()

                # Set status to acquired and add entity PKs to list
                entity_pks = []
                for row in tasked_entity_rows:
                    row.status = 1
                    entity_pks.append(row.entity_pk)

                # Commit changes
                self.session.commit()

                # Get function and parameters for processing
                fun, kwargs = cloudpickle.loads(task_row.target_fun), cloudpickle.loads(task_row.target_args)

                # TODO: adapt this for other entities
                # Create entity collection from PKs
                entity_collection = RoiCollection(analysis=self, query=self.session.query(RoiTable).filter(
                    RoiTable.pk.in_(entity_pks)))

                # Map function to collection
                try:
                    entity_collection.map_async(fun, **kwargs)

                except Exception as e:
                    # If errors were encountered, rollback all to status pending
                    # TODO: adapt this for other entities
                    for row in self.session.query(TaskedEntityTable).filter(
                            TaskedEntityTable.roi_pk.in_(entity_pks)).all():
                        row.status = 0  # pending
                    self.session.commit()

                    # Raise original error
                    raise e

                else:
                    # If no errors were encountered, set all tasked entity rows to finished
                    # TODO: adapt this for other entities
                    for row in self.session.query(TaskedEntityTable).filter(
                            TaskedEntityTable.roi_pk.in_(entity_pks)).all():
                        row.status = 2  # finished
                    self.session.commit()

            # Set task to finished if there aren't any tasked entities left which aren't finished
            #  (so neither pending nor acquired)
            if (self.session.query(TaskedEntityTable)
                    .filter(TaskedEntityTable.task_pk == task_row.pk)
                    .filter(TaskedEntityTable.status != 2).count()) == 0:
                task_row.status = 1
                self.session.commit()


def _recording_id_from_path(path: Union[Path, str]) -> Tuple[str, str]:
    return Path(path).as_posix().split('/')[-1].split('_')  # type: ignore


def _animal_id_from_path(path: Union[Path, str]) -> str:
    return Path(path).as_posix().split('/')[-2]


def scan_folder(root_path: str, recording_list: List[str]) -> List[str]:
    for fld in os.listdir(root_path):
        current_path = os.path.join(root_path, fld)

        # Skip files
        if not os.path.isdir(current_path):
            continue

        # Skip analysis directory
        if os.path.exists(os.path.join(current_path, 'metadata.db')):
            continue

        if 'suite2p' in os.listdir(current_path):
            recording_list.append(current_path)
            continue

        recording_list = scan_folder(current_path, recording_list)

    return recording_list


def create_animal(analysis: caload.analysis.Analysis, path: str) -> caload.entities.Animal:
    # Create animal
    animal_id = _animal_id_from_path(path)
    animal_path = '/'.join(Path(path).as_posix().split('/')[:-1])
    animal = analysis.add_animal(animal_id=animal_id)
    animal['animal_id'] = animal_id

    # Search for zstacks
    zstack_names = []
    for fn in os.listdir(animal_path):
        path = os.path.join(path, fn)
        if os.path.isdir(path):
            continue
        if 'zstack' in fn:
            if fn.lower().endswith(('.tif', '.tiff')):
                zstack_names.append(fn)

    if len(zstack_names) > 0:
        if len(zstack_names) > 1:
            print(f'WARNING: multiple zstacks detected, using {zstack_names[0]}')

        print(f'Add zstack {zstack_names[0]}')

        animal['zstack_fn'] = zstack_names[0]
        animal['zstack'] = tifffile.imread(os.path.join(animal_path, zstack_names[0]))

    # Add metadata
    add_metadata(animal, animal_path)

    # Commit animal
    analysis.session.commit()

    return animal


def digest_folder(folder_list: List[str], analysis: caload.analysis.Analysis):
    print(f'Process folders: {folder_list}')
    for recording_path in folder_list:

        # recording_path = Path(recording_path).as_posix()
        print(f'Recording folder {recording_path}')

        # Check if animal exists
        animal_id = _animal_id_from_path(recording_path)
        _animal_list = analysis.animals(f'animal_id == "{animal_id}"')

        if len(_animal_list) == 0:
            # Add new animal
            animal = create_animal(analysis, recording_path)
        else:
            animal = _animal_list[0]

        # Create debug folder
        debug_folder_path = os.path.join(recording_path, 'debug')
        if not os.path.exists(debug_folder_path):
            os.mkdir(debug_folder_path)

        # Get recording
        # Expected recording folder format "<rec_date('YYYY-mm-dd')>_<rec_id>_*"
        rec_date, rec_id, *_ = _recording_id_from_path(recording_path)
        rec_date = caload.utils.parse_date(rec_date)
        _recording_list = analysis.recordings(f'animal_id == "{animal.id}"',
                                              f'rec_date == {rec_date}',
                                              f'rec_id == "{rec_id}"')
        # Add recording
        if len(_recording_list) > 0:
            print('Recording already exists. Skip')
            continue

        recording = animal.add_recording(rec_date=rec_date, rec_id=rec_id)
        recording['animal_id'] = animal_id
        recording['rec_date'] = rec_date
        recording['rec_id'] = rec_id

        # Add metadata
        add_metadata(recording, recording_path)

        # Load s2p processed data
        s2p_path = os.path.join(recording_path, 'suite2p', 'plane0')

        # Load suite2p's analysis options
        print('Include suite2p ops')
        ops = np.load(os.path.join(s2p_path, 'ops.npy'), allow_pickle=True).item()
        unravel_dict(ops, recording, 's2p')

        print('Calculate frame timing of signal')
        with h5py.File(os.path.join(recording_path, 'Io.hdf5'), 'r') as io_file:

            mirror_position = np.squeeze(io_file['ai_y_mirror_in'])[:]
            mirror_time = np.squeeze(io_file['ai_y_mirror_in_time'])[:]

            # Calculate frame timing
            frame_idcs, frame_times = calculate_ca_frame_times(mirror_position, mirror_time)

            record_group_ids = io_file['__record_group_id'][:].squeeze()
            record_group_ids_time = io_file['__time'][:].squeeze()

            ca_rec_group_id_fun = scipy.interpolate.interp1d(record_group_ids_time, record_group_ids, kind='nearest')

        print('Load ROI data')
        fluorescence = np.load(os.path.join(s2p_path, 'F.npy'), allow_pickle=True)
        spikes_all = np.load(os.path.join(s2p_path, 'spks.npy'), allow_pickle=True)
        roi_stats_all = np.load(os.path.join(s2p_path, 'stat.npy'), allow_pickle=True)
        # In some suite2p versions the iscell file may be missing?
        try:
            iscell_all = np.load(os.path.join(s2p_path, 'iscell.npy'), allow_pickle=True)
        except:
            iscell_all = None

        # Check if frame times and signal match
        if frame_times.shape[0] != fluorescence.shape[1]:
            print(f'Detected frame times\' length doesn\'t match frame count. '
                  f'Detected frame times ({frame_times.shape[0]}) / Frames ({fluorescence.shape[1]})')

            # Shorten signal
            if frame_times.shape[0] < fluorescence.shape[1]:
                fluorescence = fluorescence[:, :frame_times.shape[0]]
                print('Truncated signal at end to resolve mismatch. Check debug output to verify')

            # Shorten frame times
            else:
                frame_times = frame_times[:fluorescence.shape[1]]
                print('Truncated detected frame times at end to resolve mismatch. Check debug output to verify')

        # Get imaging rate from sync signal
        imaging_rate = 1. / np.mean(np.diff(frame_times))  # Hz
        print(f'Estimated imaging rate {imaging_rate:.2f}Hz')

        # Save to recording
        recording['roi_num'] = fluorescence.shape[0]
        recording['signal_length'] = fluorescence.shape[0]
        recording['imaging_rate'] = imaging_rate
        recording['ca_times'] = frame_times
        record_group_ids = ca_rec_group_id_fun(frame_times)
        recording['record_group_ids'] = record_group_ids

        # Commit recording
        analysis.session.commit()

        # Add suite2p's analysis ROI stats
        print('Add ROI stats and signals')
        for roi_id in tqdm(range(fluorescence.shape[0])):
            # Create ROI
            roi = recording.add_roi(roi_id=roi_id)
            roi['animal_id'] = animal_id
            roi['rec_date'] = rec_date
            roi['rec_id'] = rec_id
            roi['roi_id'] = roi_id

            roi_stats = roi_stats_all[roi_id]

            # Write ROI stats
            roi.update({f's2p/{k}': v for k, v in roi_stats.items()})

            # Write data
            fluores = fluorescence[roi_id]
            spikes = spikes_all[roi_id]
            roi['fluorescence'] = fluores
            roi['spikes'] = spikes

            if iscell_all is not None:
                iscell = iscell_all[roi_id]
                roi['iscell'] = iscell

        # Commit rois
        analysis.session.commit()

        print('Add display phase data')
        with h5py.File(os.path.join(recording_path, 'Display.hdf5'), 'r') as disp_file:

            # Get attributes
            recording.update({f'display/attrs/{k}': v for k, v in disp_file.attrs.items()})

            for key1, member1 in tqdm(disp_file.items()):

                # If dataset, write to file
                if isinstance(member1, h5py.Dataset):
                    recording[f'display/{key1}'] = member1[:]
                    continue

                # Otherwise it's a group -> keep going

                # Add phase
                if 'phase' in key1:

                    phase_id = int(key1.replace('phase', ''))
                    phase = recording.add_phase(phase_id=phase_id)
                    phase['animal_id'] = animal_id
                    phase['rec_date'] = rec_date
                    phase['rec_id'] = rec_id
                    phase['phase_id'] = phase_id

                    # Add calcium start/end indices
                    in_phase_idcs = np.where(record_group_ids == phase_id)[0]
                    start_index = np.argmin(np.abs(frame_times - frame_times[in_phase_idcs[0]]))
                    end_index = np.argmin(np.abs(frame_times - frame_times[in_phase_idcs[-1]]))
                    phase['ca_start_index'] = start_index
                    phase['ca_end_index'] = end_index

                    # Write attributes
                    phase.update({k: v for k, v in member1.attrs.items()})

                    # Write datasets
                    for key2, member2 in member1.items():
                        if isinstance(member2, h5py.Dataset):
                            phase[key2] = member2[:]

                # Add other data
                else:
                    # Write attributes
                    recording.update({f'display/{key1}/{k}': v for k, v in member1.attrs.items()})

                    # Get datasets
                    for key2, member2 in member1.items():
                        if isinstance(member2, h5py.Dataset):
                            recording[f'display/{key1}/{key2}'] = member2[:]

        # Commit phases and display data
        analysis.session.commit()


def add_metadata(entity: caload.entities.Entity, folder_path: str):
    """Function searches for and returns metadata on a given folder path

    Function scans the `folder_path` for metadata yaml files (ending in `meta.yaml`)
    and returns a dictionary containing their contents
    """

    meta_files = [f for f in os.listdir(folder_path) if f.endswith('metadata.yaml')]

    log.info(f'Found {len(meta_files)} metadata files in {folder_path}.')

    metadata = {}
    for f in meta_files:
        with open(os.path.join(folder_path, f), 'r') as stream:
            try:
                metadata.update(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    # Add metadata
    unravel_dict(metadata, entity, 'metadata')


def unravel_dict(dict_data: dict, entity: caload.entities.Entity, path: str):
    for key, item in dict_data.items():
        if isinstance(item, dict):
            unravel_dict(item, entity, f'{path}/{key}')
            continue
        entity[f'{path}/{key}'] = item


def calculate_ca_frame_times(mirror_position: np.ndarray, mirror_time: np.ndarray):
    peak_prominence = (mirror_position.max() - mirror_position.min()) / 4
    peak_idcs, _ = scipy.signal.find_peaks(mirror_position, prominence=peak_prominence)
    trough_idcs, _ = scipy.signal.find_peaks(-mirror_position, prominence=peak_prominence)

    # Find first trough
    first_peak = peak_idcs[0]
    first_trough = trough_idcs[trough_idcs < first_peak][-1]

    # Discard all before (and including) first trough
    trough_idcs = trough_idcs[first_trough < trough_idcs]
    frame_idcs = np.sort(np.concatenate([trough_idcs, peak_idcs]))

    # Get corresponding times
    frame_times = mirror_time[frame_idcs]

    return frame_idcs, frame_times
