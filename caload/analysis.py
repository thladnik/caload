from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

import h5py
import numpy as np
import scipy
import tifffile
import yaml
from sqlalchemy import Engine, create_engine, and_, text
from sqlalchemy.orm import Query, Session, aliased
from sqlalchemy.schema import CreateSchema
from tqdm import tqdm

import caload
from caload import utils
from caload.entities import Animal, Recording, Roi, Phase, \
    AnimalCollection, RecordingCollection, RoiCollection, PhaseCollection
from caload.sqltables import AnimalValueTable, AttributeTable, PhaseValueTable, RecordingValueTable, RoiValueTable, \
    SQLBase, \
    AnimalTable, RecordingTable, \
    RoiTable, \
    PhaseTable

__all__ = ['Analysis', 'Mode', 'open_analysis']


class Mode(Enum):
    create = 1
    analyse = 2


log = logging.getLogger(__name__)


def create_analysis(analysis_path: str, data_root: str,
                    dbhost: str = None, dbname: str = None, dbuser: str = None, dbpassword: str = None,
                    bulk_format: str = None, compression: str = 'gzip', compression_opts: Any = None,
                    shuffle_filter: bool = True, max_blob_size: int = None,
                    **kwargs) -> Analysis:

    analysis_path = Path(analysis_path).as_posix()

    if os.path.exists(analysis_path):
        print(f'Open existing analysis at {analysis_path}')
        analysis = open_analysis(analysis_path, mode=Mode.create, **kwargs)

    else:

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

        # Create schema
        engine = create_engine(f'mysql+pymysql://{dbuser}:{dbpassword}@{dbhost}?charset=utf8mb4')
        with engine.connect() as connection:
            connection.execute(CreateSchema(dbname, if_not_exists=True))
            connection.commit()

        # Create analysis data folder
        os.mkdir(analysis_path)

        # Set config data
        config = {'dbhost': dbhost,
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
        print(f'Create new analysis at {analysis_path}')
        analysis = Analysis(analysis_path, mode=Mode.create, **kwargs)

    # Scan for data folders
    recording_folders = scan_folder(data_root, [])

    # Start digesting recordings
    digest_folder(recording_folders, analysis)

    return analysis


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
    entities: Dict[tuple, Union[Animal, Recording, Roi, Phase]]

    def __init__(self, path: str, mode: Mode = Mode.analyse, lazy_init: bool = False, echo: bool = False):

        # Set mode
        self.mode = mode
        self.is_create_mode = mode is Mode.create

        # Set path as posix
        self._analysis_path = Path(path).as_posix()

        # Set up entity dictionary
        self.entities = {}

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

    def open_session(self, pool_size: int = 20):

        # Create engine
        connstr = f'{self.config["dbuser"]}:{self.config["dbpassword"]}@{self.config["dbhost"]}/{self.config["dbname"]}'
        self.sql_engine = create_engine(f'mysql+pymysql://{connstr}?charset=utf8mb4',
                                        echo=self.echo, pool_size=pool_size)

        SQLBase.metadata.create_all(self.sql_engine)

        # Create a session
        self.session = Session(self.sql_engine)

    def close_session(self):
        self.session.close()
        self.sql_engine.dispose()

        # Session attribute *needs* to be deleted, to prevent serialization error
        #  during multiprocess' pickling, because it contains weakrefs
        del self.session
        del self.sql_engine

    def add_animal(self, animal_id: str) -> Animal:
        return Animal.create(analysis=self, animal_id=animal_id)

    def animals(self, *attr_filters, **kwargs) -> AnimalCollection:
        query = _get_entity_query_by_attributes(self, AnimalTable, AnimalValueTable, *attr_filters, **kwargs)

        return AnimalCollection(analysis=self, query=query)

    def get_animal_by_id(self, animal_id: str = None) -> Union[Animal, None]:

        # Build query
        query = _get_entity_query_by_ids(self, AnimalTable, animal_id=animal_id)

        # Return data
        if query.count() == 0:
            return None
        if query.count() == 1:
            return Animal(analysis=self, row=query.first())
        raise Exception('This should not happen')

    def recordings(self, *attr_filters, **kwargs) -> RecordingCollection:
        query = _get_entity_query_by_attributes(self, RecordingTable, RecordingValueTable, *attr_filters, **kwargs)

        return RecordingCollection(analysis=self, query=query)

    def get_recordings_by_id(self, rec_id: str = None, rec_date: Union[str, date] = None, animal_id: str = None) \
            -> Union[Recording, RecordingCollection, None]:

        # Build query
        query = _get_entity_query_by_ids(self, RecordingTable, rec_id=rec_id,
                                         rec_date=rec_date, animal_id=animal_id)

        # Return data
        if query.count() == 0:
            return None
        if query.count() == 1:
            return Recording(analysis=self, row=query.first())
        return RecordingCollection(analysis=self, query=query)

    def rois(self, *attr_filters, **kwargs) -> RoiCollection:
        query = _get_entity_query_by_attributes(self, RoiTable, RoiValueTable, *attr_filters, **kwargs)

        return RoiCollection(analysis=self, query=query)

    def get_rois_by_id(self,
                       roi_id: int = None, rec_id: str = None,
                       rec_date: Union[str, date] = None, animal_id: str = None) \
            -> Union[Roi, RoiCollection, None]:

        # Build query
        query = _get_entity_query_by_ids(self, RoiTable,
                                         roi_id=roi_id, rec_id=rec_id,
                                         rec_date=rec_date, animal_id=animal_id)

        # Return data
        if query.count() == 0:
            return None
        if query.count() == 1:
            return Roi(analysis=self, row=query.first())
        return RoiCollection(analysis=self, query=query)

    def phases(self, *attr_filters, **kwargs) -> PhaseCollection:
        query = _get_entity_query_by_attributes(self, PhaseTable, PhaseValueTable, *attr_filters, **kwargs)

        return PhaseCollection(analysis=self, query=query)

    def get_phases_by_id(self,
                         phase_id: int = None, rec_id: str = None,
                         rec_date: Union[str, date] = None, animal_id: str = None) \
            -> Union[Phase, PhaseCollection, None]:

        # Build query
        query = _get_entity_query_by_ids(self, PhaseTable,
                                         phase_id=phase_id, rec_id=rec_id,
                                         rec_date=rec_date, animal_id=animal_id)

        # Return data
        if query.count() == 0:
            return None
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


    # # Join parents and filter
    # # if base_table != AnimalTable and animal_id is not None:
    # # query = query.join(AnimalTable)
    #
    # if animal_id is not None:
    #     query = query.filter(AnimalTable.id == animal_id)
    #
    # if base_table != RecordingTable and (rec_date is not None or rec_id is not None):
    #     query = query.join(RecordingTable)
    #
    # if rec_date is not None:
    #     query = query.filter(RecordingTable.date == rec_date)
    # if rec_id is not None:
    #     query = query.filter(RecordingTable.id == rec_id)
    #
    # # Filter bottom entities
    # if roi_id is not None:
    #     query = query.filter(RoiTable.id == roi_id)
    #
    # if phase_id is not None:
    #     query = query.filter(PhaseTable.id == phase_id)

    return query


def _get_entity_query_by_attributes(analysis: Analysis,
                                    base_table: Type[AnimalTable, RecordingTable, RoiTable, PhaseTable],
                                    attr_value_table: Type[AnimalValueTable, RecordingValueTable, RoiValueTable, PhaseValueTable],
                                    *attribute_filters: Tuple[Union[str, Tuple[str, str, Any]]],
                                    entity_query: Query = None,
                                    **equality_filters):

    # Convert to list
    attribute_filters = list(attribute_filters)

    # Add kwargs as equality filters
    for key, value in equality_filters.items():
        attribute_filters.append(f'{key} == {value}')

    # Create query
    _query = analysis.session.query(base_table)

    # If an entity query was provided, use it to query all entities where Entity.pk in (pk1, pk2, ...)
    if entity_query is not None:
        _subquery = entity_query.subquery().primary_key
        _query = _query.filter(base_table.pk.in_(_subquery))

    # Filter by attributes
    for filt in attribute_filters:

        # Parse filter
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
        _animal_list = analysis.animals(animal_id=animal_id)

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
        _recording_list = analysis.recordings(animal_id=animal.id, rec_date=rec_date, rec_id=rec_id)
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
        imaging_rate = 1./np.mean(np.diff(frame_times))  # Hz
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
