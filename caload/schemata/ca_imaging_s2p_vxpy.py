"""Structure and functions for analysis of imaging experiments done with vxPy.

Expected raw data folder/file structure:

data_root
|
|   + animal_id_01
|   +-- zstack_for_animal_id_01.tif (optional)
|   +-- recording_id_01
|   |   +-- suite2p/
|   |       +-- plane0/
|   |       +-- ...
|   |       +-- planeX/
|   |   +-- Camera.hdf5
|   |   +-- ...
|   |   +-- Worker.hdf5
|   |   +-- ants_registration/ (optional)
|   ...
|   +-- recording_id_n
|   |   +-- suite2p/
|   |       +-- plane0/
|   |       +-- ...
|   |       +-- planeX/
|   |   +-- Camera.hdf5
|   |   +-- ...
|   |   +-- Worker.hdf5
|   |   +-- ants_registration/ (optional)
|
|   + animal_id_02
|   +-- zstack_for_animal_id_02.tif (optional)
|   +-- recording_id_01
|   |   +-- suite2p/
|   |       +-- plane0/
|   |       +-- ...
|   |       +-- planeX/
|   |   +-- Camera.hdf5
|   |   +-- ...
|   |   +-- Worker.hdf5
|   |   +-- ants_registration/ (optional)
|   ...
|   +-- recording_id_n
|   |   +-- suite2p/
|   |       +-- plane0/
|   |       +-- ...
|   |       +-- planeX/
|   |   +-- Camera.hdf5
|   |   +-- ...
|   |   +-- Worker.hdf5
|   |   +-- ants_registration/ (optional)
...

Output data hierarchy:

Animal
+-- Recording
    +-- Roi
    +-- Phase

# TODO: future hierarchy to better accommodate multi-plane recordings (right now recording data is duplicated instead):
        Animal
        +-- Recording
            +-- Phase
            +-- Plane
                +-- Roi


"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, List, Tuple, Type, Union

import h5py
import numpy as np
import pandas as pd
import scipy
import yaml
from tifffile import tifffile
from tqdm import tqdm

import caload
from caload.entities import *

__all__ = ['Animal', 'Recording', 'Roi', 'Phase', 'AnimalCollection',
           'RecordingCollection', 'RoiCollection', 'PhaseCollection',
           'digest', 'schema']


class AnimalCollection(EntityCollection):
    pass


class RecordingCollection(EntityCollection):
    pass


class RoiCollection(EntityCollection):
    pass


class PhaseCollection(EntityCollection):
    pass


class Animal(Entity):
    collection_type = AnimalCollection

    @property
    def recordings(self) -> RecordingCollection:
        return self.analysis.get(Recording, animal_id=self.id)

    @property
    def rois(self) -> RoiCollection:
        return self.analysis.get(Roi, animal_id=self.id)

    @property
    def phases(self) -> PhaseCollection:
        return self.analysis.get(Phase, animal_id=self.id)


class Recording(Entity):
    collection_type = RecordingCollection

    parent_type = Animal
    animal: Animal

    def load(self):
        self.animal = Animal(analysis=self.analysis, row=self.row.parent)

    @property
    def rois(self) -> RoiCollection:
        return self.analysis.get(Roi, animal_id=self.animal.id, rec_id=self.id)

    @property
    def phases(self) -> PhaseCollection:
        return self.analysis.get(Phase, animal_id=self.animal.id, rec_id=self.id)


class Roi(Entity):
    collection_type = RoiCollection

    parent_type = Recording
    animal: Animal
    recording: Recording

    def load(self):
        self.recording = Recording(analysis=self.analysis, row=self.row.parent)
        self.animal = Animal(analysis=self.analysis, row=self.row.parent.parent)


class Phase(Entity):
    collection_type = PhaseCollection

    parent_type = Recording
    animal: Animal
    recording: Recording

    def load(self):
        self.recording = Recording(analysis=self.analysis, row=self.row.parent)
        self.animal = Animal(analysis=self.analysis, row=self.row.parent.parent)


schema = [Animal, Recording, Roi, Phase]


def digest(analysis: caload.analysis.Analysis, data_path: Union[str, os.PathLike],
           sync_type: str = None, sync_signal: str = None, sync_signal_time: str = None,
           frame_avg_num: Union[int, Callable] = 1):

    if sync_signal is None:
        sync_signal = 'ai_y_mirror_in'
        sync_signal_time = 'ai_y_mirror_in_time'

    if sync_type is None:
        sync_type = 'y_mirror'

    # Scan for data folders
    folder_list = scan_folder(data_path, [])

    print(f'Process folders: {folder_list}')
    for recording_path in folder_list:

        # recording_path = Path(recording_path).as_posix()
        print(f'Recording folder {recording_path}')

        # Add animal
        animal = get_animal(analysis, recording_path)
        analysis.session.commit()

        print('Calculate frame timing of signal')
        with h5py.File(os.path.join(recording_path, 'Io.hdf5'), 'r') as io_file:

            sync_data = np.squeeze(io_file[sync_signal])[:]
            sync_data_times = np.squeeze(io_file[sync_signal_time])[:]

            # Calculate frame timing
            if sync_type == 'y_mirror':
                frame_idcs_all, frame_times_all = calculate_ca_frame_times_from_y_mirror(sync_data, sync_data_times)
            elif sync_type == 'frame_sync_toggle':
                frame_idcs_all = np.where(np.diff(sync_data) > 0)
                frame_times_all = sync_data_times[frame_idcs_all]
            else:
                raise Exception('Unknown sync type')

            # Interpolate record group IDs to imaging frame time
            try:
                record_group_ids = io_file['__record_group_id'][:].squeeze()
                record_group_ids_time = io_file['__time'][:].squeeze()
            except KeyError as _:
                # For backwards compatibility to pre-2023
                record_group_ids = io_file['record_group_id'][:].squeeze()
                record_group_ids_time = io_file['global_time'][:].squeeze()

            ca_rec_group_id_fun = scipy.interpolate.interp1d(record_group_ids_time, record_group_ids, kind='nearest')

        # Find all layers in suite2p folder
        layers  = []
        for _name in os.listdir(os.path.join(recording_path, 'suite2p')):
            if (not os.path.isdir(os.path.join(recording_path, 'suite2p', _name))
                    or not _name.startswith('plane')):
                continue
            layers.append(_name)

        layer_num = len(layers)
        for layer_str in layers:

            # Get path to plane data
            s2p_path = os.path.join(recording_path, 'suite2p', layer_str)

            # Get plane index
            layer_idx = int(layer_str.replace('plane', ''))

            # Get recording
            recording = get_recording(animal, recording_path, layer_idx)

            print(f'Process {recording}')

            if len(recording.rois) > 0:
                print(f'Recording already exists with {len(recording.rois)} ROIs. Skipping')
                continue

            if isinstance(frame_avg_num, int):
                frame_avg_num_cur = frame_avg_num
            else:
                if not callable(frame_avg_num):
                    raise Exception('frame_avg_num must be int or callable function')

                frame_avg_num_cur = frame_avg_num(animal.id, recording.id)

            # Get frame times for this layer
            frame_times = frame_times_all[int(layer_idx + frame_avg_num_cur // 2)::(layer_num * frame_avg_num_cur)]

            # Load suite2p's analysis options
            print('Include suite2p ops')
            ops = np.load(os.path.join(s2p_path, 'ops.npy'), allow_pickle=True).item()
            unravel_dict(ops, recording, 's2p')

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

            print('Load anatomical registration data')
            roi_coordinates = None
            if 'ants_registration' in os.listdir(os.path.join(recording_path, 'suite2p')):
                # Check for registration data in each registration subfolder for current plane
                for fld in os.listdir(os.path.join(recording_path, 'suite2p', 'ants_registration', layer_str)):
                    registration_path = os.path.join(recording_path, 'suite2p', 'ants_registration', layer_str, fld)

                    # Read coordinates of available
                    if 'mapped_points.h5' in os.listdir(registration_path):
                        roi_coordinates = pd.read_hdf(os.path.join(registration_path, 'mapped_points.h5'),
                                                      key='coordinates')

                        print(f'Found ANTs registration data for  ROI coordinates: {registration_path}')
                        break

            if roi_coordinates is None:
                print('WARNING: no ANTs registration data found')

            # Add suite2p's analysis ROI stats
            print('Add ROI stats and signals')
            rois = recording.add_child_entity(Roi, entity_id=[f'roi_{roi_id}' for roi_id in range(fluorescence.shape[0])])
            for roi_id, roi in tqdm(enumerate(rois)):
                # Create ROI
                # roi = recording.add_child_entity(Roi, entity_id=f'roi_{roi_id}')
                roi['animal_id'] = animal.id
                roi['rec_id'] = recording.id
                roi['roi_id'] = roi_id

                roi_stats = roi_stats_all[roi_id]

                # Write ROI stats
                roi.update({f's2p/{k}': v for k, v in roi_stats.items()})

                # Write ROI coordinates
                if roi_coordinates is not None:
                    coords = roi_coordinates.iloc[roi_id]
                    roi.update({'ants/x': float(coords.x), 'ants/y': float(coords.y), 'ants/z': float(coords.z)})

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

            print(f'Process directory data in {recording_path}')
            for data_fn in os.listdir(recording_path):
                if not any([data_fn.lower().endswith(fn) for fn in ['.h5', 'hdf5']]):
                    continue

                # Get short name for attribute names
                fn_short = data_fn.split('.')[0].lower()
                with h5py.File(os.path.join(recording_path, data_fn), 'r') as h5file:

                    print(f'> {data_fn}')
                    # Get attributes
                    recording.update({f'{fn_short}/attrs/{k}': v for k, v in h5file.attrs.items()})
                    for key1, member1 in tqdm(h5file.items()):
                        # If dataset, write to file
                        if isinstance(member1, h5py.Dataset):
                            recording[f'{fn_short}/{key1}'] = np.squeeze(member1[:])
                            continue
                        # Otherwise it's a group -> keep going
                        # Add phase
                        if 'phase' in key1:
                            phase_id = key1
                            phase_id_int = int(key1.replace('phase', ''))
                            phase = recording.add_child_entity(Phase, entity_id=phase_id)
                            phase['animal_id'] = animal.id
                            phase['rec_id'] = recording.id
                            phase['phase_id'] = phase_id
                            phase['phase_id_int'] = phase_id_int
                            # Add calcium start/end indices
                            in_phase_idcs = np.where(record_group_ids == phase_id_int)[0]
                            start_index = np.argmin(np.abs(frame_times - frame_times[in_phase_idcs[0]]))
                            end_index = np.argmin(np.abs(frame_times - frame_times[in_phase_idcs[-1]]))
                            phase['ca_start_index'] = start_index
                            phase['ca_end_index'] = end_index
                            # Write attributes
                            phase.update({k: v for k, v in member1.attrs.items()})
                            # Write datasets
                            for key2, member2 in member1.items():
                                if isinstance(member2, h5py.Dataset):
                                    phase[key2] = np.squeeze(member2[:])
                        # Add other data
                        else:
                            # Write attributes
                            recording.update({f'{fn_short}/{key1}/{k}': v for k, v in member1.attrs.items()})
                            # Get datasets
                            for key2, member2 in member1.items():
                                if isinstance(member2, h5py.Dataset):
                                    recording[f'{fn_short}/{key1}/{key2}'] = np.squeeze(member2[:])

        # Commit phases and display data
        analysis.session.commit()


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


def get_animal(analysis: caload.analysis.Analysis, path: str) -> Animal:
    # Create animal
    path_parts = Path(path).as_posix().split('/')
    animal_id = path_parts[-2]

    animal_collection = analysis.get(Animal, f'animal_id == "{animal_id}"')

    if len(animal_collection) > 0:
        if len(animal_collection) > 1:
            raise Exception(f'Got {len(animal_collection)} animals with ID {animal_id}')

        return animal_collection[0]

    # Create new animal entity
    print(f'Create new entity for animal {animal_id}')
    animal = analysis.add_entity(Animal, animal_id)
    animal['animal_id'] = animal_id

    # Get path to animal folder
    animal_path = '/'.join(Path(path).as_posix().split('/')[:-1])

    # Search for zstacks
    zstack_names = []
    for fn in os.listdir(animal_path):
        path = os.path.join(path, fn)
        if os.path.isdir(path):
            continue
        if 'zstack' in fn:
            if fn.lower().endswith(('.tif', '.tiff')):
                zstack_names.append(fn)

    # Add first stack that was detected
    if len(zstack_names) > 0:
        if len(zstack_names) > 1:
            print(f'WARNING: multiple zstacks detected, using {zstack_names[0]}')

        print(f'Add zstack {zstack_names[0]}')

        animal['zstack_fn'] = zstack_names[0]
        animal['zstack'] = tifffile.imread(os.path.join(animal_path, zstack_names[0]))

    # Add metadata
    add_metadata(animal, animal_path)

    # Search for valid registration path in animal folder
    valid_reg_path = None
    if 'ants_registration' in os.listdir(animal_path):
        for mov_folder in os.listdir(os.path.join(animal_path, 'ants_registration')):
            for ref_folder in os.listdir(os.path.join(animal_path, 'ants_registration', mov_folder)):
                reg_path = os.path.join(animal_path, 'ants_registration', mov_folder, ref_folder)

                # If there is a transform file, we'll take it
                if 'Composite.h5' in os.listdir(reg_path):
                    valid_reg_path = reg_path
                    break

    # Write registration metadata to animal entity
    if valid_reg_path is not None:
        print(f'Loading ANTs registration metadata at {valid_reg_path}')
        ants_metadata = yaml.safe_load(open(os.path.join(valid_reg_path, 'metadata.yaml'), 'r'))
        animal.update({f'ants/{n}': v for n, v in ants_metadata.items()})

    # Commit animal
    analysis.session.commit()

    return animal


def get_recording(animal: Animal, path: str, layer_idx: int) -> Recording:
    # Create debug folder
    debug_folder_path = os.path.join(path, 'debug')
    if not os.path.exists(debug_folder_path):
        os.mkdir(debug_folder_path)

    # Get recording
    rec_id = f"{Path(path).as_posix().split('/')[-1]}_layer{layer_idx}"
    # expr = f'animal_id == "{animal.id}" AND rec_id == "{rec_id}"'
    recording_collection = animal.analysis.get(Recording, animal_id=animal.id, rec_id=rec_id)
    # Add recording
    if len(recording_collection) > 0:
        if len(recording_collection) > 1:
            raise Exception(f'Got {len(recording_collection)} recordings with '
                            f'animal_id == "{animal.id}" and rec_id == "{rec_id}"')

        return recording_collection[0]

    recording = animal.add_child_entity(Recording, rec_id)
    recording['animal_id'] = animal.id
    recording['rec_id'] = rec_id

    # Add metadata
    add_metadata(recording, path)

    return recording


def add_metadata(entity: Entity, folder_path: str):
    """Function searches for and returns metadata on a given folder path

    Function scans the `folder_path` for metadata yaml files (ending in `meta.yaml`)
    and returns a dictionary containing their contents
    """

    meta_files = [f for f in os.listdir(folder_path) if f.endswith('metadata.yaml')]

    print(f'Found {len(meta_files)} metadata files in {folder_path}.')

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


def calculate_ca_frame_times_from_y_mirror(mirror_position: np.ndarray, mirror_time: np.ndarray):
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


if __name__ == '__main__':
    pass
